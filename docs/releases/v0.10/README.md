# Quasar v0.10.0 — KV cache and incremental decode

## Goal

Avoid recomputing Keys and Values for tokens that have already been processed.

The release answers one question:

> During autoregressive generation, why should an inference engine recompute K and V for `the`, `moon`, and `shines` when those tokens have not changed?

v0.10 turns the v0.9 RoPE attention path into an incremental decode path backed by a preallocated KV cache.

## Starting point: v0.9 repeats work

Take:

```text
the moon shines
```

To evaluate attention for `shines`, v0.9 computes:

```text
Q(shines)
K(the)     V(the)
K(moon)    V(moon)
K(shines)  V(shines)
```

Suppose the next token is:

```text
at
```

Without a cache, evaluating the new last token over:

```text
the moon shines at
```

computes again:

```text
Q(at)
K(the)     V(the)       <- repeated
K(moon)    V(moon)      <- repeated
K(shines)  V(shines)    <- repeated
K(at)      V(at)        <- new
```

If the next token is `night`, the old K/V projections are repeated yet again.

The earlier tokens are immutable during decode, so their projected Keys and Values can be reused.

## The KV cache

After processing `the moon shines`, v0.10 stores:

```text
K cache                       V cache

K(the@0)                      V(the)
K(moon@1)                     V(moon)
K(shines@2)                   V(shines)
```

The cached Keys are already RoPE-rotated at their sequence positions. Values are projected through `Wv` but are not rotated, matching the v0.9 attention path.

When `at` arrives, Quasar computes only:

```text
Q(at@3)
K(at@3)
V(at)
```

then appends the new K/V pair:

```text
K cache                       V cache

K(the@0)                      V(the)
K(moon@1)                     V(moon)
K(shines@2)                   V(shines)
K(at@3)                       V(at)
```

The new Query is compared with all cached Keys, softmax produces attention weights, and the output is the weighted sum of all cached Values.

## Step-by-step example: `the moon shines at night`

v0.10 exposes an incremental `Step` API.

### Step 0 — `the`

```text
position = 0
compute Q(the), K(the), V(the)
apply RoPE to Q and K
cache K(the), V(the)
attention sees 1 cached token
```

Cache length: `1`.

### Step 1 — `moon`

```text
position = 1
compute Q(moon), K(moon), V(moon)
apply RoPE to Q and K
cache K(moon), V(moon)
```

Attention for `moon` uses:

```text
Q(moon@1)
    |
    +-- K(the@0)
    +-- K(moon@1)
```

and then combines:

```text
V(the)
V(moon)
```

Cache length: `2`.

### Step 2 — `shines`

Only Q/K/V for `shines` are newly projected.

The previous K/V rows are reused:

```text
Q(shines@2)
    |
    +-- cached K(the@0)
    +-- cached K(moon@1)
    +-- new    K(shines@2)
```

Cache length: `3`.

### Step 3 — `at`

Again, only `at` gets fresh Q/K/V work.

Cache length: `4`.

### Step 4 — `night`

Only `night` gets fresh Q/K/V work.

Cache length: `5`.

This is the core decode behavior used by autoregressive Transformer inference engines.

## Prefill vs decode

The KV cache makes an important inference distinction visible.

### Prefill

The prompt already exists:

```text
the moon shines
```

An inference engine processes that prompt and fills cache rows for its tokens.

Conceptually:

```text
prompt tokens
    |
    v
compute K/V
    |
    v
KV cache
```

Production engines usually process prompt tokens in larger parallel operations rather than literally calling a one-token loop like Quasar does. v0.10 keeps the sequential implementation because it makes cache behavior obvious.

### Decode

After prefill, generation produces one new token at a time:

```text
at
night
...
```

Each decode step:

1. computes Q/K/V for the new token
2. applies RoPE to its Q and K
3. appends its K and V to the cache
4. compares the new Q with all cached Keys
5. combines all cached Values

The cache grows by one row per generated token.

## Correctness: cached and uncached attention must agree

The optimization must not change the mathematics.

Quasar tests every prefix of a small English example by comparing:

```text
cached.Step(newToken)
```

with the v0.9 implementation applied from scratch to the full prefix:

```text
uncached.LastToken(allTokensSoFar)
```

For each prefix the test compares:

- rotated Query
- scaled attention scores
- softmax weights
- final attention output

within floating-point tolerance.

This establishes an important engineering rule:

> cache reuse is an optimization, not a model-semantic change.

## What is stored?

For each token Quasar stores exactly two `D`-wide float32 rows:

```text
rotated K : D float32 values
projected V: D float32 values
```

The Query is **not cached** because during last-token decode it is needed only for the current token.

This is why the structure is called a **KV cache**, not a QKV cache.

## Memory layout

`KVCache` owns two contiguous buffers:

```text
keys:
[token0 D values][token1 D values][token2 D values]...

values:
[token0 D values][token1 D values][token2 D values]...
```

The cache is allocated once for a fixed capacity. `Append` copies one new K/V pair into the next rows. `Key(i)` and `Value(i)` return zero-copy slices over those rows.

`Reset()` only sets the logical length back to zero. It does not clear or reallocate the backing arrays; later appends overwrite the old rows.

## Memory cost

For Quasar's single-head float32 cache:

```text
bytes = 2 * capacity * D * 4
```

The `2` is for K and V.

For five tokens and `D=4`:

```text
2 * 5 * 4 * 4 = 160 bytes
```

For an illustrative larger single-head width of `D=4096` and 4096 cached positions:

```text
2 * 4096 * 4096 * 4
= 134,217,728 bytes
≈ 128 MiB
```

Real Transformer cache size is architecture-dependent because cache rows are normally organized per layer and per KV head. A useful conceptual scaling relation is:

```text
KV memory
≈ 2 * layers * sequenceLength * kvHeads * headDimension * bytesPerElement
```

This is why context length, KV-head count, data type, and cache quantization matter so much in production inference.

## Compute saved during decode

Let `N` be the number of tokens already in the cache.

Without K/V caching, a simple last-token implementation like v0.9 performs approximately:

```text
1 Q projection
N K projections
N V projections
N attention dot products
```

for every decode step.

With the v0.10 cache, the new step performs:

```text
1 Q projection
1 K projection
1 V projection
N attention dot products
```

The attention comparison against previous Keys still grows with context length; the cache does **not** make attention O(1). It removes the repeated K/V projection work for old tokens.

This distinction is important: KV caching saves recomputation but trades it for memory usage.

## Implementation

`attention.KVCache` provides the storage primitive.

Properties:

- fixed token capacity
- contiguous float32 Key buffer
- contiguous float32 Value buffer
- no per-token allocation after construction
- zero-copy row access
- O(1) logical reset
- explicit reserved-byte reporting

`attention.CachedRotary` owns:

```text
RotaryProjected
KVCache
query scratch
key scratch
value scratch
score buffer
weight buffer
output buffer
```

`Step(input)` uses the current cache length as the new token's absolute RoPE position.

The hot path is:

```text
new embedding
    |
    +-> Wq -> query scratch -> RoPE(position)
    +-> Wk -> key scratch   -> RoPE(position) -> cache append
    +-> Wv -> value scratch                  -> cache append

query
  |
  +-> dot cached K rows
  +-> softmax
  +-> weighted cached V rows
  v
output
```

The result returned by `Step` exposes read-only views into reusable internal scratch buffers. They are valid only until the next `Step` or `Reset`; this avoids allocating result vectors on every decode step.

## CLI

```bash
go run ./cmd/quasar \
  -corpus examples/corpus.txt \
  -attention "the moon shines at night" \
  -cached-rope-attention \
  -dimensions 4
```

The CLI exposes every incremental step and the cache state so the optimization is visible rather than hidden behind an abstraction.

## Deliberate limitations

v0.10 still does not include:

- trained attention parameters
- backpropagation through attention
- multi-head attention
- batched prefill kernels
- cache paging or eviction
- sliding-window attention
- KV quantization
- grouped-query or multi-query attention
- multiple Transformer layers
- residual connections
- normalization
- feed-forward networks

The cache is intentionally fixed-capacity and single-head so its purpose and memory layout remain obvious.

## Connection to real inference engines

When a production engine mentions:

```text
prefill
KV cache
past key/value
cache position
decode token
```

the underlying reason is now concrete.

During autoregressive decode, old tokens do not change. Their K/V representations are reusable state. The inference engine spends memory to retain that state so it does not repeatedly recompute it.

Quasar v0.10 is the first release where the code is not only reproducing Transformer mathematics; it is also reproducing a fundamental **inference-engine optimization**.
