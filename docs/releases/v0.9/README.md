# Quasar v0.9.0 — Rotary Position Embedding (RoPE)

## Goal

Replace the simple absolute positional embedding mechanism introduced in v0.8 with a positional mechanism that acts directly on Query and Key vectors and makes their dot product sensitive to **relative position**.

The release answers one question:

> How can attention know how far apart two tokens are without adding a learned position vector such as `P0`, `P1`, `P2`, ... to every token?

## Starting point

By v0.8 Quasar had:

```text
token embedding
      +
absolute position embedding
      ↓
 positioned representation
      ↓
   Wq / Wk / Wv
      ↓
   Q / K / V
      ↓
   attention
```

This makes order visible, but it also requires an explicit table with one row for every supported absolute position.

RoPE takes a different route.

## The core idea: rotate pairs of dimensions

Take a two-dimensional vector:

```text
[x, y]
```

A rotation by angle `θ` is:

```text
x' = x*cos(θ) - y*sin(θ)
y' = x*sin(θ) + y*cos(θ)
```

For example, starting from:

```text
[1, 0]
```

and rotating by one radian gives approximately:

```text
[cos(1), sin(1)]
≈
[0.5403, 0.8415]
```

Quasar applies this operation to every adjacent pair of vector dimensions:

```text
[d0, d1] [d2, d3] [d4, d5] ...
```

This is why the v0.9 educational implementation requires an even vector width.

## Position becomes the rotation angle

For each pair Quasar computes an inverse frequency. The pair at index `i` uses:

```text
frequency_i = 1 / base^(2*i/D)
```

with:

```text
base = 10000
D    = vector dimensions
```

The angle is:

```text
angle = position * frequency_i
```

So position changes the orientation of the vector rather than adding another vector to it.

Different dimension pairs rotate at different speeds. In a 4-dimensional example:

```text
pair 0 -> frequency 1
pair 1 -> frequency 0.01
```

The fast pair changes strongly over short distances; the slow pair changes over longer distances. Larger real models use many such pairs, giving attention multiple positional scales.

## Example: `the moon shines`

Consider the final token:

```text
the moon shines
 0    1      2
```

Suppose, only to make the arithmetic visible, that projected Query and Key vectors are two-dimensional and both start as:

```text
[1, 0]
```

For `shines`, the Query is at position `2`, so RoPE rotates it by `2` radians:

```text
Q2 = [cos(2), sin(2)]
```

For `moon`, the Key is at position `1`:

```text
K1 = [cos(1), sin(1)]
```

Their dot product is:

```text
Q2 · K1
=
cos(2)cos(1) + sin(2)sin(1)
=
cos(2 - 1)
=
cos(1)
≈ 0.5403
```

The interesting part is not the number itself. It is that the result depends on:

```text
2 - 1 = 1
```

—the **relative distance** between the Query and Key.

## Shift both tokens and the relationship stays the same

Move both hypothetical positions forward by five:

```text
Query: 2 -> 7
Key:   1 -> 6
```

Now:

```text
Q7 · K6
=
cos(7 - 6)
=
cos(1)
```

The absolute positions changed, but their relative distance did not.

Quasar has a unit test that checks this property numerically with 4-dimensional vectors: shifting both Q and K by the same amount preserves their rotated dot product within floating-point tolerance.

This is the key theoretical reason RoPE is useful in attention.

## Where RoPE sits in the attention pipeline

v0.9 uses:

```text
embedding
   ↓
  Wq
   ↓
raw Query
   ↓
RoPE(query position)
   ↓
rotated Query
```

and:

```text
embedding
   ↓
  Wk
   ↓
raw Key
   ↓
RoPE(key position)
   ↓
rotated Key
```

Then attention performs the familiar calculation:

```text
score = rotatedQ · rotatedK / sqrt(D)
```

followed by softmax.

### Values are deliberately not rotated

The Value path remains:

```text
embedding -> Wv -> Value
```

RoPE is used to change **how Query and Key compare**. The Value is the payload transferred after attention has decided how much weight a token receives.

This makes the conceptual split explicit:

```text
Q + K + RoPE -> where/how strongly to attend
V            -> what information to carry
```

## Full v0.9 attention path

```text
                         ┌─ Wq ─> Q ─> RoPE(position) ─┐
embedding(token) ────────┼─ Wk ─> K ─> RoPE(position) ─┼─> Q·K / sqrt(D)
                         └─ Wv ─> V ───────────────────┘
                                                     ↓
                                                  softmax
                                                     ↓
                                              attention weights
                                                     ↓
                                               weighted V sum
```

## Comparison with v0.8 absolute positional embeddings

### v0.8

```text
token embedding + position embedding
```

Properties:

- easy to understand
- fixed vector width
- requires a table of `maxPositions × D` values
- explicitly represents absolute positions

### v0.9

```text
projected Q/K -> position-dependent rotation
```

Properties:

- no learned absolute position table
- no `maxPositions × D` positional-weight buffer in this implementation
- position affects Q/K similarity directly
- naturally exposes relative-position structure in the Q/K dot product

RoPE does not make context length infinite. Real models still have practical and training-time context limits, and production systems use additional scaling schemes for long-context extension. v0.9 simply removes the specific fixed absolute-position table used in v0.8.

## Implementation

`position.Rotary` precomputes only one inverse frequency per dimension pair:

```text
D / 2 float64 frequencies
```

At runtime `ApplyInto` computes sine/cosine values and rotates each pair directly into a caller-provided buffer. Input and output may be the same slice, so in-place rotation is safe.

`attention.RotaryProjected` reuses scratch buffers in the last-token path:

1. project the final token through `Wq`
2. rotate the Query in place at its position
3. for each context token, project `Wk` into one key scratch buffer
4. rotate that key in place and compute its attention score
5. softmax the scores
6. project each `Wv` into one value scratch buffer
7. accumulate the weighted Value into the output

The implementation therefore does not materialize an `N × D` matrix of rotated Keys or Values for this educational last-token path.

## CLI

```bash
go run ./cmd/quasar \
  -corpus examples/corpus.txt \
  -attention "the moon shines" \
  -rope-attention \
  -dimensions 4
```

The output exposes:

- the rotated Query for `shines`
- the rotated Key for every token
- the unrotated projected Value for every token
- scaled attention scores
- softmax weights
- final weighted Value output

The numbers are deterministic but still **untrained**. They demonstrate the geometry and data flow, not learned linguistic relevance.

## Performance notes

v0.9 does not claim a performance optimization. It establishes a measurable implementation shape.

Current costs for one last-token query over `N` tokens include:

- one `Wq` projection
- `N` `Wk` projections
- `N` Q/K dot products
- `N` `Wv` projections
- RoPE sine/cosine work for Q/K

Future inference-oriented work can avoid repeating many of these operations by caching previously computed Keys and Values. That future optimization is the reason the **KV cache** will eventually become important.

## Deliberate limitations

v0.9 still does not include:

- trained token embeddings or trained Wq/Wk/Wv for the attention demos
- training/backpropagation through attention
- multi-head attention
- causal full-sequence attention
- KV cache
- model-specific RoPE bases or scaling schemes
- partial rotary dimensions
- precomputed/cached sine-cosine tables
- Transformer residual paths, normalization, or feed-forward layers

Keeping these out is intentional: this release teaches one positional mechanism cleanly.

## Connection to real inference engines

When code in a production inference engine mentions RoPE, rotary dimensions, position, frequency, or Q/K rotation, the operation is no longer abstract:

```text
project Q/K
   ↓
pair dimensions
   ↓
rotate each pair according to token position
   ↓
compute attention similarity
```

The production version may fuse this into GPU kernels, apply it only to part of each head, use model-specific scaling, or operate on cached Keys. The mathematical reason for the operation is the same one demonstrated here.
