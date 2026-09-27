# Quasar v0.8.0 — absolute positional information

## Goal

Teach Quasar that a token's identity is not enough: the model must also know **where that token appears in the sequence**.

v0.7 already had distinct Query, Key, and Value projections, but the same token embedding always produced the same Q/K/V inputs regardless of position. v0.8 adds a deliberately simple absolute positional embedding before those projections.

## The problem

Consider:

```text
the moon shines
```

and:

```text
moon the shines
```

Without positional information, the context contains the same token vectors. The final query token is still `shines`, and the attention mechanism has no explicit fact saying that `the` moved from position 0 to position 1 while `moon` moved in the opposite direction.

Attention answers **how much** one token should use another token. Positional information answers **where** each token is located. These are different problems.

## The v0.8 representation

For token position `i`, Quasar computes:

```text
x_i = token_embedding_i + position_embedding_i
```

Both vectors have width `D`, so the result also has width `D`.

For example, with two-dimensional vectors:

```text
token embedding(the) = [0.2, 0.6]
position(0)           = [0.1, 0.3]
```

then:

```text
the@0 = [0.3, 0.9]
```

If the same token appeared at position 1:

```text
position(1) = [-0.2, 0.4]
```

then:

```text
the@1 = [0.0, 1.0]
```

The token is still `the`, but its numerical representation now carries different positional information.

## Why addition does not recreate the v0.4 bug

v0.4 averaged **different token embeddings together**:

```text
mean(the, moon)
```

Because addition is commutative, swapping the tokens produced the same pooled vector.

v0.8 does something different. It combines token identity and position **inside each token representation**:

```text
the@0  = embedding(the)  + position(0)
moon@1 = embedding(moon) + position(1)
```

and keeps those token representations separate when attention processes them.

After swapping the words:

```text
moon@0 = embedding(moon) + position(0)
the@1  = embedding(the)  + position(1)
```

The position vectors are now attached to different token embeddings, so the Q/K/V projections and attention scores can change.

## Data flow

v0.8 extends the v0.7 path like this:

```text
token embedding
      +
positional embedding
      |
      v
positioned token
      |
  +---+---+
  |   |   |
 Wq  Wk  Wv
  |   |   |
  Q   K   V
   \  |  /
 scaled dot-product
      |
   softmax
      |
 weighted V sum
      |
 attention output
```

The scaled dot-product attention algorithm itself is unchanged.

## Absolute positional embeddings

The new `position.Table` stores one vector per supported absolute position:

```text
position 0 -> [ ... D floats ... ]
position 1 -> [ ... D floats ... ]
position 2 -> [ ... D floats ... ]
...
```

This is structurally similar to a token embedding table, except the lookup key is an integer sequence position rather than a token ID.

The table is deterministic but untrained in v0.8. A trainable model could later update these values through backpropagation just like other model parameters.

## Parameters and memory

If the maximum supported context length is `N` and embedding width is `D`, the positional table contains:

```text
N × D float32 values
```

For example:

```text
N = 128
D = 64

128 × 64 = 8192 float32 values
```

The storage therefore grows linearly with maximum context length.

Quasar stores the table in one contiguous `[]float32` backing buffer.

## Hot-path implementation

The attention path does not materialize a full `N × D` copy of every positioned token.

Instead it reuses one `positionedScratch` buffer:

```text
positionedScratch = token + position
keyScratch        = Wk(positionedScratch)
```

for the score pass, then reuses the same positioned buffer and a `valueScratch` buffer during the weighted-value pass.

This keeps the implementation educational while avoiding an unnecessary second copy of the whole context.

## Validation

v0.8 tests two important properties:

1. the same token gets a different representation at different positions;
2. with controlled Q/K/V and positional weights, `the moon shines` and `moon the shines` produce different attention outputs.

The second test is the behavioral proof that order has entered the attention calculation.

## CLI inspection

Run:

```bash
go run ./cmd/quasar \
  -corpus examples/corpus.txt \
  -attention "the moon shines" \
  -positioned-attention \
  -dimensions 3
```

For every token the CLI prints:

```text
position
original token embedding
positional embedding
combined token+position vector
attention score
attention weight
```

It also prints the projected Query for `shines` and the final attention output.

## Deliberate limitations

v0.8 does not yet implement:

- trained positional embeddings
- relative position information
- RoPE
- causal attention over every sequence position
- multi-head attention
- training through Q/K/V attention
- a complete Transformer block

Absolute positional embeddings also require a configured maximum number of positions and identify positions by fixed absolute indices. These limitations will motivate a more modern positional mechanism later.

## Relation to real Transformers

Early Transformer architectures commonly added positional information to token embeddings before the attention stack. Many modern decoder-only LLMs instead use mechanisms such as Rotary Position Embeddings (RoPE), which encode position into Query and Key geometry rather than simply adding a position vector to the token representation.

Quasar intentionally learns the simpler absolute approach first. Once the need for position and its effect on Q/K/V are concrete, RoPE becomes a solution to a known problem rather than an unexplained formula.
