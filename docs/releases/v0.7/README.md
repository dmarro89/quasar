# Quasar v0.7.0 — Projected Q, K, and V

## Goal

v0.6 introduced the attention algorithm with one deliberate simplification:

```text
Q = embedding
K = embedding
V = embedding
```

That made scaled dot-product attention easy to inspect, but a real Transformer gives the same token three different learned roles.

v0.7 introduces separate linear projection matrices:

```text
embedding -> Wq -> Query
embedding -> Wk -> Key
embedding -> Wv -> Value
```

The matrices are deterministic and **not trained yet** in this release. The goal is to understand the projection operation and the distinct Q/K/V spaces before adding backpropagation through attention.

## Running the example

```bash
go run ./cmd/quasar \
  -corpus examples/corpus.txt \
  -attention "the moon shines" \
  -projected-attention \
  -dimensions 3
```

The command prints:

- the projected Query for the final token `shines`
- the projected Key and Value for every token
- the scaled Q·K score for every token
- the softmax attention weight for every token
- the final weighted Value output

The old v0.6 identity-QKV mode remains available by omitting `-projected-attention`.

## 1. What is a linear projection?

Take a tiny embedding:

```text
shines = [1, 2]
```

and a 2×2 matrix:

```text
Wq =
[ 1.0  0.5 ]
[ 0.0 -1.0 ]
```

Quasar computes one dot product per matrix row:

```text
Q[0] = 1.0×1 + 0.5×2 = 2.0
Q[1] = 0.0×1 - 1.0×2 = -2.0
```

so:

```text
Q(shines) = [2, -2]
```

This is a matrix-vector multiplication. A projection does not look up a different token. It transforms the numeric representation of the same token into a different coordinate space.

For an input width `D` and output width `P`, a projection matrix contains:

```text
P × D weights
```

In v0.7 Q, K, and V all keep the same width `D`, so each matrix contains `D × D` weights.

## 2. Why three matrices?

The three roles answer different questions.

### Query

```text
What information am I looking for right now?
```

For next-token decode in our example, the final token `shines` produces the Query.

### Key

```text
What kind of queries should consider me relevant?
```

Every token exposes a Key. The Query is compared with those Keys.

### Value

```text
What information should I contribute if I receive attention weight?
```

Values are not used to decide relevance. They are the payload mixed after relevance is known.

Therefore there is no requirement that:

```text
Q == K == V
```

In a trained Transformer, `Wq`, `Wk`, and `Wv` learn different transformations because the three roles have different jobs.

## 3. Example with `the moon shines`

Assume simple two-dimensional embeddings:

```text
the    = [1, 0]
moon   = [0, 1]
shines = [1, 1]
```

For illustration, choose these matrices:

```text
Wq = identity

Wk =
[ 1  0 ]
[ 0 -1 ]

Wv =
[ 2  0 ]
[ 0  3 ]
```

The final token becomes:

```text
Q(shines) = [1, 1]
```

The Keys become:

```text
K(the)    = [ 1,  0]
K(moon)   = [ 0, -1]
K(shines) = [ 1, -1]
```

and Values become:

```text
V(the)    = [2, 0]
V(moon)   = [0, 3]
V(shines) = [2, 3]
```

Now the familiar attention calculation continues unchanged.

### Scores

```text
Q · K(the)    =  1
Q · K(moon)   = -1
Q · K(shines) =  0
```

With dimension `d = 2`, divide by `sqrt(2)`:

```text
the    ≈  0.707
moon   ≈ -0.707
shines =  0
```

### Weights

Softmax converts those scores into positive weights summing to 1.

### Output

The final attention vector is still:

```text
weight(the)    × V(the)
+ weight(moon) × V(moon)
+ weight(shines) × V(shines)
```

The important change from v0.6 is not the softmax formula. It is that relevance is measured in **Key space** and transferred information comes from **Value space**.

## 4. Parameters versus activations

This release introduces another useful distinction.

The matrices:

```text
Wq
Wk
Wv
```

are **parameters**. In a trained model they are persistent weights stored in the model file.

The vectors:

```text
Q(the)
K(the)
V(the)
Q(moon)
...
```

are **activations**. They are temporary values produced by applying those parameters to the current input.

This distinction becomes important for both memory accounting and inference optimization.

## 5. Memory layout

`projection.Linear` stores its matrix as one contiguous `[]float32` in row-major order:

```text
row 0 | row 1 | row 2 | ...
```

For each output element Quasar computes a dot product between one matrix row and the input vector.

The hot-path API also exposes `ForwardInto`, allowing callers to reuse output buffers instead of allocating a new vector on every projection.

Projected attention reuses a `keyScratch` and a `valueScratch` buffer. For the final-token query it does not need to retain a separately allocated K and V vector for every token just to compute the result.

## 6. Computational cost

For width `D`, one dense `D × D` projection costs approximately:

```text
D² multiply-add work
```

For one final-token attention over `N` context tokens, v0.7 performs roughly:

```text
1 query projection
+ N key projections
+ N value projections
+ N query-key dot products
+ N weighted value accumulations
```

This is already enough to see why matrix multiplication and memory bandwidth become dominant concerns in real inference engines.

## 7. What is still deliberately missing?

The projection matrices are initialized but not trained. Therefore the vectors printed by the CLI are structurally correct but do not yet encode useful learned Q/K/V semantics.

Position is also still unresolved. Reordering previous tokens can still leave attention unable to distinguish sequence order because no positional information has been injected.

v0.7 therefore has:

```text
linear projection primitive   yes
separate Wq/Wk/Wv             yes
scaled dot-product attention  yes
softmax attention weights     yes
trained Q/K/V matrices        no
positional information/RoPE   no
causal full-sequence mask     no
multi-head attention          no
Transformer block             no
```

## Relation to a real Transformer

We can now refine the v0.6 formula.

Given an input representation `X`:

```text
Q = X Wq
K = X Wk
V = X Wv
```

then:

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
```

Depending on matrix convention, code may write `W × x` rather than `x × W`; the underlying operation is the same linear transformation.

The next major unresolved issue is now very concrete: **none of these projections knows where a token appears in the sequence**. That gives Quasar a natural reason to introduce positional information before building a complete Transformer block.
