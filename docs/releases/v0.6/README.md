# Quasar v0.6.0 — Last-token self-attention

## Goal

v0.5 preserved order by concatenating one embedding per context position. That solved one problem but created another: the context vector grows as `embedding dimensions × context length`.

v0.6 introduces the core attention operation so a variable number of context tokens can be combined into one fixed-width vector while giving each token a different, content-dependent weight.

This release intentionally implements only **one query: the final token in the context**. That matches the piece of attention needed when reasoning about next-token decode, without introducing the full Transformer at once.

## Running the example

```bash
go run ./cmd/quasar \
  -corpus examples/corpus.txt \
  -attention "the moon shines" \
  -dimensions 3
```

The command prints:

- the final token used as the query (`shines`)
- one raw attention score for each token
- one softmax attention weight for each token
- the final weighted output vector

The vectors are deterministically initialized but **untrained**. The numerical attention calculation is real, but the weights do not yet carry learned semantic meaning.

## The concrete example

Consider:

```text
the moon shines
```

Assume tiny two-dimensional embeddings:

```text
the    = [1.0, 0.0]
moon   = [0.0, 1.0]
shines = [1.0, 1.0]
```

In v0.6 we deliberately use identity Q/K/V:

```text
query  = embedding(shines)
keys   = [embedding(the), embedding(moon), embedding(shines)]
values = [embedding(the), embedding(moon), embedding(shines)]
```

A real Transformer usually creates Q, K, and V with learned linear projections. Quasar does not do that yet because this release isolates the attention algorithm itself.

## 1. Query, Key, Value

The three names describe three roles.

### Query

The query represents what the current token is looking for.

For the final token:

```text
Q = shines = [1.0, 1.0]
```

### Keys

Each context token exposes a key that can be compared with the query:

```text
K(the)    = [1.0, 0.0]
K(moon)   = [0.0, 1.0]
K(shines) = [1.0, 1.0]
```

### Values

Values are the information that will actually be mixed together after relevance has been measured.

In v0.6 values are the same embeddings:

```text
V(the)    = [1.0, 0.0]
V(moon)   = [0.0, 1.0]
V(shines) = [1.0, 1.0]
```

## 2. Scaled dot-product scores

Attention compares the query with every key using a dot product:

```text
Q · K(the)    = 1
Q · K(moon)   = 1
Q · K(shines) = 2
```

Then each score is divided by:

```text
sqrt(d)
```

where `d` is the vector dimension.

For `d = 2`:

```text
sqrt(2) ≈ 1.414
```

so the scaled scores are approximately:

```text
the    = 0.707
moon   = 0.707
shines = 1.414
```

The scale matters because dot products tend to become larger as vector dimension grows. Very large logits make softmax extremely sharp and numerically harder to train. `1/sqrt(d)` keeps score magnitudes under better control.

The formula is:

```text
score_i = (Q · K_i) / sqrt(d)
```

## 3. Softmax produces attention weights

The scores are passed through the same softmax idea already introduced during neural training:

```text
scores -> softmax -> weights
```

The important properties are:

```text
each weight >= 0
sum(weights) = 1
```

For the example the final token receives a larger weight because its key has the largest dot product with the query.

The implementation subtracts the maximum score before exponentiation, just as Quasar already does for token probabilities, to keep the computation numerically stable.

## 4. Weighted sum of values

Attention does not simply choose one token. It blends the value vectors:

```text
output =
    weight(the)    * V(the)
  + weight(moon)   * V(moon)
  + weight(shines) * V(shines)
```

The result still has dimension `d`.

This is the major structural difference from v0.5 concatenation.

### v0.5

With embedding dimension `D` and context length `N`:

```text
context width = D × N
```

### v0.6 attention

```text
attention output width = D
```

The representation width no longer grows with the number of context tokens.

## Dynamic weighting

Mean pooling from v0.4 gave every token exactly the same importance:

```text
1/N, 1/N, 1/N, ...
```

Attention instead computes weights from the current query and keys:

```text
query changes
    -> scores change
    -> weights change
    -> context output changes
```

That is the central idea of attention: relevance is computed dynamically from the current content rather than being fixed in advance.

## Performance view

For one final-token query with `N` context vectors of width `D`, this implementation performs roughly:

```text
N dot products of D values
+
N weighted additions of D values
```

so the core work grows approximately with `N × D` for this single query.

The implementation also exposes `LastTokenInto`, which accepts reusable buffers for scores, weights, and output. That avoids forcing new allocations on every future decode step.

A full Transformer prefill computes attention for many query positions, which introduces the familiar quadratic relationship with sequence length. Quasar deliberately does not implement that full matrix yet.

## What self-attention means

It is called **self-attention** because the query, keys, and values all come from the same input sequence:

```text
the moon shines
```

rather than the query attending to a separate external sequence.

## Important limitation: position is still unresolved

Attention does not automatically know token order.

If two previous tokens are reordered while the final query remains the same, identity Q/K/V attention has the same set of key/value vectors and can produce the same weighted sum.

So v0.6 solves:

```text
dynamic relevance          yes
fixed-width context output yes
learned Q/K/V projections  no
positional information     no
multi-head attention       no
Transformer block          no
```

This separation is intentional. Position and learned projections are distinct concepts and will be introduced independently rather than hidden inside one large implementation.

## Relation to a real Transformer

The core formula now visible in Quasar is the same scaled dot-product attention idea used by Transformers:

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
```

Quasar v0.6 computes only the final row of that idea and uses the embeddings directly as Q, K, and V.

That is enough to make the formula concrete before adding the learned projection matrices, positional mechanisms such as RoPE, causal masking for all positions, multiple heads, residual connections, and feed-forward layers.
