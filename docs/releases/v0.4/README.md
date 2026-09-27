# Quasar v0.4.0 — more than one token of context

## Goal

Move beyond predicting the next token from only the immediately previous token. v0.4.0 introduces a fixed context window so Quasar can use multiple previous tokens in one prediction.

The release answers one question: **how can several token embeddings be combined into one representation that influences the next-token prediction?**

## Concepts introduced

1. **Context window** — the set of previous tokens used for one next-token prediction.
2. **Sliding window** — after Quasar generates a token, the oldest context token is dropped and the new token is appended.
3. **Context aggregation** — v0.4 averages the embeddings in the window into one fixed-width vector.
4. **Shared gradient** — because the context vector is an average, the gradient flowing back from the prediction is divided equally among the token occurrences in the window.
5. **Order invariance** — averaging loses token order. `[A,B]` and `[B,A]` produce the same context vector and therefore the same logits.

## From one token to a context

v0.3 used one embedding directly:

`A -> embedding(A) -> logits -> next token`

v0.4 with context size 2 uses:

`A B -> mean(embedding(A), embedding(B)) -> logits -> next token`

For two-dimensional embeddings:

```text
embedding(A) = [0.2, 0.6]
embedding(B) = [0.8, 0.0]
```

Quasar computes:

```text
context = [(0.2 + 0.8) / 2, (0.6 + 0.0) / 2]
        = [0.5, 0.3]
```

That context vector is then compared with every output row using the same dot-product logits introduced in v0.3.

## Sliding generation

With context size 2 and seed:

```text
[A, B]
```

if Quasar predicts `C`, the next context becomes:

```text
[B, C]
```

If it then predicts `D`, the next context becomes:

```text
[C, D]
```

The context therefore slides forward one token at a time during autoregressive generation.

## Training

Training still uses the v0.3 pipeline:

`logits -> softmax -> cross-entropy -> gradients -> SGD`

The difference is the input to that pipeline. Instead of one token embedding, Quasar first builds a context vector by averaging all embeddings in the window.

If the context contains `N` token occurrences, each occurrence receives `1/N` of the gradient with respect to the combined context vector. When the same token appears more than once, its shared embedding accumulates the contribution from every occurrence.

## The intentional limitation: order disappears

Mean pooling is commutative:

```text
(A + B) / 2 == (B + A) / 2
```

Therefore Quasar v0.4 cannot distinguish:

```text
[A, B]
```

from:

```text
[B, A]
```

This is not hidden. A unit test explicitly verifies that both contexts produce exactly the same scores.

Natural language depends heavily on order, so this representation is fundamentally insufficient. The limitation gives Quasar the next concrete problem to solve: **how can a model know where each token appears in the sequence?**

That question will motivate positional information before attention is introduced.

## Try it

```bash
go run ./cmd/quasar \
  -corpus examples/corpus.txt \
  -prompt "the moon" \
  -tokens 4 \
  -neural \
  -context-size 2 \
  -dimensions 8 \
  -epochs 120 \
  -learning-rate 0.1
```

The prompt must contain at least as many tokens as the selected context size.

## What v0.4.0 deliberately does not do

- preserve token order inside the context representation
- positional embeddings or RoPE
- variable attention weights between context tokens
- self-attention
- Transformer blocks
- batching
- persisted model weights
- GPU acceleration

## Relation to a real inference engine

Real Transformer LLMs also operate on multiple previous tokens, but they do not collapse the context into an unordered average. They preserve separate token representations, encode position, and use attention to let each token interact with the others.

Quasar v0.4 isolates only the first problem: moving from a single previous token to a multi-token context. Its deliberate failure to preserve order creates the reason for the next architectural step.

## Validation

Tests verify that:

- training loss decreases
- a two-token context can learn a next-token prediction
- generation slides the context window correctly
- mean pooling is exactly order-invariant
- invalid context sizes are rejected
- the v0.1 and v0.3 CLI paths continue to work

CI runs formatting checks, `go vet ./...`, and `go test ./...`.

## Performance notes

The model reuses training buffers for context vectors, logits, probabilities, and gradients across examples. Context construction is O(context size × embedding dimensions), while output scoring remains O(vocabulary size × embedding dimensions).

No optimization claim is made yet. The implementation stays explicit so the cost of each stage remains visible before future profiling and acceleration work.
