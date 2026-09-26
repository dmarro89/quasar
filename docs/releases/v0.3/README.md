# Quasar v0.3.0 — weights learn from data

## Goal

Turn the untrained embedding vectors introduced in v0.2 into trainable model weights and use them to predict the next token.

The release answers one question: **how can Quasar measure a wrong prediction and change numeric weights so the next prediction becomes better?**

## Concepts introduced

1. **Neural bigram** — the model still sees only one previous token, but prediction now comes from trainable numeric weights rather than transition counts.
2. **Logit** — one raw score for every possible next token.
3. **Softmax** — converts logits into a probability distribution.
4. **Cross-entropy loss** — measures how much probability the model assigned to the correct next token.
5. **Gradient** — describes how each weight should change to reduce the loss.
6. **SGD** — stochastic gradient descent applies a small update to the weights after each training pair.
7. **Epoch** — one complete pass over all adjacent token pairs in the training corpus.

## Model

For a current token `A` and candidate next token `B`, Quasar computes:

`logit(B) = dot(input_embedding(A), output_weight(B))`

If the vocabulary contains `V` tokens, the model produces `V` logits.

Softmax turns those logits into probabilities. Cross-entropy then looks at the probability assigned to the known correct next token from the corpus.

Example:

```text
current token: moon
correct next token: shines

raw logits:
shines  1.4
moves   0.9
sun    -0.2

softmax:
shines  0.56
moves   0.34
sun     0.10

loss = -log(0.56)
```

Training changes both the current token's input embedding and the output weight vectors so the correct token receives a higher score in future passes.

## Training loop

For each adjacent pair in the corpus:

```text
current TokenID
    |
    v
input embedding
    |
    +---- dot with every output row ----> logits
                                         |
                                         v
                                      softmax
                                         |
                                         v
                                  probability of target
                                         |
                                         v
                                        loss
                                         |
                                         v
                                      gradients
                                         |
                                         v
                                    SGD updates
```

The generated token is still selected with deterministic argmax, so repeated runs with the same corpus and configuration remain reproducible.

## Why keep a one-token context?

The v0.1 bigram model already established the one-token prediction problem. v0.3 deliberately keeps that same limitation so only the mechanism producing the scores changes:

```text
v0.1: token -> transition counts -> argmax
v0.3: token -> embedding -> dot products -> logits -> argmax
```

This isolates neural training from future complexity such as multi-token context and attention.

## Memory and CPU choices

Input embeddings and output weights both use contiguous `float32` tables. Lookup remains zero-copy.

The training loop allocates reusable scratch buffers for logits, probabilities, and the input gradient once per `Train` call rather than once per token pair. This keeps the implementation readable while avoiding obvious allocation churn in the inner loop.

The implementation is intentionally scalar CPU code. No optimization claim is made yet.

## Try it

The original deterministic count-based model remains the default:

```bash
go run ./cmd/quasar -corpus examples/corpus.txt -prompt "the" -tokens 4
```

To train and use the neural bigram model:

```bash
go run ./cmd/quasar \
  -corpus examples/corpus.txt \
  -prompt "the" \
  -tokens 4 \
  -neural \
  -dimensions 8 \
  -epochs 100 \
  -learning-rate 0.05
```

Training telemetry is written to stderr as:

```text
loss <first epoch> -> <last epoch>
```

Generated text is written separately to stdout.

## What v0.3.0 deliberately does not do

- use more than one previous token as context
- attention
- hidden Transformer layers
- automatic differentiation
- batching
- model persistence
- stochastic sampling
- GPU acceleration

The gradient equations are implemented explicitly so the learning mechanism is visible rather than hidden behind an autograd framework.

## Relation to a real inference engine

At inference time, a real language model also transforms learned weights into logits and chooses a next token. Its network between embedding lookup and logits is dramatically deeper, but the outer idea is the same:

`trained weights -> logits -> next-token decision`

DwarfStar does not train its models; it loads already trained weights and performs inference. Quasar v0.3 implements training only as an educational step so we can understand where those learned weights come from before returning our focus to inference.

## Validation

Tests verify that:

- average training loss decreases
- a repeated next-token relationship is learned
- an input embedding changes after SGD updates
- invalid training parameters are rejected
- the CLI can train and generate with the neural model
- the original v0.1 bigram generation path still works

CI continues to run formatting checks, `go vet ./...`, and `go test ./...` with the current stable Go toolchain.
