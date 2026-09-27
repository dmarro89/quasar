# Quasar

Quasar is an educational inference engine built from first principles in Go.

The project grows one small release at a time. Each version introduces one important inference concept, implements it with minimal code, tests it, documents it, and keeps performance visible from the beginning.

Quasar is inspired by specialized local inference engines such as DwarfStar, but deliberately starts from much simpler models so every layer of the system can be understood before it is optimized.

## Current release: v0.6.0

v0.6.0 introduces Quasar's first self-attention primitive.

The final token in a context acts as the query. It compares itself with every token key using scaled dot products, softmax converts those scores into attention weights, and the output is a weighted sum of the value vectors:

```text
the moon shines
         |
         +-> query

the ---- score ----\
moon --- score -----+-> softmax weights -> weighted value sum -> fixed-width output
shines - score ----/
```

To isolate the attention algorithm, v0.6 deliberately uses **identity Q/K/V**: token embeddings are used directly as queries, keys, and values. Learned Q/K/V projections, positional mechanisms, multi-head attention, and Transformer blocks come later.

The attention output remains the same width as one embedding regardless of context length. This addresses the width explosion of the v0.5 concatenation approach, while introducing dynamic content-dependent weighting.

## Inspect self-attention

```bash
go run ./cmd/quasar \
  -corpus examples/corpus.txt \
  -attention "the moon shines" \
  -dimensions 3
```

The command prints:

- `shines` as the query token
- one scaled query-key score per token
- one softmax attention weight per token
- the final weighted output vector

The embeddings are still untrained in this inspection mode, so the calculation is real but the resulting attention weights do not yet represent learned semantics.

## Generate with the original bigram model

```bash
go run ./cmd/quasar -corpus examples/corpus.txt -prompt "the" -tokens 4
```

With the included corpus:

```text
the moon shines at night
```

## Train a one-token neural model

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

## Train with mean-pooled context from v0.4

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

## Train with ordered context from v0.5

```bash
go run ./cmd/quasar \
  -corpus examples/corpus.txt \
  -prompt "the moon" \
  -tokens 4 \
  -neural \
  -context-size 2 \
  -ordered-context \
  -dimensions 8 \
  -epochs 120 \
  -learning-rate 0.1
```

The prompt must contain at least as many tokens as the context window. The first and final epoch loss are printed to stderr; generated text is written to stdout.

## Inspect an untrained embedding

```bash
go run ./cmd/quasar -corpus examples/corpus.txt -embedding moon -dimensions 4
```

This inspection mode intentionally shows the deterministic initialization before training.

## Development principles

- latest stable Go
- standard library by default
- minimal, focused implementations
- CPU and memory costs considered from day one
- deterministic unit tests
- small pull requests
- release-by-release technical documentation

See [AGENTS.md](AGENTS.md) for the complete development rules.

## Documentation

- [General documentation](docs/README.md)
- [v0.1.0 technical notes](docs/releases/v0.1/README.md)
- [v0.2.0 technical notes](docs/releases/v0.2/README.md)
- [v0.3.0 technical notes](docs/releases/v0.3/README.md)
- [v0.4.0 technical notes](docs/releases/v0.4/README.md)
- [v0.5.0 technical notes](docs/releases/v0.5/README.md)
- [v0.6.0 technical notes](docs/releases/v0.6/README.md)
