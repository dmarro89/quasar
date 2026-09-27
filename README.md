# Quasar

Quasar is an educational inference engine built from first principles in Go.

The project grows one small release at a time. Each version introduces one important inference concept, implements it with minimal code, tests it, documents it, and keeps performance visible from the beginning.

Quasar is inspired by specialized local inference engines such as DwarfStar, but deliberately starts from much simpler models so every layer of the system can be understood before it is optimized.

## Current release: v0.5.0

v0.5.0 teaches Quasar to preserve the order of a fixed multi-token context.

v0.4 combined context embeddings with a mean, which made `the moon` and `moon the` indistinguishable. v0.5 adds an ordered representation by concatenating one embedding slot per context position:

```text
the moon -> [embedding(the) | embedding(moon)]
moon the -> [embedding(moon) | embedding(the)]
```

The prediction and training path is still familiar:

`ordered context -> dot products -> logits -> softmax -> cross-entropy -> gradients -> SGD`

This intentionally simple solution preserves order without introducing attention yet. Its main limitation is also instructive: context width grows as `embedding dimensions × context size`, so concatenation does not scale like a real Transformer.

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

## Train with ordered context

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
