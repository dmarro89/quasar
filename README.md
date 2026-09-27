# Quasar

Quasar is an educational inference engine built from first principles in Go.

The project grows one small release at a time. Each version introduces one important inference concept, implements it with minimal code, tests it, documents it, and keeps performance visible from the beginning.

Quasar is inspired by specialized local inference engines such as DwarfStar, but deliberately starts from much simpler models so every layer of the system can be understood before it is optimized.

## Current release: v0.4.0

v0.4.0 lets Quasar predict from more than one previous token.

A configurable fixed context window is converted into one context vector by averaging its token embeddings:

`tokens -> embeddings -> mean context vector -> dot products -> logits -> next token`

Training still uses the v0.3 learning loop:

`logits -> softmax -> cross-entropy loss -> gradients -> SGD weight updates`

The key limitation is intentional: averaging loses order, so `[A,B]` and `[B,A]` produce the same context representation. Quasar now has a concrete reason to introduce positional information in a future release.

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

`-context-size` defaults to `1`, preserving the v0.3 neural bigram behavior.

## Train with two tokens of context

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
