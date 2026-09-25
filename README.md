# Quasar

Quasar is an educational inference engine built from first principles in Go.

The project grows one small release at a time. Each version introduces one important inference concept, implements it with minimal code, tests it, documents it, and keeps performance visible from the beginning.

Quasar is inspired by specialized local inference engines such as DwarfStar, but deliberately starts from much simpler models so every layer of the system can be understood before it is optimized.

## Current release: v0.1.0

v0.1.0 implements a deterministic word-level bigram generator. It teaches the smallest useful autoregressive inference loop:

`text -> tokens -> vocabulary -> transition scores -> next token -> repeat`

This is not an LLM yet. It is the baseline from which Quasar will evolve.

## Run it

```bash
go run ./cmd/quasar -corpus examples/corpus.txt -prompt "the" -tokens 4
```

With the included corpus, the deterministic output is:

```text
the moon shines at night
```

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
