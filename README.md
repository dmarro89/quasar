# Quasar

Quasar is an educational inference engine built from first principles in Go.

The project grows one small release at a time. Each version introduces one important inference concept, implements it with minimal code, tests it, documents it, and keeps performance visible from the beginning.

Quasar is inspired by specialized local inference engines such as DwarfStar, but deliberately starts from much simpler models so that every layer of the system can be understood before it is optimized.

## Current work: v0.1.0

The first release will implement the smallest useful autoregressive text-generation loop, using a deterministic word-level bigram model.

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

General engine documentation lives under [`docs/`](docs/). Each release also gets dedicated technical notes under `docs/releases/`.
