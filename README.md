# Quasar

Quasar is an educational inference engine built from first principles in Go.

The project grows one small release at a time. Each version introduces one important inference concept, implements it with minimal code, tests it, documents it, and keeps performance visible from the beginning.

Quasar is inspired by specialized local inference engines such as DwarfStar, but deliberately starts from much simpler models so every layer of the system can be understood before it is optimized.

## Current release: v0.10.0

v0.10.0 introduces the **KV cache** and the first truly incremental attention decode path.

The v0.9 implementation recomputed projected Keys and Values for every previous token whenever attention was evaluated. v0.10 stores those results once and reuses them:

```text
new token
   |
   +-> Wq -> Query -> RoPE(position)
   +-> Wk -> Key   -> RoPE(position) -> append to K cache
   +-> Wv -> Value                  -> append to V cache

rotated Query
   |
   +-> dot with all cached Keys
   +-> softmax
   +-> weighted sum of cached Values
```

Only the new token needs fresh Q/K/V projections. Earlier Keys and Values are read directly from preallocated contiguous cache buffers.

The cache has a fixed token capacity and reserves:

```text
2 * capacity * dimensions * sizeof(float32)
```

bytes in this single-head educational implementation.

For example, five tokens with four-dimensional K/V vectors reserve:

```text
2 * 5 * 4 * 4 = 160 bytes
```

The attention weights are still deterministic but untrained. v0.10 is about inference data flow and avoiding repeated work, not linguistic quality.

## Inspect incremental RoPE attention with a KV cache

```bash
go run ./cmd/quasar \
  -corpus examples/corpus.txt \
  -attention "the moon shines at night" \
  -cached-rope-attention \
  -dimensions 4
```

The command prints each incremental step, including:

- token position
- current cache length/capacity
- the new rotated Query
- the newly cached rotated Key
- the newly cached unrotated projected Value
- attention scores and softmax weights over all cached tokens
- the current attention output
- total reserved K/V cache bytes

## Inspect non-cached RoPE attention from v0.9

```bash
go run ./cmd/quasar \
  -corpus examples/corpus.txt \
  -attention "the moon shines" \
  -rope-attention \
  -dimensions 4
```

## Inspect absolute-position attention from v0.8

```bash
go run ./cmd/quasar \
  -corpus examples/corpus.txt \
  -attention "the moon shines" \
  -positioned-attention \
  -dimensions 4
```

## Inspect projected Q/K/V attention from v0.7

```bash
go run ./cmd/quasar \
  -corpus examples/corpus.txt \
  -attention "the moon shines" \
  -projected-attention \
  -dimensions 4
```

## Inspect identity-QKV self-attention from v0.6

```bash
go run ./cmd/quasar \
  -corpus examples/corpus.txt \
  -attention "the moon shines" \
  -dimensions 4
```

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
- [v0.7.0 technical notes](docs/releases/v0.7/README.md)
- [v0.8.0 technical notes](docs/releases/v0.8/README.md)
- [v0.9.0 technical notes](docs/releases/v0.9/README.md)
- [v0.10.0 technical notes](docs/releases/v0.10/README.md)
