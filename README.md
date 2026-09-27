# Quasar

Quasar is an educational inference engine built from first principles in Go.

The project grows one small release at a time. Each version introduces one important inference concept, implements it with minimal code, tests it, documents it, and keeps performance visible from the beginning.

Quasar is inspired by specialized local inference engines such as DwarfStar, but deliberately starts from much simpler models so every layer of the system can be understood before it is optimized.

## Current release: v0.7.0

v0.7.0 gives self-attention distinct Query, Key, and Value representations.

v0.6 used the token embedding directly for all three roles:

```text
Q = K = V = embedding
```

v0.7 introduces three independent linear projection matrices:

```text
embedding -> Wq -> Query
embedding -> Wk -> Key
embedding -> Wv -> Value
```

The attention algorithm itself remains familiar:

```text
Q/K projections -> scaled dot-product scores -> softmax weights -> weighted V sum
```

The matrices are deterministic but still untrained. This release isolates linear projection and the separate Q/K/V roles before adding training through attention or positional information.

## Inspect projected Q/K/V attention

```bash
go run ./cmd/quasar \
  -corpus examples/corpus.txt \
  -attention "the moon shines" \
  -projected-attention \
  -dimensions 3
```

The command prints:

- the projected Query for `shines`
- a projected Key and Value for `the`, `moon`, and `shines`
- the attention score and softmax weight for each token
- the final weighted Value output

The v0.6 identity-QKV mode is still available by omitting `-projected-attention`.

## Inspect identity-QKV self-attention from v0.6

```bash
go run ./cmd/quasar \
  -corpus examples/corpus.txt \
  -attention "the moon shines" \
  -dimensions 3
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
