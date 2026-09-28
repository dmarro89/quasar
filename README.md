# Quasar

Quasar is an educational inference engine built from first principles in Go.

The project grows one small release at a time. Each version introduces one important inference concept, implements it with minimal code, tests it, documents it, and keeps performance visible from the beginning.

Quasar is inspired by specialized local inference engines such as DwarfStar, but deliberately starts from much simpler models so every layer of the system can be understood before it is optimized.

## Current release: v0.9.0

v0.9.0 introduces **RoPE (Rotary Position Embedding)**.

Instead of adding a learned-style absolute position vector to each token as v0.8 did, Quasar now encodes position by rotating projected Query and Key vectors:

```text
embedding -> Wq -> Query -> RoPE(position)
embedding -> Wk -> Key   -> RoPE(position)
embedding -> Wv -> Value                 // not rotated
```

RoPE treats adjacent vector dimensions as 2D pairs and rotates every pair by a position-dependent angle. Different pairs rotate at different frequencies.

The important consequence is that the dot product between a rotated Query and Key carries **relative positional information**. Shifting both tokens by the same number of positions preserves that relationship.

The v0.9 implementation uses the standard base-10000 frequency schedule, requires an even vector width, and computes rotations without a `maxPositions × dimensions` positional embedding table.

The Q/K/V matrices and token embeddings are still untrained. This release isolates the geometry and runtime behavior of RoPE before Quasar adds more Transformer structure.

## Inspect RoPE projected attention

```bash
go run ./cmd/quasar \
  -corpus examples/corpus.txt \
  -attention "the moon shines" \
  -rope-attention \
  -dimensions 4
```

The command prints:

- the final token's rotated Query
- each token's rotated Key
- each token's unrotated projected Value
- attention scores and softmax weights
- the final weighted Value output

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
