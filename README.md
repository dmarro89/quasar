# Quasar

Quasar is an educational inference engine built from first principles in Go.

The project grows one small release at a time. Each version introduces one important inference concept, implements it with minimal code, tests it, documents it, and keeps performance visible from the beginning.

Quasar is inspired by specialized local inference engines such as DwarfStar, but deliberately starts from much simpler models so every layer of the system can be understood before it is optimized.

## Current release: v0.8.0

v0.8.0 adds **absolute positional information** to projected self-attention.

Each token still starts with its token embedding, but Quasar now adds a second vector that represents where the token appears in the sequence:

```text
x_i = token_embedding_i + position_embedding_i
```

The resulting fixed-width vector is then sent through the v0.7 Q/K/V projections:

```text
                         +-> Wq -> Query
token embedding          |
       +                 +-> Wk -> Key
position embedding       |
       |                 +-> Wv -> Value
       v
 positioned token
```

This means the same token at position 0 and position 1 no longer has the same representation. As a result, `the moon shines` and `moon the shines` can produce different attention outputs even though they contain the same words.

The positional embeddings in v0.8 are deterministic but untrained. They are deliberately simple absolute embeddings so the reason positional information exists is clear before Quasar introduces a more modern mechanism such as RoPE.

## Inspect positioned projected attention

```bash
go run ./cmd/quasar \
  -corpus examples/corpus.txt \
  -attention "the moon shines" \
  -positioned-attention \
  -dimensions 3
```

The command prints, for every token:

- its sequence position
- the original token embedding
- the positional embedding
- the combined token+position representation
- the attention score and softmax weight
- the final attention output

The projected Query for the final token is also shown.

## Inspect projected Q/K/V attention from v0.7

```bash
go run ./cmd/quasar \
  -corpus examples/corpus.txt \
  -attention "the moon shines" \
  -projected-attention \
  -dimensions 3
```

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
- [v0.8.0 technical notes](docs/releases/v0.8/README.md)
