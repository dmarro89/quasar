# Quasar v0.2.0 — tokens become vectors

## Goal

Move beyond representing a token only as an arbitrary integer ID. v0.2.0 introduces the numeric representation that neural inference engines operate on: a fixed-width vector for every token.

The release answers one question: **how does `TokenID(1)` become a vector of numbers that later neural layers can transform?**

## Concepts introduced

1. **Vector** — a contiguous sequence of `float32` values.
2. **Embedding** — the vector associated with a token ID.
3. **Embedding table** — a matrix with one row per vocabulary token and one column per embedding dimension.
4. **Dot product** — the first basic numeric operation between two vectors.
5. **Initialization vs learning** — v0.2 initializes vectors deterministically, but does not train them. Their values do not carry semantic meaning yet.

## Implementation

`tensor.Vector` is a `[]float32`. Using `float32` gives a simple baseline numeric representation while using half the memory of `float64`.

`embedding.Table` stores all rows in one flat contiguous `[]float32` buffer. For vocabulary size `V` and embedding dimension `D`, storage is exactly `V × D` float32 values, or approximately `V × D × 4` bytes.

`Lookup(TokenID)` computes the row offset and returns a slice backed by the existing table. It performs no per-lookup allocation or copy.

The initial values are deterministic pseudo-random numbers so examples and tests are reproducible. They are deliberately untrained.

## Data flow

`text -> TokenID -> embedding table -> float32 vector`

The existing v0.1 bigram generation path remains unchanged. v0.2 adds representation; it does not pretend that random embeddings improve generation.

## Try it

```bash
go run ./cmd/quasar -corpus examples/corpus.txt -embedding moon -dimensions 4
```

Quasar prints the token ID and its four-dimensional vector, explicitly marked as untrained.

## What v0.2.0 deliberately does not do

- learn embedding values
- infer semantic similarity
- use embeddings to predict the next token
- matrix multiplication
- neural layers
- attention
- backpropagation
- GPU acceleration

Those require additional concepts and will be introduced only when the previous layer is understood.

## Relation to a real inference engine

A real LLM also starts inference by looking up each token in an embedding weight matrix. DwarfStar exposes this same idea through model tensors such as `token_embd.weight`; its rows are trained weights with thousands of dimensions rather than Quasar's small untrained demonstration vectors.

The important structural connection is now visible:

`TokenID -> row lookup in embedding weights -> vector`

## Validation

The vector primitive, embedding table, bounds validation, deterministic initialization, zero-copy lookup behavior, generation regression path, and embedding-inspection CLI are unit tested. CI runs formatting checks, `go vet ./...`, and `go test ./...`.

## Performance baseline

The embedding table is contiguous and lookups are zero-copy. This is intentional data-layout groundwork rather than a performance claim. Future optimizations must be benchmarked before adoption.
