# Quasar v0.1.0 — the smallest inference loop

## Goal

Build the smallest useful text generator that exposes the same outer loop used by a real autoregressive inference engine without introducing neural-network math yet.

The release answers one question: **how can a program turn previous tokens into a next-token decision and repeat that decision to generate text?**

## Concepts introduced

1. **Token** — a discrete piece of input text. v0.1.0 deliberately uses whitespace-separated words.
2. **Vocabulary** — a stable mapping between tokens and compact numeric token IDs.
3. **Model score** — Quasar counts how often token B followed token A in a corpus. These counts are a simple stand-in for the scores a neural model will later produce.
4. **Argmax generation** — the next token is the candidate with the highest score. Ties use the lower token ID, making generation deterministic.
5. **Autoregressive loop** — the generated token becomes the input used to choose the following token.

## Implementation

The tokenizer stores a map from word to `TokenID` for encoding and a slice indexed by `TokenID` for decoding.

The bigram model stores one lazily allocated transition map per source token. A dense vocabulary-by-vocabulary matrix would reserve O(V²) counters even for transitions that never occur; the sparse representation stores only observed transitions.

Generation repeatedly calls `Next(current)` until it reaches the requested token limit or a token with no observed successor.

## Data flow

`text -> tokens -> vocabulary -> transition counts -> argmax -> next token -> repeat`

## What v0.1.0 deliberately does not do

- subword or byte tokenization
- probabilities or stochastic sampling
- embeddings
- trainable neural weights
- attention
- KV cache
- GPU acceleration
- persisted model files

Those concepts are intentionally postponed so future releases can introduce one new reason for complexity at a time.

## Relation to a real inference engine

A production LLM replaces Quasar's bigram transition counts with neural-network logits, but the outer loop is already recognizable:

`token -> model scores -> choose next token -> append token -> repeat`

Later releases will replace one simple component at a time while preserving that loop.

## Validation

v0.1.0 is validated by formatting checks, `go vet ./...`, and `go test ./...`. CLI behavior is tested through the same `run` function used by `main`.

## Performance baseline

No performance optimization is claimed in v0.1.0. The data layout already avoids a dense O(V²) transition table. Future performance changes must be justified by benchmarks rather than intuition alone.
