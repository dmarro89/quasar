# Quasar v0.5.0 — ordered context

## Goal

v0.4 taught Quasar to use multiple previous tokens, but mean pooling erased their order. The context `the moon` and the reversed context `moon the` produced the same representation because vector addition is commutative.

v0.5.0 fixes exactly that problem without introducing attention yet.

The release answers one question:

**How can a fixed-size neural model preserve token order while still using the training loop we already understand?**

## From mean pooling to positional slots

Assume two-dimensional embeddings:

```text
the  = [0.2, 0.6]
moon = [0.8, 0.1]
```

v0.4 used the mean:

```text
(the + moon) / 2 = [0.5, 0.35]
(moon + the) / 2 = [0.5, 0.35]
```

The order disappeared.

v0.5 concatenates the vectors instead:

```text
the moon -> [0.2, 0.6 | 0.8, 0.1]
moon the -> [0.8, 0.1 | 0.2, 0.6]
```

The two contexts now have different numeric representations.

## Why concatenation works

Each context position owns a fixed slot in the combined vector.

For embedding dimension `D` and context size `N`, the ordered context width is:

```text
D × N
```

If `D=4` and `N=2`, the context vector has eight values:

```text
position 0             position 1
[the embedding]        [moon embedding]
      4 values    +         4 values
             = 8 values
```

The output weight for every candidate next token also has width eight so Quasar can compute the same dot product used in previous releases.

## Prediction flow

For the phrase:

```text
the moon shines
```

with context size two, prediction becomes:

```text
the        moon
 |           |
 v           v
embedding  embedding
 |           |
 +--- concatenate ---+
          |
          v
ordered context vector
          |
          v
dot product with every output row
          |
          v
logits
          |
          v
softmax during training / argmax during generation
          |
          v
shines
```

The learning loop remains unchanged:

```text
logits -> softmax -> cross-entropy -> gradients -> SGD
```

## Backpropagation through concatenation

Concatenation has a particularly simple backward pass.

If the context is:

```text
[the | moon]
```

then the gradient of the combined context is split back into the same slots:

```text
context gradient
[grad for the | grad for moon]
       |              |
       v              v
embedding(the)   embedding(moon)
```

Unlike mean pooling, there is no `1/N` scaling. Each token receives the gradient associated with the slot it occupied.

## Sliding ordered context

Training still uses a sliding window. For a repeating example:

```text
the moon shines the moon shines
```

with context size two, Quasar sees examples such as:

```text
the moon    -> shines
moon shines -> the
shines the  -> moon
```

Generation slides the window in the same way:

```text
the moon
    -> shines

moon shines
    -> the

shines the
    -> moon
```

## CLI

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

Without `-ordered-context`, a multi-token context keeps the v0.4 mean-pooling behavior. This makes the two representations directly comparable.

## Theoretical lesson: sequence order is information

Natural language is a sequence, not a bag of words.

Even when two phrases contain the same tokens, changing their order can change their meaning. A model therefore needs a representation that can distinguish both token identity and where that token occurs in the sequence.

v0.5 encodes position structurally through fixed slots. It is intentionally not how modern Transformers solve the problem, but it makes the requirement concrete before introducing more sophisticated mechanisms.

## Deliberate limitations

Concatenation solves order but has serious drawbacks:

1. **Width grows with context length.** `D=4096` and `N=128` would create a context vector with 524,288 values.
2. **The model is tied to one fixed context size.** Position 0 and position 1 are different hard-wired slots.
3. **All context tokens are treated through one large static projection.** The model cannot dynamically decide that one word matters more than another for the current prediction.
4. **It is not self-attention.** There are no queries, keys, values, attention scores, or causal masks yet.

These limitations create the next problem naturally: once token representations remain distinct, how can the model dynamically decide which previous tokens are relevant?

That is the problem self-attention will eventually address.

## Performance and memory

The input embedding table remains contiguous and zero-copy on lookup.

The cost introduced by this release is explicit: output rows grow from `D` values to `D × contextSize` values. This is acceptable for a teaching model but intentionally demonstrates why naive concatenation does not scale to real LLM context lengths.

No performance optimization is claimed in this release.

## What v0.5.0 deliberately does not do

- positional embeddings
- RoPE
- queries, keys, or values
- self-attention
- causal masking
- Transformer blocks
- batching
- GPU acceleration

Those concepts remain separate learning steps.
