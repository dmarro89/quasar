package embedding

import (
	"fmt"
	"math/rand/v2"

	"github.com/dmarro89/quasar/tensor"
	"github.com/dmarro89/quasar/tokenizer"
)

// Table stores one fixed-width embedding vector for every token.
// All rows share one contiguous backing buffer to minimize allocations.
type Table struct {
	dimensions int
	weights    []float32
}

// NewTable creates a deterministically initialized embedding table.
// The initial values carry no learned meaning yet; training comes later.
func NewTable(vocabularySize, dimensions int, seed uint64) (*Table, error) {
	if vocabularySize <= 0 {
		return nil, fmt.Errorf("vocabulary size must be positive: %d", vocabularySize)
	}
	if dimensions <= 0 {
		return nil, fmt.Errorf("embedding dimensions must be positive: %d", dimensions)
	}

	weights := make([]float32, vocabularySize*dimensions)
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	for i := range weights {
		weights[i] = float32(rng.Float64() - 0.5)
	}

	return &Table{dimensions: dimensions, weights: weights}, nil
}

// Lookup returns a zero-copy view of the embedding row for id.
func (t *Table) Lookup(id tokenizer.TokenID) (tensor.Vector, error) {
	row := int(id)
	if row < 0 || row >= t.Size() {
		return nil, fmt.Errorf("token id %d exceeds embedding table size %d", id, t.Size())
	}

	start := row * t.dimensions
	return tensor.Vector(t.weights[start : start+t.dimensions]), nil
}

// Size returns the number of token rows in the table.
func (t *Table) Size() int {
	return len(t.weights) / t.dimensions
}

// Dimensions returns the number of float32 values in each embedding row.
func (t *Table) Dimensions() int {
	return t.dimensions
}
