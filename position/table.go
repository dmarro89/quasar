package position

import (
	"fmt"
	"math/rand/v2"

	"github.com/dmarro89/quasar/tensor"
)

// Table stores one fixed-width embedding vector for every supported position.
// Rows share one contiguous float32 backing buffer.
type Table struct {
	dimensions int
	weights    []float32
}

// NewTable creates deterministic positional embeddings.
// The values are intentionally untrained in v0.8; the goal is to make
// positions numerically distinguishable before introducing RoPE.
func NewTable(maxPositions, dimensions int, seed uint64) (*Table, error) {
	if maxPositions <= 0 {
		return nil, fmt.Errorf("max positions must be positive: %d", maxPositions)
	}
	if dimensions <= 0 {
		return nil, fmt.Errorf("position dimensions must be positive: %d", dimensions)
	}

	weights := make([]float32, maxPositions*dimensions)
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	for i := range weights {
		weights[i] = float32(rng.Float64() - 0.5)
	}

	return &Table{dimensions: dimensions, weights: weights}, nil
}

// Lookup returns a zero-copy view of one positional embedding row.
func (t *Table) Lookup(position int) (tensor.Vector, error) {
	if position < 0 || position >= t.Size() {
		return nil, fmt.Errorf("position %d exceeds table size %d", position, t.Size())
	}
	start := position * t.dimensions
	return tensor.Vector(t.weights[start : start+t.dimensions]), nil
}

// AddInto adds the positional embedding to input and writes the result into
// a caller-provided buffer so hot paths can reuse storage.
func (t *Table) AddInto(input tensor.Vector, position int, output tensor.Vector) error {
	if len(input) != t.dimensions {
		return fmt.Errorf("input has %d dimensions, want %d", len(input), t.dimensions)
	}
	if len(output) != t.dimensions {
		return fmt.Errorf("output has %d dimensions, want %d", len(output), t.dimensions)
	}
	positional, err := t.Lookup(position)
	if err != nil {
		return err
	}
	for i := range output {
		output[i] = input[i] + positional[i]
	}
	return nil
}

// Size returns the number of supported positions.
func (t *Table) Size() int {
	return len(t.weights) / t.dimensions
}

// Dimensions returns the width of each positional embedding.
func (t *Table) Dimensions() int {
	return t.dimensions
}

// Weights returns the contiguous row-major backing buffer.
// The zero-copy view keeps future training possible without another API.
func (t *Table) Weights() []float32 {
	return t.weights
}
