package attention

import (
	"fmt"
	"math"

	"github.com/dmarro89/quasar/tensor"
)

// Result exposes the intermediate values of one last-token attention step.
// Scores are scaled query-key dot products. Weights are their softmax values.
// Output is the weighted sum of the value vectors.
type Result struct {
	Scores  tensor.Vector
	Weights tensor.Vector
	Output  tensor.Vector
}

// LastToken computes a single-head scaled dot-product attention step for the
// final token in vectors. To isolate the attention mechanism in v0.6, the input
// embeddings are used directly as query, keys, and values (identity Q/K/V).
func LastToken(vectors []tensor.Vector) (Result, error) {
	if len(vectors) == 0 {
		return Result{}, fmt.Errorf("attention requires at least one vector")
	}
	if len(vectors[0]) == 0 {
		return Result{}, fmt.Errorf("attention vectors must not be empty")
	}

	dimensions := len(vectors[0])
	for i, vector := range vectors {
		if len(vector) != dimensions {
			return Result{}, fmt.Errorf("attention vector %d has %d dimensions, want %d", i, len(vector), dimensions)
		}
	}

	result := Result{
		Scores:  make(tensor.Vector, len(vectors)),
		Weights: make(tensor.Vector, len(vectors)),
		Output:  make(tensor.Vector, dimensions),
	}
	if err := LastTokenInto(vectors, result.Scores, result.Weights, result.Output); err != nil {
		return Result{}, err
	}
	return result, nil
}

// LastTokenInto is the allocation-free form of LastToken for callers that
// reuse score, weight, and output buffers across repeated attention steps.
func LastTokenInto(vectors []tensor.Vector, scores, weights, output tensor.Vector) error {
	if len(vectors) == 0 {
		return fmt.Errorf("attention requires at least one vector")
	}
	dimensions := len(vectors[0])
	if dimensions == 0 {
		return fmt.Errorf("attention vectors must not be empty")
	}
	for i, vector := range vectors {
		if len(vector) != dimensions {
			return fmt.Errorf("attention vector %d has %d dimensions, want %d", i, len(vector), dimensions)
		}
	}
	if len(scores) != len(vectors) {
		return fmt.Errorf("score buffer has length %d, want %d", len(scores), len(vectors))
	}
	if len(weights) != len(vectors) {
		return fmt.Errorf("weight buffer has length %d, want %d", len(weights), len(vectors))
	}
	if len(output) != dimensions {
		return fmt.Errorf("output buffer has %d dimensions, want %d", len(output), dimensions)
	}

	query := vectors[len(vectors)-1]
	scale := float32(1 / math.Sqrt(float64(dimensions)))
	maxScore := float32(-math.MaxFloat32)
	for i, key := range vectors {
		dot, err := tensor.Dot(query, key)
		if err != nil {
			return err
		}
		score := dot * scale
		scores[i] = score
		if score > maxScore {
			maxScore = score
		}
	}

	var denominator float64
	for i, score := range scores {
		value := math.Exp(float64(score - maxScore))
		weights[i] = float32(value)
		denominator += value
	}
	for i := range weights {
		weights[i] /= float32(denominator)
	}

	clear(output)
	for i, value := range vectors {
		weight := weights[i]
		for dimension := range output {
			output[dimension] += weight * value[dimension]
		}
	}
	return nil
}
