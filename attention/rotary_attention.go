package attention

import (
	"fmt"
	"math"

	"github.com/dmarro89/quasar/position"
	"github.com/dmarro89/quasar/tensor"
)

// RotaryProjected applies RoPE to projected Queries and Keys before computing
// last-token self-attention. Values remain unrotated.
type RotaryProjected struct {
	projected  *Projected
	rotary     *position.Rotary
	dimensions int
}

// NewRotaryProjected creates projected attention with rotary positional
// encoding for Queries and Keys.
func NewRotaryProjected(dimensions int, seed uint64) (*RotaryProjected, error) {
	projected, err := NewProjected(dimensions, seed)
	if err != nil {
		return nil, err
	}
	rotary, err := position.NewRotary(dimensions, position.DefaultRoPEBase)
	if err != nil {
		return nil, fmt.Errorf("create rotary encoder: %w", err)
	}
	return &RotaryProjected{projected: projected, rotary: rotary, dimensions: dimensions}, nil
}

// Project returns the position-aware Q and K plus the unmodified projected V
// for one input vector. This allocating helper is intended for inspection and
// tests, not the hot decode loop.
func (r *RotaryProjected) Project(input tensor.Vector, index int) (tensor.Vector, tensor.Vector, tensor.Vector, error) {
	query, key, value, err := r.projected.Project(input)
	if err != nil {
		return nil, nil, nil, err
	}
	if err := r.rotary.ApplyInto(query, index, query); err != nil {
		return nil, nil, nil, err
	}
	if err := r.rotary.ApplyInto(key, index, key); err != nil {
		return nil, nil, nil, err
	}
	return query, key, value, nil
}

// LastToken computes projected single-head attention for the final token after
// applying RoPE to every Query/Key position. Values are not rotated.
func (r *RotaryProjected) LastToken(vectors []tensor.Vector) (ProjectedResult, error) {
	if err := r.projected.validate(vectors); err != nil {
		return ProjectedResult{}, err
	}

	result := ProjectedResult{
		Query:   make(tensor.Vector, r.dimensions),
		Scores:  make(tensor.Vector, len(vectors)),
		Weights: make(tensor.Vector, len(vectors)),
		Output:  make(tensor.Vector, r.dimensions),
	}
	keyScratch := make(tensor.Vector, r.dimensions)
	valueScratch := make(tensor.Vector, r.dimensions)

	last := len(vectors) - 1
	if err := r.projected.query.ForwardInto(vectors[last], result.Query); err != nil {
		return ProjectedResult{}, err
	}
	if err := r.rotary.ApplyInto(result.Query, last, result.Query); err != nil {
		return ProjectedResult{}, err
	}

	scale := float32(1 / math.Sqrt(float64(r.dimensions)))
	maxScore := float32(-math.MaxFloat32)
	for i, vector := range vectors {
		if err := r.projected.key.ForwardInto(vector, keyScratch); err != nil {
			return ProjectedResult{}, err
		}
		if err := r.rotary.ApplyInto(keyScratch, i, keyScratch); err != nil {
			return ProjectedResult{}, err
		}
		dot, err := tensor.Dot(result.Query, keyScratch)
		if err != nil {
			return ProjectedResult{}, err
		}
		score := dot * scale
		result.Scores[i] = score
		if score > maxScore {
			maxScore = score
		}
	}

	var denominator float64
	for i, score := range result.Scores {
		value := math.Exp(float64(score - maxScore))
		result.Weights[i] = float32(value)
		denominator += value
	}
	for i := range result.Weights {
		result.Weights[i] /= float32(denominator)
	}

	for i, vector := range vectors {
		if err := r.projected.value.ForwardInto(vector, valueScratch); err != nil {
			return ProjectedResult{}, err
		}
		weight := result.Weights[i]
		for dimension := range result.Output {
			result.Output[dimension] += weight * valueScratch[dimension]
		}
	}
	return result, nil
}
