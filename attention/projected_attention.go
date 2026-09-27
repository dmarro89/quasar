package attention

import (
	"fmt"
	"math"

	"github.com/dmarro89/quasar/projection"
	"github.com/dmarro89/quasar/tensor"
)

// Projected applies separate linear projections to produce Q, K, and V before
// computing last-token attention.
type Projected struct {
	query      *projection.Linear
	key        *projection.Linear
	value      *projection.Linear
	dimensions int
}

// ProjectedResult exposes the query projection together with attention
// intermediates for inspection.
type ProjectedResult struct {
	Query   tensor.Vector
	Scores  tensor.Vector
	Weights tensor.Vector
	Output  tensor.Vector
}

// NewProjected creates three independent deterministic Q/K/V projections.
func NewProjected(dimensions int, seed uint64) (*Projected, error) {
	if dimensions <= 0 {
		return nil, fmt.Errorf("attention dimensions must be positive: %d", dimensions)
	}
	query, err := projection.NewLinear(dimensions, dimensions, seed)
	if err != nil {
		return nil, fmt.Errorf("create query projection: %w", err)
	}
	key, err := projection.NewLinear(dimensions, dimensions, seed+1)
	if err != nil {
		return nil, fmt.Errorf("create key projection: %w", err)
	}
	value, err := projection.NewLinear(dimensions, dimensions, seed+2)
	if err != nil {
		return nil, fmt.Errorf("create value projection: %w", err)
	}
	return &Projected{query: query, key: key, value: value, dimensions: dimensions}, nil
}

// Project returns Q, K, and V for one input vector. This allocating helper is
// intended for inspection and tests rather than a hot decode loop.
func (p *Projected) Project(input tensor.Vector) (tensor.Vector, tensor.Vector, tensor.Vector, error) {
	query, err := p.query.Forward(input)
	if err != nil {
		return nil, nil, nil, err
	}
	key, err := p.key.Forward(input)
	if err != nil {
		return nil, nil, nil, err
	}
	value, err := p.value.Forward(input)
	if err != nil {
		return nil, nil, nil, err
	}
	return query, key, value, nil
}

// LastToken computes projected single-head attention for the final vector.
func (p *Projected) LastToken(vectors []tensor.Vector) (ProjectedResult, error) {
	if err := p.validate(vectors); err != nil {
		return ProjectedResult{}, err
	}
	result := ProjectedResult{
		Query:   make(tensor.Vector, p.dimensions),
		Scores:  make(tensor.Vector, len(vectors)),
		Weights: make(tensor.Vector, len(vectors)),
		Output:  make(tensor.Vector, p.dimensions),
	}
	keyScratch := make(tensor.Vector, p.dimensions)
	valueScratch := make(tensor.Vector, p.dimensions)
	if err := p.lastTokenInto(vectors, result.Query, keyScratch, valueScratch, result.Scores, result.Weights, result.Output); err != nil {
		return ProjectedResult{}, err
	}
	return result, nil
}

func (p *Projected) lastTokenInto(vectors []tensor.Vector, query, keyScratch, valueScratch, scores, weights, output tensor.Vector) error {
	if err := p.validate(vectors); err != nil {
		return err
	}
	if len(query) != p.dimensions || len(keyScratch) != p.dimensions || len(valueScratch) != p.dimensions || len(output) != p.dimensions {
		return fmt.Errorf("projected attention vector buffer dimension mismatch")
	}
	if len(scores) != len(vectors) || len(weights) != len(vectors) {
		return fmt.Errorf("projected attention score/weight buffer length mismatch")
	}

	if err := p.query.ForwardInto(vectors[len(vectors)-1], query); err != nil {
		return err
	}

	scale := float32(1 / math.Sqrt(float64(p.dimensions)))
	maxScore := float32(-math.MaxFloat32)
	for i, vector := range vectors {
		if err := p.key.ForwardInto(vector, keyScratch); err != nil {
			return err
		}
		dot, err := tensor.Dot(query, keyScratch)
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
	for i, vector := range vectors {
		if err := p.value.ForwardInto(vector, valueScratch); err != nil {
			return err
		}
		weight := weights[i]
		for dimension := range output {
			output[dimension] += weight * valueScratch[dimension]
		}
	}
	return nil
}

func (p *Projected) validate(vectors []tensor.Vector) error {
	if len(vectors) == 0 {
		return fmt.Errorf("projected attention requires at least one vector")
	}
	for i, vector := range vectors {
		if len(vector) != p.dimensions {
			return fmt.Errorf("attention vector %d has %d dimensions, want %d", i, len(vector), p.dimensions)
		}
	}
	return nil
}
