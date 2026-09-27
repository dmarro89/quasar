package attention

import (
	"fmt"
	"math"

	"github.com/dmarro89/quasar/position"
	"github.com/dmarro89/quasar/tensor"
)

// PositionedProjected adds absolute positional embeddings before the Q/K/V
// projections used by last-token attention.
type PositionedProjected struct {
	projected  *Projected
	positions  *position.Table
	dimensions int
}

// NewPositionedProjected creates projected attention with absolute positional
// embeddings for up to maxPositions tokens.
func NewPositionedProjected(dimensions, maxPositions int, seed uint64) (*PositionedProjected, error) {
	projected, err := NewProjected(dimensions, seed)
	if err != nil {
		return nil, err
	}
	positions, err := position.NewTable(maxPositions, dimensions, seed+3)
	if err != nil {
		return nil, fmt.Errorf("create positional embeddings: %w", err)
	}
	return &PositionedProjected{
		projected:  projected,
		positions:  positions,
		dimensions: dimensions,
	}, nil
}

// Position returns the positional vector and the combined token+position
// representation for inspection. It allocates and is not a hot-path API.
func (p *PositionedProjected) Position(input tensor.Vector, index int) (tensor.Vector, tensor.Vector, error) {
	positional, err := p.positions.Lookup(index)
	if err != nil {
		return nil, nil, err
	}
	combined := make(tensor.Vector, p.dimensions)
	if err := p.positions.AddInto(input, index, combined); err != nil {
		return nil, nil, err
	}
	positionCopy := append(tensor.Vector(nil), positional...)
	return positionCopy, combined, nil
}

// LastToken computes single-head projected attention after adding one absolute
// positional embedding to every token representation.
func (p *PositionedProjected) LastToken(vectors []tensor.Vector) (ProjectedResult, error) {
	if err := p.validate(vectors); err != nil {
		return ProjectedResult{}, err
	}

	result := ProjectedResult{
		Query:   make(tensor.Vector, p.dimensions),
		Scores:  make(tensor.Vector, len(vectors)),
		Weights: make(tensor.Vector, len(vectors)),
		Output:  make(tensor.Vector, p.dimensions),
	}
	positionedScratch := make(tensor.Vector, p.dimensions)
	keyScratch := make(tensor.Vector, p.dimensions)
	valueScratch := make(tensor.Vector, p.dimensions)

	last := len(vectors) - 1
	if err := p.positions.AddInto(vectors[last], last, positionedScratch); err != nil {
		return ProjectedResult{}, err
	}
	if err := p.projected.query.ForwardInto(positionedScratch, result.Query); err != nil {
		return ProjectedResult{}, err
	}

	scale := float32(1 / math.Sqrt(float64(p.dimensions)))
	maxScore := float32(-math.MaxFloat32)
	for i, vector := range vectors {
		if err := p.positions.AddInto(vector, i, positionedScratch); err != nil {
			return ProjectedResult{}, err
		}
		if err := p.projected.key.ForwardInto(positionedScratch, keyScratch); err != nil {
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
		if err := p.positions.AddInto(vector, i, positionedScratch); err != nil {
			return ProjectedResult{}, err
		}
		if err := p.projected.value.ForwardInto(positionedScratch, valueScratch); err != nil {
			return ProjectedResult{}, err
		}
		weight := result.Weights[i]
		for dimension := range result.Output {
			result.Output[dimension] += weight * valueScratch[dimension]
		}
	}

	return result, nil
}

func (p *PositionedProjected) validate(vectors []tensor.Vector) error {
	if len(vectors) == 0 {
		return fmt.Errorf("positioned attention requires at least one vector")
	}
	if len(vectors) > p.positions.Size() {
		return fmt.Errorf("context has %d positions, maximum is %d", len(vectors), p.positions.Size())
	}
	for i, vector := range vectors {
		if len(vector) != p.dimensions {
			return fmt.Errorf("attention vector %d has %d dimensions, want %d", i, len(vector), p.dimensions)
		}
	}
	return nil
}
