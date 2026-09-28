package position

import (
	"fmt"
	"math"

	"github.com/dmarro89/quasar/tensor"
)

const DefaultRoPEBase = 10000.0

// Rotary applies rotary positional encoding (RoPE) to pairs of vector
// dimensions. Each pair rotates by a position-dependent angle; higher pairs
// rotate more slowly according to the configured base.
type Rotary struct {
	dimensions       int
	inverseFrequency []float64
}

// NewRotary creates a rotary positional encoder.
func NewRotary(dimensions int, base float64) (*Rotary, error) {
	if dimensions <= 0 {
		return nil, fmt.Errorf("rotary dimensions must be positive: %d", dimensions)
	}
	if dimensions%2 != 0 {
		return nil, fmt.Errorf("rotary dimensions must be even: %d", dimensions)
	}
	if base <= 1 {
		return nil, fmt.Errorf("rotary base must be greater than 1: %f", base)
	}

	pairs := dimensions / 2
	inverseFrequency := make([]float64, pairs)
	for pair := 0; pair < pairs; pair++ {
		exponent := float64(2*pair) / float64(dimensions)
		inverseFrequency[pair] = 1 / math.Pow(base, exponent)
	}
	return &Rotary{dimensions: dimensions, inverseFrequency: inverseFrequency}, nil
}

// Apply returns a rotated copy of input at the supplied absolute position.
func (r *Rotary) Apply(input tensor.Vector, position int) (tensor.Vector, error) {
	output := make(tensor.Vector, r.dimensions)
	if err := r.ApplyInto(input, position, output); err != nil {
		return nil, err
	}
	return output, nil
}

// ApplyInto rotates input into a caller-provided output buffer. Input and
// output may refer to the same slice because each pair is read before writing.
func (r *Rotary) ApplyInto(input tensor.Vector, position int, output tensor.Vector) error {
	if len(input) != r.dimensions {
		return fmt.Errorf("rotary input has %d dimensions, want %d", len(input), r.dimensions)
	}
	if len(output) != r.dimensions {
		return fmt.Errorf("rotary output has %d dimensions, want %d", len(output), r.dimensions)
	}
	if position < 0 {
		return fmt.Errorf("rotary position must be non-negative: %d", position)
	}

	for pair, frequency := range r.inverseFrequency {
		first := pair * 2
		second := first + 1
		angle := float64(position) * frequency
		cosine, sine := math.Cos(angle), math.Sin(angle)
		x, y := float64(input[first]), float64(input[second])
		output[first] = float32(x*cosine - y*sine)
		output[second] = float32(x*sine + y*cosine)
	}
	return nil
}

// Dimensions returns the vector width accepted by the encoder.
func (r *Rotary) Dimensions() int {
	return r.dimensions
}
