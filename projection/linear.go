package projection

import (
	"fmt"
	"math/rand/v2"

	"github.com/dmarro89/quasar/tensor"
)

// Linear applies a learned-style linear projection without bias.
// Weights are stored row-major in one contiguous float32 buffer:
// outputDimensions rows × inputDimensions columns.
type Linear struct {
	inputDimensions  int
	outputDimensions int
	weights          []float32
}

// NewLinear creates a deterministically initialized linear projection.
func NewLinear(inputDimensions, outputDimensions int, seed uint64) (*Linear, error) {
	if inputDimensions <= 0 {
		return nil, fmt.Errorf("input dimensions must be positive: %d", inputDimensions)
	}
	if outputDimensions <= 0 {
		return nil, fmt.Errorf("output dimensions must be positive: %d", outputDimensions)
	}

	weights := make([]float32, inputDimensions*outputDimensions)
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	for i := range weights {
		weights[i] = float32(rng.Float64() - 0.5)
	}

	return &Linear{
		inputDimensions:  inputDimensions,
		outputDimensions: outputDimensions,
		weights:          weights,
	}, nil
}

// Forward projects input into a newly allocated output vector.
func (l *Linear) Forward(input tensor.Vector) (tensor.Vector, error) {
	output := make(tensor.Vector, l.outputDimensions)
	if err := l.ForwardInto(input, output); err != nil {
		return nil, err
	}
	return output, nil
}

// ForwardInto projects input into a caller-provided output buffer.
func (l *Linear) ForwardInto(input, output tensor.Vector) error {
	if len(input) != l.inputDimensions {
		return fmt.Errorf("input has %d dimensions, want %d", len(input), l.inputDimensions)
	}
	if len(output) != l.outputDimensions {
		return fmt.Errorf("output has %d dimensions, want %d", len(output), l.outputDimensions)
	}

	for row := 0; row < l.outputDimensions; row++ {
		start := row * l.inputDimensions
		var sum float32
		for column, value := range input {
			sum += l.weights[start+column] * value
		}
		output[row] = sum
	}
	return nil
}

// InputDimensions returns the projection input width.
func (l *Linear) InputDimensions() int {
	return l.inputDimensions
}

// OutputDimensions returns the projection output width.
func (l *Linear) OutputDimensions() int {
	return l.outputDimensions
}

// Weights returns the contiguous row-major weight buffer.
// The returned slice is a zero-copy view so future training code can update it.
func (l *Linear) Weights() []float32 {
	return l.weights
}
