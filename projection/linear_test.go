package projection

import (
	"testing"

	"github.com/dmarro89/quasar/tensor"
)

func TestLinearForwardUsesRowMajorWeights(t *testing.T) {
	linear, err := NewLinear(2, 2, 1)
	if err != nil {
		t.Fatalf("NewLinear() error = %v", err)
	}
	copy(linear.Weights(), []float32{
		1, 0,
		0, 2,
	})

	output, err := linear.Forward(tensor.Vector{2, 3})
	if err != nil {
		t.Fatalf("Forward() error = %v", err)
	}
	if got, want := output[0], float32(2); got != want {
		t.Fatalf("output[0] = %v, want %v", got, want)
	}
	if got, want := output[1], float32(6); got != want {
		t.Fatalf("output[1] = %v, want %v", got, want)
	}
}

func TestLinearForwardIntoReusesOutput(t *testing.T) {
	linear, err := NewLinear(2, 1, 1)
	if err != nil {
		t.Fatalf("NewLinear() error = %v", err)
	}
	copy(linear.Weights(), []float32{0.5, -1})

	output := tensor.Vector{99}
	if err := linear.ForwardInto(tensor.Vector{4, 1}, output); err != nil {
		t.Fatalf("ForwardInto() error = %v", err)
	}
	if got, want := output[0], float32(1); got != want {
		t.Fatalf("output[0] = %v, want %v", got, want)
	}
}

func TestLinearRejectsInvalidShapes(t *testing.T) {
	if _, err := NewLinear(0, 2, 1); err == nil {
		t.Fatal("NewLinear() error = nil, want invalid input dimension error")
	}
	if _, err := NewLinear(2, 0, 1); err == nil {
		t.Fatal("NewLinear() error = nil, want invalid output dimension error")
	}

	linear, err := NewLinear(2, 2, 1)
	if err != nil {
		t.Fatalf("NewLinear() error = %v", err)
	}
	if err := linear.ForwardInto(tensor.Vector{1}, make(tensor.Vector, 2)); err == nil {
		t.Fatal("ForwardInto() error = nil, want input shape error")
	}
	if err := linear.ForwardInto(tensor.Vector{1, 2}, make(tensor.Vector, 1)); err == nil {
		t.Fatal("ForwardInto() error = nil, want output shape error")
	}
}
