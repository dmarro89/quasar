package position

import (
	"math"
	"testing"

	"github.com/dmarro89/quasar/tensor"
)

func TestRotaryPositionZeroIsIdentity(t *testing.T) {
	rope, err := NewRotary(4, DefaultRoPEBase)
	if err != nil {
		t.Fatalf("NewRotary() error = %v", err)
	}
	input := tensor.Vector{0.25, -0.5, 0.75, 1.0}
	output, err := rope.Apply(input, 0)
	if err != nil {
		t.Fatalf("Apply() error = %v", err)
	}
	assertCloseVector(t, output, input, 1e-6)
}

func TestRotaryTwoDimensionsRotateByOneRadianAtPositionOne(t *testing.T) {
	rope, err := NewRotary(2, DefaultRoPEBase)
	if err != nil {
		t.Fatalf("NewRotary() error = %v", err)
	}
	output, err := rope.Apply(tensor.Vector{1, 0}, 1)
	if err != nil {
		t.Fatalf("Apply() error = %v", err)
	}
	want := tensor.Vector{float32(math.Cos(1)), float32(math.Sin(1))}
	assertCloseVector(t, output, want, 1e-6)
}

func TestRotaryDotProductDependsOnRelativePosition(t *testing.T) {
	rope, err := NewRotary(4, DefaultRoPEBase)
	if err != nil {
		t.Fatalf("NewRotary() error = %v", err)
	}
	query := tensor.Vector{0.2, -0.3, 0.7, 0.1}
	key := tensor.Vector{-0.4, 0.8, 0.5, -0.2}

	q2, err := rope.Apply(query, 2)
	if err != nil {
		t.Fatalf("Apply(query, 2) error = %v", err)
	}
	k5, err := rope.Apply(key, 5)
	if err != nil {
		t.Fatalf("Apply(key, 5) error = %v", err)
	}
	q7, err := rope.Apply(query, 7)
	if err != nil {
		t.Fatalf("Apply(query, 7) error = %v", err)
	}
	k10, err := rope.Apply(key, 10)
	if err != nil {
		t.Fatalf("Apply(key, 10) error = %v", err)
	}

	first, err := tensor.Dot(q2, k5)
	if err != nil {
		t.Fatalf("Dot(first) error = %v", err)
	}
	shifted, err := tensor.Dot(q7, k10)
	if err != nil {
		t.Fatalf("Dot(shifted) error = %v", err)
	}
	if math.Abs(float64(first-shifted)) > 1e-5 {
		t.Fatalf("same relative distance changed dot product: %f vs %f", first, shifted)
	}
}

func TestRotaryRejectsOddDimensions(t *testing.T) {
	if _, err := NewRotary(3, DefaultRoPEBase); err == nil {
		t.Fatal("NewRotary() error = nil, want odd-dimension error")
	}
}

func assertCloseVector(t *testing.T, got, want tensor.Vector, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("vector length = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > tolerance {
			t.Fatalf("vector[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}
