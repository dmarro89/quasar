package attention

import (
	"math"
	"testing"

	"github.com/dmarro89/quasar/tensor"
)

func TestRotaryProjectedRotatesQKButNotV(t *testing.T) {
	rotary, err := NewRotaryProjected(2, 1)
	if err != nil {
		t.Fatalf("NewRotaryProjected() error = %v", err)
	}
	identity := []float32{
		1, 0,
		0, 1,
	}
	copy(rotary.projected.query.Weights(), identity)
	copy(rotary.projected.key.Weights(), identity)
	copy(rotary.projected.value.Weights(), identity)

	query, key, value, err := rotary.Project(tensor.Vector{1, 0}, 1)
	if err != nil {
		t.Fatalf("Project() error = %v", err)
	}
	rotated := tensor.Vector{float32(math.Cos(1)), float32(math.Sin(1))}
	assertProjectedVector(t, "query", query, rotated, 1e-6)
	assertProjectedVector(t, "key", key, rotated, 1e-6)
	assertProjectedVector(t, "value", value, tensor.Vector{1, 0}, 1e-6)
}

func TestRotaryProjectedAttentionDistinguishesEnglishWordOrder(t *testing.T) {
	rotary, err := NewRotaryProjected(2, 1)
	if err != nil {
		t.Fatalf("NewRotaryProjected() error = %v", err)
	}
	identity := []float32{
		1, 0,
		0, 1,
	}
	copy(rotary.projected.query.Weights(), identity)
	copy(rotary.projected.key.Weights(), identity)
	copy(rotary.projected.value.Weights(), identity)

	the := tensor.Vector{1, 0}
	moon := tensor.Vector{0, 1}
	shines := tensor.Vector{1, 1}

	ordered, err := rotary.LastToken([]tensor.Vector{the, moon, shines})
	if err != nil {
		t.Fatalf("LastToken(ordered) error = %v", err)
	}
	swapped, err := rotary.LastToken([]tensor.Vector{moon, the, shines})
	if err != nil {
		t.Fatalf("LastToken(swapped) error = %v", err)
	}

	if rotaryVectorsAlmostEqual(ordered.Output, swapped.Output, 1e-6) {
		t.Fatalf("RoPE attention produced the same output for different word order: %v", ordered.Output)
	}
}

func TestRotaryProjectedRejectsOddDimensions(t *testing.T) {
	if _, err := NewRotaryProjected(3, 1); err == nil {
		t.Fatal("NewRotaryProjected() error = nil, want even-dimension error")
	}
}

func rotaryVectorsAlmostEqual(a, b tensor.Vector, tolerance float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(float64(a[i]-b[i])) > tolerance {
			return false
		}
	}
	return true
}
