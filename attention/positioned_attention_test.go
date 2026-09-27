package attention

import (
	"math"
	"testing"

	"github.com/dmarro89/quasar/tensor"
)

func TestPositionedProjectedMakesTheSameTokenPositionDependent(t *testing.T) {
	positioned, err := NewPositionedProjected(2, 3, 1)
	if err != nil {
		t.Fatalf("NewPositionedProjected() error = %v", err)
	}
	copy(positioned.positions.Weights(), []float32{
		0.1, 0.2,
		0.3, 0.4,
		0.5, 0.6,
	})

	token := tensor.Vector{1, 2}
	_, atZero, err := positioned.Position(token, 0)
	if err != nil {
		t.Fatalf("Position(0) error = %v", err)
	}
	_, atOne, err := positioned.Position(token, 1)
	if err != nil {
		t.Fatalf("Position(1) error = %v", err)
	}
	if vectorsAlmostEqual(atZero, atOne, 1e-6) {
		t.Fatalf("same token has identical positioned representations: %v and %v", atZero, atOne)
	}
}

func TestPositionedProjectedDistinguishesEnglishWordOrder(t *testing.T) {
	positioned, err := NewPositionedProjected(2, 3, 1)
	if err != nil {
		t.Fatalf("NewPositionedProjected() error = %v", err)
	}
	identity := []float32{
		1, 0,
		0, 1,
	}
	copy(positioned.projected.query.Weights(), identity)
	copy(positioned.projected.key.Weights(), identity)
	copy(positioned.projected.value.Weights(), identity)
	copy(positioned.positions.Weights(), []float32{
		0.5, 0,
		0, 0.5,
		0, 0,
	})

	the := tensor.Vector{1, 0}
	moon := tensor.Vector{0, 1}
	shines := tensor.Vector{1, 0.2}

	ordered, err := positioned.LastToken([]tensor.Vector{the, moon, shines})
	if err != nil {
		t.Fatalf("LastToken(the moon shines) error = %v", err)
	}
	reversed, err := positioned.LastToken([]tensor.Vector{moon, the, shines})
	if err != nil {
		t.Fatalf("LastToken(moon the shines) error = %v", err)
	}
	if vectorsAlmostEqual(ordered.Output, reversed.Output, 1e-5) {
		t.Fatalf("word order did not change attention output: ordered=%v reversed=%v", ordered.Output, reversed.Output)
	}
}

func TestPositionedProjectedRejectsContextBeyondPositionTable(t *testing.T) {
	positioned, err := NewPositionedProjected(2, 2, 1)
	if err != nil {
		t.Fatalf("NewPositionedProjected() error = %v", err)
	}
	_, err = positioned.LastToken([]tensor.Vector{{1, 0}, {0, 1}, {1, 1}})
	if err == nil {
		t.Fatal("LastToken() error = nil, want maximum-position error")
	}
}

func vectorsAlmostEqual(a, b tensor.Vector, tolerance float64) bool {
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
