package attention

import (
	"math"
	"testing"

	"github.com/dmarro89/quasar/tensor"
)

func TestProjectedCreatesDistinctQKV(t *testing.T) {
	projected, err := NewProjected(2, 1)
	if err != nil {
		t.Fatalf("NewProjected() error = %v", err)
	}
	copy(projected.query.Weights(), []float32{
		1, 0,
		0, 1,
	})
	copy(projected.key.Weights(), []float32{
		1, 0,
		0, -1,
	})
	copy(projected.value.Weights(), []float32{
		2, 0,
		0, 3,
	})

	query, key, value, err := projected.Project(tensor.Vector{1, 1})
	if err != nil {
		t.Fatalf("Project() error = %v", err)
	}
	assertVector(t, "query", query, tensor.Vector{1, 1}, 1e-6)
	assertVector(t, "key", key, tensor.Vector{1, -1}, 1e-6)
	assertVector(t, "value", value, tensor.Vector{2, 3}, 1e-6)
}

func TestProjectedLastTokenUsesProjectedKeysAndValues(t *testing.T) {
	projected, err := NewProjected(2, 1)
	if err != nil {
		t.Fatalf("NewProjected() error = %v", err)
	}
	copy(projected.query.Weights(), []float32{
		1, 0,
		0, 1,
	})
	copy(projected.key.Weights(), []float32{
		1, 0,
		0, -1,
	})
	copy(projected.value.Weights(), []float32{
		2, 0,
		0, 3,
	})

	vectors := []tensor.Vector{
		{1, 0}, // the
		{0, 1}, // moon
		{1, 1}, // shines
	}
	result, err := projected.LastToken(vectors)
	if err != nil {
		t.Fatalf("LastToken() error = %v", err)
	}

	scale := float32(1 / math.Sqrt(2))
	assertVector(t, "scores", result.Scores, tensor.Vector{scale, -scale, 0}, 1e-5)

	var weightSum float32
	for _, weight := range result.Weights {
		weightSum += weight
	}
	if math.Abs(float64(weightSum-1)) > 1e-5 {
		t.Fatalf("weight sum = %v, want 1", weightSum)
	}

	wantOutput := tensor.Vector{
		result.Weights[0]*2 + result.Weights[2]*2,
		result.Weights[1]*3 + result.Weights[2]*3,
	}
	assertVector(t, "output", result.Output, wantOutput, 1e-5)
}

func TestProjectedRejectsInvalidInput(t *testing.T) {
	if _, err := NewProjected(0, 1); err == nil {
		t.Fatal("NewProjected() error = nil, want dimension error")
	}
	projected, err := NewProjected(2, 1)
	if err != nil {
		t.Fatalf("NewProjected() error = %v", err)
	}
	if _, err := projected.LastToken(nil); err == nil {
		t.Fatal("LastToken() error = nil, want empty input error")
	}
	if _, err := projected.LastToken([]tensor.Vector{{1}}); err == nil {
		t.Fatal("LastToken() error = nil, want shape error")
	}
}
