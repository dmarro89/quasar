package attention

import (
	"math"
	"testing"

	"github.com/dmarro89/quasar/tensor"
)

func TestLastTokenWeightsSumToOneAndFavorSimilarVectors(t *testing.T) {
	vectors := []tensor.Vector{
		{1, 0}, // the
		{0, 1}, // moon
		{1, 0}, // shines: also the query
	}

	result, err := LastToken(vectors)
	if err != nil {
		t.Fatalf("LastToken() error = %v", err)
	}

	var sum float32
	for _, weight := range result.Weights {
		sum += weight
	}
	if math.Abs(float64(sum-1)) > 1e-6 {
		t.Fatalf("attention weights sum = %f, want 1", sum)
	}
	if result.Weights[0] <= result.Weights[1] {
		t.Fatalf("weight(the) = %f, want greater than weight(moon) = %f", result.Weights[0], result.Weights[1])
	}
	if result.Weights[2] <= result.Weights[1] {
		t.Fatalf("weight(shines) = %f, want greater than weight(moon) = %f", result.Weights[2], result.Weights[1])
	}
}

func TestLastTokenOutputIsWeightedSumOfValues(t *testing.T) {
	vectors := []tensor.Vector{
		{1},
		{2},
	}

	result, err := LastToken(vectors)
	if err != nil {
		t.Fatalf("LastToken() error = %v", err)
	}

	want := result.Weights[0]*vectors[0][0] + result.Weights[1]*vectors[1][0]
	if math.Abs(float64(result.Output[0]-want)) > 1e-6 {
		t.Fatalf("output = %f, want weighted sum %f", result.Output[0], want)
	}
}

func TestLastTokenUsesScaledDotProduct(t *testing.T) {
	vectors := []tensor.Vector{
		{1, 0},
		{1, 1},
	}

	result, err := LastToken(vectors)
	if err != nil {
		t.Fatalf("LastToken() error = %v", err)
	}

	scale := float32(1 / math.Sqrt(2))
	if math.Abs(float64(result.Scores[0]-scale)) > 1e-6 {
		t.Fatalf("score[0] = %f, want %f", result.Scores[0], scale)
	}
	if math.Abs(float64(result.Scores[1]-2*scale)) > 1e-6 {
		t.Fatalf("score[1] = %f, want %f", result.Scores[1], 2*scale)
	}
}

func TestLastTokenRejectsInvalidShapes(t *testing.T) {
	if _, err := LastToken(nil); err == nil {
		t.Fatal("LastToken(nil) error = nil, want error")
	}
	if _, err := LastToken([]tensor.Vector{{}}); err == nil {
		t.Fatal("LastToken(empty vector) error = nil, want error")
	}
	if _, err := LastToken([]tensor.Vector{{1, 2}, {1}}); err == nil {
		t.Fatal("LastToken(mismatched vectors) error = nil, want error")
	}
}
