package model

import (
	"testing"

	"github.com/dmarro89/quasar/tokenizer"
)

func TestNeuralContextTrainingReducesLoss(t *testing.T) {
	m, err := NewNeuralContext(3, 6, 2, 1)
	if err != nil {
		t.Fatalf("NewNeuralContext() error = %v", err)
	}

	tokens := []tokenizer.TokenID{0, 1, 2, 0, 1, 2, 0, 1, 2}
	losses, err := m.Train(tokens, 80, 0.1)
	if err != nil {
		t.Fatalf("Train() error = %v", err)
	}
	if losses[len(losses)-1] >= losses[0] {
		t.Fatalf("loss did not decrease: first=%f last=%f", losses[0], losses[len(losses)-1])
	}
}

func TestNeuralContextLearnsTwoTokenPrediction(t *testing.T) {
	m, err := NewNeuralContext(3, 8, 2, 1)
	if err != nil {
		t.Fatalf("NewNeuralContext() error = %v", err)
	}

	tokens := []tokenizer.TokenID{0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2}
	if _, err := m.Train(tokens, 120, 0.1); err != nil {
		t.Fatalf("Train() error = %v", err)
	}

	got, err := m.Predict([]tokenizer.TokenID{0, 1})
	if err != nil {
		t.Fatalf("Predict() error = %v", err)
	}
	if got != 2 {
		t.Fatalf("Predict([0 1]) = %d, want 2", got)
	}
}

func TestNeuralContextMeanPoolingIgnoresOrder(t *testing.T) {
	m, err := NewNeuralContext(3, 4, 2, 1)
	if err != nil {
		t.Fatalf("NewNeuralContext() error = %v", err)
	}

	forward, err := m.Scores([]tokenizer.TokenID{0, 1})
	if err != nil {
		t.Fatalf("Scores([0 1]) error = %v", err)
	}
	reversed, err := m.Scores([]tokenizer.TokenID{1, 0})
	if err != nil {
		t.Fatalf("Scores([1 0]) error = %v", err)
	}

	if len(forward) != len(reversed) {
		t.Fatalf("score lengths differ: %d != %d", len(forward), len(reversed))
	}
	for i := range forward {
		if forward[i] != reversed[i] {
			t.Fatalf("scores differ at %d: %f != %f", i, forward[i], reversed[i])
		}
	}
}

func TestNeuralContextGenerateSlidesWindow(t *testing.T) {
	m, err := NewNeuralContext(3, 8, 2, 1)
	if err != nil {
		t.Fatalf("NewNeuralContext() error = %v", err)
	}

	tokens := []tokenizer.TokenID{0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2}
	if _, err := m.Train(tokens, 120, 0.1); err != nil {
		t.Fatalf("Train() error = %v", err)
	}

	got, err := m.Generate([]tokenizer.TokenID{0, 1}, 3)
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	want := []tokenizer.TokenID{2, 0, 1}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("generated[%d] = %d, want %d; full=%v", i, got[i], want[i], got)
		}
	}
}

func TestNeuralContextRejectsWrongContextSize(t *testing.T) {
	m, err := NewNeuralContext(3, 4, 2, 1)
	if err != nil {
		t.Fatalf("NewNeuralContext() error = %v", err)
	}

	if _, err := m.Predict([]tokenizer.TokenID{0}); err == nil {
		t.Fatal("Predict() error = nil, want context-size error")
	}
	if _, err := m.Train([]tokenizer.TokenID{0, 1}, 1, 0.1); err == nil {
		t.Fatal("Train() error = nil, want insufficient-training-data error")
	}
}
