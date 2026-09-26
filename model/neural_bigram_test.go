package model

import (
	"testing"

	"github.com/dmarro89/quasar/tokenizer"
)

func TestNeuralBigramTrainingReducesLoss(t *testing.T) {
	m, err := NewNeuralBigram(2, 4, 1)
	if err != nil {
		t.Fatalf("NewNeuralBigram() error = %v", err)
	}

	tokens := []tokenizer.TokenID{0, 1, 0, 1, 0, 1, 0, 1}
	losses, err := m.Train(tokens, 40, 0.1)
	if err != nil {
		t.Fatalf("Train() error = %v", err)
	}
	if losses[len(losses)-1] >= losses[0] {
		t.Fatalf("loss did not decrease: first=%f last=%f", losses[0], losses[len(losses)-1])
	}
}

func TestNeuralBigramLearnsNextToken(t *testing.T) {
	m, err := NewNeuralBigram(3, 6, 1)
	if err != nil {
		t.Fatalf("NewNeuralBigram() error = %v", err)
	}

	// Token 0 is repeatedly followed by token 2.
	tokens := []tokenizer.TokenID{0, 2, 0, 2, 0, 2, 0, 2}
	if _, err := m.Train(tokens, 60, 0.1); err != nil {
		t.Fatalf("Train() error = %v", err)
	}

	got, err := m.Predict(0)
	if err != nil {
		t.Fatalf("Predict() error = %v", err)
	}
	if got != 2 {
		t.Fatalf("Predict(0) = %d, want 2", got)
	}
}

func TestNeuralBigramTrainingChangesEmbedding(t *testing.T) {
	m, err := NewNeuralBigram(2, 4, 1)
	if err != nil {
		t.Fatalf("NewNeuralBigram() error = %v", err)
	}

	beforeView, err := m.Embedding(0)
	if err != nil {
		t.Fatalf("Embedding() error = %v", err)
	}
	before := append([]float32(nil), beforeView...)

	if _, err := m.Train([]tokenizer.TokenID{0, 1, 0, 1}, 10, 0.1); err != nil {
		t.Fatalf("Train() error = %v", err)
	}

	after, err := m.Embedding(0)
	if err != nil {
		t.Fatalf("Embedding() error = %v", err)
	}

	changed := false
	for i := range before {
		if before[i] != after[i] {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatal("embedding did not change after training")
	}
}

func TestNeuralBigramRejectsInvalidTrainingConfiguration(t *testing.T) {
	m, err := NewNeuralBigram(2, 4, 1)
	if err != nil {
		t.Fatalf("NewNeuralBigram() error = %v", err)
	}

	if _, err := m.Train([]tokenizer.TokenID{0, 1}, 0, 0.1); err == nil {
		t.Fatal("Train() epochs error = nil, want error")
	}
	if _, err := m.Train([]tokenizer.TokenID{0, 1}, 1, 0); err == nil {
		t.Fatal("Train() learning-rate error = nil, want error")
	}
}
