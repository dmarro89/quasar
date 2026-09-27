package model

import (
	"slices"
	"testing"

	"github.com/dmarro89/quasar/tokenizer"
)

func TestNeuralOrderedContextTrainingReducesLoss(t *testing.T) {
	m, err := NewNeuralOrderedContext(3, 4, 2, 1)
	if err != nil {
		t.Fatalf("NewNeuralOrderedContext() error = %v", err)
	}

	// Repeating English-shaped pattern: "the moon shines".
	tokens := []tokenizer.TokenID{0, 1, 2, 0, 1, 2, 0, 1, 2}
	losses, err := m.Train(tokens, 80, 0.1)
	if err != nil {
		t.Fatalf("Train() error = %v", err)
	}
	if losses[len(losses)-1] >= losses[0] {
		t.Fatalf("loss did not decrease: first=%f last=%f", losses[0], losses[len(losses)-1])
	}
}

func TestNeuralOrderedContextLearnsTheMoonShines(t *testing.T) {
	m, err := NewNeuralOrderedContext(3, 6, 2, 1)
	if err != nil {
		t.Fatalf("NewNeuralOrderedContext() error = %v", err)
	}

	// 0=the, 1=moon, 2=shines.
	tokens := []tokenizer.TokenID{0, 1, 2, 0, 1, 2, 0, 1, 2}
	if _, err := m.Train(tokens, 100, 0.1); err != nil {
		t.Fatalf("Train() error = %v", err)
	}

	got, err := m.Predict([]tokenizer.TokenID{0, 1})
	if err != nil {
		t.Fatalf("Predict() error = %v", err)
	}
	if got != 2 {
		t.Fatalf("Predict([the moon]) = %d, want 2 (shines)", got)
	}
}

func TestNeuralOrderedContextPreservesTokenOrder(t *testing.T) {
	m, err := NewNeuralOrderedContext(2, 3, 2, 1)
	if err != nil {
		t.Fatalf("NewNeuralOrderedContext() error = %v", err)
	}

	theMoon := make([]float32, m.contextWidth)
	moonThe := make([]float32, m.contextWidth)
	if err := m.contextInto([]tokenizer.TokenID{0, 1}, theMoon); err != nil {
		t.Fatalf("contextInto([the moon]) error = %v", err)
	}
	if err := m.contextInto([]tokenizer.TokenID{1, 0}, moonThe); err != nil {
		t.Fatalf("contextInto([moon the]) error = %v", err)
	}

	if slices.Equal(theMoon, moonThe) {
		t.Fatalf("ordered contexts are equal: [the moon]=%v [moon the]=%v", theMoon, moonThe)
	}

	the, _ := m.input.Lookup(0)
	moon, _ := m.input.Lookup(1)
	if !slices.Equal(theMoon[:m.dimensions], the) || !slices.Equal(theMoon[m.dimensions:], moon) {
		t.Fatalf("[the moon] does not preserve positional slots: got %v", theMoon)
	}
	if !slices.Equal(moonThe[:m.dimensions], moon) || !slices.Equal(moonThe[m.dimensions:], the) {
		t.Fatalf("[moon the] does not preserve positional slots: got %v", moonThe)
	}
}

func TestNeuralOrderedContextRejectsWrongContextSize(t *testing.T) {
	m, err := NewNeuralOrderedContext(3, 4, 2, 1)
	if err != nil {
		t.Fatalf("NewNeuralOrderedContext() error = %v", err)
	}

	if _, err := m.Scores([]tokenizer.TokenID{0}); err == nil {
		t.Fatal("Scores() error = nil, want context-size error")
	}
}
