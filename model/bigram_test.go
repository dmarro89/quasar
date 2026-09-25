package model

import (
	"testing"

	"github.com/dmarro89/quasar/tokenizer"
)

func TestNextReturnsMostFrequentTransition(t *testing.T) {
	m := NewBigram(3)
	if err := m.Train([]tokenizer.TokenID{0, 1, 0, 1, 0, 2}); err != nil {
		t.Fatalf("Train() error = %v", err)
	}

	got, ok := m.Next(0)
	if !ok {
		t.Fatal("Next() ok = false, want true")
	}
	if got != 1 {
		t.Fatalf("Next(0) = %d, want 1", got)
	}
}

func TestNextBreaksTiesByLowerTokenID(t *testing.T) {
	m := NewBigram(3)
	if err := m.Train([]tokenizer.TokenID{0, 2, 0, 1}); err != nil {
		t.Fatalf("Train() error = %v", err)
	}

	got, ok := m.Next(0)
	if !ok {
		t.Fatal("Next() ok = false, want true")
	}
	if got != 1 {
		t.Fatalf("Next(0) = %d, want 1", got)
	}
}

func TestGenerateStopsWhenNoTransitionExists(t *testing.T) {
	m := NewBigram(3)
	if err := m.Train([]tokenizer.TokenID{0, 1, 2}); err != nil {
		t.Fatalf("Train() error = %v", err)
	}

	got := m.Generate(0, 10)
	want := []tokenizer.TokenID{1, 2}
	if len(got) != len(want) {
		t.Fatalf("Generate() length = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Generate()[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestTrainRejectsTokenOutsideVocabulary(t *testing.T) {
	m := NewBigram(2)
	if err := m.Train([]tokenizer.TokenID{0, 2}); err == nil {
		t.Fatal("Train() error = nil, want out-of-vocabulary error")
	}
}
