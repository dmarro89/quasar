package embedding

import (
	"testing"

	"github.com/dmarro89/quasar/tokenizer"
)

func TestTableLookup(t *testing.T) {
	table, err := NewTable(3, 4, 42)
	if err != nil {
		t.Fatalf("NewTable() error = %v", err)
	}

	got, err := table.Lookup(tokenizer.TokenID(1))
	if err != nil {
		t.Fatalf("Lookup() error = %v", err)
	}
	if len(got) != 4 {
		t.Fatalf("len(Lookup()) = %d, want 4", len(got))
	}
}

func TestTableInitializationIsDeterministic(t *testing.T) {
	a, _ := NewTable(2, 4, 7)
	b, _ := NewTable(2, 4, 7)

	av, _ := a.Lookup(1)
	bv, _ := b.Lookup(1)
	for i := range av {
		if av[i] != bv[i] {
			t.Fatalf("same seed differs at dimension %d: %v != %v", i, av[i], bv[i])
		}
	}
}

func TestLookupReturnsBackingRowWithoutCopy(t *testing.T) {
	table, _ := NewTable(2, 3, 1)
	first, _ := table.Lookup(0)
	first[1] = 42

	again, _ := table.Lookup(0)
	if again[1] != 42 {
		t.Fatalf("Lookup() copied row: got %v, want 42", again[1])
	}
}

func TestTableRejectsInvalidInput(t *testing.T) {
	if _, err := NewTable(0, 4, 1); err == nil {
		t.Fatal("NewTable() accepted empty vocabulary")
	}
	if _, err := NewTable(4, 0, 1); err == nil {
		t.Fatal("NewTable() accepted zero dimensions")
	}

	table, _ := NewTable(2, 4, 1)
	if _, err := table.Lookup(2); err == nil {
		t.Fatal("Lookup() accepted out-of-range token")
	}
}
