package position

import (
	"testing"

	"github.com/dmarro89/quasar/tensor"
)

func TestAddIntoMakesTheSameTokenPositionDependent(t *testing.T) {
	table, err := NewTable(3, 2, 1)
	if err != nil {
		t.Fatalf("NewTable() error = %v", err)
	}
	copy(table.Weights(), []float32{
		0.1, 0.2,
		0.3, 0.4,
		0.5, 0.6,
	})

	token := tensor.Vector{1, 2}
	atZero := make(tensor.Vector, 2)
	atOne := make(tensor.Vector, 2)
	if err := table.AddInto(token, 0, atZero); err != nil {
		t.Fatalf("AddInto(position 0) error = %v", err)
	}
	if err := table.AddInto(token, 1, atOne); err != nil {
		t.Fatalf("AddInto(position 1) error = %v", err)
	}

	assertVector(t, "position 0", atZero, tensor.Vector{1.1, 2.2})
	assertVector(t, "position 1", atOne, tensor.Vector{1.3, 2.4})
	if atZero[0] == atOne[0] && atZero[1] == atOne[1] {
		t.Fatal("same token representation is identical across positions")
	}
}

func TestLookupRejectsOutOfRangePosition(t *testing.T) {
	table, err := NewTable(2, 2, 1)
	if err != nil {
		t.Fatalf("NewTable() error = %v", err)
	}
	if _, err := table.Lookup(2); err == nil {
		t.Fatal("Lookup() error = nil, want out-of-range error")
	}
}

func TestAddIntoRejectsShapeMismatch(t *testing.T) {
	table, err := NewTable(2, 2, 1)
	if err != nil {
		t.Fatalf("NewTable() error = %v", err)
	}
	if err := table.AddInto(tensor.Vector{1}, 0, make(tensor.Vector, 2)); err == nil {
		t.Fatal("AddInto() input error = nil, want shape error")
	}
	if err := table.AddInto(tensor.Vector{1, 2}, 0, make(tensor.Vector, 1)); err == nil {
		t.Fatal("AddInto() output error = nil, want shape error")
	}
}

func assertVector(t *testing.T, name string, got, want tensor.Vector) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length = %d, want %d", name, len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s[%d] = %v, want %v", name, i, got[i], want[i])
		}
	}
}
