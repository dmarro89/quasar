package tensor

import "testing"

func TestDot(t *testing.T) {
	got, err := Dot(Vector{1, 2, 3}, Vector{4, 5, 6})
	if err != nil {
		t.Fatalf("Dot() error = %v", err)
	}
	if got != 32 {
		t.Fatalf("Dot() = %v, want 32", got)
	}
}

func TestDotRejectsDifferentLengths(t *testing.T) {
	_, err := Dot(Vector{1, 2}, Vector{1})
	if err == nil {
		t.Fatal("Dot() error = nil, want length mismatch")
	}
}

func TestVectorUsesFloat32Storage(t *testing.T) {
	v := Vector{1, 2, 3}
	if len(v) != 3 {
		t.Fatalf("len(Vector) = %d, want 3", len(v))
	}
}
