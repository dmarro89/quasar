package attention

import (
	"testing"

	"github.com/dmarro89/quasar/tensor"
)

func TestCachedRotaryMatchesUncachedPrefixAttention(t *testing.T) {
	const (
		dimensions = 2
		capacity   = 4
		seed       = 101
	)
	cached, err := NewCachedRotary(dimensions, capacity, seed)
	if err != nil {
		t.Fatalf("NewCachedRotary() error = %v", err)
	}
	uncached, err := NewRotaryProjected(dimensions, seed)
	if err != nil {
		t.Fatalf("NewRotaryProjected() error = %v", err)
	}

	vectors := []tensor.Vector{
		{1, 0},     // the
		{0, 1},     // moon
		{1, 1},     // shines
		{0.5, 0.5}, // at
	}
	for i, vector := range vectors {
		got, err := cached.Step(vector)
		if err != nil {
			t.Fatalf("Step(%d) error = %v", i, err)
		}
		want, err := uncached.LastToken(vectors[:i+1])
		if err != nil {
			t.Fatalf("LastToken(%d) error = %v", i, err)
		}
		assertProjectedVector(t, "query", got.Query, want.Query, 1e-5)
		assertProjectedVector(t, "scores", got.Scores, want.Scores, 1e-5)
		assertProjectedVector(t, "weights", got.Weights, want.Weights, 1e-5)
		assertProjectedVector(t, "output", got.Output, want.Output, 1e-5)
		if got.Position != i {
			t.Fatalf("Step(%d) Position = %d, want %d", i, got.Position, i)
		}
		if cached.Len() != i+1 {
			t.Fatalf("Step(%d) cache Len = %d, want %d", i, cached.Len(), i+1)
		}
	}
}

func TestCachedRotaryStoresRotatedKeyAndUnrotatedValue(t *testing.T) {
	cached, err := NewCachedRotary(2, 2, 1)
	if err != nil {
		t.Fatalf("NewCachedRotary() error = %v", err)
	}
	identity := []float32{
		1, 0,
		0, 1,
	}
	copy(cached.rotary.projected.query.Weights(), identity)
	copy(cached.rotary.projected.key.Weights(), identity)
	copy(cached.rotary.projected.value.Weights(), identity)

	if _, err := cached.Step(tensor.Vector{1, 0}); err != nil {
		t.Fatalf("Step(first) error = %v", err)
	}
	if _, err := cached.Step(tensor.Vector{1, 0}); err != nil {
		t.Fatalf("Step(second) error = %v", err)
	}

	key, err := cached.CachedKey(1)
	if err != nil {
		t.Fatalf("CachedKey(1) error = %v", err)
	}
	value, err := cached.CachedValue(1)
	if err != nil {
		t.Fatalf("CachedValue(1) error = %v", err)
	}
	wantKey, err := cached.rotary.rotary.Apply(tensor.Vector{1, 0}, 1)
	if err != nil {
		t.Fatalf("Apply() error = %v", err)
	}
	assertProjectedVector(t, "cached key", key, wantKey, 1e-6)
	assertProjectedVector(t, "cached value", value, tensor.Vector{1, 0}, 1e-6)
}

func TestCachedRotaryResetRestartsPositions(t *testing.T) {
	cached, err := NewCachedRotary(2, 2, 1)
	if err != nil {
		t.Fatalf("NewCachedRotary() error = %v", err)
	}
	first, err := cached.Step(tensor.Vector{1, 0})
	if err != nil {
		t.Fatalf("Step(first) error = %v", err)
	}
	firstOutput := append(tensor.Vector(nil), first.Output...)
	if _, err := cached.Step(tensor.Vector{0, 1}); err != nil {
		t.Fatalf("Step(second) error = %v", err)
	}

	cached.Reset()
	if cached.Len() != 0 {
		t.Fatalf("Len() after Reset = %d, want 0", cached.Len())
	}
	again, err := cached.Step(tensor.Vector{1, 0})
	if err != nil {
		t.Fatalf("Step(after Reset) error = %v", err)
	}
	if again.Position != 0 {
		t.Fatalf("Position after Reset = %d, want 0", again.Position)
	}
	assertProjectedVector(t, "reset output", again.Output, firstOutput, 1e-6)
}

func TestCachedRotaryRejectsInvalidInputAndOverflow(t *testing.T) {
	cached, err := NewCachedRotary(2, 1, 1)
	if err != nil {
		t.Fatalf("NewCachedRotary() error = %v", err)
	}
	if _, err := cached.Step(tensor.Vector{1}); err == nil {
		t.Fatal("Step(short input) error = nil, want shape error")
	}
	if _, err := cached.Step(tensor.Vector{1, 0}); err != nil {
		t.Fatalf("Step() error = %v", err)
	}
	if _, err := cached.Step(tensor.Vector{0, 1}); err == nil {
		t.Fatal("Step(overflow) error = nil, want capacity error")
	}
}
