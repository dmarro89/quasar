package attention

import (
	"testing"

	"github.com/dmarro89/quasar/tensor"
)

func TestKVCacheStoresContiguousRows(t *testing.T) {
	cache, err := NewKVCache(2, 3)
	if err != nil {
		t.Fatalf("NewKVCache() error = %v", err)
	}
	if err := cache.Append(tensor.Vector{1, 2}, tensor.Vector{3, 4}); err != nil {
		t.Fatalf("Append(first) error = %v", err)
	}
	if err := cache.Append(tensor.Vector{5, 6}, tensor.Vector{7, 8}); err != nil {
		t.Fatalf("Append(second) error = %v", err)
	}

	key, err := cache.Key(1)
	if err != nil {
		t.Fatalf("Key(1) error = %v", err)
	}
	value, err := cache.Value(1)
	if err != nil {
		t.Fatalf("Value(1) error = %v", err)
	}
	assertProjectedVector(t, "key", key, tensor.Vector{5, 6}, 0)
	assertProjectedVector(t, "value", value, tensor.Vector{7, 8}, 0)
	if cache.Len() != 2 {
		t.Fatalf("Len() = %d, want 2", cache.Len())
	}
	if cache.MemoryBytes() != 48 {
		t.Fatalf("MemoryBytes() = %d, want 48", cache.MemoryBytes())
	}
}

func TestKVCacheCopiesAppendedVectors(t *testing.T) {
	cache, err := NewKVCache(2, 1)
	if err != nil {
		t.Fatalf("NewKVCache() error = %v", err)
	}
	key := tensor.Vector{1, 2}
	value := tensor.Vector{3, 4}
	if err := cache.Append(key, value); err != nil {
		t.Fatalf("Append() error = %v", err)
	}
	key[0] = 99
	value[0] = 99

	cachedKey, _ := cache.Key(0)
	cachedValue, _ := cache.Value(0)
	assertProjectedVector(t, "key", cachedKey, tensor.Vector{1, 2}, 0)
	assertProjectedVector(t, "value", cachedValue, tensor.Vector{3, 4}, 0)
}

func TestKVCacheResetReusesCapacity(t *testing.T) {
	cache, err := NewKVCache(2, 1)
	if err != nil {
		t.Fatalf("NewKVCache() error = %v", err)
	}
	if err := cache.Append(tensor.Vector{1, 2}, tensor.Vector{3, 4}); err != nil {
		t.Fatalf("Append() error = %v", err)
	}
	cache.Reset()
	if cache.Len() != 0 {
		t.Fatalf("Len() after Reset = %d, want 0", cache.Len())
	}
	if err := cache.Append(tensor.Vector{5, 6}, tensor.Vector{7, 8}); err != nil {
		t.Fatalf("Append() after Reset error = %v", err)
	}
	key, _ := cache.Key(0)
	assertProjectedVector(t, "reused key", key, tensor.Vector{5, 6}, 0)
}

func TestKVCacheRejectsInvalidShapesAndOverflow(t *testing.T) {
	if _, err := NewKVCache(0, 1); err == nil {
		t.Fatal("NewKVCache(0, 1) error = nil, want dimension error")
	}
	if _, err := NewKVCache(2, 0); err == nil {
		t.Fatal("NewKVCache(2, 0) error = nil, want capacity error")
	}

	cache, err := NewKVCache(2, 1)
	if err != nil {
		t.Fatalf("NewKVCache() error = %v", err)
	}
	if err := cache.Append(tensor.Vector{1}, tensor.Vector{1, 2}); err == nil {
		t.Fatal("Append(short key) error = nil, want shape error")
	}
	if err := cache.Append(tensor.Vector{1, 2}, tensor.Vector{1}); err == nil {
		t.Fatal("Append(short value) error = nil, want shape error")
	}
	if err := cache.Append(tensor.Vector{1, 2}, tensor.Vector{3, 4}); err != nil {
		t.Fatalf("Append() error = %v", err)
	}
	if err := cache.Append(tensor.Vector{5, 6}, tensor.Vector{7, 8}); err == nil {
		t.Fatal("Append(overflow) error = nil, want capacity error")
	}
	if _, err := cache.Key(1); err == nil {
		t.Fatal("Key(1) error = nil, want range error")
	}
	if _, err := cache.Value(-1); err == nil {
		t.Fatal("Value(-1) error = nil, want range error")
	}
}
