package attention

import (
	"fmt"

	"github.com/dmarro89/quasar/tensor"
)

// KVCache stores projected Keys and Values for previously processed tokens.
// Rows are kept in two contiguous buffers so each cached token occupies one
// fixed-width slice and no per-token allocation is required after creation.
type KVCache struct {
	dimensions int
	capacity   int
	length     int
	keys       []float32
	values     []float32
}

// NewKVCache preallocates storage for capacity tokens.
func NewKVCache(dimensions, capacity int) (*KVCache, error) {
	if dimensions <= 0 {
		return nil, fmt.Errorf("KV cache dimensions must be positive: %d", dimensions)
	}
	if capacity <= 0 {
		return nil, fmt.Errorf("KV cache capacity must be positive: %d", capacity)
	}
	return &KVCache{
		dimensions: dimensions,
		capacity:   capacity,
		keys:       make([]float32, dimensions*capacity),
		values:     make([]float32, dimensions*capacity),
	}, nil
}

// Append copies one Key/Value pair into the next cache row.
func (c *KVCache) Append(key, value tensor.Vector) error {
	if len(key) != c.dimensions {
		return fmt.Errorf("KV cache key has %d dimensions, want %d", len(key), c.dimensions)
	}
	if len(value) != c.dimensions {
		return fmt.Errorf("KV cache value has %d dimensions, want %d", len(value), c.dimensions)
	}
	if c.length == c.capacity {
		return fmt.Errorf("KV cache is full: capacity %d", c.capacity)
	}

	start := c.length * c.dimensions
	end := start + c.dimensions
	copy(c.keys[start:end], key)
	copy(c.values[start:end], value)
	c.length++
	return nil
}

// Key returns a zero-copy view of the cached Key at index.
func (c *KVCache) Key(index int) (tensor.Vector, error) {
	if index < 0 || index >= c.length {
		return nil, fmt.Errorf("KV cache key index %d out of range [0,%d)", index, c.length)
	}
	start := index * c.dimensions
	return tensor.Vector(c.keys[start : start+c.dimensions]), nil
}

// Value returns a zero-copy view of the cached Value at index.
func (c *KVCache) Value(index int) (tensor.Vector, error) {
	if index < 0 || index >= c.length {
		return nil, fmt.Errorf("KV cache value index %d out of range [0,%d)", index, c.length)
	}
	start := index * c.dimensions
	return tensor.Vector(c.values[start : start+c.dimensions]), nil
}

// Reset makes the preallocated cache reusable without clearing or reallocating
// its backing storage. Old rows become inaccessible and are overwritten by
// subsequent Append calls.
func (c *KVCache) Reset() {
	c.length = 0
}

// Len returns the number of cached tokens.
func (c *KVCache) Len() int {
	return c.length
}

// Capacity returns the maximum number of tokens the cache can hold.
func (c *KVCache) Capacity() int {
	return c.capacity
}

// Dimensions returns the width of each cached Key and Value.
func (c *KVCache) Dimensions() int {
	return c.dimensions
}

// MemoryBytes returns the bytes reserved by the two float32 backing buffers.
func (c *KVCache) MemoryBytes() int {
	return 2 * c.capacity * c.dimensions * 4
}
