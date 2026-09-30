package attention

import (
	"fmt"
	"math"

	"github.com/dmarro89/quasar/tensor"
)

// CachedRotary performs incremental last-token RoPE attention while retaining
// projected Keys and Values from earlier decode steps.
type CachedRotary struct {
	rotary       *RotaryProjected
	cache        *KVCache
	dimensions   int
	query        tensor.Vector
	keyScratch   tensor.Vector
	valueScratch tensor.Vector
	scores       tensor.Vector
	weights      tensor.Vector
	output       tensor.Vector
}

// CachedRotaryResult exposes one incremental attention step. Its vector fields
// are read-only views of reusable internal buffers and remain valid only until
// the next Step or Reset call.
type CachedRotaryResult struct {
	Position int
	Query    tensor.Vector
	Scores   tensor.Vector
	Weights  tensor.Vector
	Output   tensor.Vector
}

// NewCachedRotary creates a single-head projected RoPE attention decoder with
// preallocated KV cache storage for capacity tokens.
func NewCachedRotary(dimensions, capacity int, seed uint64) (*CachedRotary, error) {
	rotary, err := NewRotaryProjected(dimensions, seed)
	if err != nil {
		return nil, err
	}
	cache, err := NewKVCache(dimensions, capacity)
	if err != nil {
		return nil, err
	}
	return &CachedRotary{
		rotary:       rotary,
		cache:        cache,
		dimensions:   dimensions,
		query:        make(tensor.Vector, dimensions),
		keyScratch:   make(tensor.Vector, dimensions),
		valueScratch: make(tensor.Vector, dimensions),
		scores:       make(tensor.Vector, capacity),
		weights:      make(tensor.Vector, capacity),
		output:       make(tensor.Vector, dimensions),
	}, nil
}

// Step processes exactly one new token representation. Only that token's Q/K/V
// projections are computed. The rotated K and unrotated V are appended to the
// cache and reused by all later steps.
func (c *CachedRotary) Step(input tensor.Vector) (CachedRotaryResult, error) {
	if len(input) != c.dimensions {
		return CachedRotaryResult{}, fmt.Errorf("cached attention input has %d dimensions, want %d", len(input), c.dimensions)
	}
	if c.cache.Len() == c.cache.Capacity() {
		return CachedRotaryResult{}, fmt.Errorf("cached attention KV cache is full: capacity %d", c.cache.Capacity())
	}

	position := c.cache.Len()
	projected := c.rotary.projected
	if err := projected.query.ForwardInto(input, c.query); err != nil {
		return CachedRotaryResult{}, err
	}
	if err := projected.key.ForwardInto(input, c.keyScratch); err != nil {
		return CachedRotaryResult{}, err
	}
	if err := projected.value.ForwardInto(input, c.valueScratch); err != nil {
		return CachedRotaryResult{}, err
	}
	if err := c.rotary.rotary.ApplyInto(c.query, position, c.query); err != nil {
		return CachedRotaryResult{}, err
	}
	if err := c.rotary.rotary.ApplyInto(c.keyScratch, position, c.keyScratch); err != nil {
		return CachedRotaryResult{}, err
	}
	if err := c.cache.Append(c.keyScratch, c.valueScratch); err != nil {
		return CachedRotaryResult{}, err
	}

	length := c.cache.Len()
	scores := c.scores[:length:length]
	weights := c.weights[:length:length]
	scale := float32(1 / math.Sqrt(float64(c.dimensions)))
	maxScore := float32(-math.MaxFloat32)
	for i := 0; i < length; i++ {
		key, err := c.cache.Key(i)
		if err != nil {
			return CachedRotaryResult{}, err
		}
		dot, err := tensor.Dot(c.query, key)
		if err != nil {
			return CachedRotaryResult{}, err
		}
		score := dot * scale
		scores[i] = score
		if score > maxScore {
			maxScore = score
		}
	}

	var denominator float64
	for i, score := range scores {
		value := math.Exp(float64(score - maxScore))
		weights[i] = float32(value)
		denominator += value
	}
	for i := range weights {
		weights[i] /= float32(denominator)
	}

	clear(c.output)
	for i := 0; i < length; i++ {
		value, err := c.cache.Value(i)
		if err != nil {
			return CachedRotaryResult{}, err
		}
		weight := weights[i]
		for dimension := range c.output {
			c.output[dimension] += weight * value[dimension]
		}
	}

	return CachedRotaryResult{
		Position: position,
		Query:    c.query,
		Scores:   scores,
		Weights:  weights,
		Output:   c.output,
	}, nil
}

// Reset starts a new sequence while retaining all preallocated buffers.
func (c *CachedRotary) Reset() {
	c.cache.Reset()
}

// Len returns the number of tokens currently stored in the KV cache.
func (c *CachedRotary) Len() int {
	return c.cache.Len()
}

// Capacity returns the maximum number of tokens that can be cached.
func (c *CachedRotary) Capacity() int {
	return c.cache.Capacity()
}

// CacheMemoryBytes returns bytes reserved specifically for cached K/V rows.
func (c *CachedRotary) CacheMemoryBytes() int {
	return c.cache.MemoryBytes()
}

// CachedKey returns a read-only zero-copy view of a stored rotated Key.
func (c *CachedRotary) CachedKey(index int) (tensor.Vector, error) {
	return c.cache.Key(index)
}

// CachedValue returns a read-only zero-copy view of a stored projected Value.
func (c *CachedRotary) CachedValue(index int) (tensor.Vector, error) {
	return c.cache.Value(index)
}
