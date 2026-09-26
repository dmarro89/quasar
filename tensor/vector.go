package tensor

import "fmt"

// Vector is a one-dimensional contiguous collection of float32 values.
// float32 matches the common baseline representation used by inference engines
// while keeping memory usage half that of float64.
type Vector []float32

// Dot computes the dot product between two vectors of equal length.
func Dot(a, b Vector) (float32, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("dot product length mismatch: %d != %d", len(a), len(b))
	}

	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum, nil
}
