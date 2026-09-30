package main

import (
	"fmt"
	"io"

	"github.com/dmarro89/quasar/attention"
	"github.com/dmarro89/quasar/tokenizer"
)

func inspectCachedRotaryAttention(tok *tokenizer.Tokenizer, text string, dimensions int, stdout io.Writer) error {
	ids, vectors, err := attentionVectors(tok, text, dimensions)
	if err != nil {
		return err
	}

	cached, err := attention.NewCachedRotary(dimensions, len(vectors), embeddingSeed+100)
	if err != nil {
		return fmt.Errorf("create cached rotary attention: %w", err)
	}
	fmt.Fprintf(stdout, "cache_capacity=%d cache_memory_bytes=%d\n", cached.Capacity(), cached.CacheMemoryBytes())

	for i, id := range ids {
		result, err := cached.Step(vectors[i])
		if err != nil {
			return fmt.Errorf("cached attention step %d: %w", i, err)
		}
		token, err := tok.Decode([]tokenizer.TokenID{id})
		if err != nil {
			return fmt.Errorf("decode cached attention token: %w", err)
		}
		key, err := cached.CachedKey(i)
		if err != nil {
			return fmt.Errorf("read cached key: %w", err)
		}
		value, err := cached.CachedValue(i)
		if err != nil {
			return fmt.Errorf("read cached value: %w", err)
		}

		fmt.Fprintf(stdout, "step=%d position=%d token=%s cache=%d/%d\n", i, result.Position, token, cached.Len(), cached.Capacity())
		fmt.Fprintf(stdout, "q=%v\n", result.Query)
		fmt.Fprintf(stdout, "cached_k=%v cached_v=%v\n", key, value)
		fmt.Fprintf(stdout, "scores=%v weights=%v\n", result.Scores, result.Weights)
		fmt.Fprintf(stdout, "output=%v\n", result.Output)
	}
	fmt.Fprintln(stdout, "note=untrained incremental rope attention with KV cache")
	return nil
}
