package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunInspectsIncrementalKVCache(t *testing.T) {
	path := filepath.Join(t.TempDir(), "corpus.txt")
	if err := os.WriteFile(path, []byte("the moon shines at night\n"), 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	var stdout, stderr bytes.Buffer
	err := run([]string{
		"-corpus", path,
		"-attention", "the moon shines at night",
		"-cached-rope-attention",
		"-dimensions", "4",
	}, &stdout, &stderr)
	if err != nil {
		t.Fatalf("run() error = %v; stderr = %q", err, stderr.String())
	}

	output := stdout.String()
	for _, want := range []string{
		"cache_capacity=5 cache_memory_bytes=160",
		"step=0 position=0 token=the cache=1/5",
		"step=2 position=2 token=shines cache=3/5",
		"step=4 position=4 token=night cache=5/5",
		"cached_k=",
		"cached_v=",
		"note=untrained incremental rope attention with KV cache",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("output does not contain %q:\n%s", want, output)
		}
	}
}

func TestRunRejectsCachedRoPEWithoutAttention(t *testing.T) {
	path := filepath.Join(t.TempDir(), "corpus.txt")
	if err := os.WriteFile(path, []byte("the moon shines\n"), 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	var stdout, stderr bytes.Buffer
	err := run([]string{"-corpus", path, "-prompt", "the", "-cached-rope-attention"}, &stdout, &stderr)
	if err == nil || !strings.Contains(err.Error(), "-cached-rope-attention requires -attention") {
		t.Fatalf("run() error = %v, want cached-attention validation error", err)
	}
}
