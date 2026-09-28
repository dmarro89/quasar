package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunInspectsRoPEAttention(t *testing.T) {
	path := filepath.Join(t.TempDir(), "corpus.txt")
	corpus := "the moon shines at night\nthe sun shines during the day\n"
	if err := os.WriteFile(path, []byte(corpus), 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	var stdout, stderr bytes.Buffer
	err := run([]string{
		"-corpus", path,
		"-attention", "the moon shines",
		"-rope-attention",
		"-dimensions", "4",
	}, &stdout, &stderr)
	if err != nil {
		t.Fatalf("run() error = %v; stderr = %q", err, stderr.String())
	}

	output := stdout.String()
	for _, want := range []string{
		"query=shines position=2 q=[",
		"position=0 token=the k=[",
		"position=1 token=moon k=[",
		"position=2 token=shines k=[",
		" v=[",
		" score=",
		" weight=",
		"output=[",
		"note=untrained rope projected-qkv attention",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("RoPE attention output %q does not contain %q", output, want)
		}
	}
	if got := strings.Count(output, "weight="); got != 3 {
		t.Fatalf("RoPE attention output has %d weights, want 3: %q", got, output)
	}
}

func TestRunRejectsRoPEAttentionWithOddDimensions(t *testing.T) {
	path := filepath.Join(t.TempDir(), "corpus.txt")
	if err := os.WriteFile(path, []byte("the moon shines"), 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	var stdout, stderr bytes.Buffer
	err := run([]string{
		"-corpus", path,
		"-attention", "the moon shines",
		"-rope-attention",
		"-dimensions", "3",
	}, &stdout, &stderr)
	if err == nil {
		t.Fatal("run() error = nil, want even-dimension RoPE error")
	}
}
