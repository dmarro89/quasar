package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunInspectsPositionedEnglishAttention(t *testing.T) {
	path := filepath.Join(t.TempDir(), "corpus.txt")
	corpus := "the moon shines at night\nthe sun shines during the day\n"
	if err := os.WriteFile(path, []byte(corpus), 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	var stdout, stderr bytes.Buffer
	err := run([]string{
		"-corpus", path,
		"-attention", "the moon shines",
		"-positioned-attention",
		"-dimensions", "3",
	}, &stdout, &stderr)
	if err != nil {
		t.Fatalf("run() error = %v; stderr = %q", err, stderr.String())
	}

	output := stdout.String()
	for _, want := range []string{
		"query=shines q=[",
		"position=0 token=the embedding=[",
		"position=1 token=moon embedding=[",
		"position=2 token=shines embedding=[",
		" positional=[",
		" combined=[",
		" score=",
		" weight=",
		"output=[",
		"note=untrained absolute-positional projected-qkv attention",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("positioned attention output %q does not contain %q", output, want)
		}
	}
	if got := strings.Count(output, "weight="); got != 3 {
		t.Fatalf("positioned attention output has %d weights, want 3: %q", got, output)
	}
}

func TestRunRejectsPositionedAttentionWithoutAttentionText(t *testing.T) {
	var stdout, stderr bytes.Buffer
	err := run([]string{"-corpus", "unused", "-prompt", "the", "-positioned-attention"}, &stdout, &stderr)
	if err == nil {
		t.Fatal("run() error = nil, want positioned-attention mode error")
	}
}

func TestRunRejectsTwoAttentionInspectionModes(t *testing.T) {
	var stdout, stderr bytes.Buffer
	err := run([]string{
		"-corpus", "unused",
		"-attention", "the moon",
		"-projected-attention",
		"-positioned-attention",
	}, &stdout, &stderr)
	if err == nil {
		t.Fatal("run() error = nil, want conflicting attention mode error")
	}
}
