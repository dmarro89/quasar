package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunGeneratesDeterministicText(t *testing.T) {
	corpus := "the moon shines at night\nthe moon moves across the sky\n"
	path := filepath.Join(t.TempDir(), "corpus.txt")
	if err := os.WriteFile(path, []byte(corpus), 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	var stdout, stderr bytes.Buffer
	err := run([]string{"-corpus", path, "-prompt", "the", "-tokens", "4"}, &stdout, &stderr)
	if err != nil {
		t.Fatalf("run() error = %v; stderr = %q", err, stderr.String())
	}
	if got, want := strings.TrimSpace(stdout.String()), "the moon shines at night"; got != want {
		t.Fatalf("output = %q, want %q", got, want)
	}
}

func TestRunRejectsUnknownPromptToken(t *testing.T) {
	path := filepath.Join(t.TempDir(), "corpus.txt")
	if err := os.WriteFile(path, []byte("moon shines"), 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	var stdout, stderr bytes.Buffer
	err := run([]string{"-corpus", path, "-prompt", "sun"}, &stdout, &stderr)
	if err == nil {
		t.Fatal("run() error = nil, want unknown-token error")
	}
}
