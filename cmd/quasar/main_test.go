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

func TestRunTrainsAndUsesNeuralBigram(t *testing.T) {
	path := filepath.Join(t.TempDir(), "corpus.txt")
	if err := os.WriteFile(path, []byte("a b a b a b a b"), 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	var stdout, stderr bytes.Buffer
	err := run([]string{
		"-corpus", path,
		"-prompt", "a",
		"-tokens", "3",
		"-neural",
		"-dimensions", "4",
		"-epochs", "80",
		"-learning-rate", "0.1",
	}, &stdout, &stderr)
	if err != nil {
		t.Fatalf("run() error = %v; stderr = %q", err, stderr.String())
	}
	if got, want := strings.TrimSpace(stdout.String()), "a b a b"; got != want {
		t.Fatalf("output = %q, want %q", got, want)
	}
	if !strings.HasPrefix(strings.TrimSpace(stderr.String()), "loss ") || !strings.Contains(stderr.String(), " -> ") {
		t.Fatalf("stderr = %q, want loss telemetry", stderr.String())
	}
}

func TestRunTrainsAndUsesTwoTokenContext(t *testing.T) {
	path := filepath.Join(t.TempDir(), "corpus.txt")
	if err := os.WriteFile(path, []byte("a b c a b c a b c a b c"), 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	var stdout, stderr bytes.Buffer
	err := run([]string{
		"-corpus", path,
		"-prompt", "a b",
		"-tokens", "3",
		"-neural",
		"-context-size", "2",
		"-dimensions", "8",
		"-epochs", "120",
		"-learning-rate", "0.1",
	}, &stdout, &stderr)
	if err != nil {
		t.Fatalf("run() error = %v; stderr = %q", err, stderr.String())
	}
	if got, want := strings.TrimSpace(stdout.String()), "a b c a b"; got != want {
		t.Fatalf("output = %q, want %q", got, want)
	}
	if !strings.HasPrefix(strings.TrimSpace(stderr.String()), "loss ") || !strings.Contains(stderr.String(), " -> ") {
		t.Fatalf("stderr = %q, want loss telemetry", stderr.String())
	}
}

func TestRunRejectsPromptShorterThanContext(t *testing.T) {
	path := filepath.Join(t.TempDir(), "corpus.txt")
	if err := os.WriteFile(path, []byte("a b c a b c"), 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	var stdout, stderr bytes.Buffer
	err := run([]string{
		"-corpus", path,
		"-prompt", "a",
		"-neural",
		"-context-size", "2",
	}, &stdout, &stderr)
	if err == nil {
		t.Fatal("run() error = nil, want short-prompt error")
	}
}

func TestRunInspectsEmbedding(t *testing.T) {
	path := filepath.Join(t.TempDir(), "corpus.txt")
	if err := os.WriteFile(path, []byte("moon"), 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	var stdout, stderr bytes.Buffer
	err := run([]string{"-corpus", path, "-embedding", "moon", "-dimensions", "3"}, &stdout, &stderr)
	if err != nil {
		t.Fatalf("run() error = %v; stderr = %q", err, stderr.String())
	}
	output := strings.TrimSpace(stdout.String())
	if !strings.Contains(output, "token=moon id=0 embedding=[") || !strings.HasSuffix(output, "(untrained)") {
		t.Fatalf("unexpected embedding output %q", output)
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

func TestRunRejectsGenerationAndEmbeddingTogether(t *testing.T) {
	var stdout, stderr bytes.Buffer
	err := run([]string{"-corpus", "unused", "-prompt", "moon", "-embedding", "moon"}, &stdout, &stderr)
	if err == nil {
		t.Fatal("run() error = nil, want conflicting-mode error")
	}
}
