package tokenizer

import "testing"

func TestFitUsesStableFirstSeenIDs(t *testing.T) {
	tok := New()
	ids := tok.Fit("moon sun moon")

	want := []TokenID{0, 1, 0}
	if len(ids) != len(want) {
		t.Fatalf("got %d ids, want %d", len(ids), len(want))
	}
	for i := range want {
		if ids[i] != want[i] {
			t.Fatalf("ids[%d] = %d, want %d", i, ids[i], want[i])
		}
	}
	if tok.Size() != 2 {
		t.Fatalf("vocabulary size = %d, want 2", tok.Size())
	}
}

func TestEncodeRejectsUnknownToken(t *testing.T) {
	tok := New()
	tok.Fit("moon sun")

	if _, err := tok.Encode("moon earth"); err == nil {
		t.Fatal("Encode() error = nil, want unknown-token error")
	}
}

func TestDecodeRoundTrip(t *testing.T) {
	tok := New()
	ids := tok.Fit("quasar learns")

	got, err := tok.Decode(ids)
	if err != nil {
		t.Fatalf("Decode() error = %v", err)
	}
	if got != "quasar learns" {
		t.Fatalf("Decode() = %q, want %q", got, "quasar learns")
	}
}
