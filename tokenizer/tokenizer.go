package tokenizer

import (
	"fmt"
	"strings"
)

// TokenID is Quasar's compact numeric representation of a token.
type TokenID uint32

// Tokenizer is a deliberately simple word-level tokenizer.
// It assigns token IDs in first-seen order while fitting a corpus.
type Tokenizer struct {
	tokenToID map[string]TokenID
	idToToken []string
}

// New creates an empty tokenizer.
func New() *Tokenizer {
	return &Tokenizer{tokenToID: make(map[string]TokenID)}
}

// Fit learns a vocabulary from text and returns its token IDs.
func (t *Tokenizer) Fit(text string) []TokenID {
	words := strings.Fields(text)
	ids := make([]TokenID, 0, len(words))

	for _, word := range words {
		id, ok := t.tokenToID[word]
		if !ok {
			id = TokenID(len(t.idToToken))
			t.tokenToID[word] = id
			t.idToToken = append(t.idToToken, word)
		}
		ids = append(ids, id)
	}

	return ids
}

// Encode converts known words to token IDs.
func (t *Tokenizer) Encode(text string) ([]TokenID, error) {
	words := strings.Fields(text)
	ids := make([]TokenID, 0, len(words))

	for _, word := range words {
		id, ok := t.tokenToID[word]
		if !ok {
			return nil, fmt.Errorf("unknown token %q", word)
		}
		ids = append(ids, id)
	}

	return ids, nil
}

// Decode converts token IDs back to whitespace-separated text.
func (t *Tokenizer) Decode(ids []TokenID) (string, error) {
	words := make([]string, 0, len(ids))
	for _, id := range ids {
		if int(id) >= len(t.idToToken) {
			return "", fmt.Errorf("unknown token id %d", id)
		}
		words = append(words, t.idToToken[id])
	}
	return strings.Join(words, " "), nil
}

// Size returns the number of tokens in the vocabulary.
func (t *Tokenizer) Size() int {
	return len(t.idToToken)
}
