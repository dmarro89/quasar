package model

import (
	"fmt"

	"github.com/dmarro89/quasar/tokenizer"
)

// Bigram stores observed next-token counts for each token.
// Rows are allocated lazily so unused vocabulary entries consume no map storage.
type Bigram struct {
	transitions []map[tokenizer.TokenID]uint32
}

// NewBigram creates a model sized for the supplied vocabulary.
func NewBigram(vocabularySize int) *Bigram {
	return &Bigram{transitions: make([]map[tokenizer.TokenID]uint32, vocabularySize)}
}

// Train counts every adjacent token pair in a sequence.
func (m *Bigram) Train(tokens []tokenizer.TokenID) error {
	for i := 0; i+1 < len(tokens); i++ {
		from, to := tokens[i], tokens[i+1]
		if int(from) >= len(m.transitions) || int(to) >= len(m.transitions) {
			return fmt.Errorf("transition %d -> %d exceeds vocabulary size %d", from, to, len(m.transitions))
		}
		if m.transitions[from] == nil {
			m.transitions[from] = make(map[tokenizer.TokenID]uint32)
		}
		m.transitions[from][to]++
	}
	return nil
}

// Next returns the highest-scoring next token.
// Ties are resolved by the lower token ID so generation is deterministic.
func (m *Bigram) Next(current tokenizer.TokenID) (tokenizer.TokenID, bool) {
	if int(current) >= len(m.transitions) {
		return 0, false
	}

	var best tokenizer.TokenID
	var bestCount uint32
	found := false
	for candidate, count := range m.transitions[current] {
		if !found || count > bestCount || (count == bestCount && candidate < best) {
			best, bestCount, found = candidate, count, true
		}
	}
	return best, found
}

// Generate predicts at most maxNewTokens tokens after seed.
func (m *Bigram) Generate(seed tokenizer.TokenID, maxNewTokens int) []tokenizer.TokenID {
	if maxNewTokens <= 0 {
		return nil
	}

	generated := make([]tokenizer.TokenID, 0, maxNewTokens)
	current := seed
	for len(generated) < maxNewTokens {
		next, ok := m.Next(current)
		if !ok {
			break
		}
		generated = append(generated, next)
		current = next
	}
	return generated
}
