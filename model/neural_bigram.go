package model

import (
	"fmt"
	"math"

	"github.com/dmarro89/quasar/embedding"
	"github.com/dmarro89/quasar/tensor"
	"github.com/dmarro89/quasar/tokenizer"
)

// NeuralBigram predicts the next token from the embedding of the current token.
// It intentionally uses only one previous token so training can be understood
// before Quasar introduces larger contexts or attention.
type NeuralBigram struct {
	input      *embedding.Table
	output     *embedding.Table
	vocabulary int
	dimensions int
}

// NewNeuralBigram creates a deterministically initialized neural bigram model.
func NewNeuralBigram(vocabularySize, dimensions int, seed uint64) (*NeuralBigram, error) {
	input, err := embedding.NewTable(vocabularySize, dimensions, seed)
	if err != nil {
		return nil, fmt.Errorf("create input embeddings: %w", err)
	}
	output, err := embedding.NewTable(vocabularySize, dimensions, seed+1)
	if err != nil {
		return nil, fmt.Errorf("create output weights: %w", err)
	}

	return &NeuralBigram{
		input:      input,
		output:     output,
		vocabulary: vocabularySize,
		dimensions: dimensions,
	}, nil
}

// Embedding returns the current learned input embedding for a token.
func (m *NeuralBigram) Embedding(id tokenizer.TokenID) (tensor.Vector, error) {
	return m.input.Lookup(id)
}

// Scores returns one logit for every possible next token.
func (m *NeuralBigram) Scores(current tokenizer.TokenID) ([]float32, error) {
	logits := make([]float32, m.vocabulary)
	if err := m.scoresInto(current, logits); err != nil {
		return nil, err
	}
	return logits, nil
}

// Predict returns the token with the highest logit.
func (m *NeuralBigram) Predict(current tokenizer.TokenID) (tokenizer.TokenID, error) {
	logits, err := m.Scores(current)
	if err != nil {
		return 0, err
	}

	best := tokenizer.TokenID(0)
	for i := 1; i < len(logits); i++ {
		if logits[i] > logits[best] {
			best = tokenizer.TokenID(i)
		}
	}
	return best, nil
}

// Generate predicts at most maxNewTokens tokens autoregressively.
func (m *NeuralBigram) Generate(seed tokenizer.TokenID, maxNewTokens int) ([]tokenizer.TokenID, error) {
	if maxNewTokens <= 0 {
		return nil, nil
	}

	generated := make([]tokenizer.TokenID, 0, maxNewTokens)
	current := seed
	for len(generated) < maxNewTokens {
		next, err := m.Predict(current)
		if err != nil {
			return nil, err
		}
		generated = append(generated, next)
		current = next
	}
	return generated, nil
}

// Train learns adjacent-token predictions with softmax cross-entropy and SGD.
// It returns the average loss for every epoch so callers can observe learning.
func (m *NeuralBigram) Train(tokens []tokenizer.TokenID, epochs int, learningRate float32) ([]float64, error) {
	if len(tokens) < 2 {
		return nil, fmt.Errorf("training requires at least two tokens")
	}
	if epochs <= 0 {
		return nil, fmt.Errorf("epochs must be positive: %d", epochs)
	}
	if learningRate <= 0 {
		return nil, fmt.Errorf("learning rate must be positive: %f", learningRate)
	}
	for _, id := range tokens {
		if int(id) >= m.vocabulary {
			return nil, fmt.Errorf("token id %d exceeds vocabulary size %d", id, m.vocabulary)
		}
	}

	logits := make([]float32, m.vocabulary)
	probabilities := make([]float64, m.vocabulary)
	gradientInput := make([]float32, m.dimensions)
	losses := make([]float64, epochs)

	for epoch := 0; epoch < epochs; epoch++ {
		var totalLoss float64

		for i := 0; i+1 < len(tokens); i++ {
			current, target := tokens[i], tokens[i+1]
			inputVector, err := m.input.Lookup(current)
			if err != nil {
				return nil, err
			}
			if err := m.scoresInto(current, logits); err != nil {
				return nil, err
			}

			maxLogit := logits[0]
			for _, logit := range logits[1:] {
				if logit > maxLogit {
					maxLogit = logit
				}
			}

			var denominator float64
			for candidate, logit := range logits {
				value := math.Exp(float64(logit - maxLogit))
				probabilities[candidate] = value
				denominator += value
			}
			for candidate := range probabilities {
				probabilities[candidate] /= denominator
			}

			targetProbability := probabilities[target]
			totalLoss += -math.Log(targetProbability)

			// For softmax + cross-entropy, dLoss/dLogit is p - y.
			probabilities[target]--

			clear(gradientInput)
			for candidate, gradient64 := range probabilities {
				outputVector, err := m.output.Lookup(tokenizer.TokenID(candidate))
				if err != nil {
					return nil, err
				}
				gradient := float32(gradient64)
				for dimension := range gradientInput {
					gradientInput[dimension] += gradient * outputVector[dimension]
				}
			}

			// Update output rows only after computing the input gradient so that
			// every gradient is based on the same pre-update weights.
			for candidate, gradient64 := range probabilities {
				outputVector, err := m.output.Lookup(tokenizer.TokenID(candidate))
				if err != nil {
					return nil, err
				}
				gradient := float32(gradient64)
				for dimension := range outputVector {
					outputVector[dimension] -= learningRate * gradient * inputVector[dimension]
				}
			}
			for dimension := range inputVector {
				inputVector[dimension] -= learningRate * gradientInput[dimension]
			}
		}

		losses[epoch] = totalLoss / float64(len(tokens)-1)
	}

	return losses, nil
}

func (m *NeuralBigram) scoresInto(current tokenizer.TokenID, logits []float32) error {
	if len(logits) != m.vocabulary {
		return fmt.Errorf("logit buffer size %d != vocabulary size %d", len(logits), m.vocabulary)
	}

	inputVector, err := m.input.Lookup(current)
	if err != nil {
		return err
	}
	for candidate := range logits {
		outputVector, err := m.output.Lookup(tokenizer.TokenID(candidate))
		if err != nil {
			return err
		}
		logit, err := tensor.Dot(inputVector, outputVector)
		if err != nil {
			return err
		}
		logits[candidate] = logit
	}
	return nil
}
