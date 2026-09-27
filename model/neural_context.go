package model

import (
	"fmt"
	"math"

	"github.com/dmarro89/quasar/embedding"
	"github.com/dmarro89/quasar/tensor"
	"github.com/dmarro89/quasar/tokenizer"
)

// NeuralContext predicts the next token from a fixed number of previous tokens.
// It combines the context by averaging token embeddings. This intentionally
// ignores token order so Quasar can expose why positional information matters
// before introducing attention.
type NeuralContext struct {
	input       *embedding.Table
	output      *embedding.Table
	vocabulary  int
	dimensions  int
	contextSize int
}

// NewNeuralContext creates a deterministically initialized fixed-window model.
func NewNeuralContext(vocabularySize, dimensions, contextSize int, seed uint64) (*NeuralContext, error) {
	if contextSize <= 0 {
		return nil, fmt.Errorf("context size must be positive: %d", contextSize)
	}
	input, err := embedding.NewTable(vocabularySize, dimensions, seed)
	if err != nil {
		return nil, fmt.Errorf("create input embeddings: %w", err)
	}
	output, err := embedding.NewTable(vocabularySize, dimensions, seed+1)
	if err != nil {
		return nil, fmt.Errorf("create output weights: %w", err)
	}

	return &NeuralContext{
		input:       input,
		output:      output,
		vocabulary:  vocabularySize,
		dimensions:  dimensions,
		contextSize: contextSize,
	}, nil
}

// ContextSize returns the number of previous tokens used for prediction.
func (m *NeuralContext) ContextSize() int {
	return m.contextSize
}

// Embedding returns the current learned input embedding for a token.
func (m *NeuralContext) Embedding(id tokenizer.TokenID) (tensor.Vector, error) {
	return m.input.Lookup(id)
}

// Scores returns one logit for every possible next token.
func (m *NeuralContext) Scores(context []tokenizer.TokenID) ([]float32, error) {
	contextVector := make([]float32, m.dimensions)
	if err := m.contextInto(context, contextVector); err != nil {
		return nil, err
	}

	logits := make([]float32, m.vocabulary)
	if err := m.scoresFromVector(contextVector, logits); err != nil {
		return nil, err
	}
	return logits, nil
}

// Predict returns the token with the highest logit for context.
func (m *NeuralContext) Predict(context []tokenizer.TokenID) (tokenizer.TokenID, error) {
	logits, err := m.Scores(context)
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

// Generate predicts tokens autoregressively while keeping a fixed-size context.
func (m *NeuralContext) Generate(seed []tokenizer.TokenID, maxNewTokens int) ([]tokenizer.TokenID, error) {
	if len(seed) != m.contextSize {
		return nil, fmt.Errorf("seed context size %d != required size %d", len(seed), m.contextSize)
	}
	if maxNewTokens <= 0 {
		return nil, nil
	}

	context := append([]tokenizer.TokenID(nil), seed...)
	generated := make([]tokenizer.TokenID, 0, maxNewTokens)
	for len(generated) < maxNewTokens {
		next, err := m.Predict(context)
		if err != nil {
			return nil, err
		}
		generated = append(generated, next)
		copy(context, context[1:])
		context[len(context)-1] = next
	}
	return generated, nil
}

// Train learns next-token predictions from sliding fixed-size context windows.
// Context embeddings are averaged, so each token receives an equal share of the
// gradient flowing back from the combined context vector.
func (m *NeuralContext) Train(tokens []tokenizer.TokenID, epochs int, learningRate float32) ([]float64, error) {
	if len(tokens) <= m.contextSize {
		return nil, fmt.Errorf("training requires more than %d tokens", m.contextSize)
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

	contextVector := make([]float32, m.dimensions)
	logits := make([]float32, m.vocabulary)
	probabilities := make([]float64, m.vocabulary)
	gradientContext := make([]float32, m.dimensions)
	losses := make([]float64, epochs)
	examples := len(tokens) - m.contextSize

	for epoch := 0; epoch < epochs; epoch++ {
		var totalLoss float64

		for i := 0; i+m.contextSize < len(tokens); i++ {
			context := tokens[i : i+m.contextSize]
			target := tokens[i+m.contextSize]

			if err := m.contextInto(context, contextVector); err != nil {
				return nil, err
			}
			if err := m.scoresFromVector(contextVector, logits); err != nil {
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

			// Softmax + cross-entropy: dLoss/dLogit = probability - target.
			probabilities[target]--

			clear(gradientContext)
			for candidate, gradient64 := range probabilities {
				outputVector, err := m.output.Lookup(tokenizer.TokenID(candidate))
				if err != nil {
					return nil, err
				}
				gradient := float32(gradient64)
				for dimension := range gradientContext {
					gradientContext[dimension] += gradient * outputVector[dimension]
				}
			}

			// Update output weights after computing the context gradient so every
			// gradient uses the same pre-update weights.
			for candidate, gradient64 := range probabilities {
				outputVector, err := m.output.Lookup(tokenizer.TokenID(candidate))
				if err != nil {
					return nil, err
				}
				gradient := float32(gradient64)
				for dimension := range outputVector {
					outputVector[dimension] -= learningRate * gradient * contextVector[dimension]
				}
			}

			// The context is an average, so each occurrence contributes 1/N of
			// the context gradient to its shared token embedding.
			scale := learningRate / float32(m.contextSize)
			for _, id := range context {
				inputVector, err := m.input.Lookup(id)
				if err != nil {
					return nil, err
				}
				for dimension := range inputVector {
					inputVector[dimension] -= scale * gradientContext[dimension]
				}
			}
		}

		losses[epoch] = totalLoss / float64(examples)
	}

	return losses, nil
}

func (m *NeuralContext) contextInto(context []tokenizer.TokenID, dst []float32) error {
	if len(context) != m.contextSize {
		return fmt.Errorf("context size %d != required size %d", len(context), m.contextSize)
	}
	if len(dst) != m.dimensions {
		return fmt.Errorf("context buffer size %d != embedding dimensions %d", len(dst), m.dimensions)
	}

	clear(dst)
	for _, id := range context {
		vector, err := m.input.Lookup(id)
		if err != nil {
			return err
		}
		for dimension, value := range vector {
			dst[dimension] += value
		}
	}

	scale := float32(1) / float32(m.contextSize)
	for dimension := range dst {
		dst[dimension] *= scale
	}
	return nil
}

func (m *NeuralContext) scoresFromVector(contextVector, logits []float32) error {
	if len(logits) != m.vocabulary {
		return fmt.Errorf("logit buffer size %d != vocabulary size %d", len(logits), m.vocabulary)
	}
	for candidate := range logits {
		outputVector, err := m.output.Lookup(tokenizer.TokenID(candidate))
		if err != nil {
			return err
		}
		logit, err := tensor.Dot(contextVector, outputVector)
		if err != nil {
			return err
		}
		logits[candidate] = logit
	}
	return nil
}
