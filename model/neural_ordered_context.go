package model

import (
	"fmt"
	"math"

	"github.com/dmarro89/quasar/embedding"
	"github.com/dmarro89/quasar/tensor"
	"github.com/dmarro89/quasar/tokenizer"
)

// NeuralOrderedContext predicts the next token from an ordered fixed-size
// context. It preserves token order by concatenating the embedding for each
// context position instead of averaging them.
type NeuralOrderedContext struct {
	input        *embedding.Table
	output       *embedding.Table
	vocabulary   int
	dimensions   int
	contextSize  int
	contextWidth int
}

// NewNeuralOrderedContext creates a deterministically initialized ordered model.
func NewNeuralOrderedContext(vocabularySize, dimensions, contextSize int, seed uint64) (*NeuralOrderedContext, error) {
	if contextSize <= 0 {
		return nil, fmt.Errorf("context size must be positive: %d", contextSize)
	}
	input, err := embedding.NewTable(vocabularySize, dimensions, seed)
	if err != nil {
		return nil, fmt.Errorf("create input embeddings: %w", err)
	}
	contextWidth := dimensions * contextSize
	output, err := embedding.NewTable(vocabularySize, contextWidth, seed+1)
	if err != nil {
		return nil, fmt.Errorf("create output weights: %w", err)
	}

	return &NeuralOrderedContext{
		input:        input,
		output:       output,
		vocabulary:   vocabularySize,
		dimensions:   dimensions,
		contextSize:  contextSize,
		contextWidth: contextWidth,
	}, nil
}

// ContextSize returns the required number of previous tokens.
func (m *NeuralOrderedContext) ContextSize() int {
	return m.contextSize
}

// Scores returns one logit for every possible next token.
func (m *NeuralOrderedContext) Scores(context []tokenizer.TokenID) ([]float32, error) {
	contextVector := make([]float32, m.contextWidth)
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
func (m *NeuralOrderedContext) Predict(context []tokenizer.TokenID) (tokenizer.TokenID, error) {
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

// Generate predicts tokens autoregressively with an ordered sliding context.
func (m *NeuralOrderedContext) Generate(seed []tokenizer.TokenID, maxNewTokens int) ([]tokenizer.TokenID, error) {
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

// Train learns next-token predictions from ordered fixed-size context windows.
func (m *NeuralOrderedContext) Train(tokens []tokenizer.TokenID, epochs int, learningRate float32) ([]float64, error) {
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

	contextVector := make([]float32, m.contextWidth)
	logits := make([]float32, m.vocabulary)
	probabilities := make([]float64, m.vocabulary)
	gradientContext := make([]float32, m.contextWidth)
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

			totalLoss += -math.Log(probabilities[target])
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

			for position, id := range context {
				inputVector, err := m.input.Lookup(id)
				if err != nil {
					return nil, err
				}
				start := position * m.dimensions
				for dimension := range inputVector {
					inputVector[dimension] -= learningRate * gradientContext[start+dimension]
				}
			}
		}
		losses[epoch] = totalLoss / float64(examples)
	}

	return losses, nil
}

func (m *NeuralOrderedContext) contextInto(context []tokenizer.TokenID, dst []float32) error {
	if len(context) != m.contextSize {
		return fmt.Errorf("context size %d != required size %d", len(context), m.contextSize)
	}
	if len(dst) != m.contextWidth {
		return fmt.Errorf("context buffer size %d != ordered context width %d", len(dst), m.contextWidth)
	}

	for position, id := range context {
		vector, err := m.input.Lookup(id)
		if err != nil {
			return err
		}
		start := position * m.dimensions
		copy(dst[start:start+m.dimensions], vector)
	}
	return nil
}

func (m *NeuralOrderedContext) scoresFromVector(contextVector, logits []float32) error {
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
