package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/dmarro89/quasar/attention"
	"github.com/dmarro89/quasar/embedding"
	"github.com/dmarro89/quasar/model"
	"github.com/dmarro89/quasar/tensor"
	"github.com/dmarro89/quasar/tokenizer"
)

const embeddingSeed uint64 = 1

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, "quasar:", err)
		os.Exit(1)
	}
}

func run(args []string, stdout, stderr io.Writer) error {
	flags := flag.NewFlagSet("quasar", flag.ContinueOnError)
	flags.SetOutput(stderr)

	corpusPath := flags.String("corpus", "", "path to the training corpus")
	prompt := flags.String("prompt", "", "prompt used to seed generation")
	maxTokens := flags.Int("tokens", 8, "maximum number of new tokens to generate")
	embeddingWord := flags.String("embedding", "", "show the untrained embedding for one token")
	attentionText := flags.String("attention", "", "inspect last-token self-attention for a token sequence")
	dimensions := flags.Int("dimensions", 4, "embedding dimensions")
	neural := flags.Bool("neural", false, "train and use a neural next-token model")
	contextSize := flags.Int("context-size", 1, "number of previous tokens used by the neural model")
	orderedContext := flags.Bool("ordered-context", false, "preserve token order by concatenating context embeddings")
	epochs := flags.Int("epochs", 100, "number of neural training epochs")
	learningRate := flags.Float64("learning-rate", 0.05, "neural SGD learning rate")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *corpusPath == "" {
		return errors.New("-corpus is required")
	}

	modes := 0
	if *prompt != "" {
		modes++
	}
	if *embeddingWord != "" {
		modes++
	}
	if *attentionText != "" {
		modes++
	}
	if modes != 1 {
		return errors.New("use exactly one of -prompt, -embedding, or -attention")
	}
	if *maxTokens < 0 {
		return errors.New("-tokens must be non-negative")
	}
	if *contextSize <= 0 {
		return errors.New("-context-size must be positive")
	}
	if *neural && *prompt == "" {
		return errors.New("-neural requires -prompt")
	}
	if !*neural && *contextSize != 1 {
		return errors.New("-context-size requires -neural")
	}
	if *orderedContext && !*neural {
		return errors.New("-ordered-context requires -neural")
	}
	if *orderedContext && *contextSize == 1 {
		return errors.New("-ordered-context requires -context-size greater than 1")
	}

	corpus, err := os.ReadFile(*corpusPath)
	if err != nil {
		return fmt.Errorf("read corpus: %w", err)
	}

	tok := tokenizer.New()
	corpusIDs := tok.Fit(string(corpus))
	if len(corpusIDs) == 0 {
		return errors.New("corpus must contain at least one token")
	}

	if *embeddingWord != "" {
		return inspectEmbedding(tok, *embeddingWord, *dimensions, stdout)
	}
	if *attentionText != "" {
		return inspectAttention(tok, *attentionText, *dimensions, stdout)
	}

	if len(corpusIDs) < 2 {
		return errors.New("generation corpus must contain at least two tokens")
	}
	promptIDs, err := tok.Encode(*prompt)
	if err != nil {
		return fmt.Errorf("encode prompt: %w", err)
	}
	if len(promptIDs) == 0 {
		return errors.New("prompt must contain at least one token")
	}

	var generated []tokenizer.TokenID
	if *neural {
		if len(promptIDs) < *contextSize {
			return fmt.Errorf("prompt has %d tokens but context size is %d", len(promptIDs), *contextSize)
		}
		if *contextSize == 1 {
			generated, err = runNeuralBigram(corpusIDs, promptIDs[len(promptIDs)-1], tok.Size(), *dimensions, *epochs, float32(*learningRate), *maxTokens, stderr)
		} else {
			seed := promptIDs[len(promptIDs)-*contextSize:]
			if *orderedContext {
				generated, err = runNeuralOrderedContext(corpusIDs, seed, tok.Size(), *dimensions, *contextSize, *epochs, float32(*learningRate), *maxTokens, stderr)
			} else {
				generated, err = runNeuralContext(corpusIDs, seed, tok.Size(), *dimensions, *contextSize, *epochs, float32(*learningRate), *maxTokens, stderr)
			}
		}
		if err != nil {
			return err
		}
	} else {
		m := model.NewBigram(tok.Size())
		if err := m.Train(corpusIDs); err != nil {
			return fmt.Errorf("train model: %w", err)
		}
		generated = m.Generate(promptIDs[len(promptIDs)-1], *maxTokens)
	}

	outputIDs := make([]tokenizer.TokenID, 0, len(promptIDs)+len(generated))
	outputIDs = append(outputIDs, promptIDs...)
	outputIDs = append(outputIDs, generated...)
	text, err := tok.Decode(outputIDs)
	if err != nil {
		return fmt.Errorf("decode output: %w", err)
	}
	fmt.Fprintln(stdout, text)
	return nil
}

func runNeuralBigram(corpusIDs []tokenizer.TokenID, seed tokenizer.TokenID, vocabularySize, dimensions, epochs int, learningRate float32, maxTokens int, stderr io.Writer) ([]tokenizer.TokenID, error) {
	m, err := model.NewNeuralBigram(vocabularySize, dimensions, embeddingSeed)
	if err != nil {
		return nil, fmt.Errorf("create neural bigram: %w", err)
	}
	losses, err := m.Train(corpusIDs, epochs, learningRate)
	if err != nil {
		return nil, fmt.Errorf("train neural bigram: %w", err)
	}
	printLoss(stderr, losses)

	generated, err := m.Generate(seed, maxTokens)
	if err != nil {
		return nil, fmt.Errorf("generate with neural bigram: %w", err)
	}
	return generated, nil
}

func runNeuralContext(corpusIDs, seed []tokenizer.TokenID, vocabularySize, dimensions, contextSize, epochs int, learningRate float32, maxTokens int, stderr io.Writer) ([]tokenizer.TokenID, error) {
	m, err := model.NewNeuralContext(vocabularySize, dimensions, contextSize, embeddingSeed)
	if err != nil {
		return nil, fmt.Errorf("create neural context model: %w", err)
	}
	losses, err := m.Train(corpusIDs, epochs, learningRate)
	if err != nil {
		return nil, fmt.Errorf("train neural context model: %w", err)
	}
	printLoss(stderr, losses)

	generated, err := m.Generate(seed, maxTokens)
	if err != nil {
		return nil, fmt.Errorf("generate with neural context model: %w", err)
	}
	return generated, nil
}

func runNeuralOrderedContext(corpusIDs, seed []tokenizer.TokenID, vocabularySize, dimensions, contextSize, epochs int, learningRate float32, maxTokens int, stderr io.Writer) ([]tokenizer.TokenID, error) {
	m, err := model.NewNeuralOrderedContext(vocabularySize, dimensions, contextSize, embeddingSeed)
	if err != nil {
		return nil, fmt.Errorf("create ordered neural context model: %w", err)
	}
	losses, err := m.Train(corpusIDs, epochs, learningRate)
	if err != nil {
		return nil, fmt.Errorf("train ordered neural context model: %w", err)
	}
	printLoss(stderr, losses)

	generated, err := m.Generate(seed, maxTokens)
	if err != nil {
		return nil, fmt.Errorf("generate with ordered neural context model: %w", err)
	}
	return generated, nil
}

func printLoss(stderr io.Writer, losses []float64) {
	fmt.Fprintf(stderr, "loss %.6f -> %.6f\n", losses[0], losses[len(losses)-1])
}

func inspectEmbedding(tok *tokenizer.Tokenizer, word string, dimensions int, stdout io.Writer) error {
	if dimensions <= 0 {
		return errors.New("-dimensions must be positive")
	}
	ids, err := tok.Encode(word)
	if err != nil {
		return fmt.Errorf("encode embedding token: %w", err)
	}
	if len(ids) != 1 {
		return errors.New("-embedding must contain exactly one token")
	}

	table, err := embedding.NewTable(tok.Size(), dimensions, embeddingSeed)
	if err != nil {
		return fmt.Errorf("create embedding table: %w", err)
	}
	vector, err := table.Lookup(ids[0])
	if err != nil {
		return fmt.Errorf("lookup embedding: %w", err)
	}

	fmt.Fprintf(stdout, "token=%s id=%d embedding=%v (untrained)\n", word, ids[0], vector)
	return nil
}

func inspectAttention(tok *tokenizer.Tokenizer, text string, dimensions int, stdout io.Writer) error {
	if dimensions <= 0 {
		return errors.New("-dimensions must be positive")
	}
	ids, err := tok.Encode(text)
	if err != nil {
		return fmt.Errorf("encode attention text: %w", err)
	}
	if len(ids) < 2 {
		return errors.New("-attention must contain at least two tokens")
	}

	table, err := embedding.NewTable(tok.Size(), dimensions, embeddingSeed)
	if err != nil {
		return fmt.Errorf("create embedding table: %w", err)
	}
	vectors := make([]tensor.Vector, len(ids))
	for i, id := range ids {
		vectors[i], err = table.Lookup(id)
		if err != nil {
			return fmt.Errorf("lookup attention embedding: %w", err)
		}
	}

	result, err := attention.LastToken(vectors)
	if err != nil {
		return fmt.Errorf("compute attention: %w", err)
	}
	query, err := tok.Decode([]tokenizer.TokenID{ids[len(ids)-1]})
	if err != nil {
		return fmt.Errorf("decode attention query: %w", err)
	}
	fmt.Fprintf(stdout, "query=%s embedding=%v\n", query, vectors[len(vectors)-1])
	for i, id := range ids {
		token, err := tok.Decode([]tokenizer.TokenID{id})
		if err != nil {
			return fmt.Errorf("decode attention token: %w", err)
		}
		fmt.Fprintf(stdout, "token=%s score=%.6f weight=%.6f\n", token, result.Scores[i], result.Weights[i])
	}
	fmt.Fprintf(stdout, "output=%v\n", result.Output)
	fmt.Fprintln(stdout, "note=untrained identity-qkv attention")
	return nil
}
