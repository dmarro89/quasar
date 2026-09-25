package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/dmarro89/quasar/model"
	"github.com/dmarro89/quasar/tokenizer"
)

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
	prompt := flags.String("prompt", "", "prompt whose last token seeds generation")
	maxTokens := flags.Int("tokens", 8, "maximum number of new tokens to generate")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *corpusPath == "" {
		return errors.New("-corpus is required")
	}
	if *prompt == "" {
		return errors.New("-prompt is required")
	}
	if *maxTokens < 0 {
		return errors.New("-tokens must be non-negative")
	}

	corpus, err := os.ReadFile(*corpusPath)
	if err != nil {
		return fmt.Errorf("read corpus: %w", err)
	}

	tok := tokenizer.New()
	corpusIDs := tok.Fit(string(corpus))
	if len(corpusIDs) < 2 {
		return errors.New("corpus must contain at least two tokens")
	}

	promptIDs, err := tok.Encode(*prompt)
	if err != nil {
		return fmt.Errorf("encode prompt: %w", err)
	}
	if len(promptIDs) == 0 {
		return errors.New("prompt must contain at least one token")
	}

	m := model.NewBigram(tok.Size())
	if err := m.Train(corpusIDs); err != nil {
		return fmt.Errorf("train model: %w", err)
	}
	generated := m.Generate(promptIDs[len(promptIDs)-1], *maxTokens)

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
