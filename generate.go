package lfg

import "context"

// GenerateToken represents a single generated token streamed via a channel.
type GenerateToken struct {
	Token Token
	Text  string
}

// GenerateResult is sent once on the result channel when generation completes.
type GenerateResult struct {
	Tokens []Token
	Text   string
	Err    error
}

// Generate streams tokens on a channel. The token channel is buffered (cap 16)
// and closed when generation finishes. The result channel receives exactly one
// GenerateResult and is then closed.
// Generation stops when maxTokens is reached, an EOG token is sampled, or the
// context is cancelled.
func (s *Session) Generate(ctx context.Context, prompt string, maxTokens int) (<-chan GenerateToken, <-chan GenerateResult) {
	tokenCh := make(chan GenerateToken, 16)
	resultCh := make(chan GenerateResult, 1)

	go func() {
		defer close(tokenCh)
		defer close(resultCh)

		var allTokens []Token
		var allText string

		// Tokenize the prompt.
		vocab := s.model.Vocab()
		if vocab == nil {
			resultCh <- GenerateResult{Err: &Error{Code: ErrorInvalidArgument, Message: "model has no vocab"}}
			return
		}
		promptTokens, err := vocab.Tokenize(prompt, true, true)
		if err != nil {
			resultCh <- GenerateResult{Err: err}
			return
		}

		// Ingest prompt tokens.
		s.mu.Lock()
		if s.c == nil {
			s.mu.Unlock()
			resultCh <- GenerateResult{Err: &Error{Code: ErrorInvalidArgument, Message: "session is closed"}}
			return
		}
		s.mu.Unlock()

		if err := s.IngestTokens(promptTokens, true); err != nil {
			resultCh <- GenerateResult{Err: err}
			return
		}

		for i := 0; i < maxTokens; i++ {
			// Check for cancellation.
			select {
			case <-ctx.Done():
				resultCh <- GenerateResult{Tokens: allTokens, Text: allText, Err: ctx.Err()}
				return
			default:
			}

			// Decode + sample.
			if err := s.Decode(); err != nil {
				resultCh <- GenerateResult{Tokens: allTokens, Text: allText, Err: err}
				return
			}

			token := s.Sample()

			// Check for EOG.
			if vocab.IsEOG(token) {
				break
			}

			piece := vocab.TokenToPiece(token, false)
			allTokens = append(allTokens, token)
			allText += piece

			// Ingest the sampled token for the next step.
			if err := s.IngestTokens([]Token{token}, true); err != nil {
				resultCh <- GenerateResult{Tokens: allTokens, Text: allText, Err: err}
				return
			}

			select {
			case tokenCh <- GenerateToken{Token: token, Text: piece}:
			case <-ctx.Done():
				resultCh <- GenerateResult{Tokens: allTokens, Text: allText, Err: ctx.Err()}
				return
			}
		}

		resultCh <- GenerateResult{Tokens: allTokens, Text: allText}
	}()

	return tokenCh, resultCh
}

// GenerateAll is a convenience wrapper that collects all generated tokens and
// returns the complete text. Blocks until generation completes or the context
// is cancelled.
func (s *Session) GenerateAll(ctx context.Context, prompt string, maxTokens int) (string, error) {
	_, resultCh := s.Generate(ctx, prompt, maxTokens)
	result := <-resultCh
	return result.Text, result.Err
}
