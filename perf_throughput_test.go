package lfg

import (
	"testing"
	"time"
)

func TestPerfThroughput(t *testing.T) {
	m := requireModel(t)

	messages := []ChatMessage{
		{Role: "system", Content: "You are a helpful assistant. Write detailed responses."},
		{Role: "user", Content: "Explain how CPUs work, from transistors to instruction execution."},
	}

	// --- Test 1: ChatGenerate, no callback (pure C-side loop throughput) ---
	s1, err := NewSession(m,
		WithSessionNCtx(2048),
		WithSessionSampling(SamplingConfig{Seed: 42, Temp: 0.0}),
	)
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}

	start := time.Now()
	r1, err := s1.ChatGenerate(messages, GenerateConfig{MaxTokens: 256})
	elapsed1 := time.Since(start)
	if err != nil {
		t.Fatalf("ChatGenerate (no cb): %v", err)
	}
	tps1 := float64(r1.TokenCount) / elapsed1.Seconds()
	t.Logf("No callback:   %d tokens in %v  (%.1f tok/s)", r1.TokenCount, elapsed1.Round(time.Millisecond), tps1)
	s1.Close()

	// --- Test 2: ChatGenerate WITH token callback (purego callback overhead) ---
	s2, err := NewSession(m,
		WithSessionNCtx(2048),
		WithSessionSampling(SamplingConfig{Seed: 42, Temp: 0.0}),
	)
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}

	cbCount := 0
	start = time.Now()
	r2, err := s2.ChatGenerate(messages, GenerateConfig{
		MaxTokens: 256,
		TokenCallback: func(token Token, piece string) GenerateAction {
			cbCount++
			return GenerateContinue
		},
	})
	elapsed2 := time.Since(start)
	if err != nil {
		t.Fatalf("ChatGenerate (with cb): %v", err)
	}
	tps2 := float64(r2.TokenCount) / elapsed2.Seconds()
	t.Logf("With callback: %d tokens in %v  (%.1f tok/s, %d callbacks)", r2.TokenCount, elapsed2.Round(time.Millisecond), tps2, cbCount)
	s2.Close()

	// --- Summary ---
	if tps2 > 0 && tps1 > 0 {
		overhead := (1.0 - tps2/tps1) * 100
		t.Logf("\nCallback overhead: %.1f%%", overhead)
	}
}
