package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"

	tea "github.com/charmbracelet/bubbletea"

	lfg "github.com/agentflare-ai/lfg.go"
)

// ---------------------------------------------------------------------------
// Test-only constants (removed from main.go, kept for lfg.go API tests)
// ---------------------------------------------------------------------------

const testToolCallSchema = `{
  "type": "object",
  "properties": {
    "name": { "type": "string", "enum": ["respond", "recall", "think"] },
    "arguments": {
      "type": "object",
      "properties": { "text": { "type": "string" } },
      "required": ["text"]
    }
  },
  "required": ["name", "arguments"]
}`

type toolCall struct {
	Name      string `json:"name"`
	Arguments struct {
		Text string `json:"text"`
	} `json:"arguments"`
}

// Old system prompt for structured output tests.
const testStructuredSystemPrompt = `You are a helpful AI assistant. You MUST respond exclusively using tool calls.
Available tools:
- respond(text): Send a message to the user. Use for all final answers.
- recall(text): Search memory for relevant past information.
- think(text): Internal reasoning step. Plan before acting.

Always output a single valid JSON tool call. Never output free text.`

// ---------------------------------------------------------------------------
// Shared model + session (loaded once for the entire test run)
// ---------------------------------------------------------------------------

var (
	testModel   *lfg.Model
	testSession *lfg.Session
	testMemory  *vectorMemory
)

func testModelPath() string {
	if p := os.Getenv("LFG_MODEL_PATH"); p != "" {
		return p
	}
	return "../models/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf"
}

func TestMain(m *testing.M) {
	path := testModelPath()
	if _, err := os.Stat(path); err == nil {
		var loadErr error
		testModel, loadErr = lfg.LoadModelSimple(path, lfg.WithGPULayers(0))
		if loadErr != nil {
			fmt.Fprintf(os.Stderr, "FATAL: LoadModelSimple: %v\n", loadErr)
			os.Exit(1)
		}

		var sessionErr error
		testSession, sessionErr = lfg.NewSession(testModel,
			lfg.WithSessionNCtx(2048),
			lfg.WithSessionThreads(4),
			lfg.WithSessionMaxTokens(512),
			lfg.WithSessionHealing(true),
			lfg.WithSessionStructuredCheckpointing(true),
			lfg.WithSessionReasoningBudget(256),
			lfg.WithSessionSampling(lfg.SamplingConfig{
				Temp: 0.7,
				TopK: 40,
				TopP: 0.9,
				MinP: 0.05,
			}),
		)
		if sessionErr != nil {
			fmt.Fprintf(os.Stderr, "FATAL: NewSession: %v\n", sessionErr)
			testModel.Close()
			os.Exit(1)
		}

		testMemory = newVectorMemory(testSession)
	}

	code := m.Run()

	if testSession != nil {
		testSession.Close()
	}
	if testModel != nil {
		testModel.Close()
	}

	os.Exit(code)
}

func requireModel(t *testing.T) *lfg.Model {
	t.Helper()
	if testModel == nil {
		t.Skipf("Model not found at %s — set LFG_MODEL_PATH or download the model", testModelPath())
	}
	return testModel
}

func requireSession(t *testing.T) *lfg.Session {
	t.Helper()
	requireModel(t)
	if testSession == nil {
		t.Skip("Session not available")
	}
	return testSession
}

func requireMemory(t *testing.T) *vectorMemory {
	t.Helper()
	requireSession(t)
	if testMemory == nil {
		t.Skip("Memory not available")
	}
	return testMemory
}

// freshSession creates a new session for tests that need isolated state.
func freshSession(t *testing.T, opts ...lfg.SessionOption) *lfg.Session {
	t.Helper()
	m := requireModel(t)
	defaults := []lfg.SessionOption{
		lfg.WithSessionNCtx(2048),
		lfg.WithSessionThreads(4),
		lfg.WithSessionMaxTokens(512),
		lfg.WithSessionHealing(true),
		lfg.WithSessionStructuredCheckpointing(true),
		lfg.WithSessionReasoningBudget(256),
		lfg.WithSessionSampling(lfg.SamplingConfig{
			Temp: 0.7,
			TopK: 40,
			TopP: 0.9,
			MinP: 0.05,
		}),
	}
	allOpts := append(defaults, opts...)
	s, err := lfg.NewSession(m, allOpts...)
	if err != nil {
		t.Fatalf("freshSession: %v", err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

// ---------------------------------------------------------------------------
// Model Loading
// ---------------------------------------------------------------------------

func TestLoadModelSimple(t *testing.T) {
	m := requireModel(t)

	stats := m.Stats()
	t.Logf("Model stats: params=%d, size=%d, vocab=%d, ctx_train=%d",
		stats.ParameterCount, stats.SizeBytes, stats.VocabSize, stats.ContextSize)

	if stats.ParameterCount == 0 {
		t.Fatal("model has 0 parameters")
	}
	if stats.VocabSize == 0 {
		t.Fatal("model has 0 vocab size")
	}
}

func TestModelDescription(t *testing.T) {
	m := requireModel(t)

	desc := m.Description()
	if desc == "" {
		t.Fatal("model description is empty")
	}
	t.Logf("Model description: %s", desc)
}

func TestModelChatTemplate(t *testing.T) {
	m := requireModel(t)

	tmpl, ok := m.ChatTemplate("")
	if !ok {
		t.Skip("Model has no default chat template")
	}
	if tmpl == "" {
		t.Fatal("chat template is empty")
	}
	t.Logf("Chat template (first 300 chars): %.300s", tmpl)
}

// ---------------------------------------------------------------------------
// Session Creation with Agent Config
// ---------------------------------------------------------------------------

func TestSessionCreation(t *testing.T) {
	s := freshSession(t)

	vocabSize := s.VocabSize()
	if vocabSize <= 0 {
		t.Fatalf("VocabSize = %d, want > 0", vocabSize)
	}
	t.Logf("Session vocab size: %d", vocabSize)
}

func TestSessionReasoningConfiguration(t *testing.T) {
	m := requireModel(t)
	s := freshSession(t)

	vocab := m.Vocab()
	startTokens, err := vocab.Tokenize("<think>", false, true)
	if err != nil {
		t.Fatalf("Tokenize <think>: %v", err)
	}
	endTokens, err := vocab.Tokenize("</think>", false, true)
	if err != nil {
		t.Fatalf("Tokenize </think>: %v", err)
	}

	// ConfigureReasoning should not panic.
	s.ConfigureReasoning(startTokens, endTokens)
	t.Logf("Reasoning configured: start=%v, end=%v", startTokens, endTokens)
}

// ---------------------------------------------------------------------------
// Entropy Monitor
// ---------------------------------------------------------------------------

func TestEntropyMonitorSetup(t *testing.T) {
	s := freshSession(t)

	nEmbd, err := s.ConfigureEntropyMonitor(&lfg.EntropyMonitorConfig{
		Threshold:      0.6,
		CooldownTokens: 5,
		RingSize:       8,
	})
	if err != nil {
		t.Fatalf("ConfigureEntropyMonitor: %v", err)
	}
	if nEmbd <= 0 {
		t.Fatalf("n_embd = %d, want > 0", nEmbd)
	}
	t.Logf("Entropy monitor configured, n_embd=%d", nEmbd)

	counter := s.EntropyCounter()
	if counter == nil {
		t.Log("EntropyCounter is nil (may be implementation-specific)")
	} else {
		t.Log("EntropyCounter is available")
	}
}

func TestEntropyMonitorWithGeneration(t *testing.T) {
	s := freshSession(t)

	s.ConfigureEntropyMonitor(&lfg.EntropyMonitorConfig{
		Threshold:      0.1, // low threshold to trigger events
		CooldownTokens: 1,
		RingSize:       8,
	})

	var text string
	result, err := s.PromptGenerate("The meaning of life is", true, lfg.GenerateConfig{
		MaxTokens: 30,
		TokenCallback: func(token lfg.Token, piece string) lfg.GenerateAction {
			text += piece
			return lfg.GenerateContinue
		},
	})
	if err != nil {
		t.Fatalf("PromptGenerate: %v", err)
	}

	t.Logf("Generated %d tokens: %q", result.TokenCount, text)

	pending := s.EntropyPending()
	t.Logf("Pending entropy events: %d", pending)

	var events []lfg.EntropyEvent
	for {
		ev, ok := s.EntropyPop(nil)
		if !ok {
			break
		}
		events = append(events, ev)
	}
	t.Logf("Popped %d entropy events", len(events))

	for i, ev := range events {
		t.Logf("  Entropy[%d]: normalized=%.4f topLogprob=%.4f token=%d n_past=%d",
			i, ev.Normalized, ev.TopLogprob, ev.Token, ev.NPast)
	}
}

// ---------------------------------------------------------------------------
// Confidence Monitor
// ---------------------------------------------------------------------------

func TestConfidenceMonitorSetup(t *testing.T) {
	s := freshSession(t)

	nEmbd, err := s.ConfigureConfidenceMonitor(&lfg.ConfidenceMonitorConfig{
		Threshold: 0.3,
		MinSpan:   5,
		RingSize:  8,
	})
	if err != nil {
		t.Fatalf("ConfigureConfidenceMonitor: %v", err)
	}
	if nEmbd <= 0 {
		t.Fatalf("n_embd = %d, want > 0", nEmbd)
	}
	t.Logf("Confidence monitor configured, n_embd=%d", nEmbd)
}

func TestConfidenceMonitorWithGeneration(t *testing.T) {
	s := freshSession(t)

	s.ConfigureConfidenceMonitor(&lfg.ConfidenceMonitorConfig{
		Threshold: 0.5, // generous threshold to trigger events
		MinSpan:   3,
		RingSize:  8,
	})

	var text string
	result, err := s.PromptGenerate("The capital of France is Paris, which is also known as", true, lfg.GenerateConfig{
		MaxTokens: 40,
		TokenCallback: func(token lfg.Token, piece string) lfg.GenerateAction {
			text += piece
			return lfg.GenerateContinue
		},
	})
	if err != nil {
		t.Fatalf("PromptGenerate: %v", err)
	}

	t.Logf("Generated %d tokens: %q", result.TokenCount, text)

	pending := s.ConfidencePending()
	t.Logf("Pending confidence events: %d", pending)

	var events []lfg.ConfidenceEvent
	for {
		ev, ok := s.ConfidencePop(nil)
		if !ok {
			break
		}
		events = append(events, ev)
	}
	t.Logf("Popped %d confidence events", len(events))

	for i, ev := range events {
		t.Logf("  Confidence[%d]: meanEntropy=%.4f minEntropy=%.4f span=%d start=%d end=%d",
			i, ev.MeanEntropy, ev.MinEntropy, ev.SpanLength, ev.StartPos, ev.EndPos)
	}
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

func TestSessionEmbed(t *testing.T) {
	s := freshSession(t)

	emb, err := s.Embed("Hello, world!")
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(emb) == 0 {
		t.Fatal("Embed returned empty vector")
	}
	t.Logf("Embedding dimension: %d, first 5: %v", len(emb), emb[:min(5, len(emb))])
}

func TestEmbeddingSimilarity(t *testing.T) {
	s := freshSession(t)

	emb1, err := s.Embed("The weather is sunny today")
	if err != nil {
		t.Fatalf("Embed 1: %v", err)
	}

	emb2, err := s.Embed("It is a bright and sunny day")
	if err != nil {
		t.Fatalf("Embed 2: %v", err)
	}

	emb3, err := s.Embed("Quantum computing uses qubits")
	if err != nil {
		t.Fatalf("Embed 3: %v", err)
	}

	sim12 := cosineSimilarity(emb1, emb2)
	sim13 := cosineSimilarity(emb1, emb3)

	t.Logf("Similarity (similar sentences): %.4f", sim12)
	t.Logf("Similarity (different topics): %.4f", sim13)

	// Similar sentences should have higher similarity than unrelated ones.
	if sim12 <= sim13 {
		t.Logf("Warning: expected similar sentences to score higher (%.4f <= %.4f)", sim12, sim13)
	}
}

// ---------------------------------------------------------------------------
// Vector Memory Store
// ---------------------------------------------------------------------------

func TestVectorMemoryStoreAndSearchIntegration(t *testing.T) {
	s := freshSession(t)
	vm := newVectorMemory(s)

	vm.store("Paris is the capital of France", "user")
	vm.store("Tokyo is the capital of Japan", "user")
	vm.store("Python is a programming language", "confidence")

	if vm.count() != 3 {
		t.Fatalf("count = %d, want 3", vm.count())
	}
	if vm.autoCount() != 1 {
		t.Fatalf("autoCount = %d, want 1", vm.autoCount())
	}

	results := vm.searchByText("What is the capital of France?", 2)
	t.Logf("Search results for 'capital of France': %v", results)

	if len(results) == 0 {
		t.Fatal("search returned no results")
	}
	// The Paris entry should rank highest for this query.
	if !strings.Contains(results[0], "Paris") && !strings.Contains(results[0], "France") {
		t.Logf("Warning: expected Paris/France related result first, got %q", results[0])
	}
}

func TestVectorMemorySearchByText(t *testing.T) {
	s := freshSession(t)
	vm := newVectorMemory(s)

	entries := []string{
		"The quick brown fox jumps over the lazy dog",
		"Machine learning is a subset of artificial intelligence",
		"Go is a statically typed compiled language",
		"Neural networks process data in layers",
	}

	for _, e := range entries {
		vm.store(e, "user")
	}

	results := vm.searchByText("AI and deep learning", 2)
	t.Logf("Search for 'AI and deep learning': %v", results)

	if len(results) == 0 {
		t.Fatal("search returned no results")
	}
}

func TestVectorMemoryEmpty(t *testing.T) {
	vm := &vectorMemory{}

	results := vm.search([]float32{1, 0, 0}, 5)
	if len(results) != 0 {
		t.Fatalf("search on empty memory returned %d results", len(results))
	}

	if vm.count() != 0 {
		t.Fatalf("count = %d, want 0", vm.count())
	}
	if vm.autoCount() != 0 {
		t.Fatalf("autoCount = %d, want 0", vm.autoCount())
	}
}

// ---------------------------------------------------------------------------
// Structured Output (JSON Schema)
// ---------------------------------------------------------------------------

func TestStructuredOutputToolCall(t *testing.T) {
	s := freshSession(t)

	s.Reset()
	if err := s.ConfigureStructured(testToolCallSchema, ""); err != nil {
		t.Fatalf("ConfigureStructured: %v", err)
	}

	var text string
	result, err := s.ChatGenerate(
		[]lfg.ChatMessage{
			{Role: "system", Content: "You are a helpful assistant. Respond using the respond tool."},
			{Role: "user", Content: "Say hello!"},
		},
		lfg.GenerateConfig{
			MaxTokens: 128,
			TokenCallback: func(token lfg.Token, piece string) lfg.GenerateAction {
				text += piece
				return lfg.GenerateContinue
			},
		},
	)
	if err != nil {
		t.Fatalf("ChatGenerate with structured output: %v", err)
	}

	t.Logf("Structured output: %d tokens, text=%q", result.TokenCount, text)

	var tc toolCall
	if err := json.Unmarshal([]byte(text), &tc); err != nil {
		t.Fatalf("output is not valid tool call JSON: %v\nraw: %q", err, text)
	}
	if tc.Name == "" {
		t.Fatal("tool call name is empty")
	}
	t.Logf("Tool call: name=%q, text=%q", tc.Name, tc.Arguments.Text)
}

// ---------------------------------------------------------------------------
// Tool Registration
// ---------------------------------------------------------------------------

func TestToolRegistration(t *testing.T) {
	s := freshSession(t)

	tools := []lfg.ToolDesc{
		{Name: "respond", Description: "Send a message to the user"},
		{Name: "recall", Description: "Search memory for relevant information"},
		{Name: "think", Description: "Internal reasoning step"},
	}

	n, err := s.RegisterTools(tools, 3)
	if err != nil {
		t.Fatalf("RegisterTools: %v", err)
	}
	if n != 3 {
		t.Fatalf("RegisterTools returned %d, want 3", n)
	}
	t.Logf("Registered %d tools", n)
}

func TestToolRegistrationWithDecode(t *testing.T) {
	s := freshSession(t)

	tools := []lfg.ToolDesc{
		{Name: "respond", Description: "Send a message to the user"},
		{Name: "recall", Description: "Search memory for relevant information"},
		{Name: "think", Description: "Internal reasoning step"},
	}

	_, err := s.RegisterTools(tools, 3)
	if err != nil {
		t.Fatalf("RegisterTools: %v", err)
	}

	var text string
	result, err := s.PromptGenerate("What is the weather today?", true, lfg.GenerateConfig{
		MaxTokens: 20,
		TokenCallback: func(token lfg.Token, piece string) lfg.GenerateAction {
			text += piece
			return lfg.GenerateContinue
		},
	})
	if err != nil {
		t.Fatalf("PromptGenerate with tools: %v", err)
	}

	t.Logf("Generated with tools (%d tokens): %q", result.TokenCount, text)
}

// ---------------------------------------------------------------------------
// Chat Generate
// ---------------------------------------------------------------------------

func TestChatGenerateBasic(t *testing.T) {
	s := freshSession(t)

	var text string
	result, err := s.ChatGenerate(
		[]lfg.ChatMessage{
			{Role: "user", Content: "What is 2+2?"},
		},
		lfg.GenerateConfig{
			MaxTokens: 50,
			TokenCallback: func(token lfg.Token, piece string) lfg.GenerateAction {
				text += piece
				return lfg.GenerateContinue
			},
		},
	)
	if err != nil {
		t.Fatalf("ChatGenerate: %v", err)
	}

	t.Logf("ChatGenerate: %d tokens, text=%q", result.TokenCount, text)
	if result.TokenCount == 0 {
		t.Fatal("generated no tokens")
	}
	if text == "" {
		t.Fatal("generated text is empty")
	}
}

func TestChatGenerateMultiTurn(t *testing.T) {
	s := freshSession(t)

	var text string
	result, err := s.ChatGenerate(
		[]lfg.ChatMessage{
			{Role: "system", Content: "You are a helpful assistant. Be concise."},
			{Role: "user", Content: "What is the capital of France?"},
			{Role: "assistant", Content: "Paris."},
			{Role: "user", Content: "And Germany?"},
		},
		lfg.GenerateConfig{
			MaxTokens: 30,
			TokenCallback: func(token lfg.Token, piece string) lfg.GenerateAction {
				text += piece
				return lfg.GenerateContinue
			},
		},
	)
	if err != nil {
		t.Fatalf("ChatGenerate multi-turn: %v", err)
	}

	t.Logf("ChatGenerate multi-turn: %d tokens, text=%q", result.TokenCount, text)
	if result.TokenCount == 0 {
		t.Fatal("generated no tokens")
	}
}

func TestChatGenerateWithSystemPrompt(t *testing.T) {
	s := freshSession(t)

	s.Reset()
	if err := s.ConfigureStructured(testToolCallSchema, ""); err != nil {
		t.Fatalf("ConfigureStructured: %v", err)
	}

	var text string
	result, err := s.ChatGenerate(
		[]lfg.ChatMessage{
			{Role: "system", Content: testStructuredSystemPrompt},
			{Role: "user", Content: "Hello, how are you?"},
		},
		lfg.GenerateConfig{
			MaxTokens: 128,
			TokenCallback: func(token lfg.Token, piece string) lfg.GenerateAction {
				text += piece
				return lfg.GenerateContinue
			},
		},
	)
	if err != nil {
		t.Fatalf("ChatGenerate: %v", err)
	}

	t.Logf("ChatGenerate with system prompt: %d tokens, text=%q", result.TokenCount, text)

	var tc toolCall
	if err := json.Unmarshal([]byte(text), &tc); err != nil {
		t.Logf("Warning: output is not valid tool call JSON: %v", err)
	} else {
		t.Logf("Tool call: name=%q, text=%q", tc.Name, tc.Arguments.Text)
	}
}

// ---------------------------------------------------------------------------
// ChatGenerate with Reasoning
// ---------------------------------------------------------------------------

func TestChatGenerateWithReasoning(t *testing.T) {
	m := requireModel(t)
	s := freshSession(t, lfg.WithSessionReasoningBudget(128))

	vocab := m.Vocab()
	startTokens, _ := vocab.Tokenize("<think>", false, true)
	endTokens, _ := vocab.Tokenize("</think>", false, true)
	s.ConfigureReasoning(startTokens, endTokens)

	var text string
	result, err := s.ChatGenerate(
		[]lfg.ChatMessage{
			{Role: "user", Content: "What is 15 * 7?"},
		},
		lfg.GenerateConfig{
			MaxTokens: 200,
			TokenCallback: func(token lfg.Token, piece string) lfg.GenerateAction {
				text += piece
				return lfg.GenerateContinue
			},
		},
	)
	if err != nil {
		t.Fatalf("ChatGenerate with reasoning: %v", err)
	}

	t.Logf("ChatGenerate with reasoning: %d tokens", result.TokenCount)
	t.Logf("Output (first 500 chars): %.500s", text)

	hasThinkTag := strings.Contains(text, "<think>") || strings.Contains(text, "</think>")
	t.Logf("Contains thinking tags: %v", hasThinkTag)
}

// ---------------------------------------------------------------------------
// Generate Loop with Entropy + Confidence Callbacks
// ---------------------------------------------------------------------------

func TestGenerateLoopWithEntropyCallback(t *testing.T) {
	s := freshSession(t)

	s.ConfigureEntropyMonitor(&lfg.EntropyMonitorConfig{
		Threshold:      0.3,
		CooldownTokens: 3,
		RingSize:       8,
	})

	var (
		text           string
		entropyEvents  int
		entropyMu      sync.Mutex
	)

	result, err := s.PromptGenerate("Tell me about the history of computing", true, lfg.GenerateConfig{
		MaxTokens: 50,
		TokenCallback: func(token lfg.Token, piece string) lfg.GenerateAction {
			text += piece
			return lfg.GenerateContinue
		},
		EntropyCallback: func(event lfg.EntropyEvent, embedding []float32) string {
			entropyMu.Lock()
			entropyEvents++
			entropyMu.Unlock()
			return "" // don't inject anything
		},
	})
	if err != nil {
		t.Fatalf("PromptGenerate: %v", err)
	}

	t.Logf("Generated %d tokens, %d entropy events, %d retrievals",
		result.TokenCount, entropyEvents, result.Retrievals)
	t.Logf("Text: %q", text)
}

func TestGenerateLoopWithConfidenceCallback(t *testing.T) {
	s := freshSession(t)

	s.ConfigureConfidenceMonitor(&lfg.ConfidenceMonitorConfig{
		Threshold: 0.5,
		MinSpan:   3,
		RingSize:  8,
	})

	var (
		text       string
		confEvents int
		confMu     sync.Mutex
	)

	result, err := s.PromptGenerate("The capital of France is Paris, and it is known for", true, lfg.GenerateConfig{
		MaxTokens: 40,
		TokenCallback: func(token lfg.Token, piece string) lfg.GenerateAction {
			text += piece
			return lfg.GenerateContinue
		},
		ConfidenceCallback: func(event lfg.ConfidenceEvent, embedding []float32) {
			confMu.Lock()
			confEvents++
			confMu.Unlock()
		},
	})
	if err != nil {
		t.Fatalf("PromptGenerate: %v", err)
	}

	t.Logf("Generated %d tokens, %d confidence events, %d confidence spans",
		result.TokenCount, confEvents, result.ConfidenceSpans)
	t.Logf("Text: %q", text)
}

func TestGenerateLoopWithEntropyInjection(t *testing.T) {
	s := freshSession(t)

	s.ConfigureEntropyMonitor(&lfg.EntropyMonitorConfig{
		Threshold:      0.2, // low threshold to trigger
		CooldownTokens: 2,
		RingSize:       8,
	})

	var text string
	injections := 0

	result, err := s.PromptGenerate("What are the main features of", true, lfg.GenerateConfig{
		MaxTokens: 60,
		TokenCallback: func(token lfg.Token, piece string) lfg.GenerateAction {
			text += piece
			return lfg.GenerateContinue
		},
		EntropyCallback: func(event lfg.EntropyEvent, embedding []float32) string {
			injections++
			if injections <= 1 {
				return "Context: Go is a programming language created by Google."
			}
			return ""
		},
	})
	if err != nil {
		t.Fatalf("PromptGenerate: %v", err)
	}

	t.Logf("Generated %d tokens, %d injection callbacks, %d retrievals",
		result.TokenCount, injections, result.Retrievals)
	t.Logf("Text: %q", text)
}

func TestGenerateLoopBothCallbacks(t *testing.T) {
	s := freshSession(t)

	s.ConfigureEntropyMonitor(&lfg.EntropyMonitorConfig{
		Threshold:      0.3,
		CooldownTokens: 3,
		RingSize:       8,
	})
	s.ConfigureConfidenceMonitor(&lfg.ConfidenceMonitorConfig{
		Threshold: 0.4,
		MinSpan:   3,
		RingSize:  8,
	})

	var (
		text           string
		entropyEvents  int
		confEvents     int
		mu             sync.Mutex
	)

	result, err := s.PromptGenerate(
		"Artificial intelligence is transforming many industries including healthcare and finance",
		true,
		lfg.GenerateConfig{
			MaxTokens: 50,
			TokenCallback: func(token lfg.Token, piece string) lfg.GenerateAction {
				text += piece
				return lfg.GenerateContinue
			},
			EntropyCallback: func(event lfg.EntropyEvent, embedding []float32) string {
				mu.Lock()
				entropyEvents++
				mu.Unlock()
				return ""
			},
			ConfidenceCallback: func(event lfg.ConfidenceEvent, embedding []float32) {
				mu.Lock()
				confEvents++
				mu.Unlock()
			},
		},
	)
	if err != nil {
		t.Fatalf("PromptGenerate: %v", err)
	}

	t.Logf("Generated %d tokens, entropy_events=%d, confidence_events=%d",
		result.TokenCount, entropyEvents, confEvents)
	t.Logf("Result: retrievals=%d, confidence_spans=%d, stop=%d",
		result.Retrievals, result.ConfidenceSpans, result.StopReason)
}

// ---------------------------------------------------------------------------
// ChatGenerate with Structured Output + Monitors (Full Agent Pipeline)
// ---------------------------------------------------------------------------

func TestChatGenerateWithMonitors(t *testing.T) {
	m := requireModel(t)
	s := freshSession(t)

	vocab := m.Vocab()
	startTokens, _ := vocab.Tokenize("<think>", false, true)
	endTokens, _ := vocab.Tokenize("</think>", false, true)
	s.ConfigureReasoning(startTokens, endTokens)

	s.ConfigureEntropyMonitor(&lfg.EntropyMonitorConfig{
		Threshold:      0.6,
		CooldownTokens: 5,
		RingSize:       8,
	})
	s.ConfigureConfidenceMonitor(&lfg.ConfidenceMonitorConfig{
		Threshold: 0.3,
		MinSpan:   5,
		RingSize:  8,
	})

	vm := newVectorMemory(s)
	vm.store("The user's name is Alice", "user")
	vm.store("Alice works at a tech company", "confidence")

	var (
		text           string
		entropyEvents  int
		confEvents     int
		mu             sync.Mutex
	)

	s.Reset()
	result, err := s.ChatGenerate(
		[]lfg.ChatMessage{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: "Hello! What's my name?"},
		},
		lfg.GenerateConfig{
			MaxTokens: 128,
			TokenCallback: func(token lfg.Token, piece string) lfg.GenerateAction {
				text += piece
				return lfg.GenerateContinue
			},
			EntropyCallback: func(event lfg.EntropyEvent, embedding []float32) string {
				mu.Lock()
				entropyEvents++
				mu.Unlock()
				if embedding != nil && vm.count() > 0 {
					results := vm.search(embedding, 3)
					if len(results) > 0 {
						return "Relevant context:\n" + strings.Join(results, "\n")
					}
				}
				return ""
			},
			ConfidenceCallback: func(event lfg.ConfidenceEvent, embedding []float32) {
				mu.Lock()
				confEvents++
				mu.Unlock()
			},
		},
	)
	if err != nil {
		t.Fatalf("ChatGenerate: %v", err)
	}

	t.Logf("Chat with monitors: %d tokens, entropy=%d, confidence=%d, retrievals=%d",
		result.TokenCount, entropyEvents, confEvents, result.Retrievals)
	t.Logf("Output: %q", text)

	if text == "" {
		t.Fatal("generated empty response")
	}
}

// ---------------------------------------------------------------------------
// Agent Turn Integration (runAgentTurn)
// ---------------------------------------------------------------------------

func TestAgentTurnRespond(t *testing.T) {
	m := requireModel(t)
	s := freshSession(t)

	vocab := m.Vocab()
	startTokens, _ := vocab.Tokenize("<think>", false, true)
	endTokens, _ := vocab.Tokenize("</think>", false, true)
	s.ConfigureReasoning(startTokens, endTokens)

	s.ConfigureEntropyMonitor(&lfg.EntropyMonitorConfig{
		Threshold:      0.6,
		CooldownTokens: 5,
		RingSize:       8,
	})
	s.ConfigureConfidenceMonitor(&lfg.ConfidenceMonitorConfig{
		Threshold: 0.3,
		MinSpan:   5,
		RingSize:  8,
	})

	vm := newVectorMemory(s)

	cfg := agentConfig{
		maxTokens:           512,
		entropyThreshold:    0.6,
		confidenceThreshold: 0.3,
		confidenceMinSpan:   5,
		reasoningBudget:     256,
	}

	eventCh := make(chan tea.Msg, 256)
	go runAgentTurn(s, vm, nil, "Hello, how are you?", eventCh, cfg)

	var (
		gotResponse    bool
		gotDone        bool
		tokens         []string
		responseText   string
	)

	for msg := range eventCh {
		switch msg := msg.(type) {
		case tokenMsg:
			tokens = append(tokens, string(msg))
		case responseMsg:
			gotResponse = true
			responseText = string(msg)
			t.Logf("Response: %q", responseText)
		case generationDone:
			gotDone = true
			if msg.err != nil {
				t.Logf("Generation error: %v", msg.err)
			}
			goto done
		case streamClearMsg:
			tokens = nil
		case memRetrievalMsg:
			t.Logf("Memory retrieval: %d items", msg.count)
		case memStoredMsg:
			t.Logf("Memory stored: %q (%dt)", msg.text, msg.spanLength)
		case statusMsg:
			t.Logf("Status: %s", string(msg))
		}
	}

done:
	if !gotDone {
		t.Fatal("never received generationDone message")
	}
	if !gotResponse {
		t.Fatal("never received responseMsg")
	}
	if responseText == "" {
		t.Fatal("response text is empty")
	}
}

// ---------------------------------------------------------------------------
// Multi-Turn Agent Turn (simulates TUI history flow)
// ---------------------------------------------------------------------------

func TestAgentTurnMultiTurn(t *testing.T) {
	m := requireModel(t)
	s := freshSession(t)

	vocab := m.Vocab()
	startTokens, _ := vocab.Tokenize("<think>", false, true)
	endTokens, _ := vocab.Tokenize("</think>", false, true)
	s.ConfigureReasoning(startTokens, endTokens)

	s.ConfigureEntropyMonitor(&lfg.EntropyMonitorConfig{
		Threshold:      0.6,
		CooldownTokens: 5,
		RingSize:       8,
	})
	s.ConfigureConfidenceMonitor(&lfg.ConfidenceMonitorConfig{
		Threshold: 0.3,
		MinSpan:   5,
		RingSize:  8,
	})

	vm := newVectorMemory(s)

	cfg := agentConfig{
		maxTokens:           512,
		entropyThreshold:    0.6,
		confidenceThreshold: 0.3,
		confidenceMinSpan:   5,
		reasoningBudget:     256,
	}

	// --- Turn 1: "Hello?" ---
	eventCh1 := make(chan tea.Msg, 256)
	go runAgentTurn(s, vm, nil, "Hello?", eventCh1, cfg)

	var history []lfg.ChatMessage
	var turn1Text string
	for msg := range eventCh1 {
		switch msg := msg.(type) {
		case responseMsg:
			turn1Text = string(msg)
			// Simulate what the TUI responseMsg handler does:
			// store plain text in history.
			history = append(history,
				lfg.ChatMessage{Role: "user", Content: "Hello?"},
				lfg.ChatMessage{Role: "assistant", Content: turn1Text},
			)
		case generationDone:
			goto turn2
		default:
			// absorb tokenMsg, memStoredMsg, etc.
		}
	}

turn2:
	t.Logf("Turn 1: %q", turn1Text)
	if turn1Text == "" {
		t.Fatal("turn 1 produced no response")
	}
	if len(history) == 0 {
		t.Fatal("history is empty after turn 1")
	}
	t.Logf("History after turn 1: %d messages", len(history))

	// --- Turn 2: "I'm Gabe" with history ---
	eventCh2 := make(chan tea.Msg, 256)
	go runAgentTurn(s, vm, history, "I'm Gabe", eventCh2, cfg)

	var turn2Text string
	for msg := range eventCh2 {
		switch msg := msg.(type) {
		case responseMsg:
			turn2Text = string(msg)
		case generationDone:
			goto done
		default:
		}
	}

done:
	t.Logf("Turn 2: %q", turn2Text)
	if turn2Text == "" {
		t.Fatal("turn 2 produced no response")
	}

	// Different user messages with history MUST produce different responses.
	if turn1Text == turn2Text {
		t.Fatal("identical responses across turns — history not being used")
	}
}

// ---------------------------------------------------------------------------
// Embed + Reset + ChatGenerate interaction test
// ---------------------------------------------------------------------------

func TestEmbedDoesNotCorruptChatGenerate(t *testing.T) {
	s := freshSession(t)

	// Configure monitors (same as the TUI agent).
	s.ConfigureEntropyMonitor(&lfg.EntropyMonitorConfig{
		Threshold:      0.6,
		CooldownTokens: 5,
		RingSize:       8,
	})
	s.ConfigureConfidenceMonitor(&lfg.ConfidenceMonitorConfig{
		Threshold: 0.3,
		MinSpan:   5,
		RingSize:  8,
	})

	genCfg := func(text *string) lfg.GenerateConfig {
		return lfg.GenerateConfig{
			MaxTokens: 128,
			TokenCallback: func(token lfg.Token, piece string) lfg.GenerateAction {
				*text += piece
				return lfg.GenerateContinue
			},
			EntropyCallback: func(event lfg.EntropyEvent, embedding []float32) string {
				return "" // no injection
			},
			ConfidenceCallback: func(event lfg.ConfidenceEvent, embedding []float32) {
				// no-op
			},
		}
	}

	// Turn 1: baseline.
	s.Reset()
	s.ConfigureStructured(testToolCallSchema, "")
	var text1 string
	s.ChatGenerate(
		[]lfg.ChatMessage{
			{Role: "system", Content: testStructuredSystemPrompt},
			{Role: "user", Content: "Hello?"},
		},
		genCfg(&text1),
	)
	t.Logf("Turn 1 (monitors, no embed): %q", text1)

	// Turn 2: Embed + Reset + ChatGenerate with history.
	s.Embed("I'm Gabe")
	s.Reset()
	s.ConfigureStructured(testToolCallSchema, "")
	var text2 string
	s.ChatGenerate(
		[]lfg.ChatMessage{
			{Role: "system", Content: testStructuredSystemPrompt},
			{Role: "user", Content: "Hello?"},
			{Role: "assistant", Content: text1},
			{Role: "user", Content: "I'm Gabe"},
		},
		genCfg(&text2),
	)
	t.Logf("Turn 2 (monitors + embed + history): %q", text2)

	// Turn 3: same prompt without Embed.
	s.Reset()
	s.ConfigureStructured(testToolCallSchema, "")
	var text3 string
	s.ChatGenerate(
		[]lfg.ChatMessage{
			{Role: "system", Content: testStructuredSystemPrompt},
			{Role: "user", Content: "Hello?"},
			{Role: "assistant", Content: text1},
			{Role: "user", Content: "I'm Gabe"},
		},
		genCfg(&text3),
	)
	t.Logf("Turn 3 (monitors, no embed + history): %q", text3)

	t.Logf("Turn 2 == Turn 1: %v", text2 == text1)
	t.Logf("Turn 2 == Turn 3: %v", text2 == text3)

	if text1 == text2 && text1 == text3 {
		t.Fatal("all turns identical — monitors break multi-turn structured generation")
	}
	if text1 == text2 {
		t.Fatal("Embed before Reset corrupts ChatGenerate")
	}
}

// ---------------------------------------------------------------------------
// Regression: RegisterTools + multi-turn
// ---------------------------------------------------------------------------

// TestRegisterToolsMultiTurn verifies that RegisterTools + structured
// output + multi-turn history works correctly together.
func TestRegisterToolsMultiTurn(t *testing.T) {

	s := freshSession(t)

	tools := []lfg.ToolDesc{
		{Name: "respond", Description: "Send a message to the user"},
		{Name: "recall", Description: "Search memory for relevant information"},
		{Name: "think", Description: "Internal reasoning step"},
	}
	s.RegisterTools(tools, 3)

	genCfg := func(text *string) lfg.GenerateConfig {
		return lfg.GenerateConfig{
			MaxTokens: 128,
			TokenCallback: func(token lfg.Token, piece string) lfg.GenerateAction {
				*text += piece
				return lfg.GenerateContinue
			},
		}
	}

	s.Reset()
	s.ConfigureStructured(testToolCallSchema, "")
	var text1 string
	s.ChatGenerate(
		[]lfg.ChatMessage{
			{Role: "system", Content: testStructuredSystemPrompt},
			{Role: "user", Content: "Hello?"},
		},
		genCfg(&text1),
	)
	t.Logf("Turn 1: %q", text1)

	s.Reset()
	s.ConfigureStructured(testToolCallSchema, "")
	var text2 string
	s.ChatGenerate(
		[]lfg.ChatMessage{
			{Role: "system", Content: testStructuredSystemPrompt},
			{Role: "user", Content: "Hello?"},
			{Role: "assistant", Content: text1},
			{Role: "user", Content: "I'm Gabe"},
		},
		genCfg(&text2),
	)
	t.Logf("Turn 2: %q", text2)

	if text1 == text2 {
		t.Fatal("identical — RegisterTools + multi-turn regression")
	}
}

// ---------------------------------------------------------------------------
// Reset and Re-generate (Session Reuse)
// ---------------------------------------------------------------------------

func TestSessionResetAndRegenerate(t *testing.T) {
	s := freshSession(t)

	// First generation.
	var text1 string
	s.PromptGenerate("Once upon a time", true, lfg.GenerateConfig{
		MaxTokens: 10,
		TokenCallback: func(token lfg.Token, piece string) lfg.GenerateAction {
			text1 += piece
			return lfg.GenerateContinue
		},
	})

	// Reset.
	s.Reset()

	// Second generation.
	var text2 string
	s.PromptGenerate("The quick brown fox", true, lfg.GenerateConfig{
		MaxTokens: 10,
		TokenCallback: func(token lfg.Token, piece string) lfg.GenerateAction {
			text2 += piece
			return lfg.GenerateContinue
		},
	})

	t.Logf("Gen 1: %q", text1)
	t.Logf("Gen 2: %q", text2)

	if text1 == "" || text2 == "" {
		t.Fatal("one or both generations produced empty text")
	}
}

func TestChatGenerateResetCycle(t *testing.T) {
	s := freshSession(t)

	for i := 0; i < 3; i++ {
		s.Reset()
		if err := s.ConfigureStructured(testToolCallSchema, ""); err != nil {
			t.Fatalf("ConfigureStructured cycle %d: %v", i, err)
		}

		var text string
		result, err := s.ChatGenerate(
			[]lfg.ChatMessage{
				{Role: "system", Content: testStructuredSystemPrompt},
				{Role: "user", Content: fmt.Sprintf("Say hello (cycle %d)", i)},
			},
			lfg.GenerateConfig{
				MaxTokens: 100,
				TokenCallback: func(token lfg.Token, piece string) lfg.GenerateAction {
					text += piece
					return lfg.GenerateContinue
				},
			},
		)
		if err != nil {
			t.Fatalf("ChatGenerate cycle %d: %v", i, err)
		}

		t.Logf("Cycle %d: %d tokens, text=%q", i, result.TokenCount, text)
		if result.TokenCount == 0 {
			t.Fatalf("cycle %d generated no tokens", i)
		}
	}
}

// ---------------------------------------------------------------------------
// Multi-Turn History with Reset+Structured (verifies history flows through)
// ---------------------------------------------------------------------------

func TestMultiTurnHistoryWithStructured(t *testing.T) {
	s := freshSession(t)

	// Turn 1: ask "Hello?"
	s.Reset()
	if err := s.ConfigureStructured(testToolCallSchema, ""); err != nil {
		t.Fatalf("ConfigureStructured turn 1: %v", err)
	}

	var text1 string
	_, err := s.ChatGenerate(
		[]lfg.ChatMessage{
			{Role: "system", Content: testStructuredSystemPrompt},
			{Role: "user", Content: "Hello?"},
		},
		lfg.GenerateConfig{
			MaxTokens: 128,
			TokenCallback: func(token lfg.Token, piece string) lfg.GenerateAction {
				text1 += piece
				return lfg.GenerateContinue
			},
		},
	)
	if err != nil {
		t.Fatalf("ChatGenerate turn 1: %v", err)
	}
	t.Logf("Turn 1 output: %q", text1)

	// Turn 2: include turn 1 as raw JSON history, ask a different question.
	s.Reset()
	if err := s.ConfigureStructured(testToolCallSchema, ""); err != nil {
		t.Fatalf("ConfigureStructured turn 2: %v", err)
	}

	var text2 string
	_, err = s.ChatGenerate(
		[]lfg.ChatMessage{
			{Role: "system", Content: testStructuredSystemPrompt},
			{Role: "user", Content: "Hello?"},
			{Role: "assistant", Content: text1}, // raw JSON from turn 1
			{Role: "user", Content: "What is 2+2?"},
		},
		lfg.GenerateConfig{
			MaxTokens: 128,
			TokenCallback: func(token lfg.Token, piece string) lfg.GenerateAction {
				text2 += piece
				return lfg.GenerateContinue
			},
		},
	)
	if err != nil {
		t.Fatalf("ChatGenerate turn 2: %v", err)
	}
	t.Logf("Turn 2 output: %q", text2)

	// The outputs must differ — different user message means different response.
	if text1 == text2 {
		t.Fatal("identical output across turns — history not being used by ChatGenerate")
	}

	// Both should be valid JSON tool calls.
	var tc1, tc2 toolCall
	if err := json.Unmarshal([]byte(text1), &tc1); err != nil {
		t.Fatalf("turn 1 not valid JSON: %v", err)
	}
	if err := json.Unmarshal([]byte(text2), &tc2); err != nil {
		t.Fatalf("turn 2 not valid JSON: %v", err)
	}
	t.Logf("Turn 1 tool: name=%q text=%q", tc1.Name, tc1.Arguments.Text)
	t.Logf("Turn 2 tool: name=%q text=%q", tc2.Name, tc2.Arguments.Text)
}

// ---------------------------------------------------------------------------
// Token Accumulator with Real Data
// ---------------------------------------------------------------------------

func TestTokenAccumulatorWithGeneration(t *testing.T) {
	s := freshSession(t)
	accum := &tokenAccumulator{}

	result, err := s.PromptGenerate("Hello world", true, lfg.GenerateConfig{
		MaxTokens: 15,
		TokenCallback: func(token lfg.Token, piece string) lfg.GenerateAction {
			accum.add(piece)
			return lfg.GenerateContinue
		},
	})
	if err != nil {
		t.Fatalf("PromptGenerate: %v", err)
	}

	fullText := accum.text()
	lastN := accum.extractLastN(5)

	t.Logf("Full text: %q", fullText)
	t.Logf("Last 5 pieces: %q", lastN)
	t.Logf("Token count: %d", result.TokenCount)

	if fullText == "" && result.TokenCount > 0 {
		t.Fatal("accumulator text is empty despite generating tokens")
	}
}

// ---------------------------------------------------------------------------
// TUI Model Integration (with real model data)
// ---------------------------------------------------------------------------

func TestModelLoadedTransitionWithRealModel(t *testing.T) {
	m := requireModel(t)
	s := requireSession(t)
	mem := requireMemory(t)

	tuiModel := TestableModel()

	updated, _ := tuiModel.Update(modelLoaded{
		model:   m,
		session: s,
		memory:  mem,
	})
	um := updated.(model)

	if !um.loaded {
		t.Fatal("model should be loaded after modelLoaded message")
	}
	if um.session == nil {
		t.Fatal("session should not be nil after modelLoaded")
	}
	if um.lfgModel == nil {
		t.Fatal("lfgModel should not be nil after modelLoaded")
	}
	if um.memory == nil {
		t.Fatal("memory should not be nil after modelLoaded")
	}
	if um.status != "Ready" {
		t.Fatalf("status = %q, want %q", um.status, "Ready")
	}
}

func TestTUIStreamingTokenDisplay(t *testing.T) {
	tuiModel := TestableModel()
	tuiModel.loaded = true

	// Simulate streaming plain text tokens (thinking already done).
	tuiModel.streaming = "Hello from the agent!"
	tuiModel.thinkingDone = true

	view := tuiModel.View()
	if !strings.Contains(view, "Hello from the agent!") {
		t.Fatalf("streaming text not found in view:\n%s", view)
	}
}

func TestTUIMessageTypes(t *testing.T) {
	tuiModel := TestableModel()
	tuiModel.loaded = true

	// Chat messages stay in messages.
	tuiModel.messages = []chatEntry{
		{role: "user", text: "Hello"},
		{role: "assistant", text: "Hi there!"},
	}

	// Memory events go to memoryEntries.
	tuiModel.memoryEntries = []chatEntry{
		{role: "system", text: "[recall] Retrieved 2 items"},
		{role: "stored", text: `[stored] "important fact" (5t)`},
	}
	tuiModel.memViewport.SetContent(renderMemoryContent(tuiModel.memoryEntries, tuiModel.memViewport.Width))

	// Check chat tab (tab 0).
	chatView := tuiModel.View()
	if !strings.Contains(chatView, "Hello") {
		t.Fatal("user message not found on chat tab")
	}
	if !strings.Contains(chatView, "Hi there!") {
		t.Fatal("assistant message not found on chat tab")
	}

	// Check memory tab (tab 1).
	tuiModel.activeTab = 1
	memView := tuiModel.View()
	if !strings.Contains(memView, "[recall]") {
		t.Fatal("system message not found on memory tab")
	}
	if !strings.Contains(memView, "important fact") {
		t.Fatal("stored message not found on memory tab")
	}
}

func TestTUIGenerationDoneUpdatesStatus(t *testing.T) {
	tuiModel := TestableModel()
	tuiModel.loaded = true
	tuiModel.generating = true
	tuiModel.memory = &vectorMemory{}

	// Store some fake entries.
	tuiModel.memory.storeWithEmbedding("fact1", []float32{1, 0, 0}, "user")
	tuiModel.memory.storeWithEmbedding("fact2", []float32{0, 1, 0}, "confidence")

	updated, _ := tuiModel.Update(generationDone{})
	um := updated.(model)

	if um.generating {
		t.Fatal("generating should be false after generationDone")
	}
	if !strings.Contains(um.status, "memories") {
		t.Fatalf("status should contain memory count, got %q", um.status)
	}
	t.Logf("Status after generation: %q", um.status)
}

// ---------------------------------------------------------------------------
// Cosine Similarity Edge Cases
// ---------------------------------------------------------------------------

func TestCosineSimilarityEdgeCases(t *testing.T) {
	// Identical vectors.
	sim := cosineSimilarity([]float32{1, 2, 3}, []float32{1, 2, 3})
	if sim < 0.99 {
		t.Fatalf("identical vectors: sim=%.4f, want ~1.0", sim)
	}

	// Orthogonal vectors.
	sim = cosineSimilarity([]float32{1, 0}, []float32{0, 1})
	if sim > 0.01 {
		t.Fatalf("orthogonal vectors: sim=%.4f, want ~0.0", sim)
	}

	// Opposite vectors.
	sim = cosineSimilarity([]float32{1, 0}, []float32{-1, 0})
	if sim > -0.99 {
		t.Fatalf("opposite vectors: sim=%.4f, want ~-1.0", sim)
	}

	// Empty vectors.
	sim = cosineSimilarity([]float32{}, []float32{})
	if sim != 0 {
		t.Fatalf("empty vectors: sim=%.4f, want 0", sim)
	}

	// Mismatched lengths.
	sim = cosineSimilarity([]float32{1, 2}, []float32{1, 2, 3})
	if sim != 0 {
		t.Fatalf("mismatched lengths: sim=%.4f, want 0", sim)
	}

	// Zero vector.
	sim = cosineSimilarity([]float32{0, 0, 0}, []float32{1, 2, 3})
	if sim != 0 {
		t.Fatalf("zero vector: sim=%.4f, want 0", sim)
	}
}

// ---------------------------------------------------------------------------
// Truncate Edge Cases
// ---------------------------------------------------------------------------

func TestTruncateEdgeCases(t *testing.T) {
	if truncate("", 10) != "" {
		t.Fatal("empty string should return empty")
	}
	if truncate("abc", 3) != "abc" {
		t.Fatal("exactly max length should not truncate")
	}
	if truncate("abcd", 3) != "abc..." {
		t.Fatalf("got %q", truncate("abcd", 3))
	}
	if truncate("x", 0) != "..." {
		t.Fatalf("max=0 should truncate to '...', got %q", truncate("x", 0))
	}
}

// ---------------------------------------------------------------------------
// Checkpoint Integration
// ---------------------------------------------------------------------------

func TestCheckpointWithStructuredOutput(t *testing.T) {
	s := freshSession(t)

	m := requireModel(t)
	vocab := m.Vocab()
	tokens, _ := vocab.Tokenize("Hello", true, false)
	s.IngestTokens(tokens, true)
	s.Decode()
	s.Sample()

	cp := s.CreateCheckpoint()
	if cp == nil {
		t.Fatal("CreateCheckpoint returned nil")
	}
	defer cp.Close()

	// Generate from checkpoint state.
	var text1 string
	s.GenerateFromState(lfg.GenerateConfig{
		MaxTokens: 5,
		TokenCallback: func(token lfg.Token, piece string) lfg.GenerateAction {
			text1 += piece
			return lfg.GenerateContinue
		},
	})

	// Restore and generate again.
	if err := s.RestoreCheckpoint(cp); err != nil {
		t.Fatalf("RestoreCheckpoint: %v", err)
	}

	var text2 string
	s.GenerateFromState(lfg.GenerateConfig{
		MaxTokens: 5,
		TokenCallback: func(token lfg.Token, piece string) lfg.GenerateAction {
			text2 += piece
			return lfg.GenerateContinue
		},
	})

	t.Logf("Path 1: %q", text1)
	t.Logf("Path 2: %q", text2)
}

// ---------------------------------------------------------------------------
// JSON Schema to Grammar
// ---------------------------------------------------------------------------

func TestJSONSchemaToGrammar(t *testing.T) {
	grammar, err := lfg.JSONSchemaToGrammar(testToolCallSchema, false)
	if err != nil {
		t.Fatalf("JSONSchemaToGrammar: %v", err)
	}
	if grammar == "" {
		t.Fatal("grammar is empty")
	}
	t.Logf("Grammar (first 300 chars): %.300s", grammar)
}

// ---------------------------------------------------------------------------
// Model Stats
// ---------------------------------------------------------------------------

func TestModelStats(t *testing.T) {
	m := requireModel(t)

	stats := m.Stats()
	t.Logf("ParameterCount: %d", stats.ParameterCount)
	t.Logf("SizeBytes: %d", stats.SizeBytes)
	t.Logf("VocabSize: %d", stats.VocabSize)
	t.Logf("ContextSize: %d", stats.ContextSize)

	if stats.ParameterCount == 0 {
		t.Fatal("ParameterCount is 0")
	}
	if stats.VocabSize == 0 {
		t.Fatal("VocabSize is 0")
	}
}

// ---------------------------------------------------------------------------
// Callback Stop Integration
// ---------------------------------------------------------------------------

func TestCallbackStopDuringChatGenerate(t *testing.T) {
	s := freshSession(t)

	stopAfter := 5
	count := 0

	result, err := s.ChatGenerate(
		[]lfg.ChatMessage{
			{Role: "user", Content: "Count from 1 to 100."},
		},
		lfg.GenerateConfig{
			MaxTokens: 200,
			TokenCallback: func(token lfg.Token, piece string) lfg.GenerateAction {
				count++
				if count >= stopAfter {
					return lfg.GenerateStop
				}
				return lfg.GenerateContinue
			},
		},
	)
	if err != nil {
		t.Fatalf("ChatGenerate: %v", err)
	}

	t.Logf("Stopped after %d tokens, stop_reason=%d", result.TokenCount, result.StopReason)
	if result.TokenCount != stopAfter {
		t.Fatalf("expected %d tokens, got %d", stopAfter, result.TokenCount)
	}
	if result.StopReason != lfg.StopReasonCallback {
		t.Fatalf("expected StopReasonCallback, got %d", result.StopReason)
	}
}
