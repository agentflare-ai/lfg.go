package main

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
)

func TestInitialRender(t *testing.T) {
	m := TestableModel()
	view := m.View()

	if !strings.Contains(view, "LFG Chat Agent") {
		t.Fatal("header not found in view")
	}
	// TestableModel has no session, so it shows "Loading model...".
	if !strings.Contains(view, "Loading model...") {
		t.Fatal("loading message not found in view")
	}
	if !strings.Contains(view, "Test Mode") {
		t.Fatal("status not found in view")
	}

	// After simulating model load, input should be visible.
	m2, _ := m.Update(modelLoaded{})
	view2 := m2.(model).View()
	if strings.Contains(view2, "Loading model...") {
		t.Fatal("loading message should be gone after model load")
	}
}

func TestUserMessageDisplay(t *testing.T) {
	m := TestableModel()
	m.messages = append(m.messages, chatEntry{role: "user", text: "Hello!"})
	m.messages = append(m.messages, chatEntry{role: "assistant", text: "Hi there!"})

	view := m.View()

	if !strings.Contains(view, "Hello!") {
		t.Fatal("user message not in view")
	}
	if !strings.Contains(view, "Hi there!") {
		t.Fatal("assistant message not in view")
	}
}

func TestStoredMessageDisplay(t *testing.T) {
	m := TestableModel()
	m.memoryEntries = append(m.memoryEntries, chatEntry{
		role: "stored",
		text: `[stored] "Paris is the capital" (5t)`,
	})
	m.activeTab = 1
	m.memViewport.SetContent(renderMemoryContent(m.memoryEntries))

	view := m.View()
	if !strings.Contains(view, "Paris is the capital") {
		t.Fatal("stored message not in view")
	}
}

func TestQuit(t *testing.T) {
	m := TestableModel()

	updated, cmd := m.Update(tea.KeyMsg{Type: tea.KeyCtrlC})
	um := updated.(model)

	if !um.quitting {
		t.Fatal("model should be quitting after Ctrl+C")
	}
	if cmd == nil {
		t.Fatal("quit command should not be nil")
	}
}

func TestStreamingDisplay(t *testing.T) {
	m := TestableModel()
	m.streaming = "Hello world"

	view := m.View()
	if !strings.Contains(view, "Hello world") {
		t.Fatal("streaming text not found in view")
	}
}

func TestTruncate(t *testing.T) {
	if truncate("short", 10) != "short" {
		t.Fatal("should not truncate short strings")
	}
	if truncate("a long string here", 6) != "a long..." {
		t.Fatalf("truncate mismatch: %q", truncate("a long string here", 6))
	}
}

func TestCosineSimilarity(t *testing.T) {
	a := []float32{1, 0, 0}
	b := []float32{1, 0, 0}
	sim := cosineSimilarity(a, b)
	if sim < 0.99 {
		t.Fatalf("identical vectors should have similarity ~1, got %f", sim)
	}

	c := []float32{0, 1, 0}
	sim = cosineSimilarity(a, c)
	if sim > 0.01 {
		t.Fatalf("orthogonal vectors should have similarity ~0, got %f", sim)
	}
}

func TestVectorMemoryStoreAndSearch(t *testing.T) {
	vm := &vectorMemory{}

	vm.storeWithEmbedding("hello", []float32{1, 0, 0}, "user")
	vm.storeWithEmbedding("world", []float32{0, 1, 0}, "confidence")

	if vm.count() != 2 {
		t.Fatalf("count = %d, want 2", vm.count())
	}
	if vm.autoCount() != 1 {
		t.Fatalf("autoCount = %d, want 1", vm.autoCount())
	}

	// Search with a query close to "hello".
	results := vm.search([]float32{0.9, 0.1, 0}, 1)
	if len(results) != 1 || results[0] != "hello" {
		t.Fatalf("search returned %v, want [hello]", results)
	}
}

func TestTokenAccumulator(t *testing.T) {
	a := &tokenAccumulator{}
	a.add("hello")
	a.add(" ")
	a.add("world")

	if a.text() != "hello world" {
		t.Fatalf("text = %q, want %q", a.text(), "hello world")
	}

	last2 := a.extractLastN(2)
	if last2 != " world" {
		t.Fatalf("extractLastN(2) = %q, want %q", last2, " world")
	}
}

func TestStripThinking(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{"no thinking", "Hello world", "Hello world"},
		{"thinking only", "<think>internal reasoning</think>", ""},
		{"thinking then response", "<think>let me think</think>Hello!", "Hello!"},
		{"response then thinking", "Hello<think>hmm</think> world", "Hello world"},
		{"unclosed thinking", "Hello<think>still thinking...", "Hello"},
		{"multiple blocks", "<think>a</think>Hello<think>b</think> world", "Hello world"},
		{"empty", "", ""},
		{"nested tags", "<think>outer<think>inner</think>rest", "rest"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := stripThinking(tt.input)
			if got != tt.want {
				t.Fatalf("stripThinking(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestStreamingDisplayWithThinking(t *testing.T) {
	m := TestableModel()
	m.streaming = "<think>let me reason about this</think>The answer is 42"

	view := m.View()
	if !strings.Contains(view, "The answer is 42") {
		t.Fatal("stripped response not found in view")
	}
	if strings.Contains(view, "let me reason") {
		t.Fatal("thinking content should not appear in view")
	}
}

func TestStreamingDisplayThinkingOnly(t *testing.T) {
	m := TestableModel()
	m.streaming = "<think>still reasoning..."

	view := m.View()
	if strings.Contains(view, "still reasoning") {
		t.Fatal("thinking content should not appear in view")
	}
	if !strings.Contains(view, "Thinking...") {
		t.Fatal("should show 'Thinking...' when only thinking tokens received")
	}
}

func TestModelLoadedTransition(t *testing.T) {
	m := TestableModel()

	if m.session != nil {
		t.Fatal("session should be nil initially")
	}
	if m.status != "Test Mode" {
		t.Fatalf("status = %q, want %q", m.status, "Test Mode")
	}

	// Simulate modelLoaded without a real model.
	updated, _ := m.Update(modelLoaded{})
	um := updated.(model)

	if um.status != "Ready" {
		t.Fatalf("status after load = %q, want %q", um.status, "Ready")
	}
}
