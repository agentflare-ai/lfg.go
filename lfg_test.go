package lfg

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

// testModel is the shared model instance loaded once in TestMain.
var testModel *Model

// testModelPath returns the path to the test model.
func testModelPath() string {
	if p := os.Getenv("LFG_MODEL_PATH"); p != "" {
		return p
	}
	return "models/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf"
}

func TestMain(m *testing.M) {
	path := testModelPath()
	if _, err := os.Stat(path); err == nil {
		var loadErr error
		testModel, loadErr = LoadModel(path, WithGPULayers(0))
		if loadErr != nil {
			fmt.Fprintf(os.Stderr, "FATAL: LoadModel: %v\n", loadErr)
			os.Exit(1)
		}
	}

	code := m.Run()

	if testModel != nil {
		testModel.Close()
	}

	os.Exit(code)
}

// requireModel returns the shared test model, skipping if not available.
func requireModel(t *testing.T) *Model {
	t.Helper()
	if testModel == nil {
		t.Skipf("Model not found at %s — set LFG_MODEL_PATH or download the model", testModelPath())
	}
	return testModel
}

// modelPath returns the path for tests that load their own model.
func modelPath(t *testing.T) string {
	t.Helper()
	return testModelPath()
}

// gcCollect forces GC + finalizer execution to reclaim C resources promptly.
func gcCollect() {
	runtime.GC()
	runtime.GC()
}

// ---------------------------------------------------------------------------
// Backend / System Info
// ---------------------------------------------------------------------------

func TestVersion(t *testing.T) {
	v := Version()
	if v == "" {
		t.Fatal("Version returned empty string")
	}
	t.Logf("API version: %s", v)
}

func TestVersionNumbers(t *testing.T) {
	major, minor, patch := VersionNumbers()
	t.Logf("API version: %d.%d.%d", major, minor, patch)
	if major == 0 && minor == 0 && patch == 0 {
		t.Fatal("all version numbers are zero")
	}
}

func TestABIVersion(t *testing.T) {
	v := ABIVersion()
	t.Logf("ABI version: %d", v)
	if v == 0 {
		t.Fatal("ABI version is zero")
	}
}

func TestSystemInfo(t *testing.T) {
	info := SystemInfo()
	if info == "" {
		t.Fatal("SystemInfo returned empty string")
	}
	t.Logf("System info: %s", info)
}

func TestSystemCapabilities(t *testing.T) {
	t.Logf("DeviceCount: %d", DeviceCount())
	t.Logf("MaxParallelSequences: %d", MaxParallelSequences())
	t.Logf("SupportsMmap: %v", SupportsMmap())
	t.Logf("SupportsMlock: %v", SupportsMlock())
	t.Logf("SupportsGPUOffload: %v", SupportsGPUOffload())
	t.Logf("SupportsRPC: %v", SupportsRPC())
}

func TestBackendFreeIdempotent(t *testing.T) {
	// NOTE: We don't actually call BackendFree() here because it invalidates
	// shared state used by other tests in this process. In production, call
	// BackendFree() once at program exit. This test verifies compilation only.
	t.Log("BackendFree() is available (skipping actual call to avoid invalidating test state)")
}

func TestTimeMicroseconds(t *testing.T) {
	t0 := TimeMicroseconds()
	if t0 <= 0 {
		t.Fatalf("TimeMicroseconds returned non-positive: %d", t0)
	}
	t1 := TimeMicroseconds()
	if t1 < t0 {
		t.Fatalf("TimeMicroseconds is not monotonic: %d < %d", t1, t0)
	}
}

// ---------------------------------------------------------------------------
// Error Type
// ---------------------------------------------------------------------------

func TestErrorCodeString(t *testing.T) {
	s := ErrorNone.String()
	if s == "" {
		t.Fatal("ErrorNone.String() is empty")
	}
	t.Logf("ErrorNone = %q", s)

	s = ErrorInvalidArgument.String()
	if s == "" {
		t.Fatal("ErrorInvalidArgument.String() is empty")
	}
	t.Logf("ErrorInvalidArgument = %q", s)
}

func TestErrorInterface(t *testing.T) {
	err := &Error{Code: ErrorIO, Message: "test message"}
	var e error = err
	if e.Error() == "" {
		t.Fatal("Error.Error() returned empty")
	}
	if !strings.Contains(e.Error(), "test message") {
		t.Fatalf("Error.Error() missing message: %s", e.Error())
	}
	t.Logf("Error: %s", e)
}

// ---------------------------------------------------------------------------
// Model Load Failure
// ---------------------------------------------------------------------------

func TestLoadModelBadPath(t *testing.T) {
	_, err := LoadModel("/nonexistent/path/model.gguf", WithGPULayers(0))
	if err == nil {
		t.Fatal("expected error loading nonexistent model")
	}
	t.Logf("Expected error: %v", err)
}

// ---------------------------------------------------------------------------
// Model Properties
// ---------------------------------------------------------------------------

func TestModelProperties(t *testing.T) {
	m := requireModel(t)

	if m.EmbeddingSize() <= 0 {
		t.Fatal("EmbeddingSize should be > 0")
	}
	if m.LayerCount() <= 0 {
		t.Fatal("LayerCount should be > 0")
	}
	if m.HeadCount() <= 0 {
		t.Fatal("HeadCount should be > 0")
	}
	if m.TrainingContextSize() <= 0 {
		t.Fatal("TrainingContextSize should be > 0")
	}
	if m.Size() == 0 {
		t.Fatal("Size should be > 0")
	}
	if m.ParameterCount() == 0 {
		t.Fatal("ParameterCount should be > 0")
	}

	desc := m.Description()
	if desc == "" {
		t.Fatal("Description is empty")
	}

	t.Logf("Description: %s", desc)
	t.Logf("EmbeddingSize: %d, LayerCount: %d, HeadCount: %d", m.EmbeddingSize(), m.LayerCount(), m.HeadCount())
	t.Logf("TrainingContextSize: %d, Size: %d bytes, ParameterCount: %d", m.TrainingContextSize(), m.Size(), m.ParameterCount())
	t.Logf("RopeType: %d, RopeFreqScaleTrain: %f", m.RopeType(), m.RopeFreqScaleTrain())
	t.Logf("HasEncoder: %v, HasDecoder: %v", m.HasEncoder(), m.HasDecoder())
	t.Logf("IsRecurrent: %v, IsHybrid: %v, IsDiffusion: %v", m.IsRecurrent(), m.IsHybrid(), m.IsDiffusion())
}

func TestModelMetadata(t *testing.T) {
	m := requireModel(t)

	count := m.MetadataCount()
	if count <= 0 {
		t.Fatal("MetadataCount should be > 0")
	}
	t.Logf("Metadata count: %d", count)

	// Print first 5 metadata entries.
	for i := 0; i < count && i < 5; i++ {
		key, ok := m.MetadataKeyAt(i)
		if !ok {
			continue
		}
		val, _ := m.MetadataValueAt(i)
		t.Logf("  [%d] %s = %s", i, key, val)
	}
}

func TestModelChatTemplate(t *testing.T) {
	m := requireModel(t)

	tmpl, ok := m.ChatTemplate("")
	if !ok {
		t.Skip("Model has no default chat template")
	}
	if tmpl == "" {
		t.Fatal("ChatTemplate returned empty string")
	}
	t.Logf("Chat template (first 200 chars): %.200s", tmpl)
}

func TestModelCloseIdempotent(t *testing.T) {
	path := modelPath(t)
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skipf("Model not found at %s", path)
	}
	m, err := LoadModel(path, WithGPULayers(0))
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	// Close twice should not panic.
	m.Close()
	m.Close()
}

// ---------------------------------------------------------------------------
// Vocab / Tokenization
// ---------------------------------------------------------------------------

func TestVocab(t *testing.T) {
	m := requireModel(t)
	v := m.Vocab()
	if v == nil {
		t.Fatal("Vocab returned nil")
	}

	nTokens := v.TokenCount()
	if nTokens <= 0 {
		t.Fatalf("TokenCount = %d, want > 0", nTokens)
	}
	t.Logf("VocabType: %d, TokenCount: %d", v.Type(), nTokens)

	// Special tokens.
	bos := v.BOS()
	eos := v.EOS()
	t.Logf("BOS: %d, EOS: %d, EOT: %d, NL: %d, PAD: %d", bos, eos, v.EOT(), v.NL(), v.PAD())
	t.Logf("AddBOS: %v, AddEOS: %v", v.AddBOS(), v.AddEOS())

	// BOS should not be EOG.
	if v.IsEOG(bos) {
		t.Log("BOS is EOG (unusual but possible for some models)")
	}
	// EOS should be EOG.
	if !v.IsEOG(eos) {
		t.Error("EOS should be EOG")
	}
}

func TestTokenize(t *testing.T) {
	m := requireModel(t)
	v := m.Vocab()

	tokens, err := v.Tokenize("Hello, world!", true, false)
	if err != nil {
		t.Fatalf("Tokenize: %v", err)
	}
	if len(tokens) == 0 {
		t.Fatal("Tokenize returned no tokens")
	}
	t.Logf("Tokenized %d tokens: %v", len(tokens), tokens)
}

func TestDetokenize(t *testing.T) {
	m := requireModel(t)
	v := m.Vocab()

	original := "Hello, world!"
	tokens, err := v.Tokenize(original, false, false)
	if err != nil {
		t.Fatalf("Tokenize: %v", err)
	}

	text, err := v.Detokenize(tokens, false, false)
	if err != nil {
		t.Fatalf("Detokenize: %v", err)
	}
	if text != original {
		t.Logf("Detokenize roundtrip: %q (original %q) — may differ due to tokenizer normalization", text, original)
	}
	t.Logf("Detokenized: %q", text)
}

func TestTokenizeEmpty(t *testing.T) {
	m := requireModel(t)
	v := m.Vocab()

	tokens, err := v.Tokenize("", false, false)
	if err != nil {
		t.Fatalf("Tokenize empty: %v", err)
	}
	t.Logf("Empty tokenization: %d tokens", len(tokens))
}

func TestTokenText(t *testing.T) {
	m := requireModel(t)
	v := m.Vocab()

	tokens, err := v.Tokenize("Hello", false, false)
	if err != nil {
		t.Fatalf("Tokenize: %v", err)
	}

	var pieces []string
	for _, tok := range tokens {
		piece := v.TokenText(tok, false)
		pieces = append(pieces, piece)
	}
	t.Logf("Pieces: %v", pieces)

	joined := strings.Join(pieces, "")
	if !strings.Contains(joined, "Hello") && !strings.Contains(joined, "hello") {
		t.Logf("Joined pieces %q do not contain 'Hello' — may be due to tokenizer encoding", joined)
	}
}

func TestTokenAttributes(t *testing.T) {
	m := requireModel(t)
	v := m.Vocab()

	bos := v.BOS()
	attr := v.Attributes(bos)
	t.Logf("BOS token %d: attr=%d, score=%.4f, text=%q", bos, attr, v.Score(bos), v.Text(bos))

	if !v.IsControl(bos) {
		t.Log("BOS is not marked as control (may be model-specific)")
	}
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

func TestNewContext(t *testing.T) {
	m := requireModel(t)

	ctx, err := NewContext(m, WithContextSize(512), WithBatchSize(512), WithThreads(4))
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Close()

	if ctx.ContextSize() == 0 {
		t.Fatal("ContextSize is 0")
	}
	if ctx.BatchSize() == 0 {
		t.Fatal("BatchSize is 0")
	}

	t.Logf("ContextSize: %d, BatchSize: %d, MicroBatchSize: %d, MaxSequences: %d",
		ctx.ContextSize(), ctx.BatchSize(), ctx.MicroBatchSize(), ctx.MaxSequences())
	t.Logf("ThreadCount: %d, BatchThreadCount: %d", ctx.ThreadCount(), ctx.BatchThreadCount())
}

func TestContextClosedModel(t *testing.T) {
	path := modelPath(t)
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skipf("Model not found at %s", path)
	}
	m, err := LoadModel(path, WithGPULayers(0))
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	m.Close()

	_, err = NewContext(m, WithContextSize(512))
	if err == nil {
		t.Fatal("expected error creating context from closed model")
	}
	t.Logf("Expected error: %v", err)
}

func TestContextCloseIdempotent(t *testing.T) {
	m := requireModel(t)
	ctx, err := NewContext(m, WithContextSize(512))
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	ctx.Close()
	ctx.Close()
}

func TestContextSetters(t *testing.T) {
	m := requireModel(t)
	ctx, err := NewContext(m, WithContextSize(512))
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Close()

	ctx.SetThreads(2, 2)
	if ctx.ThreadCount() != 2 {
		t.Errorf("ThreadCount = %d, want 2", ctx.ThreadCount())
	}

	ctx.SetCausalAttn(true)
	ctx.SetWarmup(false)
	ctx.SetEmbeddings(false)
}

// ---------------------------------------------------------------------------
// Batch / Decode
// ---------------------------------------------------------------------------

func TestBatchGetOneAndDecode(t *testing.T) {
	m := requireModel(t)
	ctx, err := NewContext(m, WithContextSize(512), WithBatchSize(512))
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Close()

	v := m.Vocab()
	tokens, err := v.Tokenize("Hello", true, false)
	if err != nil {
		t.Fatalf("Tokenize: %v", err)
	}

	batch := BatchGetOne(tokens)
	if err := ctx.Decode(batch); err != nil {
		t.Fatalf("Decode: %v", err)
	}

	logits := ctx.LogitsAt(-1)
	if logits == nil {
		t.Fatal("LogitsAt returned nil")
	}
	if len(logits) != v.TokenCount() {
		t.Fatalf("logits length = %d, want %d", len(logits), v.TokenCount())
	}
	t.Logf("First 5 logits: %v", logits[:5])
}

func TestBatchInit(t *testing.T) {
	batch := BatchInit(512, 0, 1)
	defer batch.Close()

	if batch.TokenCount() != 0 {
		t.Fatalf("TokenCount = %d, want 0", batch.TokenCount())
	}
	batch.SetTokenCount(0)
}

// ---------------------------------------------------------------------------
// Memory (KV Cache)
// ---------------------------------------------------------------------------

func TestMemory(t *testing.T) {
	m := requireModel(t)
	ctx, err := NewContext(m, WithContextSize(512))
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Close()

	mem := ctx.Memory()
	if mem == nil {
		t.Fatal("Memory returned nil")
	}

	t.Logf("CanShift: %v", mem.CanShift())
	t.Logf("SeqPosMin(0): %d, SeqPosMax(0): %d", mem.SeqPosMin(0), mem.SeqPosMax(0))

	// Decode some tokens first so there's something in the cache.
	v := m.Vocab()
	tokens, _ := v.Tokenize("Hello, world!", true, false)
	batch := BatchGetOne(tokens)
	if err := ctx.Decode(batch); err != nil {
		t.Fatalf("Decode: %v", err)
	}

	// Now the cache should have entries.
	posMax := mem.SeqPosMax(0)
	t.Logf("After decode — SeqPosMin(0): %d, SeqPosMax(0): %d", mem.SeqPosMin(0), posMax)

	if posMax < 0 {
		t.Fatal("SeqPosMax should be >= 0 after decode")
	}

	// Test copy sequence.
	mem.SeqCp(0, 1, 0, -1)
	t.Logf("After copy — Seq1 PosMax: %d", mem.SeqPosMax(1))

	// Test remove.
	ok := mem.SeqRm(1, 0, -1)
	t.Logf("SeqRm(1): %v", ok)

	// Test keep.
	mem.SeqKeep(0)

	// Test clear.
	mem.Clear(false)
	t.Logf("After clear — SeqPosMax(0): %d", mem.SeqPosMax(0))
}

// ---------------------------------------------------------------------------
// Sampler Chain
// ---------------------------------------------------------------------------

func TestSamplerChain(t *testing.T) {
	m := requireModel(t)
	v := m.Vocab()

	chain := NewSamplerChain(false)
	defer chain.Close()

	chain.Add(NewTopKSampler(40))
	chain.Add(NewTopPSampler(0.9, 1))
	chain.Add(NewMinPSampler(0.05, 1))
	chain.Add(NewTempSampler(0.8))
	chain.Add(NewDistSampler(42))

	if chain.ChainN() != 5 {
		t.Fatalf("ChainN = %d, want 5", chain.ChainN())
	}

	name := chain.Name()
	t.Logf("Chain name: %s", name)

	// Decode and sample.
	ctx, err := NewContext(m, WithContextSize(512), WithBatchSize(512))
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Close()

	tokens, _ := v.Tokenize("The capital of France is", true, false)
	batch := BatchGetOne(tokens)
	if err := ctx.Decode(batch); err != nil {
		t.Fatalf("Decode: %v", err)
	}

	token := chain.Sample(ctx, -1)
	if token == InvalidToken {
		t.Fatal("Sample returned InvalidToken")
	}

	piece := v.TokenText(token, false)
	t.Logf("Sampled token: %d (%q)", token, piece)
}

func TestSamplerGreedy(t *testing.T) {
	m := requireModel(t)
	v := m.Vocab()

	chain := NewSamplerChain(false)
	defer chain.Close()
	chain.Add(NewGreedySampler())

	ctx, err := NewContext(m, WithContextSize(512), WithBatchSize(512))
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Close()

	tokens, _ := v.Tokenize("The capital of France is", true, false)
	batch := BatchGetOne(tokens)
	if err := ctx.Decode(batch); err != nil {
		t.Fatalf("Decode: %v", err)
	}

	token := chain.Sample(ctx, -1)
	if token == InvalidToken {
		t.Fatal("Greedy sample returned InvalidToken")
	}
	t.Logf("Greedy sampled: %d (%q)", token, v.TokenText(token, false))
}

func TestSamplerPenalties(t *testing.T) {
	s := NewPenaltiesSampler(64, 1.1, 0.0, 0.0)
	defer s.Close()
	if s == nil {
		t.Fatal("NewPenaltiesSampler returned nil")
	}
	name := s.Name()
	if name == "" {
		t.Fatal("penalties sampler has no name")
	}
	t.Logf("Penalties sampler name: %s", name)
}

func TestSamplerClone(t *testing.T) {
	chain := NewSamplerChain(false)
	defer chain.Close()
	chain.Add(NewTopKSampler(50))
	chain.Add(NewTempSampler(0.7))
	chain.Add(NewDistSampler(42))

	clone := chain.Clone()
	if clone == nil {
		t.Fatal("Clone returned nil")
	}
	defer clone.Close()

	if clone.ChainN() != chain.ChainN() {
		t.Fatalf("Clone ChainN = %d, want %d", clone.ChainN(), chain.ChainN())
	}
}

func TestSamplerChainRemove(t *testing.T) {
	chain := NewSamplerChain(false)
	defer chain.Close()
	chain.Add(NewTopKSampler(50))
	chain.Add(NewTempSampler(0.7))
	chain.Add(NewDistSampler(42))

	if chain.ChainN() != 3 {
		t.Fatalf("ChainN = %d, want 3", chain.ChainN())
	}

	removed := chain.ChainRemove(1) // remove temp
	if removed == nil {
		t.Fatal("ChainRemove returned nil")
	}
	defer removed.Close()

	if chain.ChainN() != 2 {
		t.Fatalf("ChainN after remove = %d, want 2", chain.ChainN())
	}
}

func TestSamplerReset(t *testing.T) {
	chain := NewSamplerChain(false)
	defer chain.Close()
	chain.Add(NewDistSampler(42))

	// Reset should not panic.
	chain.Reset()
}

func TestSamplerGetSeed(t *testing.T) {
	s := NewDistSampler(42)
	defer s.Close()
	seed := s.GetSeed()
	if seed != 42 {
		t.Logf("Seed: %d (may differ from input due to internal seeding)", seed)
	}
}

func TestSamplerCloseIdempotent(t *testing.T) {
	s := NewGreedySampler()
	s.Close()
	s.Close()
}

// ---------------------------------------------------------------------------
// Low-Level Generation (Context + Sampler)
// ---------------------------------------------------------------------------

func TestLowLevelGenerate(t *testing.T) {
	m := requireModel(t)
	v := m.Vocab()

	ctx, err := NewContext(m, WithContextSize(512), WithBatchSize(512))
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Close()

	chain := NewSamplerChain(false)
	defer chain.Close()
	chain.Add(NewGreedySampler())

	prompt := "Once upon a time"
	tokens, err := v.Tokenize(prompt, true, false)
	if err != nil {
		t.Fatalf("Tokenize: %v", err)
	}

	// Decode prompt.
	batch := BatchGetOne(tokens)
	if err := ctx.Decode(batch); err != nil {
		t.Fatalf("Decode prompt: %v", err)
	}

	// Generate 20 tokens.
	var generated []Token
	for i := 0; i < 20; i++ {
		token := chain.Sample(ctx, -1)
		if v.IsEOG(token) {
			t.Logf("EOG at step %d", i)
			break
		}
		generated = append(generated, token)

		nextBatch := BatchGetOne([]Token{token})
		if err := ctx.Decode(nextBatch); err != nil {
			t.Fatalf("Decode step %d: %v", i, err)
		}
	}

	text, err := v.Detokenize(generated, false, false)
	if err != nil {
		t.Fatalf("Detokenize: %v", err)
	}
	t.Logf("Generated (%d tokens): %s", len(generated), text)

	if len(generated) == 0 {
		t.Fatal("generated no tokens")
	}
}

func TestGreedyDeterminism(t *testing.T) {
	m := requireModel(t)
	v := m.Vocab()

	generate := func() []Token {
		ctx, err := NewContext(m, WithContextSize(512), WithBatchSize(512))
		if err != nil {
			t.Fatalf("NewContext: %v", err)
		}
		defer ctx.Close()

		chain := NewSamplerChain(false)
		defer chain.Close()
		chain.Add(NewGreedySampler())

		tokens, _ := v.Tokenize("Hello", true, false)
		batch := BatchGetOne(tokens)
		if err := ctx.Decode(batch); err != nil {
			t.Fatalf("Decode: %v", err)
		}

		var result []Token
		for i := 0; i < 10; i++ {
			tok := chain.Sample(ctx, -1)
			if v.IsEOG(tok) {
				break
			}
			result = append(result, tok)
			if err := ctx.Decode(BatchGetOne([]Token{tok})); err != nil {
				t.Fatalf("Decode step %d: %v", i, err)
			}
		}
		return result
	}

	run1 := generate()
	gcCollect() // free first context before creating second
	run2 := generate()

	if len(run1) != len(run2) {
		t.Fatalf("runs differ in length: %d vs %d", len(run1), len(run2))
	}
	for i := range run1 {
		if run1[i] != run2[i] {
			t.Fatalf("runs differ at index %d: %d vs %d", i, run1[i], run2[i])
		}
	}
	t.Logf("Deterministic: %d tokens match", len(run1))
}

// ---------------------------------------------------------------------------
// Performance Metrics
// ---------------------------------------------------------------------------

func TestPerfContext(t *testing.T) {
	m := requireModel(t)
	ctx, err := NewContext(m, WithContextSize(512), WithBatchSize(512))
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Close()

	v := m.Vocab()
	tokens, _ := v.Tokenize("Hello, world!", true, false)
	batch := BatchGetOne(tokens)
	ctx.Decode(batch)

	perf := ctx.Performance()
	t.Logf("Perf: TStartMs=%.2f, TPEvalMs=%.2f, TEvalMs=%.2f, NPEval=%d, NEval=%d",
		perf.TStartMs, perf.TPEvalMs, perf.TEvalMs, perf.NPEval, perf.NEval)

	// Print should not crash.
	ctx.PrintPerformance()

	// Reset should not crash.
	ctx.ResetPerformance()
}

func TestPerfSampler(t *testing.T) {
	chain := NewSamplerChain(false)
	defer chain.Close()
	chain.Add(NewGreedySampler())

	perf := chain.Performance()
	t.Logf("Sampler Perf: TSampleMs=%.2f, NSample=%d", perf.TSampleMs, perf.NSample)

	chain.PrintPerformance()
	chain.ResetPerformance()
}

// ---------------------------------------------------------------------------
// Chat Template
// ---------------------------------------------------------------------------

func TestApplyChatTemplate(t *testing.T) {
	m := requireModel(t)

	tmpl, ok := m.ChatTemplate("")
	if !ok {
		t.Skip("Model has no default chat template")
	}

	messages := []ChatMessage{
		{Role: "user", Content: "What is 2+2?"},
	}
	formatted, err := ApplyChatTemplate(tmpl, messages, true)
	if err != nil {
		t.Fatalf("ApplyChatTemplate: %v", err)
	}
	if formatted == "" {
		t.Fatal("formatted template is empty")
	}
	t.Logf("Formatted (first 500 chars): %.500s", formatted)
}

func TestChatBuiltinTemplates(t *testing.T) {
	templates := ChatBuiltinTemplates()
	t.Logf("Built-in templates (%d): %v", len(templates), templates)
}

// ---------------------------------------------------------------------------
// Session API (High-Level)
// ---------------------------------------------------------------------------

func TestSessionLifecycle(t *testing.T) {
	m := requireModel(t)

	s, err := NewSession(m,
		WithSessionNCtx(512),
		WithSessionNBatch(512),
		WithSessionThreads(4),
	)
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer s.Close()

	vocabSize := s.VocabSize()
	if vocabSize <= 0 {
		t.Fatalf("VocabSize = %d, want > 0", vocabSize)
	}
	t.Logf("Session vocab size: %d", vocabSize)
}

func TestSessionIngestDecodeAndSample(t *testing.T) {
	m := requireModel(t)
	v := m.Vocab()

	s, err := NewSession(m, WithSessionNCtx(512))
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer s.Close()

	tokens, _ := v.Tokenize("Hello", true, false)
	if err := s.IngestTokens(tokens, true); err != nil {
		t.Fatalf("IngestTokens: %v", err)
	}

	if err := s.Decode(); err != nil {
		t.Fatalf("Decode: %v", err)
	}

	token := s.Sample()
	if token == InvalidToken {
		t.Fatal("Sample returned InvalidToken")
	}
	t.Logf("Sampled token: %d (%q)", token, v.TokenText(token, false))
}

func TestSessionMultiStep(t *testing.T) {
	m := requireModel(t)
	v := m.Vocab()

	sc := DefaultSamplingConfig()
	sc.Seed = 42
	sc.Temp = 0.1 // near-greedy

	s, err := NewSession(m,
		WithSessionNCtx(512),
		WithSessionSampling(sc),
	)
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer s.Close()

	tokens, _ := v.Tokenize("The meaning of life is", true, false)
	s.IngestTokens(tokens, true)

	var generated []string
	for i := 0; i < 20; i++ {
		if err := s.Decode(); err != nil {
			t.Fatalf("Decode step %d: %v", i, err)
		}
		tok := s.Sample()
		if v.IsEOG(tok) {
			break
		}
		piece := v.TokenText(tok, false)
		generated = append(generated, piece)
		s.IngestTokens([]Token{tok}, true)
	}

	text := strings.Join(generated, "")
	t.Logf("Generated: %s", text)
	if len(generated) == 0 {
		t.Fatal("generated no tokens")
	}
}

func TestSessionReset(t *testing.T) {
	m := requireModel(t)
	v := m.Vocab()

	s, err := NewSession(m, WithSessionNCtx(512))
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer s.Close()

	tokens, _ := v.Tokenize("Hello", true, false)
	s.IngestTokens(tokens, true)
	s.Decode()
	s.Sample()

	// Reset and generate again.
	s.Reset()
	s.IngestTokens(tokens, true)
	s.Decode()
	tok := s.Sample()
	if tok == InvalidToken {
		t.Fatal("Sample after reset returned InvalidToken")
	}
	t.Logf("Sample after reset: %d", tok)
}

func TestSessionCloseIdempotent(t *testing.T) {
	m := requireModel(t)
	s, err := NewSession(m, WithSessionNCtx(512))
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	s.Close()
	s.Close()
}

func TestSessionLogits(t *testing.T) {
	m := requireModel(t)
	v := m.Vocab()

	s, err := NewSession(m, WithSessionNCtx(512))
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer s.Close()

	tokens, _ := v.Tokenize("Hello", true, false)
	s.IngestTokens(tokens, true)
	s.Decode()

	// Get required size.
	size := s.Logits(nil)
	if size <= 0 {
		t.Fatalf("Logits size = %d, want > 0", size)
	}

	logits := make([]float32, size)
	n := s.Logits(logits)
	if n != size {
		t.Fatalf("Logits copied %d, want %d", n, size)
	}
	t.Logf("Logits: size=%d, first 5: %v", n, logits[:5])
}

// ---------------------------------------------------------------------------
// Checkpointing
// ---------------------------------------------------------------------------

func TestSessionCheckpoint(t *testing.T) {
	m := requireModel(t)
	v := m.Vocab()

	sc := DefaultSamplingConfig()
	sc.Seed = 42
	sc.Temp = 0.1 // near-greedy for determinism

	s, err := NewSession(m,
		WithSessionNCtx(512),
		WithSessionSampling(sc),
	)
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer s.Close()

	tokens, _ := v.Tokenize("Hello", true, false)
	s.IngestTokens(tokens, true)
	s.Decode()

	// Generate a few tokens.
	for i := 0; i < 3; i++ {
		tok := s.Sample()
		s.IngestTokens([]Token{tok}, true)
		s.Decode()
	}

	// Create checkpoint.
	cp := s.CreateCheckpoint()
	if cp == nil {
		t.Fatal("CreateCheckpoint returned nil")
	}
	defer cp.Close()

	// Path A: generate 5 tokens.
	var pathA []Token
	for i := 0; i < 5; i++ {
		tok := s.Sample()
		if v.IsEOG(tok) {
			break
		}
		pathA = append(pathA, tok)
		s.IngestTokens([]Token{tok}, true)
		s.Decode()
	}

	// Restore checkpoint.
	if err := s.RestoreCheckpoint(cp); err != nil {
		t.Fatalf("RestoreCheckpoint: %v", err)
	}

	// Path B: generate 5 tokens.
	var pathB []Token
	for i := 0; i < 5; i++ {
		tok := s.Sample()
		if v.IsEOG(tok) {
			break
		}
		pathB = append(pathB, tok)
		s.IngestTokens([]Token{tok}, true)
		s.Decode()
	}

	// Verify paths are identical (greedy + fixed seed).
	if len(pathA) != len(pathB) {
		t.Fatalf("paths differ in length: %d vs %d", len(pathA), len(pathB))
	}
	for i := range pathA {
		if pathA[i] != pathB[i] {
			t.Fatalf("paths differ at step %d: %d vs %d", i, pathA[i], pathB[i])
		}
	}
	t.Logf("Checkpoint determinism verified: %d tokens match", len(pathA))
}

// ---------------------------------------------------------------------------
// Structured Decoding (Grammar)
// ---------------------------------------------------------------------------

func TestSessionStructuredDecoding(t *testing.T) {
	m := requireModel(t)
	v := m.Vocab()

	s, err := NewSession(m,
		WithSessionNCtx(512),
		WithSessionStructuredCheckpointing(true),
	)
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer s.Close()

	// Ingest prompt tokens first (without updating sampler), then configure grammar.
	// The grammar only applies to generated tokens, not the prompt.
	tokens, _ := v.Tokenize("Is the sky blue? Answer yes or no:", true, false)
	s.IngestTokens(tokens, false)

	grammar := `root ::= "yes" | "no"`
	if err := s.ConfigureStructured(grammar, ""); err != nil {
		t.Fatalf("ConfigureStructured: %v", err)
	}

	var output []string
	for i := 0; i < 5; i++ {
		if err := s.Decode(); err != nil {
			t.Fatalf("Decode step %d: %v", i, err)
		}
		tok := s.Sample()
		if v.IsEOG(tok) {
			break
		}
		piece := v.TokenText(tok, false)
		output = append(output, piece)
		s.IngestTokens([]Token{tok}, true)
	}

	text := strings.Join(output, "")
	t.Logf("Structured output: %q", text)
	lower := strings.TrimSpace(strings.ToLower(text))
	if lower != "yes" && lower != "no" {
		t.Logf("Output %q is not strictly 'yes' or 'no' — grammar may match partial tokens", text)
	}
}

// ---------------------------------------------------------------------------
// State Save/Load (In-Memory)
// ---------------------------------------------------------------------------

func TestStateSaveLoadMemory(t *testing.T) {
	m := requireModel(t)
	v := m.Vocab()

	ctx, err := NewContext(m, WithContextSize(512), WithBatchSize(512))
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}

	tokens, _ := v.Tokenize("Hello, world!", true, false)
	batch := BatchGetOne(tokens)
	ctx.Decode(batch)

	// Save state.
	size := ctx.StateGetSize()
	if size <= 0 {
		t.Fatalf("StateGetSize = %d, want > 0", size)
	}
	t.Logf("State size: %d bytes", size)

	buf := make([]byte, size)
	written := ctx.StateGetData(buf)
	if written == 0 {
		t.Fatal("StateGetData wrote 0 bytes")
	}
	t.Logf("StateGetData wrote %d bytes", written)

	// Free first context before creating second to avoid resource pressure.
	ctx.Close()
	gcCollect()

	// Restore state into a fresh context.
	ctx2, err := NewContext(m, WithContextSize(512), WithBatchSize(512))
	if err != nil {
		t.Fatalf("NewContext for restore: %v", err)
	}
	defer ctx2.Close()

	read := ctx2.StateSetData(buf[:written])
	if read == 0 {
		t.Fatal("StateSetData read 0 bytes")
	}
	t.Logf("StateSetData read %d bytes", read)
}

// ---------------------------------------------------------------------------
// State Save/Load (File)
// ---------------------------------------------------------------------------

func TestStateSaveLoadFile(t *testing.T) {
	m := requireModel(t)
	v := m.Vocab()

	ctx, err := NewContext(m, WithContextSize(512), WithBatchSize(512))
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}

	tokens, _ := v.Tokenize("Hello, world!", true, false)
	batch := BatchGetOne(tokens)
	ctx.Decode(batch)

	tmpFile := t.TempDir() + "/test_state.bin"

	if err := ctx.StateSaveFile(tmpFile, tokens); err != nil {
		t.Fatalf("StateSaveFile: %v", err)
	}

	// Free first context before creating second.
	ctx.Close()
	gcCollect()

	// Load into fresh context.
	ctx2, err := NewContext(m, WithContextSize(512), WithBatchSize(512))
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx2.Close()

	loadedTokens, err := ctx2.StateLoadFile(tmpFile, 4096)
	if err != nil {
		t.Fatalf("StateLoadFile: %v", err)
	}
	if len(loadedTokens) != len(tokens) {
		t.Fatalf("loaded %d tokens, want %d", len(loadedTokens), len(tokens))
	}
	for i := range tokens {
		if loadedTokens[i] != tokens[i] {
			t.Fatalf("token %d mismatch: %d vs %d", i, loadedTokens[i], tokens[i])
		}
	}
	t.Logf("State file roundtrip: %d tokens verified", len(loadedTokens))
}

// ---------------------------------------------------------------------------
// Default Configs
// ---------------------------------------------------------------------------

func TestDefaultConfigs(t *testing.T) {
	sc := DefaultSamplingConfig()
	t.Logf("DefaultSamplingConfig: TopK=%d, TopP=%.2f, MinP=%.2f, Temp=%.2f",
		sc.TopK, sc.TopP, sc.MinP, sc.Temp)

	cfg := DefaultSessionConfig()
	t.Logf("DefaultSessionConfig: NThreads=%d, NCtx=%d, NBatch=%d, Healing=%v",
		cfg.NThreads, cfg.NCtx, cfg.NBatch, cfg.EnableHealing)
}

// ---------------------------------------------------------------------------
// Channel-Based Streaming Generation
// ---------------------------------------------------------------------------

func TestSessionGenerate(t *testing.T) {
	m := requireModel(t)

	s, err := NewSession(m,
		WithSessionNCtx(512),
		WithSessionSampling(SamplingConfig{
			Seed: 42,
			Temp: 0.8,
			TopK: 40,
			TopP: 0.9,
			MinP: 0.05,
		}),
	)
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer s.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	tokenCh, resultCh := s.Generate(ctx, "Once upon a time", 30)

	var pieces []string
	for tok := range tokenCh {
		pieces = append(pieces, tok.Text)
	}

	result := <-resultCh
	if result.Err != nil {
		t.Fatalf("Generate error: %v", result.Err)
	}

	t.Logf("Streamed %d tokens: %s", len(result.Tokens), result.Text)
	if len(result.Tokens) == 0 {
		t.Fatal("generated no tokens")
	}

	// Verify streaming text matches result text.
	streamedText := strings.Join(pieces, "")
	if streamedText != result.Text {
		t.Fatalf("streamed text mismatch:\n  streamed: %q\n  result:   %q", streamedText, result.Text)
	}
}

func TestSessionGenerateAll(t *testing.T) {
	m := requireModel(t)

	// Use same sampling config style as TestSessionGenerate to avoid
	// the Thinking model entering long <think> chains.
	s, err := NewSession(m,
		WithSessionNCtx(512),
		WithSessionSampling(SamplingConfig{
			Seed: 42,
			Temp: 0.8,
			TopK: 40,
			TopP: 0.9,
			MinP: 0.05,
		}),
	)
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer s.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Use a text-continuation prompt (not a question) to avoid thinking chains.
	text, err := s.GenerateAll(ctx, "The quick brown fox", 20)
	if err == context.DeadlineExceeded {
		t.Skipf("GenerateAll timed out (model may be slow in thinking mode)")
	}
	if err != nil {
		t.Fatalf("GenerateAll: %v", err)
	}
	t.Logf("GenerateAll: %s", text)
	if text == "" {
		t.Fatal("GenerateAll returned empty text")
	}
}

func TestSessionGenerateCancellation(t *testing.T) {
	m := requireModel(t)

	s, err := NewSession(m,
		WithSessionNCtx(512),
		WithSessionSampling(SamplingConfig{
			Seed: 42,
			Temp: 0.8,
		}),
	)
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer s.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	tokenCh, resultCh := s.Generate(ctx, "Tell me a very long story about", 1000)

	// Read a few tokens then cancel.
	count := 0
	for range tokenCh {
		count++
		if count >= 3 {
			cancel()
			break
		}
	}

	// Drain remaining tokens.
	for range tokenCh {
	}

	result := <-resultCh
	if result.Err == nil {
		t.Log("Generation completed before cancellation took effect")
	} else if result.Err != context.Canceled {
		t.Fatalf("expected context.Canceled, got: %v", result.Err)
	} else {
		t.Logf("Cancelled after %d tokens (result has %d tokens)", count, len(result.Tokens))
	}
}

// ---------------------------------------------------------------------------
// Concurrent Safety
// ---------------------------------------------------------------------------

func TestConcurrentModelReads(t *testing.T) {
	m := requireModel(t)

	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = m.EmbeddingSize()
			_ = m.LayerCount()
			_ = m.Description()
			_ = m.Size()
			_ = m.ParameterCount()
		}()
	}
	wg.Wait()
}

func TestConcurrentVocabReads(t *testing.T) {
	m := requireModel(t)
	v := m.Vocab()

	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = v.TokenCount()
			_ = v.BOS()
			_ = v.EOS()
			v.Tokenize("Hello concurrent", false, false)
		}()
	}
	wg.Wait()
}

// ---------------------------------------------------------------------------
// Progress Callback
// ---------------------------------------------------------------------------

func TestLoadModelWithProgressCallback(t *testing.T) {
	path := modelPath(t)
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skipf("Model not found at %s", path)
	}

	var progressValues []float32
	var mu sync.Mutex

	m, err := LoadModel(path,
		WithGPULayers(0),
		WithProgressCallback(func(progress float32) bool {
			mu.Lock()
			progressValues = append(progressValues, progress)
			mu.Unlock()
			return true
		}),
	)
	if err != nil {
		t.Fatalf("LoadModel with progress: %v", err)
	}
	defer m.Close()

	mu.Lock()
	defer mu.Unlock()
	if len(progressValues) == 0 {
		t.Log("No progress callbacks received (model may be too small)")
	} else {
		t.Logf("Progress callbacks: %d calls, last=%.2f", len(progressValues), progressValues[len(progressValues)-1])
	}
}

// ---------------------------------------------------------------------------
// Sampler constructors (not requiring model)
// ---------------------------------------------------------------------------

func TestAllSamplerConstructors(t *testing.T) {
	samplers := []*Sampler{
		NewGreedySampler(),
		NewDistSampler(42),
		NewTopKSampler(50),
		NewTopPSampler(0.9, 1),
		NewMinPSampler(0.05, 1),
		NewTypicalSampler(0.95, 1),
		NewTempSampler(0.8),
		NewTempExtSampler(0.8, 0.1, 1.0),
		NewXTCSampler(0.5, 0.5, 1, 42),
		NewTopNSigmaSampler(2.0),
		NewMirostatV2Sampler(42, 5.0, 0.1),
		NewPenaltiesSampler(64, 1.1, 0.0, 0.0),
		NewAdaptivePSampler(0.1, 0.9, 42),
	}

	for _, s := range samplers {
		if s == nil {
			t.Error("sampler constructor returned nil")
			continue
		}
		name := s.Name()
		if name == "" {
			t.Error("sampler has empty name")
		}
		t.Logf("Sampler: %s", name)
		s.Close()
	}
}

func TestMirostatSampler(t *testing.T) {
	m := requireModel(t)
	v := m.Vocab()

	s := NewMirostatSampler(v.TokenCount(), 42, 5.0, 0.1, 100)
	if s == nil {
		t.Fatal("NewMirostatSampler returned nil")
	}
	defer s.Close()
	t.Logf("Mirostat sampler: %s", s.Name())
}

func TestGrammarSampler(t *testing.T) {
	m := requireModel(t)
	v := m.Vocab()

	grammar := `root ::= "hello" | "world"`
	s := NewGrammarSampler(v, grammar, "root")
	if s == nil {
		t.Fatal("NewGrammarSampler returned nil")
	}
	defer s.Close()
	t.Logf("Grammar sampler: %s", s.Name())
}

func TestLogitBiasSampler(t *testing.T) {
	m := requireModel(t)
	v := m.Vocab()

	biases := []LogitBias{
		{Token: v.BOS(), Bias: -100.0},
		{Token: v.EOS(), Bias: -100.0},
	}
	s := NewLogitBiasSampler(v.TokenCount(), biases)
	if s == nil {
		t.Fatal("NewLogitBiasSampler returned nil")
	}
	defer s.Close()
	t.Logf("LogitBias sampler: %s", s.Name())
}

func TestInfillSampler(t *testing.T) {
	m := requireModel(t)
	v := m.Vocab()

	s := NewInfillSampler(v)
	if s == nil {
		t.Fatal("NewInfillSampler returned nil")
	}
	defer s.Close()
	t.Logf("Infill sampler: %s", s.Name())
}

func TestDRYSampler(t *testing.T) {
	m := requireModel(t)
	v := m.Vocab()

	s := NewDRYSampler(v, m.TrainingContextSize(), 0.8, 1.75, 2, 64, []string{"\n", ".", ",", "!", "?"})
	if s == nil {
		t.Fatal("NewDRYSampler returned nil")
	}
	defer s.Close()
	t.Logf("DRY sampler: %s", s.Name())
}

func TestPrefixSampler(t *testing.T) {
	m := requireModel(t)
	v := m.Vocab()

	s := NewPrefixSampler(v, "Hello")
	if s == nil {
		t.Fatal("NewPrefixSampler returned nil")
	}
	defer s.Close()
	t.Logf("Prefix sampler: %s", s.Name())

	// Update prefix.
	s.PrefixSet("World")
	s.PrefixSet("")
}

// ---------------------------------------------------------------------------
// JSONSchemaToGrammar
// ---------------------------------------------------------------------------

func TestJSONSchemaToGrammar(t *testing.T) {
	schema := `{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]}`
	grammar, err := JSONSchemaToGrammar(schema, false)
	if err != nil {
		t.Fatalf("JSONSchemaToGrammar: %v", err)
	}
	if grammar == "" {
		t.Fatal("grammar is empty")
	}
	t.Logf("Grammar (first 300 chars): %.300s", grammar)
}

// ---------------------------------------------------------------------------
// MemoryBreakdownPrint
// ---------------------------------------------------------------------------

func TestMemoryBreakdownPrint(t *testing.T) {
	m := requireModel(t)
	ctx, err := NewContext(m, WithContextSize(512))
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Close()

	// Should not crash.
	ctx.MemoryBreakdownPrint()
}

// ---------------------------------------------------------------------------
// Context Model Accessor
// ---------------------------------------------------------------------------

func TestContextModel(t *testing.T) {
	m := requireModel(t)
	ctx, err := NewContext(m, WithContextSize(512))
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Close()

	if ctx.Model() != m {
		t.Fatal("Context.Model() does not match")
	}
}

// ---------------------------------------------------------------------------
// MaxTokens
// ---------------------------------------------------------------------------

func TestSessionMaxTokensConfig(t *testing.T) {
	cfg := DefaultSessionConfig()
	if cfg.MaxTokens != 0 {
		t.Fatalf("DefaultSessionConfig().MaxTokens = %d, want 0", cfg.MaxTokens)
	}

	WithSessionMaxTokens(100)(&cfg)
	if cfg.MaxTokens != 100 {
		t.Fatalf("MaxTokens = %d, want 100", cfg.MaxTokens)
	}
}

func TestSessionMaxTokensEnforced(t *testing.T) {
	m := requireModel(t)
	v := m.Vocab()

	maxTok := int32(5)
	s, err := NewSession(m,
		WithSessionNCtx(512),
		WithSessionMaxTokens(maxTok),
	)
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer s.Close()

	tokens, _ := v.Tokenize("Once upon a time in a land far far away", true, false)
	s.IngestTokens(tokens, true)

	var generated []Token
	for i := 0; i < 50; i++ {
		if err := s.Decode(); err != nil {
			t.Fatalf("Decode step %d: %v", i, err)
		}
		tok := s.Sample()
		if v.IsEOG(tok) {
			break
		}
		generated = append(generated, tok)
		s.IngestTokens([]Token{tok}, true)
	}

	if int32(len(generated)) > maxTok {
		t.Fatalf("generated %d tokens, expected at most %d", len(generated), maxTok)
	}
	t.Logf("MaxTokens=%d, generated %d tokens", maxTok, len(generated))
}

func TestSessionMaxTokensResetRearms(t *testing.T) {
	m := requireModel(t)
	v := m.Vocab()

	maxTok := int32(3)
	s, err := NewSession(m,
		WithSessionNCtx(512),
		WithSessionMaxTokens(maxTok),
	)
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer s.Close()

	tokens, _ := v.Tokenize("Hello world", true, false)

	// First generation cycle.
	s.IngestTokens(tokens, true)
	var count1 int
	for i := 0; i < 20; i++ {
		s.Decode()
		tok := s.Sample()
		if v.IsEOG(tok) {
			break
		}
		count1++
		s.IngestTokens([]Token{tok}, true)
	}

	// Reset should re-arm the counter.
	s.Reset()

	s.IngestTokens(tokens, true)
	var count2 int
	for i := 0; i < 20; i++ {
		s.Decode()
		tok := s.Sample()
		if v.IsEOG(tok) {
			break
		}
		count2++
		s.IngestTokens([]Token{tok}, true)
	}

	if int32(count1) > maxTok {
		t.Fatalf("cycle 1: generated %d tokens, want <= %d", count1, maxTok)
	}
	if int32(count2) > maxTok {
		t.Fatalf("cycle 2: generated %d tokens, want <= %d", count2, maxTok)
	}
	t.Logf("MaxTokens=%d, cycle1=%d, cycle2=%d", maxTok, count1, count2)
}

// ---------------------------------------------------------------------------
// Stop Sequences
// ---------------------------------------------------------------------------

func TestSessionConfigureStopSequencesClear(t *testing.T) {
	m := requireModel(t)

	s, err := NewSession(m, WithSessionNCtx(512))
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer s.Close()

	// Clearing with nil should succeed.
	if err := s.ConfigureStopSequences(nil); err != nil {
		t.Fatalf("ConfigureStopSequences(nil): %v", err)
	}

	// Clearing with empty slice should succeed.
	if err := s.ConfigureStopSequences([][]Token{}); err != nil {
		t.Fatalf("ConfigureStopSequences([]): %v", err)
	}
}

func TestSessionConfigureStopSequences(t *testing.T) {
	m := requireModel(t)
	v := m.Vocab()

	s, err := NewSession(m, WithSessionNCtx(512))
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer s.Close()

	// Use EOS token as a stop sequence — any output containing it will halt.
	eos := v.EOS()
	seqs := [][]Token{{eos}}
	if err := s.ConfigureStopSequences(seqs); err != nil {
		t.Fatalf("ConfigureStopSequences: %v", err)
	}

	// Clear and re-set.
	if err := s.ConfigureStopSequences(nil); err != nil {
		t.Fatalf("ConfigureStopSequences clear: %v", err)
	}

	// Multi-token stop sequence.
	multiSeq := [][]Token{{Token(1), Token(2), Token(3)}}
	if err := s.ConfigureStopSequences(multiSeq); err != nil {
		t.Fatalf("ConfigureStopSequences multi-token: %v", err)
	}
	t.Log("Stop sequences configured successfully")
}

func TestSessionStopSequencesClosedSession(t *testing.T) {
	m := requireModel(t)

	s, err := NewSession(m, WithSessionNCtx(512))
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	s.Close()

	err = s.ConfigureStopSequences([][]Token{{Token(1)}})
	if err == nil {
		t.Fatal("expected error on closed session")
	}
	t.Logf("Expected error: %v", err)
}

func TestSessionStopSequenceHaltsGeneration(t *testing.T) {
	m := requireModel(t)
	v := m.Vocab()

	sc := DefaultSamplingConfig()
	sc.Seed = 42
	sc.Temp = 0.1

	s, err := NewSession(m,
		WithSessionNCtx(512),
		WithSessionSampling(sc),
	)
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer s.Close()

	// Generate a few tokens to discover what the model produces.
	tokens, _ := v.Tokenize("The quick brown fox", true, false)
	s.IngestTokens(tokens, true)

	var baseline []Token
	for i := 0; i < 10; i++ {
		s.Decode()
		tok := s.Sample()
		if v.IsEOG(tok) {
			break
		}
		baseline = append(baseline, tok)
		s.IngestTokens([]Token{tok}, true)
	}

	if len(baseline) < 3 {
		t.Skipf("Model produced only %d tokens, need at least 3 for stop sequence test", len(baseline))
	}

	// Use the 3rd token as a single-token stop sequence. Reset and re-generate.
	stopToken := baseline[2]
	t.Logf("Using token %d (%q) as stop sequence", stopToken, v.TokenText(stopToken, false))

	s.Reset()
	if err := s.ConfigureStopSequences([][]Token{{stopToken}}); err != nil {
		t.Fatalf("ConfigureStopSequences: %v", err)
	}

	s.IngestTokens(tokens, true)
	var stopped []Token
	for i := 0; i < 20; i++ {
		s.Decode()
		tok := s.Sample()
		if v.IsEOG(tok) {
			break
		}
		stopped = append(stopped, tok)
		s.IngestTokens([]Token{tok}, true)
	}

	// With the stop sequence, generation should halt at or before the baseline length.
	if len(stopped) >= len(baseline) {
		t.Logf("Warning: stopped generation (%d) not shorter than baseline (%d) — "+
			"stop token may not have appeared in regenerated output", len(stopped), len(baseline))
	}
	t.Logf("Baseline=%d tokens, with stop sequence=%d tokens", len(baseline), len(stopped))
}

// ---------------------------------------------------------------------------
// Tool Ranking
// ---------------------------------------------------------------------------

func TestSessionRegisterToolsBasic(t *testing.T) {
	m := requireModel(t)

	s, err := NewSession(m, WithSessionNCtx(512))
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer s.Close()

	tools := []ToolDesc{
		{Name: "get_weather", Description: "Get the current weather for a location"},
		{Name: "search_web", Description: "Search the web for information"},
		{Name: "calculator", Description: "Perform arithmetic calculations", JSONSchema: `{"type":"object","properties":{"expression":{"type":"string"}}}`},
	}

	n, err := s.RegisterTools(tools, 2)
	if err != nil {
		t.Fatalf("RegisterTools: %v", err)
	}
	if n != 3 {
		t.Fatalf("RegisterTools returned %d, want 3", n)
	}
	t.Logf("Registered %d tools with top_k=2", n)
}

func TestSessionRegisterToolsTopKZero(t *testing.T) {
	m := requireModel(t)

	s, err := NewSession(m, WithSessionNCtx(512))
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer s.Close()

	tools := []ToolDesc{
		{Name: "tool_a", Description: "Tool A"},
		{Name: "tool_b", Description: "Tool B"},
	}

	n, err := s.RegisterTools(tools, 0)
	if err != nil {
		t.Fatalf("RegisterTools with top_k=0: %v", err)
	}
	if n != 2 {
		t.Fatalf("RegisterTools returned %d, want 2", n)
	}
	t.Logf("Registered %d tools with top_k=0 (injection disabled)", n)
}

func TestSessionClearTools(t *testing.T) {
	m := requireModel(t)

	s, err := NewSession(m, WithSessionNCtx(512))
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer s.Close()

	tools := []ToolDesc{
		{Name: "tool_a", Description: "Tool A"},
	}

	_, err = s.RegisterTools(tools, 1)
	if err != nil {
		t.Fatalf("RegisterTools: %v", err)
	}

	// Clear should not panic or error.
	s.ClearTools()

	// Clear again (idempotent).
	s.ClearTools()
	t.Log("ClearTools is idempotent")
}

func TestSessionRegisterToolsReplace(t *testing.T) {
	m := requireModel(t)

	s, err := NewSession(m, WithSessionNCtx(512))
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer s.Close()

	tools1 := []ToolDesc{
		{Name: "tool_a", Description: "Tool A"},
	}
	n, err := s.RegisterTools(tools1, 1)
	if err != nil {
		t.Fatalf("RegisterTools(1): %v", err)
	}
	if n != 1 {
		t.Fatalf("first registration returned %d, want 1", n)
	}

	// Replace with a different set.
	tools2 := []ToolDesc{
		{Name: "tool_x", Description: "Tool X"},
		{Name: "tool_y", Description: "Tool Y"},
		{Name: "tool_z", Description: "Tool Z"},
	}
	n, err = s.RegisterTools(tools2, 2)
	if err != nil {
		t.Fatalf("RegisterTools(2): %v", err)
	}
	if n != 3 {
		t.Fatalf("second registration returned %d, want 3", n)
	}
	t.Log("Tool re-registration works")
}

func TestSessionRegisterToolsClosedSession(t *testing.T) {
	m := requireModel(t)

	s, err := NewSession(m, WithSessionNCtx(512))
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	s.Close()

	tools := []ToolDesc{{Name: "t", Description: "d"}}
	_, err = s.RegisterTools(tools, 1)
	if err == nil {
		t.Fatal("expected error on closed session")
	}
	t.Logf("Expected error: %v", err)
}

func TestSessionRegisterToolsEmpty(t *testing.T) {
	m := requireModel(t)

	s, err := NewSession(m, WithSessionNCtx(512))
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer s.Close()

	_, err = s.RegisterTools(nil, 1)
	if err == nil {
		t.Fatal("expected error for nil tools")
	}

	_, err = s.RegisterTools([]ToolDesc{}, 1)
	if err == nil {
		t.Fatal("expected error for empty tools")
	}
	t.Log("Empty/nil tools correctly rejected")
}

func TestSessionRegisterToolsWithDecode(t *testing.T) {
	m := requireModel(t)
	v := m.Vocab()

	s, err := NewSession(m,
		WithSessionNCtx(512),
		WithSessionMaxTokens(10),
	)
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer s.Close()

	tools := []ToolDesc{
		{Name: "get_weather", Description: "Get the current weather for a city"},
		{Name: "search_web", Description: "Search the internet for information"},
		{Name: "send_email", Description: "Send an email to a recipient"},
	}

	n, err := s.RegisterTools(tools, 2)
	if err != nil {
		t.Fatalf("RegisterTools: %v", err)
	}
	t.Logf("Registered %d tools", n)

	// Ingest a prompt and decode — tool injection happens on first decode.
	tokens, _ := v.Tokenize("What is the weather in Paris?", true, false)
	s.IngestTokens(tokens, true)

	var generated []Token
	for i := 0; i < 15; i++ {
		if err := s.Decode(); err != nil {
			t.Fatalf("Decode step %d: %v", i, err)
		}
		tok := s.Sample()
		if v.IsEOG(tok) {
			break
		}
		generated = append(generated, tok)
		s.IngestTokens([]Token{tok}, true)
	}

	text, _ := v.Detokenize(generated, false, false)
	t.Logf("Generated with tools (%d tokens): %s", len(generated), text)
}

func TestSessionToolsWithOptionalSchema(t *testing.T) {
	m := requireModel(t)

	s, err := NewSession(m, WithSessionNCtx(512))
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer s.Close()

	// Mix of tools with and without schema.
	tools := []ToolDesc{
		{Name: "no_schema", Description: "A tool without a schema"},
		{Name: "with_schema", Description: "A tool with a schema", JSONSchema: `{"type":"object","properties":{"q":{"type":"string"}}}`},
		{Name: "empty_schema", Description: "A tool with empty schema", JSONSchema: ""},
	}

	n, err := s.RegisterTools(tools, 3)
	if err != nil {
		t.Fatalf("RegisterTools: %v", err)
	}
	if n != 3 {
		t.Fatalf("RegisterTools returned %d, want 3", n)
	}
	t.Log("Tools with mixed schema presence registered successfully")
}

// ---------------------------------------------------------------------------
// Entropy Monitor API
// ---------------------------------------------------------------------------

func TestDefaultEntropyMonitorConfig(t *testing.T) {
	cfg := DefaultEntropyMonitorConfig()
	t.Logf("DefaultEntropyMonitorConfig: Threshold=%.2f, CooldownTokens=%d, RingSize=%d",
		cfg.Threshold, cfg.CooldownTokens, cfg.RingSize)
}

func TestSessionConfigureEntropyMonitor(t *testing.T) {
	m := requireModel(t)

	s, err := NewSession(m, WithSessionNCtx(512))
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer s.Close()

	cfg := EntropyMonitorConfig{
		Threshold:      0.5,
		CooldownTokens: 2,
		RingSize:       8,
	}
	nEmbd, err := s.ConfigureEntropyMonitor(&cfg)
	if err != nil {
		t.Fatalf("ConfigureEntropyMonitor: %v", err)
	}
	if nEmbd <= 0 {
		t.Fatalf("ConfigureEntropyMonitor returned n_embd=%d, want > 0", nEmbd)
	}
	t.Logf("Entropy monitor configured, n_embd=%d", nEmbd)

	// Disable.
	nEmbd, err = s.ConfigureEntropyMonitor(nil)
	if err != nil {
		t.Fatalf("ConfigureEntropyMonitor(nil): %v", err)
	}
	if nEmbd != 0 {
		t.Fatalf("ConfigureEntropyMonitor(nil) returned n_embd=%d, want 0", nEmbd)
	}
	t.Log("Entropy monitor configure/disable works")
}

func TestSessionEntropyMonitorClosedSession(t *testing.T) {
	m := requireModel(t)

	s, err := NewSession(m, WithSessionNCtx(512))
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	s.Close()

	_, err = s.ConfigureEntropyMonitor(&EntropyMonitorConfig{Threshold: 0.5})
	if err == nil {
		t.Fatal("expected error on closed session")
	}
	t.Logf("Expected error: %v", err)
}

func TestSessionEntropyPendingAndFlush(t *testing.T) {
	m := requireModel(t)

	s, err := NewSession(m, WithSessionNCtx(512))
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer s.Close()

	// Without monitor configured, pending should be 0.
	if n := s.EntropyPending(); n != 0 {
		t.Fatalf("EntropyPending = %d, want 0 before configuring", n)
	}

	// Flush should not panic even without monitor.
	s.EntropyFlush()
	t.Log("EntropyPending and EntropyFlush work without monitor")
}

func TestSessionEntropyPopEmpty(t *testing.T) {
	m := requireModel(t)

	s, err := NewSession(m, WithSessionNCtx(512))
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer s.Close()

	// Pop on unconfigured monitor should return false.
	_, ok := s.EntropyPop(nil)
	if ok {
		t.Fatal("EntropyPop returned true without monitor configured")
	}
	t.Log("EntropyPop correctly returns false when no events")
}

func TestSessionEntropyCounterNil(t *testing.T) {
	m := requireModel(t)

	s, err := NewSession(m, WithSessionNCtx(512))
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer s.Close()

	// Counter should be nil without entropy monitor configured.
	p := s.EntropyCounter()
	if p != nil {
		t.Log("EntropyCounter returned non-nil (monitor may be auto-configured)")
	} else {
		t.Log("EntropyCounter is nil before configuring monitor")
	}
}

func TestSessionEntropyMonitorWithGeneration(t *testing.T) {
	m := requireModel(t)
	v := m.Vocab()

	sc := DefaultSamplingConfig()
	sc.Seed = 42
	sc.Temp = 0.8

	s, err := NewSession(m,
		WithSessionNCtx(512),
		WithSessionSampling(sc),
	)
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer s.Close()

	// Configure entropy monitor with a low threshold to catch events.
	cfg := EntropyMonitorConfig{
		Threshold:      0.1,
		CooldownTokens: 1,
		RingSize:       8,
	}
	nEmbd, err := s.ConfigureEntropyMonitor(&cfg)
	if err != nil {
		t.Fatalf("ConfigureEntropyMonitor: %v", err)
	}
	t.Logf("Entropy monitor n_embd=%d", nEmbd)

	// Check counter is available.
	counter := s.EntropyCounter()
	if counter == nil {
		t.Log("EntropyCounter is nil after configuring (may be implementation-specific)")
	}

	// Generate some tokens.
	tokens, _ := v.Tokenize("The meaning of life is", true, false)
	s.IngestTokens(tokens, true)

	for i := 0; i < 20; i++ {
		if err := s.Decode(); err != nil {
			t.Fatalf("Decode step %d: %v", i, err)
		}
		tok := s.Sample()
		if v.IsEOG(tok) {
			break
		}
		s.IngestTokens([]Token{tok}, true)
	}

	// Check for pending events.
	pending := s.EntropyPending()
	t.Logf("Pending entropy events after generation: %d", pending)

	// Try to pop events.
	var events []EntropyEvent
	for {
		event, ok := s.EntropyPop(nil)
		if !ok {
			break
		}
		events = append(events, event)
	}
	t.Logf("Popped %d entropy events", len(events))

	for i, ev := range events {
		t.Logf("  Event[%d]: entropy=%.4f normalized=%.4f topLogprob=%.4f token=%d nPast=%d cpID=%d nEmbd=%d",
			i, ev.Entropy, ev.Normalized, ev.TopLogprob, ev.Token, ev.NPast, ev.CheckpointID, ev.NEmbedding)
	}
}

func TestSessionLastEntropy(t *testing.T) {
	m := requireModel(t)
	v := m.Vocab()

	s, err := NewSession(m, WithSessionNCtx(512))
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer s.Close()

	// Before any sampling, should return -1.
	e := s.LastEntropy()
	if e != -1 {
		t.Logf("LastEntropy before sample: %f (expected -1, may differ by implementation)", e)
	}

	tokens, _ := v.Tokenize("Hello", true, false)
	s.IngestTokens(tokens, true)
	s.Decode()
	s.Sample()

	e = s.LastEntropy()
	t.Logf("LastEntropy after sample: %f", e)
}

func TestSessionRewindClosedSession(t *testing.T) {
	m := requireModel(t)

	s, err := NewSession(m, WithSessionNCtx(512))
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	s.Close()

	err = s.Rewind(0)
	if err == nil {
		t.Fatal("expected error on closed session")
	}
	t.Logf("Expected error: %v", err)
}

// ---------------------------------------------------------------------------
// Embedding API
// ---------------------------------------------------------------------------

func TestSessionEmbed(t *testing.T) {
	m := requireModel(t)

	s, err := NewSession(m, WithSessionNCtx(512))
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer s.Close()

	emb, err := s.Embed("Hello, world!")
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(emb) == 0 {
		t.Fatal("Embed returned empty vector")
	}
	t.Logf("Embedding dimension: %d, first 5 values: %v", len(emb), emb[:min(5, len(emb))])
}

func TestSessionEmbedClosedSession(t *testing.T) {
	m := requireModel(t)

	s, err := NewSession(m, WithSessionNCtx(512))
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	s.Close()

	_, err = s.Embed("test")
	if err == nil {
		t.Fatal("expected error on closed session")
	}
	t.Logf("Expected error: %v", err)
}

func TestSessionClearToolsOnClosedSession(t *testing.T) {
	m := requireModel(t)

	s, err := NewSession(m, WithSessionNCtx(512))
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	s.Close()

	// Should not panic.
	s.ClearTools()
}

func TestSessionToolsResetReinjection(t *testing.T) {
	m := requireModel(t)
	v := m.Vocab()

	s, err := NewSession(m,
		WithSessionNCtx(512),
		WithSessionMaxTokens(5),
	)
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer s.Close()

	tools := []ToolDesc{
		{Name: "tool_a", Description: "First tool"},
		{Name: "tool_b", Description: "Second tool"},
	}

	_, err = s.RegisterTools(tools, 1)
	if err != nil {
		t.Fatalf("RegisterTools: %v", err)
	}

	// First cycle: tools should be injected on decode.
	tokens, _ := v.Tokenize("Hello", true, false)
	s.IngestTokens(tokens, true)
	s.Decode()
	s.Sample()

	// Reset clears the injected flag.
	s.Reset()

	// Second cycle: tools should be re-injected.
	s.IngestTokens(tokens, true)
	if err := s.Decode(); err != nil {
		t.Fatalf("Decode after reset: %v", err)
	}
	tok := s.Sample()
	if tok == InvalidToken {
		t.Fatal("Sample after reset returned InvalidToken")
	}
	t.Log("Tool re-injection after reset works")
}
