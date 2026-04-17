# lfg.go

Go bindings for [LFG (Liquid Foundation Generation)](deps/lfg.cpp), the native inference runtime for Liquid AI's Liquid Foundation Models (LFMs).

## Overview

`lfg.go` exposes the `lfg.cpp` session API to Go via [purego](https://github.com/ebitengine/purego), so applications can load the native runtime dynamically without CGO. Standard `CGO_ENABLED=0` builds work out of the box, and the vendored runtime in [`deps/lfg.cpp`](deps/lfg.cpp) tracks the upstream open-source release.

The package is intended for Liquid Foundation Models. Although models are loaded from GGUF files, compatibility should be treated as LFM-specific rather than as support for arbitrary GGUF models.

## Requirements

- **Go**: 1.22+
- **Native runtime**: A prebuilt `liblfg` shared library for your platform

### Supported Platforms

| OS | Arch | Library |
|---|---|---|
| macOS | arm64 | `liblfg-macos-aarch64.dylib` |
| Linux | amd64 | `liblfg-linux-x86_64.so` |
| Linux | arm64 | `liblfg-linux-aarch64.so` |

### Library Discovery

At runtime, `lfg.go` resolves the shared library in this order:

1. `LFG_LIB_PATH` environment variable (explicit path to the shared library)
2. `deps/lfg.cpp/dist/lib/` relative to the source directory
3. Legacy `deps/lfg.cpp/dist/` relative to the source directory
4. System search paths (`dlopen` defaults)

## Building

No C toolchain is needed. Standard `go build` works:

```bash
go build ./...
```

Or use the Makefile:

```bash
make build   # go build
make test    # go test
make vet     # go vet (with -unsafeptr=false for purego)
```

## Quick Start

### Chat Generation

The highest-level API formats messages with the model chat template, tokenizes them, and runs generation in a single call:

```go
package main

import (
	"fmt"
	"log"

	"github.com/agentflare-ai/lfg.go"
)

func main() {
	model, err := lfg.LoadModelSimple("models/model.gguf", lfg.WithGPULayers(0))
	if err != nil {
		log.Fatal(err)
	}
	defer model.Close()

	session, err := lfg.NewSession(model,
		lfg.WithSessionNCtx(2048),
		lfg.WithSessionSampling(lfg.SamplingConfig{
			Seed: 42,
			Temp: 0.7,
			TopK: 40,
			TopP: 0.9,
			MinP: 0.05,
		}),
	)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()

	result, err := session.ChatGenerate(
		[]lfg.ChatMessage{
			{Role: "system", Content: "You are a helpful assistant."},
			{Role: "user", Content: "What is the capital of France?"},
		},
		lfg.GenerateConfig{
			MaxTokens: 256,
			TokenCallback: func(token lfg.Token, piece string) lfg.GenerateAction {
				fmt.Print(piece)
				return lfg.GenerateContinue
			},
		},
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("\n[%d tokens, stop: %d]\n", result.TokenCount, result.StopReason)
}
```

### Prompt Completion

For instruction/completion-style generation:

```go
result, err := session.PromptGenerate("The capital of France is", true, lfg.GenerateConfig{
	MaxTokens: 64,
	TokenCallback: func(token lfg.Token, piece string) lfg.GenerateAction {
		fmt.Print(piece)
		return lfg.GenerateContinue
	},
})
```

### Stopping Generation Early

Return `GenerateStop` from the token callback to halt immediately:

```go
var tokens int
result, _ := session.PromptGenerate("Count to 100:", true, lfg.GenerateConfig{
	MaxTokens: 1000,
	TokenCallback: func(token lfg.Token, piece string) lfg.GenerateAction {
		tokens++
		if tokens >= 10 {
			return lfg.GenerateStop
		}
		fmt.Print(piece)
		return lfg.GenerateContinue
	},
})
// result.StopReason == lfg.StopReasonCallback
```

### Low-Level: Generate from Pre-Ingested State

For manual control over tokenization and ingestion:

```go
vocab := model.Vocab()
tokens, _ := vocab.Tokenize("Once upon a time", true, false)
session.IngestTokens(tokens, true)

result, err := session.GenerateFromState(lfg.GenerateConfig{
	MaxTokens: 100,
	TokenCallback: func(token lfg.Token, piece string) lfg.GenerateAction {
		fmt.Print(piece)
		return lfg.GenerateContinue
	},
})
```

### Channel-Based Streaming

For `context.Context` integration and concurrent token consumption:

```go
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()

tokenCh, resultCh := session.Generate(ctx, "Once upon a time", 256)
for tok := range tokenCh {
	fmt.Print(tok.Text)
}
result := <-resultCh
if result.Err != nil {
	log.Fatal(result.Err)
}
```

## API Overview

### Model Loading

| Function | Description |
|---|---|
| `LoadModelSimple(path, opts...)` | Load with simplified config (recommended) |
| `LoadModel(path, opts...)` | Load with full parameter control |

### Session

| Method | Description |
|---|---|
| `NewSession(model, opts...)` | Create session with sampling/context config |
| `ChatGenerate(messages, config)` | Chat template + tokenize + generate |
| `PromptGenerate(prompt, addBOS, config)` | Tokenize + generate |
| `GenerateFromState(config)` | Generate from pre-ingested state |
| `Generate(ctx, prompt, maxTokens)` | Channel-based streaming with context support |
| `GenerateAll(ctx, prompt, maxTokens)` | Blocking convenience wrapper for `Generate` |

### Generate Config

```go
lfg.GenerateConfig{
	MaxTokens:               256,
	IncludeHistoryReasoning: false,          // include <think> blocks in chat history
	TokenCallback:           func(...) ...,  // per token (optional)
	EntropyCallback:         func(...) ...,  // live entropy rewind/injection hook (optional)
	ToolCallCallback:        func(...) ...,  // auto-executed tool observation (optional)
	MaxToolRounds:           5,              // max auto-execution rounds (0 = default 5)
}
```

### Stop Reasons

| Constant | Meaning |
|---|---|
| `StopReasonEOS` | End-of-generation token reached |
| `StopReasonMaxTokens` | Hit the max_tokens limit |
| `StopReasonCallback` | Token callback returned `GenerateStop` |
| `StopReasonToolCall` | Model emitted a tool call end token |

### Session Options

```go
lfg.WithSessionNCtx(2048)                    // context size
lfg.WithSessionNBatch(512)                   // batch size
lfg.WithSessionThreads(4)                    // thread count
lfg.WithSessionMaxTokens(256)                // per-cycle token limit
lfg.WithSessionHealing(true)                 // token healing
lfg.WithSessionReasoningBudget(1024)         // reasoning token budget
lfg.WithSessionStructuredCheckpointing(true) // grammar checkpoint support
lfg.WithSessionToolScoreMode(mode)           // tool injection gating
lfg.WithSessionToolMinScore(0.5)             // tool score threshold
lfg.WithSessionSampling(lfg.SamplingConfig{...})
```

### Structured Output

```go
// JSON schema constraint
session.ConfigureStructured(`{"type":"object","properties":{"name":{"type":"string"}}}`, "")

// GBNF grammar
session.ConfigureStructured(`root ::= "yes" | "no"`, "root")

// Convert JSON schema to GBNF
grammar, _ := lfg.JSONSchemaToGrammar(jsonSchema, true)
```

### Stop Sequences and Stop Strings

```go
// Token-level stop sequences
vocab := model.Vocab()
stopTokens, _ := vocab.Tokenize("\n\n", false, false)
session.ConfigureStopSequences([][]lfg.Token{stopTokens})

// Text-level stop strings (encoding-independent)
session.ConfigureStopStrings([]string{"\n\n", "END"})
```

### Tool Use

Register tools for embedding-based ranking and optional auto-execution:

```go
// Register tools with auto-execution functions
session.RegisterTools([]lfg.ToolDesc{
	{
		Name:        "get_weather",
		Description: "Get current weather for a location",
		Parameters:  `{"type":"object","properties":{"location":{"type":"string"}}}`,
		Fn: func(args string) (string, error) {
			// auto-executed by the engine during generation
			return `{"temp": 72, "condition": "sunny"}`, nil
		},
	},
}, 3) // topK=3: inject top 3 tools into context

// Generate with tool call observation
result, _ := session.ChatGenerate(messages, lfg.GenerateConfig{
	MaxTokens: 512,
	ToolCallCallback: func(call lfg.ToolCall, result string, round int) {
		fmt.Printf("Tool: %s(%s) -> %s\n", call.Name, call.Arguments, result)
	},
	MaxToolRounds: 3,
})

// Inspect tool calls after generation
calls := session.ToolCalls()

// Rank tools against a query
ranking, _ := session.RankTools("what's the weather like?")
```

### Embeddings

```go
embedding, err := session.Embed("some text to embed")
// embedding is []float32, L2-normalized
```

## Architecture

The library is organized around these core types:

- **`Model`** — Loaded model weights (thread-safe with `sync.RWMutex`)
- **`Session`** — High-level inference session with built-in sampling, KV cache, and generate loop
- **`Vocab`** — Tokenizer access (borrows parent Model's lock)
- **`Context`** / **`Batch`** / **`Sampler`** — Low-level building blocks

### purego (No CGO)

The bindings use [purego](https://github.com/ebitengine/purego) to call into the shared library at runtime via `dlopen`. This eliminates CGO entirely:

- No C compiler needed at build time
- `CGO_ENABLED=0` works out of the box
- Cross-compilation is trivial (just `GOOS`/`GOARCH`)
- No CGO pointer rules to worry about

### Generate Loop

The C-side generate loop (`ChatGenerate`, `PromptGenerate`, `GenerateFromState`) is the recommended way to generate text. It runs the entire decode+sample+ingest loop in C with a single FFI crossing. The channel-based `Generate` makes multiple crossings per token and is provided for `context.Context` integration.

### Entropy Monitor

Detect high-entropy regions during generation:

```go
cfg := lfg.EntropyMonitorConfig{Threshold: 0.5, CooldownTokens: 2, RingSize: 8}
nEmbd, _ := session.ConfigureEntropyMonitor(&cfg)

_, _ = session.PromptGenerate("prompt", true, lfg.GenerateConfig{MaxTokens: 256})
for {
	emb := make([]float32, nEmbd)
	event, ok := session.EntropyPop(emb)
	if !ok {
		break
	}
	fmt.Printf("entropy=%.3f norm=%.3f token=%d\n", event.Entropy, event.Normalized, event.Token)
}
```

Or intercept high-entropy events inline during generation and inject context immediately:

```go
_, _ = session.PromptGenerate("prompt", true, lfg.GenerateConfig{
	MaxTokens: 256,
	EntropyCallback: func(event lfg.EntropyEvent, embedding []float32) string {
		if len(embedding) == 0 {
			return ""
		}
		// Return text to rewind to event.CheckpointID and inject it before continuing.
		return "Relevant retrieved context goes here."
	},
})
```

### Confidence Monitor

Detect sustained low-entropy spans (where the model is confident) via queue pop:

```go
cfg := lfg.ConfidenceMonitorConfig{Threshold: 0.3, MinSpan: 5, RingSize: 4}
nEmbd, _ := session.ConfigureConfidenceMonitor(&cfg)

_, _ = session.PromptGenerate("prompt", true, lfg.GenerateConfig{MaxTokens: 256})
for {
	emb := make([]float32, nEmbd)
	event, ok := session.ConfidencePop(emb)
	if !ok {
		break
	}
	fmt.Printf("Confident span: %d tokens, text: %q\n", event.SpanLength, event.SpanText)
}
```

### Surprise Monitor

Measure how surprising the input prompt is to the model:

```go
cfg := lfg.SurpriseMonitorConfig{Threshold: 0.5, RingSize: 8}
nEmbd, _ := session.ConfigureSurpriseMonitor(&cfg)

_, _ = session.PromptGenerate("prompt", true, lfg.GenerateConfig{MaxTokens: 256})
for {
	emb := make([]float32, nEmbd)
	event, ok := session.SurprisePop(emb)
	if !ok {
		break
	}
	fmt.Printf("Surprise: mean=%.3f, %d/%d tokens above threshold\n",
		event.MeanSurprise, event.NAboveThreshold, event.NTokensEvaluated)
}
```

## License

Apache-2.0. See [LICENSE](LICENSE) for details.
