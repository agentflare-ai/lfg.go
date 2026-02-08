# lfg.go

Go bindings for [LFG (Liquid Foundation Generation)](deps/lfg.cpp), a high-performance, CPU-only inference engine for Liquid AI LFM2.5 models.

## Overview

`lfg.go` provides a Go interface to the `lfg.cpp` library, allowing you to run LFM2.5 (Text, Vision, Audio) models directly from Go applications.

**Note:** This project relies on `lfg.cpp`, which is included as a submodule in `deps/`.

## Prerequisites

*   **Go**: 1.22+
*   **Zig**: 0.15+ (Used as the C/C++ compiler for cross-compilation)

## Building

This project uses `zig cc` and `zig c++` to handle CGO compilation, enabling easy cross-compilation.

### Using Make

The `Makefile` provides targets for common platforms:

```bash
# Build for macOS (ARM64) - Default
make build

# Build for Linux (AMD64)
make build-linux-amd64

# Build for Linux (ARM64)
make build-linux-arm64

# Build for Windows (AMD64)
make build-windows-amd64
```

### Manual Build

You can also build manually using `go build` with the appropriate `CC` and `CXX` flags:

```bash
CGO_ENABLED=1 CC="zig cc" CXX="zig c++" go build -v .
```

## Quick Start

### Chat Generation (Recommended)

The highest-level API. Formats messages with the model's chat template, tokenizes, and generates in a **single CGo call**:

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

For instruction/completion-style generation. Tokenizes and generates in a single CGo call:

```go
var output string
result, err := session.PromptGenerate("The capital of France is", true, lfg.GenerateConfig{
	MaxTokens: 64,
	TokenCallback: func(token lfg.Token, piece string) lfg.GenerateAction {
		output += piece
		return lfg.GenerateContinue
	},
})
```

### Stopping Generation Early

Return `GenerateStop` from the token callback to halt generation immediately:

```go
var tokens int
result, _ := session.PromptGenerate("Count to 100:", true, lfg.GenerateConfig{
	MaxTokens: 1000,
	TokenCallback: func(token lfg.Token, piece string) lfg.GenerateAction {
		tokens++
		if tokens >= 10 {
			return lfg.GenerateStop // stops immediately
		}
		fmt.Print(piece)
		return lfg.GenerateContinue
	},
})
// result.StopReason == lfg.StopReasonCallback
```

### Low-Level: Generate from Pre-Ingested State

If you need manual control over tokenization and ingestion, use `GenerateFromState`:

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

## API Overview

### Model Loading

| Function | Description |
|---|---|
| `LoadModelSimple(path, opts...)` | Load a model with simplified config (recommended) |
| `LoadModel(path, opts...)` | Load a model with full parameter control |

### Session

| Method | Description |
|---|---|
| `NewSession(model, opts...)` | Create a new session with sampling/context config |
| `ChatGenerate(messages, config)` | Chat template + tokenize + generate (1 CGo call) |
| `PromptGenerate(prompt, addBOS, config)` | Tokenize + generate (1 CGo call) |
| `GenerateFromState(config)` | Generate from pre-ingested state (1 CGo call) |
| `Generate(ctx, prompt, maxTokens)` | Channel-based streaming with Go context support |
| `GenerateAll(ctx, prompt, maxTokens)` | Blocking convenience wrapper for `Generate` |

### Generate Config

```go
lfg.GenerateConfig{
	MaxTokens:          256,           // 0 = use session config
	TokenCallback:      func(...) ..., // called per token (optional)
	EntropyCallback:    func(...) ..., // called on high entropy (optional)
	ConfidenceCallback: func(...) ..., // called on confident span end (optional)
}
```

### Stop Reasons

| Constant | Meaning |
|---|---|
| `StopReasonEOS` | End-of-generation token reached |
| `StopReasonMaxTokens` | Hit the max_tokens limit |
| `StopReasonCallback` | Token callback returned `GenerateStop` |

### Session Options

```go
lfg.WithSessionNCtx(2048)                    // context size
lfg.WithSessionNBatch(512)                   // batch size
lfg.WithSessionThreads(4)                    // thread count
lfg.WithSessionMaxTokens(256)                // per-cycle token limit
lfg.WithSessionHealing(true)                 // token healing
lfg.WithSessionReasoningBudget(1024)         // reasoning token budget
lfg.WithSessionStructuredCheckpointing(true) // for grammar support
lfg.WithSessionSampling(lfg.SamplingConfig{...})
```

### Structured Output

```go
// JSON schema constraint
session.ConfigureStructured(`{"type":"object","properties":{"name":{"type":"string"}}}`, "")

// GBNF grammar
session.ConfigureStructured(`root ::= "yes" | "no"`, "root")
```

### Stop Sequences

```go
vocab := model.Vocab()
stopTokens, _ := vocab.Tokenize("\n\n", false, false)
session.ConfigureStopSequences([][]lfg.Token{stopTokens})
```

## Architecture

The library is organized around these core types:

- **`Model`** - Loaded model weights (thread-safe with `sync.RWMutex`)
- **`Session`** - High-level inference session with built-in sampling, KV cache, and generate loop
- **`Vocab`** - Tokenizer access (borrows parent Model's lock)
- **`Context`** / **`Batch`** / **`Sampler`** - Low-level building blocks

The C-side generate loop (`ChatGenerate`, `PromptGenerate`, `GenerateFromState`) is the recommended way to generate text. It runs the entire decode+sample+ingest loop in C with a single CGo crossing, compared to the channel-based `Generate` which makes 3 CGo calls per token.

### Entropy Monitor

Detect high-entropy tokens during generation and optionally inject context:

```go
// Configure the entropy monitor.
cfg := lfg.EntropyMonitorConfig{Threshold: 0.5, CooldownTokens: 2, RingSize: 8}
nEmbd, _ := session.ConfigureEntropyMonitor(&cfg)

// Use EntropyCallback in GenerateConfig for automatic rewind+inject.
result, _ := session.PromptGenerate("prompt", true, lfg.GenerateConfig{
    MaxTokens: 256,
    EntropyCallback: func(event lfg.EntropyEvent, embedding []float32) string {
        return "injected context" // or "" to skip
    },
})
// result.Retrievals = number of rewind+inject cycles

// Or poll manually: EntropyPop, EntropyPending, EntropyFlush, EntropyCounter
```

### Confidence Monitor

Detect sustained low-entropy spans (where the model is confident). The inverse of the entropy monitor — fires events when confident spans **end**:

```go
// Configure the confidence monitor.
cfg := lfg.ConfidenceMonitorConfig{Threshold: 0.3, MinSpan: 5, RingSize: 4}
nEmbd, _ := session.ConfigureConfidenceMonitor(&cfg)

// Use ConfidenceCallback in GenerateConfig for real-time notifications.
result, _ := session.PromptGenerate("prompt", true, lfg.GenerateConfig{
    MaxTokens: 256,
    ConfidenceCallback: func(event lfg.ConfidenceEvent, embedding []float32) {
        fmt.Printf("Confident span: %d tokens, mean entropy %.3f\n",
            event.SpanLength, event.MeanEntropy)
    },
})
// result.ConfidenceSpans = number of confidence events fired

// Or poll manually: ConfidencePop, ConfidencePending, ConfidenceFlush, ConfidenceCounter
```
