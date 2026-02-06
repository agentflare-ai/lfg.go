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

## Usage

```go
package main

import (
	"fmt"
	"github.com/agentflare-ai/lfg.go"
)

func main() {
	lfg.Init()
	fmt.Println("LFG API Version:", lfg.ApiVersion())
	fmt.Println("System Info:", lfg.SystemInfo())
}
```
