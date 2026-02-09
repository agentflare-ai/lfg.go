//go:build (darwin && arm64) || (linux && amd64) || (linux && arm64)

package lfg

import (
	"runtime"
	"unsafe"
)

// AdapterLoRA wraps a LoRA adapter loaded from a file.
// The adapter is valid as long as its parent model is not freed.
// All adapters must be loaded before context creation.
type AdapterLoRA struct {
	c     uintptr
	model *Model // prevent GC of parent
}

// LoadAdapterLoRA loads a LoRA adapter from the given file path.
func LoadAdapterLoRA(model *Model, path string) (*AdapterLoRA, error) {
	model.mu.RLock()
	defer model.mu.RUnlock()
	if model.c == 0 {
		return nil, &Error{Code: ErrorInvalidArgument, Message: "model is closed"}
	}

	registerAdapterFuncs()
	pathBytes := cString(path)
	adapter := _lfg_adapter_lora_init(model.c, cStringPtr(pathBytes))
	runtime.KeepAlive(pathBytes)

	if adapter == 0 {
		if err := getLastError(); err != nil {
			return nil, err
		}
		return nil, &Error{Code: ErrorInternal, Message: "failed to load LoRA adapter"}
	}

	return &AdapterLoRA{c: adapter, model: model}, nil
}

// Metadata returns a metadata value by key name.
func (a *AdapterLoRA) Metadata(key string) (string, bool) {
	a.model.mu.RLock()
	defer a.model.mu.RUnlock()
	if a.c == 0 {
		return "", false
	}
	registerAdapterFuncs()
	keyBytes := cString(key)
	var buf [512]byte
	n := _lfg_adapter_meta_val_str(a.c, cStringPtr(keyBytes), uintptr(unsafe.Pointer(&buf[0])), uintptr(512))
	runtime.KeepAlive(keyBytes)
	if n < 0 {
		return "", false
	}
	return goStringN(uintptr(unsafe.Pointer(&buf[0])), int(n)), true
}

// MetadataCount returns the number of metadata key/value pairs.
func (a *AdapterLoRA) MetadataCount() int {
	a.model.mu.RLock()
	defer a.model.mu.RUnlock()
	if a.c == 0 {
		return 0
	}
	registerAdapterFuncs()
	return int(_lfg_adapter_meta_count(a.c))
}

// MetadataKeyAt returns the metadata key at the given index.
func (a *AdapterLoRA) MetadataKeyAt(i int) (string, bool) {
	a.model.mu.RLock()
	defer a.model.mu.RUnlock()
	if a.c == 0 {
		return "", false
	}
	registerAdapterFuncs()
	var buf [256]byte
	n := _lfg_adapter_meta_key_by_index(a.c, int32(i), uintptr(unsafe.Pointer(&buf[0])), uintptr(256))
	if n < 0 {
		return "", false
	}
	return goStringN(uintptr(unsafe.Pointer(&buf[0])), int(n)), true
}

// MetadataValueAt returns the metadata value at the given index.
func (a *AdapterLoRA) MetadataValueAt(i int) (string, bool) {
	a.model.mu.RLock()
	defer a.model.mu.RUnlock()
	if a.c == 0 {
		return "", false
	}
	registerAdapterFuncs()
	var buf [512]byte
	n := _lfg_adapter_meta_val_str_by_index(a.c, int32(i), uintptr(unsafe.Pointer(&buf[0])), uintptr(512))
	if n < 0 {
		return "", false
	}
	return goStringN(uintptr(unsafe.Pointer(&buf[0])), int(n)), true
}

// InvocationTokenCount returns the number of invocation tokens if this is an aLoRA.
func (a *AdapterLoRA) InvocationTokenCount() uint64 {
	a.model.mu.RLock()
	defer a.model.mu.RUnlock()
	if a.c == 0 {
		return 0
	}
	registerAdapterFuncs()
	return _lfg_adapter_get_alora_n_invocation_tokens(a.c)
}

// InvocationTokens returns the invocation tokens if this is an aLoRA.
func (a *AdapterLoRA) InvocationTokens() []Token {
	a.model.mu.RLock()
	defer a.model.mu.RUnlock()
	if a.c == 0 {
		return nil
	}
	registerAdapterFuncs()
	n := int(_lfg_adapter_get_alora_n_invocation_tokens(a.c))
	if n == 0 {
		return nil
	}
	ptr := _lfg_adapter_get_alora_invocation_tokens(a.c)
	if ptr == 0 {
		return nil
	}
	cTokens := unsafe.Slice((*int32)(unsafe.Pointer(ptr)), n)
	tokens := make([]Token, n)
	for i, t := range cTokens {
		tokens[i] = Token(t)
	}
	return tokens
}

// SetLoRA adds a LoRA adapter to the context with the given scale.
func (ctx *Context) SetLoRA(adapter *AdapterLoRA, scale float32) error {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == 0 {
		return &Error{Code: ErrorInvalidArgument, Message: "context is closed"}
	}
	if adapter == nil || adapter.c == 0 {
		return &Error{Code: ErrorInvalidArgument, Message: "adapter is nil"}
	}
	registerAdapterFuncs()
	rc := _lfg_set_adapter_lora(ctx.c, adapter.c, scale)
	if rc != 0 {
		return &Error{Code: ErrorInternal, Message: "failed to set LoRA adapter"}
	}
	return nil
}

// RemoveLoRA removes a specific LoRA adapter from the context.
func (ctx *Context) RemoveLoRA(adapter *AdapterLoRA) error {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == 0 {
		return &Error{Code: ErrorInvalidArgument, Message: "context is closed"}
	}
	if adapter == nil || adapter.c == 0 {
		return &Error{Code: ErrorInvalidArgument, Message: "adapter is nil"}
	}
	registerAdapterFuncs()
	rc := _lfg_rm_adapter_lora(ctx.c, adapter.c)
	if rc != 0 {
		return &Error{Code: ErrorInternal, Message: "adapter not present in context"}
	}
	return nil
}

// ClearLoRA removes all LoRA adapters from the context.
func (ctx *Context) ClearLoRA() {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == 0 {
		return
	}
	registerAdapterFuncs()
	_lfg_clear_adapter_lora(ctx.c)
}

// ApplyControlVector applies a control vector to the context.
// data should be an n_embd x n_layers buffer starting from layer 1.
// Pass nil data to clear the current control vector.
func (ctx *Context) ApplyControlVector(data []float32, nEmbd, ilStart, ilEnd int) error {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == 0 {
		return &Error{Code: ErrorInvalidArgument, Message: "context is closed"}
	}

	registerAdapterFuncs()
	var dataPtr uintptr
	var dataLen uintptr
	if len(data) > 0 {
		dataPtr = uintptr(unsafe.Pointer(&data[0]))
		dataLen = uintptr(len(data))
	}

	rc := _lfg_apply_adapter_cvec(ctx.c, dataPtr, dataLen, int32(nEmbd), int32(ilStart), int32(ilEnd))
	if rc != 0 {
		return &Error{Code: ErrorInternal, Message: "failed to apply control vector"}
	}
	return nil
}
