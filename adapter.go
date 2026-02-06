package lfg

/*
#include "lfg_inference.h"
#include <stdlib.h>
*/
import "C"
import "unsafe"

// AdapterLoRA wraps a LoRA adapter loaded from a file.
// The adapter is valid as long as its parent model is not freed.
// All adapters must be loaded before context creation.
type AdapterLoRA struct {
	c     *C.struct_lfg_adapter_lora
	model *Model // prevent GC of parent
}

// LoadAdapterLoRA loads a LoRA adapter from the given file path.
func LoadAdapterLoRA(model *Model, path string) (*AdapterLoRA, error) {
	model.mu.RLock()
	defer model.mu.RUnlock()
	if model.c == nil {
		return nil, &Error{Code: ErrorInvalidArgument, Message: "model is closed"}
	}

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	adapter := C.lfg_adapter_lora_init(model.c, cPath)
	if adapter == nil {
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
	if a.c == nil {
		return "", false
	}
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))

	var buf [512]C.char
	n := C.lfg_adapter_meta_val_str(a.c, cKey, &buf[0], 512)
	if n < 0 {
		return "", false
	}
	return C.GoStringN(&buf[0], n), true
}

// MetadataCount returns the number of metadata key/value pairs.
func (a *AdapterLoRA) MetadataCount() int {
	a.model.mu.RLock()
	defer a.model.mu.RUnlock()
	if a.c == nil {
		return 0
	}
	return int(C.lfg_adapter_meta_count(a.c))
}

// MetadataKeyAt returns the metadata key at the given index.
func (a *AdapterLoRA) MetadataKeyAt(i int) (string, bool) {
	a.model.mu.RLock()
	defer a.model.mu.RUnlock()
	if a.c == nil {
		return "", false
	}
	var buf [256]C.char
	n := C.lfg_adapter_meta_key_by_index(a.c, C.int32_t(i), &buf[0], 256)
	if n < 0 {
		return "", false
	}
	return C.GoStringN(&buf[0], n), true
}

// MetadataValueAt returns the metadata value at the given index.
func (a *AdapterLoRA) MetadataValueAt(i int) (string, bool) {
	a.model.mu.RLock()
	defer a.model.mu.RUnlock()
	if a.c == nil {
		return "", false
	}
	var buf [512]C.char
	n := C.lfg_adapter_meta_val_str_by_index(a.c, C.int32_t(i), &buf[0], 512)
	if n < 0 {
		return "", false
	}
	return C.GoStringN(&buf[0], n), true
}

// InvocationTokenCount returns the number of invocation tokens if this is an aLoRA.
func (a *AdapterLoRA) InvocationTokenCount() uint64 {
	a.model.mu.RLock()
	defer a.model.mu.RUnlock()
	if a.c == nil {
		return 0
	}
	return uint64(C.lfg_adapter_get_alora_n_invocation_tokens(a.c))
}

// InvocationTokens returns the invocation tokens if this is an aLoRA.
func (a *AdapterLoRA) InvocationTokens() []Token {
	a.model.mu.RLock()
	defer a.model.mu.RUnlock()
	if a.c == nil {
		return nil
	}
	n := int(C.lfg_adapter_get_alora_n_invocation_tokens(a.c))
	if n == 0 {
		return nil
	}
	ptr := C.lfg_adapter_get_alora_invocation_tokens(a.c)
	if ptr == nil {
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
	if ctx.c == nil {
		return &Error{Code: ErrorInvalidArgument, Message: "context is closed"}
	}
	if adapter == nil || adapter.c == nil {
		return &Error{Code: ErrorInvalidArgument, Message: "adapter is nil"}
	}
	rc := C.lfg_set_adapter_lora(ctx.c, adapter.c, C.float(scale))
	if rc != 0 {
		return &Error{Code: ErrorInternal, Message: "failed to set LoRA adapter"}
	}
	return nil
}

// RemoveLoRA removes a specific LoRA adapter from the context.
func (ctx *Context) RemoveLoRA(adapter *AdapterLoRA) error {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == nil {
		return &Error{Code: ErrorInvalidArgument, Message: "context is closed"}
	}
	if adapter == nil || adapter.c == nil {
		return &Error{Code: ErrorInvalidArgument, Message: "adapter is nil"}
	}
	rc := C.lfg_rm_adapter_lora(ctx.c, adapter.c)
	if rc != 0 {
		return &Error{Code: ErrorInternal, Message: "adapter not present in context"}
	}
	return nil
}

// ClearLoRA removes all LoRA adapters from the context.
func (ctx *Context) ClearLoRA() {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == nil {
		return
	}
	C.lfg_clear_adapter_lora(ctx.c)
}

// ApplyControlVector applies a control vector to the context.
// data should be an n_embd x n_layers buffer starting from layer 1.
// Pass nil data to clear the current control vector.
func (ctx *Context) ApplyControlVector(data []float32, nEmbd, ilStart, ilEnd int) error {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == nil {
		return &Error{Code: ErrorInvalidArgument, Message: "context is closed"}
	}

	var dataPtr *C.float
	var dataLen C.size_t
	if len(data) > 0 {
		dataPtr = (*C.float)(unsafe.Pointer(&data[0]))
		dataLen = C.size_t(len(data))
	}

	rc := C.lfg_apply_adapter_cvec(ctx.c, dataPtr, dataLen, C.int32_t(nEmbd), C.int32_t(ilStart), C.int32_t(ilEnd))
	if rc != 0 {
		return &Error{Code: ErrorInternal, Message: "failed to apply control vector"}
	}
	return nil
}
