package lfg

/*
typedef struct lfm_model lfm_model;
typedef struct lfm_context lfm_context;
typedef struct lfm_vocab lfm_vocab;
typedef struct lfm_sampler lfm_sampler;
#include "lfm_inference.h"
#include <stdlib.h>
*/
import "C"
import "unsafe"

// AdapterLoRA wraps a LoRA adapter loaded from a file.
// The adapter is valid as long as its parent model is not freed.
// All adapters must be loaded before context creation.
type AdapterLoRA struct {
	c     *C.struct_lfm_adapter_lora
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

	adapter := C.lfm_adapter_lora_init(model.c, cPath)
	if adapter == nil {
		if err := getLastError(); err != nil {
			return nil, err
		}
		return nil, &Error{Code: ErrorInternal, Message: "failed to load LoRA adapter"}
	}

	return &AdapterLoRA{c: adapter, model: model}, nil
}

// MetaValStr returns a metadata value by key name.
func (a *AdapterLoRA) MetaValStr(key string) (string, bool) {
	a.model.mu.RLock()
	defer a.model.mu.RUnlock()
	if a.c == nil {
		return "", false
	}
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))

	var buf [512]C.char
	n := C.lfm_adapter_meta_val_str(a.c, cKey, &buf[0], 512)
	if n < 0 {
		return "", false
	}
	return C.GoStringN(&buf[0], n), true
}

// MetaCount returns the number of metadata key/value pairs.
func (a *AdapterLoRA) MetaCount() int {
	a.model.mu.RLock()
	defer a.model.mu.RUnlock()
	if a.c == nil {
		return 0
	}
	return int(C.lfm_adapter_meta_count(a.c))
}

// MetaKeyByIndex returns the metadata key at the given index.
func (a *AdapterLoRA) MetaKeyByIndex(i int) (string, bool) {
	a.model.mu.RLock()
	defer a.model.mu.RUnlock()
	if a.c == nil {
		return "", false
	}
	var buf [256]C.char
	n := C.lfm_adapter_meta_key_by_index(a.c, C.int32_t(i), &buf[0], 256)
	if n < 0 {
		return "", false
	}
	return C.GoStringN(&buf[0], n), true
}

// MetaValStrByIndex returns the metadata value at the given index.
func (a *AdapterLoRA) MetaValStrByIndex(i int) (string, bool) {
	a.model.mu.RLock()
	defer a.model.mu.RUnlock()
	if a.c == nil {
		return "", false
	}
	var buf [512]C.char
	n := C.lfm_adapter_meta_val_str_by_index(a.c, C.int32_t(i), &buf[0], 512)
	if n < 0 {
		return "", false
	}
	return C.GoStringN(&buf[0], n), true
}

// NInvocationTokens returns the number of invocation tokens if this is an aLoRA.
func (a *AdapterLoRA) NInvocationTokens() uint64 {
	a.model.mu.RLock()
	defer a.model.mu.RUnlock()
	if a.c == nil {
		return 0
	}
	return uint64(C.lfm_adapter_get_alora_n_invocation_tokens(a.c))
}

// InvocationTokens returns the invocation tokens if this is an aLoRA.
func (a *AdapterLoRA) InvocationTokens() []Token {
	a.model.mu.RLock()
	defer a.model.mu.RUnlock()
	if a.c == nil {
		return nil
	}
	n := int(C.lfm_adapter_get_alora_n_invocation_tokens(a.c))
	if n == 0 {
		return nil
	}
	ptr := C.lfm_adapter_get_alora_invocation_tokens(a.c)
	if ptr == nil {
		return nil
	}
	tokens := make([]Token, n)
	copy(tokens, unsafe.Slice((*Token)(unsafe.Pointer(ptr)), n))
	return tokens
}

// SetAdapterLoRA adds a LoRA adapter to the context with the given scale.
func (ctx *Context) SetAdapterLoRA(adapter *AdapterLoRA, scale float32) error {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == nil {
		return &Error{Code: ErrorInvalidArgument, Message: "context is closed"}
	}
	if adapter == nil || adapter.c == nil {
		return &Error{Code: ErrorInvalidArgument, Message: "adapter is nil"}
	}
	rc := C.lfm_set_adapter_lora(ctx.c, adapter.c, C.float(scale))
	if rc != 0 {
		return &Error{Code: ErrorInternal, Message: "failed to set LoRA adapter"}
	}
	return nil
}

// RmAdapterLoRA removes a specific LoRA adapter from the context.
func (ctx *Context) RmAdapterLoRA(adapter *AdapterLoRA) error {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == nil {
		return &Error{Code: ErrorInvalidArgument, Message: "context is closed"}
	}
	if adapter == nil || adapter.c == nil {
		return &Error{Code: ErrorInvalidArgument, Message: "adapter is nil"}
	}
	rc := C.lfm_rm_adapter_lora(ctx.c, adapter.c)
	if rc != 0 {
		return &Error{Code: ErrorInternal, Message: "adapter not present in context"}
	}
	return nil
}

// ClearAdapterLoRA removes all LoRA adapters from the context.
func (ctx *Context) ClearAdapterLoRA() {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == nil {
		return
	}
	C.lfm_clear_adapter_lora(ctx.c)
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

	rc := C.lfm_apply_adapter_cvec(ctx.c, dataPtr, dataLen, C.int32_t(nEmbd), C.int32_t(ilStart), C.int32_t(ilEnd))
	if rc != 0 {
		return &Error{Code: ErrorInternal, Message: "failed to apply control vector"}
	}
	return nil
}
