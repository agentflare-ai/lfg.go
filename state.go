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

// StateGetSize returns the actual size in bytes needed to save the full state.
func (ctx *Context) StateGetSize() int {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == nil {
		return 0
	}
	return int(C.lfm_state_get_size(ctx.c))
}

// StateGetData copies the full context state into dst.
// dst must be large enough (use StateGetSize).
// Returns the number of bytes written.
func (ctx *Context) StateGetData(dst []byte) int {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == nil || len(dst) == 0 {
		return 0
	}
	return int(C.lfm_state_get_data(ctx.c, (*C.uint8_t)(unsafe.Pointer(&dst[0])), C.size_t(len(dst))))
}

// StateSetData restores context state from src.
// Returns the number of bytes read.
func (ctx *Context) StateSetData(src []byte) int {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == nil || len(src) == 0 {
		return 0
	}
	return int(C.lfm_state_set_data(ctx.c, (*C.uint8_t)(unsafe.Pointer(&src[0])), C.size_t(len(src))))
}

// StateSaveFile saves the session state to a file.
func (ctx *Context) StateSaveFile(path string, tokens []Token) error {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == nil {
		return &Error{Code: ErrorInvalidArgument, Message: "context is closed"}
	}

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var tokPtr *C.lfm_token
	if len(tokens) > 0 {
		tokPtr = (*C.lfm_token)(&tokens[0])
	}

	ok := C.lfm_state_save_file(ctx.c, cPath, tokPtr, C.size_t(len(tokens)))
	if !bool(ok) {
		if err := getLastError(); err != nil {
			return err
		}
		return &Error{Code: ErrorIO, Message: "failed to save state file"}
	}
	return nil
}

// StateLoadFile loads session state from a file.
// Returns the tokens that were stored in the session file.
func (ctx *Context) StateLoadFile(path string, maxTokens int) ([]Token, error) {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == nil {
		return nil, &Error{Code: ErrorInvalidArgument, Message: "context is closed"}
	}

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	tokens := make([]Token, maxTokens)
	var nTokens C.size_t

	ok := C.lfm_state_load_file(ctx.c, cPath, (*C.lfm_token)(&tokens[0]), C.size_t(maxTokens), &nTokens)
	if !bool(ok) {
		if err := getLastError(); err != nil {
			return nil, err
		}
		return nil, &Error{Code: ErrorIO, Message: "failed to load state file"}
	}
	return tokens[:nTokens], nil
}

// StateSeqGetSize returns the size needed to copy a single sequence's state.
func (ctx *Context) StateSeqGetSize(seqID SeqID) int {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == nil {
		return 0
	}
	return int(C.lfm_state_seq_get_size(ctx.c, C.lfm_seq_id(seqID)))
}

// StateSeqGetData copies a single sequence's state into dst.
// Returns the number of bytes written.
func (ctx *Context) StateSeqGetData(dst []byte, seqID SeqID) int {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == nil || len(dst) == 0 {
		return 0
	}
	return int(C.lfm_state_seq_get_data(ctx.c, (*C.uint8_t)(unsafe.Pointer(&dst[0])), C.size_t(len(dst)), C.lfm_seq_id(seqID)))
}

// StateSeqSetData restores a single sequence's state from src.
// Returns the number of bytes read, or 0 on failure.
func (ctx *Context) StateSeqSetData(src []byte, destSeqID SeqID) int {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == nil || len(src) == 0 {
		return 0
	}
	return int(C.lfm_state_seq_set_data(ctx.c, (*C.uint8_t)(unsafe.Pointer(&src[0])), C.size_t(len(src)), C.lfm_seq_id(destSeqID)))
}

// StateSeqSaveFile saves a single sequence's state to a file.
func (ctx *Context) StateSeqSaveFile(path string, seqID SeqID, tokens []Token) error {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == nil {
		return &Error{Code: ErrorInvalidArgument, Message: "context is closed"}
	}

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var tokPtr *C.lfm_token
	if len(tokens) > 0 {
		tokPtr = (*C.lfm_token)(&tokens[0])
	}

	n := C.lfm_state_seq_save_file(ctx.c, cPath, C.lfm_seq_id(seqID), tokPtr, C.size_t(len(tokens)))
	if n == 0 {
		return &Error{Code: ErrorIO, Message: "failed to save sequence state file"}
	}
	return nil
}

// StateSeqLoadFile loads a single sequence's state from a file.
func (ctx *Context) StateSeqLoadFile(path string, destSeqID SeqID, maxTokens int) ([]Token, error) {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == nil {
		return nil, &Error{Code: ErrorInvalidArgument, Message: "context is closed"}
	}

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	tokens := make([]Token, maxTokens)
	var nTokens C.size_t

	n := C.lfm_state_seq_load_file(ctx.c, cPath, C.lfm_seq_id(destSeqID), (*C.lfm_token)(&tokens[0]), C.size_t(maxTokens), &nTokens)
	if n == 0 {
		if err := getLastError(); err != nil {
			return nil, err
		}
		return nil, &Error{Code: ErrorIO, Message: "failed to load sequence state file"}
	}
	return tokens[:nTokens], nil
}
