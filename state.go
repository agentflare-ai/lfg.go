//go:build (darwin && arm64) || (linux && amd64) || (linux && arm64)

package lfg

import (
	"runtime"
	"unsafe"
)

// StateGetSize returns the actual size in bytes needed to save the full state.
func (ctx *Context) StateGetSize() int {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == 0 {
		return 0
	}
	registerStateFuncs()
	return int(_lfg_state_get_size(ctx.c))
}

// StateGetData copies the full context state into dst.
// dst must be large enough (use StateGetSize).
// Returns the number of bytes written.
func (ctx *Context) StateGetData(dst []byte) int {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == 0 || len(dst) == 0 {
		return 0
	}
	registerStateFuncs()
	return int(_lfg_state_get_data(ctx.c, uintptr(unsafe.Pointer(&dst[0])), uintptr(len(dst))))
}

// StateSetData restores context state from src.
// Returns the number of bytes read.
func (ctx *Context) StateSetData(src []byte) int {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == 0 || len(src) == 0 {
		return 0
	}
	registerStateFuncs()
	return int(_lfg_state_set_data(ctx.c, uintptr(unsafe.Pointer(&src[0])), uintptr(len(src))))
}

// StateSaveFile saves the session state to a file.
func (ctx *Context) StateSaveFile(path string, tokens []Token) error {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == 0 {
		return &Error{Code: ErrorInvalidArgument, Message: "context is closed"}
	}

	registerStateFuncs()
	pathBytes := cString(path)

	ok := _lfg_state_save_file(ctx.c, cStringPtr(pathBytes), tokenPtr(tokens), uintptr(len(tokens)))
	runtime.KeepAlive(pathBytes)
	if !ok {
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
	if ctx.c == 0 {
		return nil, &Error{Code: ErrorInvalidArgument, Message: "context is closed"}
	}

	registerStateFuncs()
	pathBytes := cString(path)

	tokens := make([]Token, maxTokens)
	var nTokens uintptr

	ok := _lfg_state_load_file(ctx.c, cStringPtr(pathBytes), tokenPtr(tokens), uintptr(maxTokens), uintptr(unsafe.Pointer(&nTokens)))
	runtime.KeepAlive(pathBytes)
	if !ok {
		if err := getLastError(); err != nil {
			return nil, err
		}
		return nil, &Error{Code: ErrorIO, Message: "failed to load state file"}
	}
	return tokens[:nTokens], nil
}

// StateSeqGetSize returns the size needed to copy a single sequence's state.
func (ctx *Context) StateSeqGetSize(seqID SequenceID) int {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == 0 {
		return 0
	}
	registerStateFuncs()
	return int(_lfg_state_seq_get_size(ctx.c, int32(seqID)))
}

// StateSeqGetData copies a single sequence's state into dst.
// Returns the number of bytes written.
func (ctx *Context) StateSeqGetData(dst []byte, seqID SequenceID) int {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == 0 || len(dst) == 0 {
		return 0
	}
	registerStateFuncs()
	return int(_lfg_state_seq_get_data(ctx.c, uintptr(unsafe.Pointer(&dst[0])), uintptr(len(dst)), int32(seqID)))
}

// StateSeqSetData restores a single sequence's state from src.
// Returns the number of bytes read, or 0 on failure.
func (ctx *Context) StateSeqSetData(src []byte, destSeqID SequenceID) int {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == 0 || len(src) == 0 {
		return 0
	}
	registerStateFuncs()
	return int(_lfg_state_seq_set_data(ctx.c, uintptr(unsafe.Pointer(&src[0])), uintptr(len(src)), int32(destSeqID)))
}

// StateSeqSaveFile saves a single sequence's state to a file.
func (ctx *Context) StateSeqSaveFile(path string, seqID SequenceID, tokens []Token) error {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == 0 {
		return &Error{Code: ErrorInvalidArgument, Message: "context is closed"}
	}

	registerStateFuncs()
	pathBytes := cString(path)

	n := _lfg_state_seq_save_file(ctx.c, cStringPtr(pathBytes), int32(seqID), tokenPtr(tokens), uintptr(len(tokens)))
	runtime.KeepAlive(pathBytes)
	if n == 0 {
		return &Error{Code: ErrorIO, Message: "failed to save sequence state file"}
	}
	return nil
}

// StateSeqLoadFile loads a single sequence's state from a file.
func (ctx *Context) StateSeqLoadFile(path string, destSeqID SequenceID, maxTokens int) ([]Token, error) {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == 0 {
		return nil, &Error{Code: ErrorInvalidArgument, Message: "context is closed"}
	}

	registerStateFuncs()
	pathBytes := cString(path)

	tokens := make([]Token, maxTokens)
	var nTokens uintptr

	n := _lfg_state_seq_load_file(ctx.c, cStringPtr(pathBytes), int32(destSeqID), tokenPtr(tokens), uintptr(maxTokens), uintptr(unsafe.Pointer(&nTokens)))
	runtime.KeepAlive(pathBytes)
	if n == 0 {
		if err := getLastError(); err != nil {
			return nil, err
		}
		return nil, &Error{Code: ErrorIO, Message: "failed to load sequence state file"}
	}
	return tokens[:nTokens], nil
}
