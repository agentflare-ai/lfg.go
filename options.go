package lfg

/*
#include "lfg_inference.h"

// C trampoline for the progress callback.
// Calls back into Go via the goProgressCallback function.
extern _Bool goProgressCallback(float progress, void *user_data);

static _Bool progress_callback_trampoline(float progress, void *user_data) {
    return goProgressCallback(progress, user_data);
}

static lfg_progress_callback get_progress_trampoline(void) {
    return progress_callback_trampoline;
}
*/
import "C"
import (
	"sync"
	"unsafe"
)

// ProgressCallback is called during model loading with a progress value between 0.0 and 1.0.
// Return true to continue loading, false to abort.
type ProgressCallback func(progress float32) bool

// Global registry for progress callbacks, keyed by a unique ID.
var (
	progressMu        sync.Mutex
	progressCallbacks = make(map[uintptr]ProgressCallback)
	progressNextID    uintptr
)

func registerProgressCallback(cb ProgressCallback) uintptr {
	progressMu.Lock()
	defer progressMu.Unlock()
	progressNextID++
	id := progressNextID
	progressCallbacks[id] = cb
	return id
}

func unregisterProgressCallback(id uintptr) {
	progressMu.Lock()
	defer progressMu.Unlock()
	delete(progressCallbacks, id)
}

// getProgressTrampoline returns the C function pointer for the progress callback.
func getProgressTrampoline() C.lfg_progress_callback {
	return C.get_progress_trampoline()
}

//export goProgressCallback
func goProgressCallback(progress C.float, userData unsafe.Pointer) C._Bool {
	if userData == nil {
		return C._Bool(true)
	}
	id := *(*uintptr)(userData)
	progressMu.Lock()
	cb, ok := progressCallbacks[id]
	progressMu.Unlock()
	if !ok {
		return C._Bool(true)
	}
	return C._Bool(cb(float32(progress)))
}
