package lfg

import (
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
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

// progressTrampoline holds the purego callback pointer, initialized once.
var (
	progressTrampolineOnce sync.Once
	progressTrampoline     uintptr
)

// getProgressTrampoline returns the purego callback function pointer for the
// progress callback. The callback is created once via purego.NewCallback and
// matches the C signature: bool (*)(float, void*).
func getProgressTrampoline() uintptr {
	progressTrampolineOnce.Do(func() {
		progressTrampoline = purego.NewCallback(func(progress float32, userData uintptr) uintptr {
			if userData == 0 {
				return 1 // true — continue
			}
			id := *(*uintptr)(unsafe.Pointer(userData))
			progressMu.Lock()
			cb, ok := progressCallbacks[id]
			progressMu.Unlock()
			if !ok {
				return 1 // true — continue
			}
			if cb(progress) {
				return 1 // true — continue
			}
			return 0 // false — abort
		})
	})
	return progressTrampoline
}
