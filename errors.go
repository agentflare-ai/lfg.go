//go:build (darwin && arm64) || (linux && amd64) || (linux && arm64)

package lfg

import (
	"fmt"
	"unsafe"
)

// ErrorCode represents an error code returned by the C API.
type ErrorCode int

const (
	ErrorNone            ErrorCode = 0
	ErrorInvalidArgument ErrorCode = 1
	ErrorIO              ErrorCode = 2
	ErrorOutOfMemory     ErrorCode = 3
	ErrorUnsupported     ErrorCode = 4
	ErrorCancelled       ErrorCode = 5
	ErrorInternal        ErrorCode = 6
)

// String returns the human-readable name of the error code.
func (c ErrorCode) String() string {
	registerBackendFuncs()
	return goString(_lfg_error_string(int32(c)))
}

// Error represents an error from the lfg C library.
type Error struct {
	Code    ErrorCode
	Message string
}

func (e *Error) Error() string {
	if e.Message != "" {
		return fmt.Sprintf("lfg: %s: %s", e.Code, e.Message)
	}
	return fmt.Sprintf("lfg: %s", e.Code)
}

// getLastError reads the thread-local C error state. Must be called immediately
// after a failed C call (CGO pins the goroutine to the OS thread during the call).
func getLastError() error {
	registerBackendFuncs()
	var buf [1024]byte
	code := _lfg_get_last_error(uintptr(unsafe.Pointer(&buf[0])), uintptr(1024))
	if code == int32(ErrorNone) {
		return nil
	}
	// Find length of the null-terminated string in the buffer.
	n := 0
	for n < len(buf) && buf[n] != 0 {
		n++
	}
	return &Error{
		Code:    ErrorCode(code),
		Message: goStringN(uintptr(unsafe.Pointer(&buf[0])), n),
	}
}
