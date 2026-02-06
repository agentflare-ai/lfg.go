package lfg

/*
typedef struct lfm_model lfm_model;
typedef struct lfm_context lfm_context;
typedef struct lfm_vocab lfm_vocab;
typedef struct lfm_sampler lfm_sampler;
#include "lfm_inference.h"
*/
import "C"
import "fmt"

// ErrorCode represents an error code returned by the C API.
type ErrorCode int

const (
	ErrorNone            ErrorCode = C.LFM_ERROR_NONE
	ErrorInvalidArgument ErrorCode = C.LFM_ERROR_INVALID_ARGUMENT
	ErrorIO              ErrorCode = C.LFM_ERROR_IO
	ErrorOutOfMemory     ErrorCode = C.LFM_ERROR_OUT_OF_MEMORY
	ErrorUnsupported     ErrorCode = C.LFM_ERROR_UNSUPPORTED
	ErrorCancelled       ErrorCode = C.LFM_ERROR_CANCELLED
	ErrorInternal        ErrorCode = C.LFM_ERROR_INTERNAL
)

// String returns the human-readable name of the error code.
func (c ErrorCode) String() string {
	return C.GoString(C.lfm_error_string(C.enum_lfm_error(c)))
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
	var buf [1024]C.char
	code := C.lfm_get_last_error(&buf[0], 1024)
	if code == C.LFM_ERROR_NONE {
		return nil
	}
	return &Error{
		Code:    ErrorCode(code),
		Message: C.GoString(&buf[0]),
	}
}
