package lfg

/*
typedef struct lfm_model lfm_model;
typedef struct lfm_context lfm_context;
typedef struct lfm_vocab lfm_vocab;
typedef struct lfm_sampler lfm_sampler;
#include "lfm_api.h"
#include <stdlib.h>
*/
import "C"


// ApiVersion returns the version string of the underlying liquid.cpp library.
func ApiVersion() string {
	return C.GoString(C.lfm_api_version_string())
}

// Init initializes the liquid + ggml backend.
// Should be called once at the start of the program.
func Init() {
	C.lfm_backend_init()
}

// GetLastError returns the last error message from the library for the current thread.
// Returns an empty string if no error occurred.
func GetLastError() string {
	var buf [1024]C.char
	errCode := C.lfm_get_last_error(&buf[0], 1024)
	if errCode == C.LFM_ERROR_NONE {
		return ""
	}
	return C.GoString(&buf[0])
}

// SystemInfo returns the system information string.
func SystemInfo() string {
	return C.GoString(C.lfm_print_system_info())
}
