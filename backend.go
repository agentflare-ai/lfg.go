package lfg

/*
typedef struct lfm_model lfm_model;
typedef struct lfm_context lfm_context;
typedef struct lfm_vocab lfm_vocab;
typedef struct lfm_sampler lfm_sampler;
#include "lfm_inference.h"
*/
import "C"
import "sync"

var backendOnce sync.Once

// ensureBackend initializes the backend exactly once. Called automatically by
// LoadModel, NewContext, and NewSession.
func ensureBackend() {
	backendOnce.Do(func() {
		C.lfm_backend_init()
	})
}

// BackendFree releases resources allocated by the backend.
// Safe to call multiple times or before any initialization.
func BackendFree() {
	C.lfm_backend_free()
}

// APIVersion returns the version string of the underlying C library.
func APIVersion() string {
	return C.GoString(C.lfm_api_version_string())
}

// APIVersionNumbers returns the major, minor, and patch version numbers.
func APIVersionNumbers() (major, minor, patch uint32) {
	var ma, mi, pa C.uint32_t
	C.lfm_api_version(&ma, &mi, &pa)
	return uint32(ma), uint32(mi), uint32(pa)
}

// ABIVersion returns the ABI version number.
func ABIVersion() uint32 {
	return uint32(C.lfm_abi_version())
}

// SystemInfo returns a string describing the system capabilities.
func SystemInfo() string {
	return C.GoString(C.lfm_print_system_info())
}

// TimeUS returns the current time in microseconds.
func TimeUS() int64 {
	return int64(C.lfm_time_us())
}

// MaxDevices returns the maximum number of devices available.
func MaxDevices() int {
	return int(C.lfm_max_devices())
}

// MaxParallelSequences returns the maximum number of parallel sequences.
func MaxParallelSequences() int {
	return int(C.lfm_max_parallel_sequences())
}

// SupportsMmap returns whether the system supports mmap.
func SupportsMmap() bool {
	return bool(C.lfm_supports_mmap())
}

// SupportsMlock returns whether the system supports mlock.
func SupportsMlock() bool {
	return bool(C.lfm_supports_mlock())
}

// SupportsGPUOffload returns whether GPU offloading is supported.
func SupportsGPUOffload() bool {
	return bool(C.lfm_supports_gpu_offload())
}

// SupportsRPC returns whether RPC is supported.
func SupportsRPC() bool {
	return bool(C.lfm_supports_rpc())
}
