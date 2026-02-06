package lfg

/*
#include "lfg_inference.h"
*/
import "C"
import "sync"

var backendOnce sync.Once

// ensureBackend initializes the backend exactly once. Called automatically by
// LoadModel, NewContext, and NewSession.
func ensureBackend() {
	backendOnce.Do(func() {
		C.lfg_backend_init()
	})
}

// BackendFree releases resources allocated by the backend.
// Safe to call multiple times or before any initialization.
func BackendFree() {
	C.lfg_backend_free()
}

// Version returns the version string of the underlying C library.
func Version() string {
	return C.GoString(C.lfg_api_version_string())
}

// VersionNumbers returns the major, minor, and patch version numbers.
func VersionNumbers() (major, minor, patch uint32) {
	var ma, mi, pa C.uint32_t
	C.lfg_api_version(&ma, &mi, &pa)
	return uint32(ma), uint32(mi), uint32(pa)
}

// ABIVersion returns the ABI version number.
func ABIVersion() uint32 {
	return uint32(C.lfg_abi_version())
}

// SystemInfo returns a string describing the system capabilities.
func SystemInfo() string {
	return C.GoString(C.lfg_print_system_info())
}

// TimeMicroseconds returns the current time in microseconds.
func TimeMicroseconds() int64 {
	return int64(C.lfg_time_us())
}

// DeviceCount returns the maximum number of devices available.
func DeviceCount() int {
	return int(C.lfg_max_devices())
}

// MaxParallelSequences returns the maximum number of parallel sequences.
func MaxParallelSequences() int {
	return int(C.lfg_max_parallel_sequences())
}

// SupportsMmap returns whether the system supports mmap.
func SupportsMmap() bool {
	return bool(C.lfg_supports_mmap())
}

// SupportsMlock returns whether the system supports mlock.
func SupportsMlock() bool {
	return bool(C.lfg_supports_mlock())
}

// SupportsGPUOffload returns whether GPU offloading is supported.
func SupportsGPUOffload() bool {
	return bool(C.lfg_supports_gpu_offload())
}

// SupportsRPC returns whether RPC is supported.
func SupportsRPC() bool {
	return bool(C.lfg_supports_rpc())
}
