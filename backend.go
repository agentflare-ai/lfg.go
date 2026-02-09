//go:build (darwin && arm64) || (linux && amd64) || (linux && arm64)

package lfg

import "sync"

var backendOnce sync.Once

// ensureBackend initializes the backend exactly once. Called automatically by
// LoadModel, NewContext, and NewSession.
func ensureBackend() {
	backendOnce.Do(func() {
		registerBackendFuncs()
		_lfg_backend_init()
	})
}

// BackendFree releases resources allocated by the backend.
// Safe to call multiple times or before any initialization.
func BackendFree() {
	registerBackendFuncs()
	_lfg_backend_free()
}

// Version returns the version string of the underlying C library.
func Version() string {
	registerBackendFuncs()
	return goString(_lfg_api_version_string())
}

// VersionNumbers returns the major, minor, and patch version numbers.
func VersionNumbers() (major, minor, patch uint32) {
	registerBackendFuncs()
	var ma, mi, pa uint32
	_lfg_api_version(&ma, &mi, &pa)
	return ma, mi, pa
}

// ABIVersion returns the ABI version number.
func ABIVersion() uint32 {
	registerBackendFuncs()
	return _lfg_abi_version()
}

// SystemInfo returns a string describing the system capabilities.
func SystemInfo() string {
	registerBackendFuncs()
	return goString(_lfg_print_system_info())
}

// TimeMicroseconds returns the current time in microseconds.
func TimeMicroseconds() int64 {
	registerBackendFuncs()
	return _lfg_time_us()
}

// DeviceCount returns the maximum number of devices available.
func DeviceCount() int {
	registerBackendFuncs()
	return int(_lfg_max_devices())
}

// MaxParallelSequences returns the maximum number of parallel sequences.
func MaxParallelSequences() int {
	registerBackendFuncs()
	return int(_lfg_max_parallel_sequences())
}

// SupportsMmap returns whether the system supports mmap.
func SupportsMmap() bool {
	registerBackendFuncs()
	return _lfg_supports_mmap()
}

// SupportsMlock returns whether the system supports mlock.
func SupportsMlock() bool {
	registerBackendFuncs()
	return _lfg_supports_mlock()
}

// SupportsGPUOffload returns whether GPU offloading is supported.
func SupportsGPUOffload() bool {
	registerBackendFuncs()
	return _lfg_supports_gpu_offload()
}

// SupportsRPC returns whether RPC is supported.
func SupportsRPC() bool {
	registerBackendFuncs()
	return _lfg_supports_rpc()
}
