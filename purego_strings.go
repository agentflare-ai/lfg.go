//go:build (darwin && arm64) || (linux && amd64) || (linux && arm64)

package lfg

import "unsafe"

// goString converts a C null-terminated string pointer (uintptr) to a Go string.
// Returns empty string if ptr is 0.
func goString(ptr uintptr) string {
	if ptr == 0 {
		return ""
	}
	// Walk to null terminator.
	p := ptr
	n := 0
	for *(*byte)(unsafe.Pointer(p)) != 0 {
		n++
		p++
	}
	if n == 0 {
		return ""
	}
	return string(unsafe.Slice((*byte)(unsafe.Pointer(ptr)), n))
}

// goStringN converts a C string pointer (uintptr) with known length to a Go string.
// Returns empty string if ptr is 0 or n <= 0.
func goStringN(ptr uintptr, n int) string {
	if ptr == 0 || n <= 0 {
		return ""
	}
	return string(unsafe.Slice((*byte)(unsafe.Pointer(ptr)), n))
}

// cString allocates a null-terminated byte slice from a Go string.
// The returned slice is pinned in Go memory — use with purego which
// does not enforce CGO pointer rules.
func cString(s string) []byte {
	b := make([]byte, len(s)+1)
	copy(b, s)
	// b[len(s)] is already 0
	return b
}

// cStringPtr returns a uintptr to a null-terminated Go byte slice.
// The caller must keep the byte slice alive (runtime.KeepAlive) until
// the C function returns.
func cStringPtr(b []byte) uintptr {
	if len(b) == 0 {
		return 0
	}
	return uintptr(unsafe.Pointer(&b[0]))
}

// mallocCString allocates a NUL-terminated C string via native malloc.
// The caller is responsible for ensuring native code eventually frees it.
func mallocCString(s string) uintptr {
	registerSessionFuncs()

	n := len(s)
	ptr := _malloc(uintptr(n + 1))
	if ptr == 0 {
		return 0
	}

	dst := unsafe.Slice((*byte)(unsafe.Pointer(ptr)), n+1)
	copy(dst, s)
	dst[n] = 0
	return ptr
}
