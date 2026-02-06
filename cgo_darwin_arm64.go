//go:build darwin && arm64

package lfg

/*
#cgo CFLAGS: -I${SRCDIR}/deps/include -D_XOPEN_SOURCE=600
#cgo LDFLAGS: -L${SRCDIR}/deps/lib -llfg-macos-aarch64 -framework Accelerate -framework Metal -framework Foundation -lstdc++
*/
import "C"
