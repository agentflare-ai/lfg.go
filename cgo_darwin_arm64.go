//go:build darwin && arm64

package lfg

/*
#cgo CFLAGS: -I${SRCDIR}/deps/lfg.cpp/src/inference -I${SRCDIR}/deps/lfg.cpp/src/ggml -D_XOPEN_SOURCE=600
#cgo LDFLAGS: -L${SRCDIR}/deps/lfg.cpp/dist -llfg-macos-aarch64 -framework Accelerate -framework Metal -framework Foundation -lstdc++
*/
import "C"
