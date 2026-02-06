//go:build windows && amd64

package lfg

/*
#cgo CFLAGS: -I${SRCDIR}/deps/lfg.cpp/src/inference -I${SRCDIR}/deps/lfg.cpp/src/ggml
#cgo LDFLAGS: -L${SRCDIR}/deps/lfg.cpp/dist -llfg-windows-x86_64-static -lws2_32 -lwinmm -liphlpapi -lstdc++
*/
import "C"
