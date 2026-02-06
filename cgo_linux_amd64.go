//go:build linux && amd64

package lfg

/*
#cgo CFLAGS: -I${SRCDIR}/deps/lfg.cpp/src/inference -I${SRCDIR}/deps/lfg.cpp/src/ggml -D_GNU_SOURCE
#cgo LDFLAGS: -L${SRCDIR}/deps/lfg.cpp/dist -llfg-linux-x86_64 -lm -lstdc++ -lpthread -ldl
*/
import "C"
