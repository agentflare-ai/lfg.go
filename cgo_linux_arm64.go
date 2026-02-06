//go:build linux && arm64

package lfg

/*
#cgo CFLAGS: -I${SRCDIR}/deps/include -D_GNU_SOURCE
#cgo LDFLAGS: -L${SRCDIR}/deps/lib -llfg-linux-aarch64 -lm -lstdc++ -lpthread -ldl
*/
import "C"
