//go:build linux && amd64

package lfg

/*
#cgo CFLAGS: -I${SRCDIR}/deps/include -D_GNU_SOURCE
#cgo LDFLAGS: -L${SRCDIR}/deps/lib -llfg-linux-x86_64 -lm -lstdc++ -lpthread -ldl
*/
import "C"
