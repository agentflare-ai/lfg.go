//go:build (darwin && arm64) || (linux && amd64) || (linux && arm64)

package lfg

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"github.com/ebitengine/purego"
)

var (
	libOnce  sync.Once
	libHandle uintptr
	libErr   error
)

func libraryName() string {
	switch runtime.GOOS {
	case "darwin":
		return "liblfg.dylib"
	case "linux":
		switch runtime.GOARCH {
		case "amd64":
			return "liblfg-linux-x86_64.so"
		case "arm64":
			return "liblfg-linux-aarch64.so"
		}
	}
	return ""
}

func loadLibrary() (uintptr, error) {
	libOnce.Do(func() {
		name := libraryName()
		if name == "" {
			libErr = fmt.Errorf("lfg: unsupported platform %s/%s", runtime.GOOS, runtime.GOARCH)
			return
		}

		// 1. LFG_LIB_PATH env var
		if envPath := os.Getenv("LFG_LIB_PATH"); envPath != "" {
			libHandle, libErr = purego.Dlopen(envPath, purego.RTLD_NOW|purego.RTLD_GLOBAL)
			if libErr == nil {
				return
			}
		}

		// 2. Relative to source directory (deps/lfg.cpp/dist/)
		_, thisFile, _, _ := runtime.Caller(0)
		distDir := filepath.Join(filepath.Dir(thisFile), "deps", "lfg.cpp", "dist")
		relPath := filepath.Join(distDir, name)
		libHandle, libErr = purego.Dlopen(relPath, purego.RTLD_NOW|purego.RTLD_GLOBAL)
		if libErr == nil {
			return
		}

		// 3. System search (dlopen default paths)
		libHandle, libErr = purego.Dlopen(name, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	})
	return libHandle, libErr
}
