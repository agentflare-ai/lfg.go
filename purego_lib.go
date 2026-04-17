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
	libOnce   sync.Once
	libHandle uintptr
	libErr    error
)

func libraryNames() []string {
	switch runtime.GOOS {
	case "darwin":
		return []string{"liblfg-macos-aarch64.dylib", "liblfg.dylib"}
	case "linux":
		switch runtime.GOARCH {
		case "amd64":
			return []string{"liblfg-linux-x86_64.so"}
		case "arm64":
			return []string{"liblfg-linux-aarch64.so"}
		}
	}
	return nil
}

func loadLibrary() (uintptr, error) {
	libOnce.Do(func() {
		names := libraryNames()
		if len(names) == 0 {
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

		// 2. Relative to source directory (deps/lfg.cpp/dist/lib/, then legacy dist/)
		_, thisFile, _, _ := runtime.Caller(0)
		baseDir := filepath.Join(filepath.Dir(thisFile), "deps", "lfg.cpp", "dist")
		searchDirs := []string{
			filepath.Join(baseDir, "lib"),
			baseDir,
		}
		for _, dir := range searchDirs {
			for _, name := range names {
				relPath := filepath.Join(dir, name)
				libHandle, libErr = purego.Dlopen(relPath, purego.RTLD_NOW|purego.RTLD_GLOBAL)
				if libErr == nil {
					return
				}
			}
		}

		// 3. System search (dlopen default paths)
		for _, name := range names {
			libHandle, libErr = purego.Dlopen(name, purego.RTLD_NOW|purego.RTLD_GLOBAL)
			if libErr == nil {
				return
			}
		}
	})
	return libHandle, libErr
}
