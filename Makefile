.PHONY: build build-linux-amd64 build-linux-arm64 build-macos-arm64 clean

ZIG_CC := zig cc
ZIG_CXX := zig c++

build: build-macos-arm64

build-linux-amd64:
	CGO_ENABLED=1 GOOS=linux GOARCH=amd64 CC="$(ZIG_CC) -target x86_64-linux-gnu" CXX="$(ZIG_CXX) -target x86_64-linux-gnu" go build -v -o lfg-linux-amd64 .

build-linux-arm64:
	CGO_ENABLED=1 GOOS=linux GOARCH=arm64 CC="$(ZIG_CC) -target aarch64-linux-gnu" CXX="$(ZIG_CXX) -target aarch64-linux-gnu" go build -v -o lfg-linux-arm64 .

build-macos-arm64:
	CGO_ENABLED=1 GOOS=darwin GOARCH=arm64 CC="$(ZIG_CC)" CXX="$(ZIG_CXX)" go build -v -o lfg-macos-arm64 .

clean:
	rm -f lfg-linux-amd64 lfg-linux-arm64 lfg-macos-arm64
