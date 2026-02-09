.PHONY: build test vet clean

build:
	CGO_ENABLED=0 go build ./...

test:
	CGO_ENABLED=0 go test -count=1 ./...

# unsafeptr=false: purego requires unsafe.Pointer(uintptr) conversions
# to dereference C pointers in callbacks — these are safe and expected.
vet:
	CGO_ENABLED=0 go vet -unsafeptr=false ./...

clean:
	go clean ./...
