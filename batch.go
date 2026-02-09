//go:build (darwin && arm64) || (linux && amd64) || (linux && arm64)

package lfg

import "runtime"

// Batch wraps an lfg_batch for submitting tokens to encode/decode.
type Batch struct {
	c     cBatch
	owned bool // true if allocated via BatchInit (needs BatchFree)
}

// BatchGetOne creates a batch for a single sequence of tokens.
// The batch does not own the tokens slice — it must remain alive during Decode/Encode.
// The sequence ID is fixed to 0 and positions are tracked automatically.
func BatchGetOne(tokens []Token) Batch {
	if len(tokens) == 0 {
		return Batch{}
	}
	registerBatchFuncs()
	b := _lfg_batch_get_one(tokenPtr(tokens), int32(len(tokens)))
	return Batch{c: b, owned: false}
}

// BatchInit allocates a batch that can hold up to nTokens tokens.
// Each token can be assigned up to nSeqMax sequence IDs.
// If embd > 0, embeddings storage is allocated instead of token storage.
// Must be freed with Close().
func BatchInit(nTokens, embd, nSeqMax int) *Batch {
	registerBatchFuncs()
	b := _lfg_batch_init(int32(nTokens), int32(embd), int32(nSeqMax))
	batch := &Batch{c: b, owned: true}
	runtime.SetFinalizer(batch, func(b *Batch) { b.Close() })
	return batch
}

// Close frees the batch if it was allocated with BatchInit.
func (b *Batch) Close() {
	if b.owned {
		registerBatchFuncs()
		_lfg_batch_free(b.c)
		b.owned = false
		runtime.SetFinalizer(b, nil)
	}
}

// TokenCount returns the number of tokens in the batch.
func (b *Batch) TokenCount() int {
	return int(b.c.NTokens)
}

// SetTokenCount sets the number of tokens in the batch.
func (b *Batch) SetTokenCount(n int) {
	b.c.NTokens = int32(n)
}

// Decode processes a batch of tokens using the decoder with KV cache.
// Returns nil on success.
func (ctx *Context) Decode(batch Batch) error {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == 0 {
		return &Error{Code: ErrorInvalidArgument, Message: "context is closed"}
	}
	registerBatchFuncs()
	rc := _lfg_decode(ctx.c, batch.c)
	if rc != 0 {
		switch rc {
		case 1:
			return &Error{Code: ErrorInternal, Message: "could not find a KV slot for the batch"}
		case 2:
			return &Error{Code: ErrorCancelled, Message: "decode aborted"}
		case -1:
			return &Error{Code: ErrorInvalidArgument, Message: "invalid input batch"}
		default:
			if err := getLastError(); err != nil {
				return err
			}
			return &Error{Code: ErrorInternal, Message: "decode failed"}
		}
	}
	return nil
}

// Encode processes a batch of tokens using the encoder (no KV cache).
// For encoder-decoder models, stores encoder output for cross-attention.
func (ctx *Context) Encode(batch Batch) error {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == 0 {
		return &Error{Code: ErrorInvalidArgument, Message: "context is closed"}
	}
	registerBatchFuncs()
	rc := _lfg_encode(ctx.c, batch.c)
	if rc != 0 {
		if err := getLastError(); err != nil {
			return err
		}
		return &Error{Code: ErrorInternal, Message: "encode failed"}
	}
	return nil
}
