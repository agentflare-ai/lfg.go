package lfg

/*
typedef struct lfm_model lfm_model;
typedef struct lfm_context lfm_context;
typedef struct lfm_vocab lfm_vocab;
typedef struct lfm_sampler lfm_sampler;
#include "lfm_inference.h"
*/
import "C"
import (
	"runtime"
	"sync"
	"unsafe"
)

// ContextOption configures context creation parameters.
type ContextOption func(*C.struct_lfm_context_params)

// WithNCtx sets the text context size. 0 means use model default.
func WithNCtx(n uint32) ContextOption {
	return func(p *C.struct_lfm_context_params) {
		p.n_ctx = C.uint32_t(n)
	}
}

// WithNBatch sets the maximum logical batch size for lfm_decode.
func WithNBatch(n uint32) ContextOption {
	return func(p *C.struct_lfm_context_params) {
		p.n_batch = C.uint32_t(n)
	}
}

// WithNUBatch sets the physical maximum batch size.
func WithNUBatch(n uint32) ContextOption {
	return func(p *C.struct_lfm_context_params) {
		p.n_ubatch = C.uint32_t(n)
	}
}

// WithNSeqMax sets the maximum number of sequences.
func WithNSeqMax(n uint32) ContextOption {
	return func(p *C.struct_lfm_context_params) {
		p.n_seq_max = C.uint32_t(n)
	}
}

// WithNThreads sets the number of threads for generation.
func WithNThreads(n int) ContextOption {
	return func(p *C.struct_lfm_context_params) {
		p.n_threads = C.int32_t(n)
	}
}

// WithNThreadsBatch sets the number of threads for batch processing.
func WithNThreadsBatch(n int) ContextOption {
	return func(p *C.struct_lfm_context_params) {
		p.n_threads_batch = C.int32_t(n)
	}
}

// WithRopeScaling sets the RoPE scaling type.
func WithRopeScaling(t RopeScalingType) ContextOption {
	return func(p *C.struct_lfm_context_params) {
		p.rope_scaling_type = C.enum_lfm_rope_scaling_type(t)
	}
}

// WithPoolingType sets the embedding pooling type.
func WithPoolingType(t PoolingType) ContextOption {
	return func(p *C.struct_lfm_context_params) {
		p.pooling_type = C.enum_lfm_pooling_type(t)
	}
}

// WithAttentionType sets the attention type for embeddings.
func WithAttentionType(t AttentionType) ContextOption {
	return func(p *C.struct_lfm_context_params) {
		p.attention_type = C.enum_lfm_attention_type(t)
	}
}

// WithFlashAttn sets the flash attention type.
func WithFlashAttn(t FlashAttnType) ContextOption {
	return func(p *C.struct_lfm_context_params) {
		p.flash_attn_type = C.enum_lfm_flash_attn_type(t)
	}
}

// WithRopeFreqBase sets the RoPE base frequency.
func WithRopeFreqBase(f float32) ContextOption {
	return func(p *C.struct_lfm_context_params) {
		p.rope_freq_base = C.float(f)
	}
}

// WithRopeFreqScale sets the RoPE frequency scaling factor.
func WithRopeFreqScale(f float32) ContextOption {
	return func(p *C.struct_lfm_context_params) {
		p.rope_freq_scale = C.float(f)
	}
}

// WithEmbeddings enables extraction of embeddings together with logits.
func WithEmbeddings(v bool) ContextOption {
	return func(p *C.struct_lfm_context_params) {
		p.embeddings = C.bool(v)
	}
}

// WithOffloadKQV offloads KQV operations to GPU.
func WithOffloadKQV(v bool) ContextOption {
	return func(p *C.struct_lfm_context_params) {
		p.offload_kqv = C.bool(v)
	}
}

// WithNoPerf disables performance timing measurements.
func WithNoPerf(v bool) ContextOption {
	return func(p *C.struct_lfm_context_params) {
		p.no_perf = C.bool(v)
	}
}

// Context wraps an lfm_context pointer with thread-safe access.
type Context struct {
	mu    sync.RWMutex
	c     *C.struct_lfm_context
	model *Model // prevent GC of parent
}

// NewContext creates a new inference context from a model. Automatically initializes the backend.
func NewContext(model *Model, opts ...ContextOption) (*Context, error) {
	ensureBackend()

	model.mu.RLock()
	if model.c == nil {
		model.mu.RUnlock()
		return nil, &Error{Code: ErrorInvalidArgument, Message: "model is closed"}
	}

	params := C.lfm_context_default_params()
	for _, opt := range opts {
		opt(&params)
	}

	cCtx := C.lfm_init_from_model(model.c, params)
	model.mu.RUnlock()

	if cCtx == nil {
		if err := getLastError(); err != nil {
			return nil, err
		}
		return nil, &Error{Code: ErrorInternal, Message: "failed to create context"}
	}

	ctx := &Context{c: cCtx, model: model}
	runtime.SetFinalizer(ctx, func(ctx *Context) { ctx.Close() })
	return ctx, nil
}

// Close frees the context resources. Safe to call multiple times.
func (ctx *Context) Close() error {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c != nil {
		C.lfm_free(ctx.c)
		ctx.c = nil
		runtime.SetFinalizer(ctx, nil)
	}
	return nil
}

// Model returns the model associated with this context.
func (ctx *Context) Model() *Model {
	return ctx.model
}

// NCtx returns the actual context size.
func (ctx *Context) NCtx() uint32 {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == nil {
		return 0
	}
	return uint32(C.lfm_n_ctx(ctx.c))
}

// NCtxSeq returns the per-sequence context size.
func (ctx *Context) NCtxSeq() uint32 {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == nil {
		return 0
	}
	return uint32(C.lfm_n_ctx_seq(ctx.c))
}

// NBatch returns the actual batch size.
func (ctx *Context) NBatch() uint32 {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == nil {
		return 0
	}
	return uint32(C.lfm_n_batch(ctx.c))
}

// NUBatch returns the actual micro-batch size.
func (ctx *Context) NUBatch() uint32 {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == nil {
		return 0
	}
	return uint32(C.lfm_n_ubatch(ctx.c))
}

// NSeqMax returns the maximum number of sequences.
func (ctx *Context) NSeqMax() uint32 {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == nil {
		return 0
	}
	return uint32(C.lfm_n_seq_max(ctx.c))
}

// SetNThreads sets the number of threads for generation and batch processing.
func (ctx *Context) SetNThreads(nThreads, nThreadsBatch int) {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == nil {
		return
	}
	C.lfm_set_n_threads(ctx.c, C.int32_t(nThreads), C.int32_t(nThreadsBatch))
}

// NThreads returns the number of threads used for generation.
func (ctx *Context) NThreads() int {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == nil {
		return 0
	}
	return int(C.lfm_n_threads(ctx.c))
}

// NThreadsBatch returns the number of threads used for batch processing.
func (ctx *Context) NThreadsBatch() int {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == nil {
		return 0
	}
	return int(C.lfm_n_threads_batch(ctx.c))
}

// SetEmbeddings sets whether the context outputs embeddings.
func (ctx *Context) SetEmbeddings(v bool) {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == nil {
		return
	}
	C.lfm_set_embeddings(ctx.c, C.bool(v))
}

// SetCausalAttn sets whether to use causal attention.
func (ctx *Context) SetCausalAttn(v bool) {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == nil {
		return
	}
	C.lfm_set_causal_attn(ctx.c, C.bool(v))
}

// SetWarmup sets whether the model is in warmup mode.
func (ctx *Context) SetWarmup(v bool) {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == nil {
		return
	}
	C.lfm_set_warmup(ctx.c, C.bool(v))
}

// Synchronize waits until all computations are finished.
func (ctx *Context) Synchronize() {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == nil {
		return
	}
	C.lfm_synchronize(ctx.c)
}

// GetPoolingType returns the pooling type of the context.
func (ctx *Context) GetPoolingType() PoolingType {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == nil {
		return PoolingTypeUnspecified
	}
	return PoolingType(C.lfm_pooling_type(ctx.c))
}

// GetLogits returns all output logits from the last decode call.
// The returned slice is a view into C memory and is only valid until the next decode call.
func (ctx *Context) GetLogits() []float32 {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == nil {
		return nil
	}
	ptr := C.lfm_get_logits(ctx.c)
	if ptr == nil {
		return nil
	}
	nVocab := int(C.lfm_vocab_n_tokens(C.lfm_model_get_vocab(C.lfm_get_model(ctx.c))))
	return unsafe.Slice((*float32)(unsafe.Pointer(ptr)), nVocab)
}

// GetLogitsIth returns logits for the ith token. Negative indices access in reverse (-1 = last).
// The returned slice is a view into C memory and is only valid until the next decode call.
func (ctx *Context) GetLogitsIth(i int) []float32 {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == nil {
		return nil
	}
	ptr := C.lfm_get_logits_ith(ctx.c, C.int32_t(i))
	if ptr == nil {
		return nil
	}
	nVocab := int(C.lfm_vocab_n_tokens(C.lfm_model_get_vocab(C.lfm_get_model(ctx.c))))
	return unsafe.Slice((*float32)(unsafe.Pointer(ptr)), nVocab)
}

// GetEmbeddings returns all output embeddings from the last decode call.
// The returned slice is a view into C memory.
func (ctx *Context) GetEmbeddings() []float32 {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == nil {
		return nil
	}
	ptr := C.lfm_get_embeddings(ctx.c)
	if ptr == nil {
		return nil
	}
	nEmbd := int(C.lfm_model_n_embd(C.lfm_get_model(ctx.c)))
	return unsafe.Slice((*float32)(unsafe.Pointer(ptr)), nEmbd)
}

// GetEmbeddingsIth returns embeddings for the ith token.
// The returned slice is a view into C memory.
func (ctx *Context) GetEmbeddingsIth(i int) []float32 {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == nil {
		return nil
	}
	ptr := C.lfm_get_embeddings_ith(ctx.c, C.int32_t(i))
	if ptr == nil {
		return nil
	}
	nEmbd := int(C.lfm_model_n_embd(C.lfm_get_model(ctx.c)))
	return unsafe.Slice((*float32)(unsafe.Pointer(ptr)), nEmbd)
}

// GetEmbeddingsSeq returns embeddings for a sequence ID.
// The returned slice is a view into C memory.
func (ctx *Context) GetEmbeddingsSeq(seqID SeqID) []float32 {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == nil {
		return nil
	}
	ptr := C.lfm_get_embeddings_seq(ctx.c, C.lfm_seq_id(seqID))
	if ptr == nil {
		return nil
	}
	nEmbd := int(C.lfm_model_n_embd(C.lfm_get_model(ctx.c)))
	return unsafe.Slice((*float32)(unsafe.Pointer(ptr)), nEmbd)
}
