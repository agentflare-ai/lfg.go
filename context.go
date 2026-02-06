package lfg

/*
#include "lfg_inference.h"
*/
import "C"
import (
	"runtime"
	"sync"
	"unsafe"
)

// ContextConfig holds Go-native context creation parameters.
type ContextConfig struct {
	ContextSize    uint32
	BatchSize      uint32
	MicroBatchSize uint32
	MaxSequences   uint32
	Threads        int
	ThreadsBatch   int
	RopeScaling    RopeScalingType
	PoolingType    PoolingType
	AttentionType  AttentionType
	FlashAttnType  FlashAttnType
	RopeFreqBase   float32
	RopeFreqScale  float32
	Embeddings     bool
	OffloadKQV     bool
	NoPerf         bool
}

// ContextOption configures context creation parameters.
type ContextOption func(*ContextConfig)

// WithContextSize sets the text context size. 0 means use model default.
func WithContextSize(n uint32) ContextOption {
	return func(cfg *ContextConfig) {
		cfg.ContextSize = n
	}
}

// WithBatchSize sets the maximum logical batch size for decode.
func WithBatchSize(n uint32) ContextOption {
	return func(cfg *ContextConfig) {
		cfg.BatchSize = n
	}
}

// WithMicroBatchSize sets the physical maximum batch size.
func WithMicroBatchSize(n uint32) ContextOption {
	return func(cfg *ContextConfig) {
		cfg.MicroBatchSize = n
	}
}

// WithMaxSequences sets the maximum number of sequences.
func WithMaxSequences(n uint32) ContextOption {
	return func(cfg *ContextConfig) {
		cfg.MaxSequences = n
	}
}

// WithThreads sets the number of threads for generation.
func WithThreads(n int) ContextOption {
	return func(cfg *ContextConfig) {
		cfg.Threads = n
	}
}

// WithThreadsBatch sets the number of threads for batch processing.
func WithThreadsBatch(n int) ContextOption {
	return func(cfg *ContextConfig) {
		cfg.ThreadsBatch = n
	}
}

// WithRopeScaling sets the RoPE scaling type.
func WithRopeScaling(t RopeScalingType) ContextOption {
	return func(cfg *ContextConfig) {
		cfg.RopeScaling = t
	}
}

// WithPoolingType sets the embedding pooling type.
func WithPoolingType(t PoolingType) ContextOption {
	return func(cfg *ContextConfig) {
		cfg.PoolingType = t
	}
}

// WithAttentionType sets the attention type for embeddings.
func WithAttentionType(t AttentionType) ContextOption {
	return func(cfg *ContextConfig) {
		cfg.AttentionType = t
	}
}

// WithFlashAttn sets the flash attention type.
func WithFlashAttn(t FlashAttnType) ContextOption {
	return func(cfg *ContextConfig) {
		cfg.FlashAttnType = t
	}
}

// WithRopeFreqBase sets the RoPE base frequency.
func WithRopeFreqBase(f float32) ContextOption {
	return func(cfg *ContextConfig) {
		cfg.RopeFreqBase = f
	}
}

// WithRopeFreqScale sets the RoPE frequency scaling factor.
func WithRopeFreqScale(f float32) ContextOption {
	return func(cfg *ContextConfig) {
		cfg.RopeFreqScale = f
	}
}

// WithEmbeddings enables extraction of embeddings together with logits.
func WithEmbeddings(v bool) ContextOption {
	return func(cfg *ContextConfig) {
		cfg.Embeddings = v
	}
}

// WithOffloadKQV offloads KQV operations to GPU.
func WithOffloadKQV(v bool) ContextOption {
	return func(cfg *ContextConfig) {
		cfg.OffloadKQV = v
	}
}

// WithNoPerf disables performance timing measurements.
func WithNoPerf(v bool) ContextOption {
	return func(cfg *ContextConfig) {
		cfg.NoPerf = v
	}
}

// Context wraps an lfg_context pointer with thread-safe access.
type Context struct {
	mu    sync.RWMutex
	c     *C.struct_lfg_context
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

	defaults := C.lfg_context_default_params()
	cfg := ContextConfig{
		ContextSize:    uint32(defaults.n_ctx),
		BatchSize:      uint32(defaults.n_batch),
		MicroBatchSize: uint32(defaults.n_ubatch),
		MaxSequences:   uint32(defaults.n_seq_max),
		Threads:        int(defaults.n_threads),
		ThreadsBatch:   int(defaults.n_threads_batch),
		RopeScaling:    RopeScalingType(defaults.rope_scaling_type),
		PoolingType:    PoolingType(defaults.pooling_type),
		AttentionType:  AttentionType(defaults.attention_type),
		FlashAttnType:  FlashAttnType(defaults.flash_attn_type),
		RopeFreqBase:   float32(defaults.rope_freq_base),
		RopeFreqScale:  float32(defaults.rope_freq_scale),
		Embeddings:     bool(defaults.embeddings),
		OffloadKQV:     bool(defaults.offload_kqv),
		NoPerf:         bool(defaults.no_perf),
	}
	for _, opt := range opts {
		opt(&cfg)
	}

	// Convert Go config back to C params.
	params := defaults
	params.n_ctx = C.uint32_t(cfg.ContextSize)
	params.n_batch = C.uint32_t(cfg.BatchSize)
	params.n_ubatch = C.uint32_t(cfg.MicroBatchSize)
	params.n_seq_max = C.uint32_t(cfg.MaxSequences)
	params.n_threads = C.int32_t(cfg.Threads)
	params.n_threads_batch = C.int32_t(cfg.ThreadsBatch)
	params.rope_scaling_type = C.enum_lfg_rope_scaling_type(cfg.RopeScaling)
	params.pooling_type = C.enum_lfg_pooling_type(cfg.PoolingType)
	params.attention_type = C.enum_lfg_attention_type(cfg.AttentionType)
	params.flash_attn_type = C.enum_lfg_flash_attn_type(cfg.FlashAttnType)
	params.rope_freq_base = C.float(cfg.RopeFreqBase)
	params.rope_freq_scale = C.float(cfg.RopeFreqScale)
	params.embeddings = C.bool(cfg.Embeddings)
	params.offload_kqv = C.bool(cfg.OffloadKQV)
	params.no_perf = C.bool(cfg.NoPerf)

	cCtx := C.lfg_init_from_model(model.c, params)
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
		C.lfg_free(ctx.c)
		ctx.c = nil
		runtime.SetFinalizer(ctx, nil)
	}
	return nil
}

// Model returns the model associated with this context.
func (ctx *Context) Model() *Model {
	return ctx.model
}

// ContextSize returns the actual context size.
func (ctx *Context) ContextSize() uint32 {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == nil {
		return 0
	}
	return uint32(C.lfg_n_ctx(ctx.c))
}

// SequenceContextSize returns the per-sequence context size.
func (ctx *Context) SequenceContextSize() uint32 {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == nil {
		return 0
	}
	return uint32(C.lfg_n_ctx_seq(ctx.c))
}

// BatchSize returns the actual batch size.
func (ctx *Context) BatchSize() uint32 {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == nil {
		return 0
	}
	return uint32(C.lfg_n_batch(ctx.c))
}

// MicroBatchSize returns the actual micro-batch size.
func (ctx *Context) MicroBatchSize() uint32 {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == nil {
		return 0
	}
	return uint32(C.lfg_n_ubatch(ctx.c))
}

// MaxSequences returns the maximum number of sequences.
func (ctx *Context) MaxSequences() uint32 {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == nil {
		return 0
	}
	return uint32(C.lfg_n_seq_max(ctx.c))
}

// SetThreads sets the number of threads for generation and batch processing.
func (ctx *Context) SetThreads(nThreads, nThreadsBatch int) {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == nil {
		return
	}
	C.lfg_set_n_threads(ctx.c, C.int32_t(nThreads), C.int32_t(nThreadsBatch))
}

// ThreadCount returns the number of threads used for generation.
func (ctx *Context) ThreadCount() int {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == nil {
		return 0
	}
	return int(C.lfg_n_threads(ctx.c))
}

// BatchThreadCount returns the number of threads used for batch processing.
func (ctx *Context) BatchThreadCount() int {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == nil {
		return 0
	}
	return int(C.lfg_n_threads_batch(ctx.c))
}

// SetEmbeddings sets whether the context outputs embeddings.
func (ctx *Context) SetEmbeddings(v bool) {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == nil {
		return
	}
	C.lfg_set_embeddings(ctx.c, C.bool(v))
}

// SetCausalAttn sets whether to use causal attention.
func (ctx *Context) SetCausalAttn(v bool) {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == nil {
		return
	}
	C.lfg_set_causal_attn(ctx.c, C.bool(v))
}

// SetWarmup sets whether the model is in warmup mode.
func (ctx *Context) SetWarmup(v bool) {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == nil {
		return
	}
	C.lfg_set_warmup(ctx.c, C.bool(v))
}

// Synchronize waits until all computations are finished.
func (ctx *Context) Synchronize() {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == nil {
		return
	}
	C.lfg_synchronize(ctx.c)
}

// PoolingType returns the pooling type of the context.
func (ctx *Context) GetPoolingType() PoolingType {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == nil {
		return PoolingTypeUnspecified
	}
	return PoolingType(C.lfg_pooling_type(ctx.c))
}

// Logits returns all output logits from the last decode call.
// The returned slice is a view into C memory and is only valid until the next decode call.
func (ctx *Context) Logits() []float32 {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == nil {
		return nil
	}
	ptr := C.lfg_get_logits(ctx.c)
	if ptr == nil {
		return nil
	}
	nVocab := int(C.lfg_vocab_n_tokens(C.lfg_model_get_vocab(C.lfg_get_model(ctx.c))))
	return unsafe.Slice((*float32)(unsafe.Pointer(ptr)), nVocab)
}

// LogitsAt returns logits for the ith token. Negative indices access in reverse (-1 = last).
// The returned slice is a view into C memory and is only valid until the next decode call.
func (ctx *Context) LogitsAt(i int) []float32 {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == nil {
		return nil
	}
	ptr := C.lfg_get_logits_ith(ctx.c, C.int32_t(i))
	if ptr == nil {
		return nil
	}
	nVocab := int(C.lfg_vocab_n_tokens(C.lfg_model_get_vocab(C.lfg_get_model(ctx.c))))
	return unsafe.Slice((*float32)(unsafe.Pointer(ptr)), nVocab)
}

// Embeddings returns all output embeddings from the last decode call.
// The returned slice is a view into C memory.
func (ctx *Context) Embeddings() []float32 {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == nil {
		return nil
	}
	ptr := C.lfg_get_embeddings(ctx.c)
	if ptr == nil {
		return nil
	}
	nEmbd := int(C.lfg_model_n_embd(C.lfg_get_model(ctx.c)))
	return unsafe.Slice((*float32)(unsafe.Pointer(ptr)), nEmbd)
}

// EmbeddingsAt returns embeddings for the ith token.
// The returned slice is a view into C memory.
func (ctx *Context) EmbeddingsAt(i int) []float32 {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == nil {
		return nil
	}
	ptr := C.lfg_get_embeddings_ith(ctx.c, C.int32_t(i))
	if ptr == nil {
		return nil
	}
	nEmbd := int(C.lfg_model_n_embd(C.lfg_get_model(ctx.c)))
	return unsafe.Slice((*float32)(unsafe.Pointer(ptr)), nEmbd)
}

// SequenceEmbeddings returns embeddings for a sequence ID.
// The returned slice is a view into C memory.
func (ctx *Context) SequenceEmbeddings(seqID SequenceID) []float32 {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == nil {
		return nil
	}
	ptr := C.lfg_get_embeddings_seq(ctx.c, C.lfg_seq_id(seqID))
	if ptr == nil {
		return nil
	}
	nEmbd := int(C.lfg_model_n_embd(C.lfg_get_model(ctx.c)))
	return unsafe.Slice((*float32)(unsafe.Pointer(ptr)), nEmbd)
}
