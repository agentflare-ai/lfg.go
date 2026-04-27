//go:build (darwin && arm64) || (linux && amd64) || (linux && arm64)

package lfg

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

// byteToBool converts a C struct bool byte field to a Go bool.
func byteToBool(b byte) bool {
	return b != 0
}

// Context wraps an lfg_context pointer with thread-safe access.
type Context struct {
	mu    sync.RWMutex
	c     uintptr // *lfg_context (C pointer as uintptr)
	model *Model  // prevent GC of parent
}

// NewContext creates a new inference context from a model. Automatically initializes the backend.
func NewContext(model *Model, opts ...ContextOption) (*Context, error) {
	ensureBackend()
	registerContextFuncs()
	registerModelFuncs()
	registerVocabFuncs()

	if model == nil {
		return nil, &Error{Code: ErrorInvalidArgument, Message: "model is nil"}
	}

	model.mu.RLock()
	if model.c == 0 {
		model.mu.RUnlock()
		return nil, &Error{Code: ErrorInvalidArgument, Message: "model is closed"}
	}

	defaults := _lfg_context_default_params()
	cfg := ContextConfig{
		ContextSize:    defaults.NCtx,
		BatchSize:      defaults.NBatch,
		MicroBatchSize: defaults.NUBatch,
		MaxSequences:   defaults.NSeqMax,
		Threads:        int(defaults.NThreads),
		ThreadsBatch:   int(defaults.NThreadsBatch),
		RopeScaling:    RopeScalingType(defaults.RopeScalingType),
		PoolingType:    PoolingType(defaults.PoolingType),
		AttentionType:  AttentionType(defaults.AttentionType),
		FlashAttnType:  FlashAttnType(defaults.FlashAttnType),
		RopeFreqBase:   defaults.RopeFreqBase,
		RopeFreqScale:  defaults.RopeFreqScale,
		Embeddings:     byteToBool(defaults.Embeddings),
		OffloadKQV:     byteToBool(defaults.OffloadKQV),
		NoPerf:         byteToBool(defaults.NoPerf),
	}
	for _, opt := range opts {
		opt(&cfg)
	}

	// Convert Go config back to C params.
	params := defaults
	params.NCtx = cfg.ContextSize
	params.NBatch = cfg.BatchSize
	params.NUBatch = cfg.MicroBatchSize
	params.NSeqMax = cfg.MaxSequences
	params.NThreads = int32(cfg.Threads)
	params.NThreadsBatch = int32(cfg.ThreadsBatch)
	params.RopeScalingType = int32(cfg.RopeScaling)
	params.PoolingType = int32(cfg.PoolingType)
	params.AttentionType = int32(cfg.AttentionType)
	params.FlashAttnType = int32(cfg.FlashAttnType)
	params.RopeFreqBase = cfg.RopeFreqBase
	params.RopeFreqScale = cfg.RopeFreqScale
	params.Embeddings = boolToByte(cfg.Embeddings)
	params.OffloadKQV = boolToByte(cfg.OffloadKQV)
	params.NoPerf = boolToByte(cfg.NoPerf)

	cCtx := _lfg_init_from_model(model.c, params)
	model.mu.RUnlock()

	if cCtx == 0 {
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
	if ctx.c != 0 {
		registerContextFuncs()
		_lfg_free(ctx.c)
		ctx.c = 0
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
	if ctx.c == 0 {
		return 0
	}
	registerContextFuncs()
	return _lfg_n_ctx(ctx.c)
}

// SequenceContextSize returns the per-sequence context size.
func (ctx *Context) SequenceContextSize() uint32 {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == 0 {
		return 0
	}
	registerContextFuncs()
	return _lfg_n_ctx_seq(ctx.c)
}

// BatchSize returns the actual batch size.
func (ctx *Context) BatchSize() uint32 {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == 0 {
		return 0
	}
	registerContextFuncs()
	return _lfg_n_batch(ctx.c)
}

// MicroBatchSize returns the actual micro-batch size.
func (ctx *Context) MicroBatchSize() uint32 {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == 0 {
		return 0
	}
	registerContextFuncs()
	return _lfg_n_ubatch(ctx.c)
}

// MaxSequences returns the maximum number of sequences.
func (ctx *Context) MaxSequences() uint32 {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == 0 {
		return 0
	}
	registerContextFuncs()
	return _lfg_n_seq_max(ctx.c)
}

// SetThreads sets the number of threads for generation and batch processing.
func (ctx *Context) SetThreads(nThreads, nThreadsBatch int) {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == 0 {
		return
	}
	registerContextFuncs()
	_lfg_set_n_threads(ctx.c, int32(nThreads), int32(nThreadsBatch))
}

// ThreadCount returns the number of threads used for generation.
func (ctx *Context) ThreadCount() int {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == 0 {
		return 0
	}
	registerContextFuncs()
	return int(_lfg_n_threads(ctx.c))
}

// BatchThreadCount returns the number of threads used for batch processing.
func (ctx *Context) BatchThreadCount() int {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == 0 {
		return 0
	}
	registerContextFuncs()
	return int(_lfg_n_threads_batch(ctx.c))
}

// SetEmbeddings sets whether the context outputs embeddings.
func (ctx *Context) SetEmbeddings(v bool) {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == 0 {
		return
	}
	registerContextFuncs()
	_lfg_set_embeddings(ctx.c, v)
}

// SetCausalAttn sets whether to use causal attention.
func (ctx *Context) SetCausalAttn(v bool) {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == 0 {
		return
	}
	registerContextFuncs()
	_lfg_set_causal_attn(ctx.c, v)
}

// SetWarmup sets whether the model is in warmup mode.
func (ctx *Context) SetWarmup(v bool) {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == 0 {
		return
	}
	registerContextFuncs()
	_lfg_set_warmup(ctx.c, v)
}

// Synchronize waits until all computations are finished.
func (ctx *Context) Synchronize() {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == 0 {
		return
	}
	registerContextFuncs()
	_lfg_synchronize(ctx.c)
}

// GetPoolingType returns the pooling type of the context.
func (ctx *Context) GetPoolingType() PoolingType {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == 0 {
		return PoolingTypeUnspecified
	}
	registerContextFuncs()
	return PoolingType(_lfg_pooling_type(ctx.c))
}

// Logits returns all output logits from the last decode call.
// The returned slice is a view into C memory and is only valid until the next decode call.
func (ctx *Context) Logits() []float32 {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == 0 {
		return nil
	}
	registerContextFuncs()
	registerModelFuncs()
	registerVocabFuncs()
	ptr := _lfg_get_logits(ctx.c)
	if ptr == 0 {
		return nil
	}
	nVocab := int(_lfg_vocab_n_tokens(_lfg_model_get_vocab(_lfg_get_model(ctx.c))))
	return unsafe.Slice((*float32)(unsafe.Pointer(ptr)), nVocab)
}

// LogitsAt returns logits for the ith token. Negative indices access in reverse (-1 = last).
// The returned slice is a view into C memory and is only valid until the next decode call.
func (ctx *Context) LogitsAt(i int) []float32 {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == 0 {
		return nil
	}
	registerContextFuncs()
	registerModelFuncs()
	registerVocabFuncs()
	ptr := _lfg_get_logits_ith(ctx.c, int32(i))
	if ptr == 0 {
		return nil
	}
	nVocab := int(_lfg_vocab_n_tokens(_lfg_model_get_vocab(_lfg_get_model(ctx.c))))
	return unsafe.Slice((*float32)(unsafe.Pointer(ptr)), nVocab)
}

// Embeddings returns all output embeddings from the last decode call.
// The returned slice is a view into C memory.
func (ctx *Context) Embeddings() []float32 {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == 0 {
		return nil
	}
	registerContextFuncs()
	registerModelFuncs()
	ptr := _lfg_get_embeddings(ctx.c)
	if ptr == 0 {
		return nil
	}
	nEmbd := int(_lfg_model_n_embd(_lfg_get_model(ctx.c)))
	return unsafe.Slice((*float32)(unsafe.Pointer(ptr)), nEmbd)
}

// EmbeddingsAt returns embeddings for the ith token.
// The returned slice is a view into C memory.
func (ctx *Context) EmbeddingsAt(i int) []float32 {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == 0 {
		return nil
	}
	registerContextFuncs()
	registerModelFuncs()
	ptr := _lfg_get_embeddings_ith(ctx.c, int32(i))
	if ptr == 0 {
		return nil
	}
	nEmbd := int(_lfg_model_n_embd(_lfg_get_model(ctx.c)))
	return unsafe.Slice((*float32)(unsafe.Pointer(ptr)), nEmbd)
}

// SequenceEmbeddings returns embeddings for a sequence ID.
// The returned slice is a view into C memory.
func (ctx *Context) SequenceEmbeddings(seqID SequenceID) []float32 {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == 0 {
		return nil
	}
	registerContextFuncs()
	registerModelFuncs()
	ptr := _lfg_get_embeddings_seq(ctx.c, int32(seqID))
	if ptr == 0 {
		return nil
	}
	nEmbd := int(_lfg_model_n_embd(_lfg_get_model(ctx.c)))
	return unsafe.Slice((*float32)(unsafe.Pointer(ptr)), nEmbd)
}

// SetSampler binds a sampler to a sequence for backend-side sampling.
// This is an EXPERIMENTAL API. The sampler is not owned by the context;
// the caller must keep it alive and free it separately.
func (ctx *Context) SetSampler(seqID SequenceID, sampler *Sampler) bool {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == 0 || sampler == nil {
		return false
	}
	sampler.mu.Lock()
	defer sampler.mu.Unlock()
	if sampler.c == 0 {
		return false
	}
	registerContextFuncs()
	return _lfg_set_sampler(ctx.c, int32(seqID), sampler.c)
}

// SampledToken returns the backend-sampled token at position i after a decode
// call with backend sampling enabled via SetSampler.
func (ctx *Context) SampledToken(i int) Token {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == 0 {
		return InvalidToken
	}
	registerContextFuncs()
	return Token(_lfg_get_sampled_token_ith(ctx.c, int32(i)))
}

// SampledProbs returns the backend-sampled probability distribution at position i.
// The returned slice is a view into C memory and is only valid until the next decode call.
func (ctx *Context) SampledProbs(i int) []float32 {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == 0 {
		return nil
	}
	registerContextFuncs()
	n := _lfg_get_sampled_probs_count_ith(ctx.c, int32(i))
	if n == 0 {
		return nil
	}
	ptr := _lfg_get_sampled_probs_ith(ctx.c, int32(i))
	if ptr == 0 {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(ptr)), int(n))
}

// SampledLogits returns the backend-sampled logits at position i.
// The returned slice is a view into C memory and is only valid until the next decode call.
func (ctx *Context) SampledLogits(i int) []float32 {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == 0 {
		return nil
	}
	registerContextFuncs()
	n := _lfg_get_sampled_logits_count_ith(ctx.c, int32(i))
	if n == 0 {
		return nil
	}
	ptr := _lfg_get_sampled_logits_ith(ctx.c, int32(i))
	if ptr == 0 {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(ptr)), int(n))
}

// SampledCandidates returns the backend-sampled candidate token IDs at position i.
// The returned slice is a view into C memory and is only valid until the next decode call.
func (ctx *Context) SampledCandidates(i int) []Token {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == 0 {
		return nil
	}
	registerContextFuncs()
	n := _lfg_get_sampled_candidates_count_ith(ctx.c, int32(i))
	if n == 0 {
		return nil
	}
	ptr := _lfg_get_sampled_candidates_ith(ctx.c, int32(i))
	if ptr == 0 {
		return nil
	}
	return unsafe.Slice((*Token)(unsafe.Pointer(ptr)), int(n))
}
