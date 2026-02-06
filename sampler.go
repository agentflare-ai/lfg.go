package lfg

/*
typedef struct lfm_model lfm_model;
typedef struct lfm_context lfm_context;
typedef struct lfm_vocab lfm_vocab;
typedef struct lfm_sampler lfm_sampler;
#include "lfm_inference.h"
#include <stdlib.h>
*/
import "C"
import (
	"runtime"
	"sync"
	"unsafe"
)

// Sampler wraps an lfm_sampler pointer with thread-safe access.
type Sampler struct {
	mu    sync.Mutex
	c     *C.struct_lfm_sampler
	owned bool // false if ownership was transferred to a chain
}

// newSampler wraps a C sampler pointer.
func newSampler(c *C.struct_lfm_sampler) *Sampler {
	if c == nil {
		return nil
	}
	s := &Sampler{c: c, owned: true}
	runtime.SetFinalizer(s, func(s *Sampler) { s.Close() })
	return s
}

// Close frees the sampler. Safe to call multiple times.
// Do NOT call Close on a sampler that has been added to a chain.
func (s *Sampler) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c != nil && s.owned {
		C.lfm_sampler_free(s.c)
		s.c = nil
		runtime.SetFinalizer(s, nil)
	}
}

// Name returns the sampler's name.
func (s *Sampler) Name() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return ""
	}
	return C.GoString(C.lfm_sampler_name(s.c))
}

// Accept notifies the sampler that a token has been selected.
func (s *Sampler) Accept(token Token) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return
	}
	C.lfm_sampler_accept(s.c, C.lfm_token(token))
}

// Reset resets the sampler state.
func (s *Sampler) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return
	}
	C.lfm_sampler_reset(s.c)
}

// Clone creates a copy of the sampler.
func (s *Sampler) Clone() *Sampler {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return nil
	}
	return newSampler(C.lfm_sampler_clone(s.c))
}

// GetSeed returns the seed used by the sampler, or DefaultSeed if not applicable.
func (s *Sampler) GetSeed() uint32 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return DefaultSeed
	}
	return uint32(C.lfm_sampler_get_seed(s.c))
}

// Sample samples and accepts a token from the idx-th output of the last evaluation.
func (s *Sampler) Sample(ctx *Context, idx int) Token {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil || ctx == nil {
		return TokenNull
	}
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == nil {
		return TokenNull
	}
	return Token(C.lfm_sampler_sample(s.c, ctx.c, C.int32_t(idx)))
}

// --- Sampler Chain ---

// NewSamplerChain creates a new sampler chain.
// If noPerf is true, performance timings are not measured.
func NewSamplerChain(noPerf bool) *Sampler {
	params := C.lfm_sampler_chain_default_params()
	params.no_perf = C.bool(noPerf)
	return newSampler(C.lfm_sampler_chain_init(params))
}

// Add appends a sampler to the chain. The chain takes ownership of the sampler.
func (s *Sampler) Add(child *Sampler) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil || child == nil {
		return
	}
	child.mu.Lock()
	defer child.mu.Unlock()
	if child.c == nil {
		return
	}
	C.lfm_sampler_chain_add(s.c, child.c)
	child.owned = false
	runtime.SetFinalizer(child, nil)
}

// ChainGet returns the sampler at index i in the chain.
// If i == -1, returns the chain itself (useful to check if it's a chain).
func (s *Sampler) ChainGet(i int) *C.struct_lfm_sampler {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return nil
	}
	return C.lfm_sampler_chain_get(s.c, C.int32_t(i))
}

// ChainN returns the total number of samplers in the chain.
func (s *Sampler) ChainN() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return 0
	}
	return int(C.lfm_sampler_chain_n(s.c))
}

// ChainRemove removes and returns the sampler at index i from the chain.
// The caller takes ownership and must call Close().
func (s *Sampler) ChainRemove(i int) *Sampler {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return nil
	}
	removed := C.lfm_sampler_chain_remove(s.c, C.int32_t(i))
	if removed == nil {
		return nil
	}
	return newSampler(removed)
}

// --- Built-in Sampler Constructors ---

// NewGreedySampler creates a greedy sampler (always picks the highest probability token).
func NewGreedySampler() *Sampler {
	return newSampler(C.lfm_sampler_init_greedy())
}

// NewDistSampler creates a distribution-based sampler.
// Use DefaultSeed for a random seed.
func NewDistSampler(seed uint32) *Sampler {
	return newSampler(C.lfm_sampler_init_dist(C.uint32_t(seed)))
}

// NewTopKSampler creates a top-K sampler. Setting k <= 0 makes it a noop.
func NewTopKSampler(k int) *Sampler {
	return newSampler(C.lfm_sampler_init_top_k(C.int32_t(k)))
}

// NewTopPSampler creates a nucleus (top-p) sampler.
func NewTopPSampler(p float32, minKeep int) *Sampler {
	return newSampler(C.lfm_sampler_init_top_p(C.float(p), C.size_t(minKeep)))
}

// NewMinPSampler creates a minimum-P sampler.
func NewMinPSampler(p float32, minKeep int) *Sampler {
	return newSampler(C.lfm_sampler_init_min_p(C.float(p), C.size_t(minKeep)))
}

// NewTypicalSampler creates a locally typical sampler.
func NewTypicalSampler(p float32, minKeep int) *Sampler {
	return newSampler(C.lfm_sampler_init_typical(C.float(p), C.size_t(minKeep)))
}

// NewTempSampler creates a temperature sampler.
// When t <= 0, the max logit is kept and the rest are set to -inf.
func NewTempSampler(t float32) *Sampler {
	return newSampler(C.lfm_sampler_init_temp(C.float(t)))
}

// NewTempExtSampler creates a dynamic temperature (entropy) sampler.
func NewTempExtSampler(t, delta, exponent float32) *Sampler {
	return newSampler(C.lfm_sampler_init_temp_ext(C.float(t), C.float(delta), C.float(exponent)))
}

// NewXTCSampler creates an XTC sampler.
func NewXTCSampler(p, t float32, minKeep int, seed uint32) *Sampler {
	return newSampler(C.lfm_sampler_init_xtc(C.float(p), C.float(t), C.size_t(minKeep), C.uint32_t(seed)))
}

// NewTopNSigmaSampler creates a Top-n sigma sampler.
func NewTopNSigmaSampler(n float32) *Sampler {
	return newSampler(C.lfm_sampler_init_top_n_sigma(C.float(n)))
}

// NewMirostatSampler creates a Mirostat 1.0 sampler.
func NewMirostatSampler(nVocab int, seed uint32, tau, eta float32, m int) *Sampler {
	return newSampler(C.lfm_sampler_init_mirostat(
		C.int32_t(nVocab), C.uint32_t(seed), C.float(tau), C.float(eta), C.int32_t(m)))
}

// NewMirostatV2Sampler creates a Mirostat 2.0 sampler.
func NewMirostatV2Sampler(seed uint32, tau, eta float32) *Sampler {
	return newSampler(C.lfm_sampler_init_mirostat_v2(C.uint32_t(seed), C.float(tau), C.float(eta)))
}

// NewGrammarSampler creates a GBNF grammar sampler.
// Returns nil if grammar parsing fails.
func NewGrammarSampler(vocab *Vocab, grammarStr, grammarRoot string) *Sampler {
	cStr := C.CString(grammarStr)
	defer C.free(unsafe.Pointer(cStr))
	cRoot := C.CString(grammarRoot)
	defer C.free(unsafe.Pointer(cRoot))
	return newSampler(C.lfm_sampler_init_grammar(vocab.c, cStr, cRoot))
}

// NewPenaltiesSampler creates a repetition/frequency/presence penalty sampler.
func NewPenaltiesSampler(penaltyLastN int, penaltyRepeat, penaltyFreq, penaltyPresent float32) *Sampler {
	return newSampler(C.lfm_sampler_init_penalties(
		C.int32_t(penaltyLastN), C.float(penaltyRepeat), C.float(penaltyFreq), C.float(penaltyPresent)))
}

// NewDRYSampler creates a DRY (Don't Repeat Yourself) sampler.
func NewDRYSampler(vocab *Vocab, nCtxTrain int, multiplier, base float32, allowedLength, penaltyLastN int, seqBreakers []string) *Sampler {
	cBreakers := make([]*C.char, len(seqBreakers))
	for i, s := range seqBreakers {
		cBreakers[i] = C.CString(s)
	}
	defer func() {
		for _, p := range cBreakers {
			C.free(unsafe.Pointer(p))
		}
	}()

	var breakersPtr **C.char
	if len(cBreakers) > 0 {
		breakersPtr = &cBreakers[0]
	}

	return newSampler(C.lfm_sampler_init_dry(
		vocab.c,
		C.int32_t(nCtxTrain),
		C.float(multiplier),
		C.float(base),
		C.int32_t(allowedLength),
		C.int32_t(penaltyLastN),
		breakersPtr,
		C.size_t(len(seqBreakers))))
}

// NewAdaptivePSampler creates an adaptive-p sampler.
func NewAdaptivePSampler(target, decay float32, seed uint32) *Sampler {
	return newSampler(C.lfm_sampler_init_adaptive_p(C.float(target), C.float(decay), C.uint32_t(seed)))
}

// NewLogitBiasSampler creates a logit bias sampler.
func NewLogitBiasSampler(nVocab int, biases []LogitBias) *Sampler {
	if len(biases) == 0 {
		return newSampler(C.lfm_sampler_init_logit_bias(C.int32_t(nVocab), 0, nil))
	}
	cBiases := make([]C.struct_lfm_logit_bias, len(biases))
	for i, b := range biases {
		cBiases[i].token = C.lfm_token(b.Token)
		cBiases[i].bias = C.float(b.Bias)
	}
	return newSampler(C.lfm_sampler_init_logit_bias(
		C.int32_t(nVocab), C.int32_t(len(biases)), &cBiases[0]))
}

// NewInfillSampler creates an infill sampler for fill-in-the-middle completion.
func NewInfillSampler(vocab *Vocab) *Sampler {
	return newSampler(C.lfm_sampler_init_infill(vocab.c))
}

// NewPrefixSampler creates a sampler that masks tokens not matching the given prefix.
func NewPrefixSampler(vocab *Vocab, prefix string) *Sampler {
	cPrefix := C.CString(prefix)
	defer C.free(unsafe.Pointer(cPrefix))
	return newSampler(C.lfm_sampler_init_prefix(vocab.c, cPrefix))
}

// PrefixSet updates the prefix for an existing prefix sampler.
func (s *Sampler) PrefixSet(prefix string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return
	}
	cPrefix := C.CString(prefix)
	defer C.free(unsafe.Pointer(cPrefix))
	C.lfm_sampler_prefix_set(s.c, cPrefix)
}
