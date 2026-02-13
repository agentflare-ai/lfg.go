//go:build (darwin && arm64) || (linux && amd64) || (linux && arm64)

package lfg

import (
	"runtime"
	"sync"
	"unsafe"
)

// Sampler wraps an lfg_sampler pointer with thread-safe access.
type Sampler struct {
	mu    sync.Mutex
	c     uintptr
	owned bool // false if ownership was transferred to a chain
}

// newSampler wraps a C sampler pointer.
func newSampler(c uintptr) *Sampler {
	if c == 0 {
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
	if s.c != 0 && s.owned {
		registerSamplerFuncs()
		_lfg_sampler_free(s.c)
		s.c = 0
		runtime.SetFinalizer(s, nil)
	}
}

// Name returns the sampler's name.
func (s *Sampler) Name() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return ""
	}
	registerSamplerFuncs()
	return goString(_lfg_sampler_name(s.c))
}

// Accept notifies the sampler that a token has been selected.
func (s *Sampler) Accept(token Token) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return
	}
	registerSamplerFuncs()
	_lfg_sampler_accept(s.c, int32(token))
}

// Reset resets the sampler state.
func (s *Sampler) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return
	}
	registerSamplerFuncs()
	_lfg_sampler_reset(s.c)
}

// Clone creates a copy of the sampler.
func (s *Sampler) Clone() *Sampler {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return nil
	}
	registerSamplerFuncs()
	return newSampler(_lfg_sampler_clone(s.c))
}

// GetSeed returns the seed used by the sampler, or RandomSeed if not applicable.
func (s *Sampler) GetSeed() uint32 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return RandomSeed
	}
	registerSamplerFuncs()
	return _lfg_sampler_get_seed(s.c)
}

// Sample samples and accepts a token from the idx-th output of the last evaluation.
func (s *Sampler) Sample(ctx *Context, idx int) Token {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 || ctx == nil {
		return InvalidToken
	}
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == 0 {
		return InvalidToken
	}
	registerSamplerFuncs()
	return Token(_lfg_sampler_sample(s.c, ctx.c, int32(idx)))
}

// --- Sampler Chain ---

// NewSamplerChain creates a new sampler chain.
// If noPerf is true, performance timings are not measured.
func NewSamplerChain(noPerf bool) *Sampler {
	registerSamplerFuncs()
	params := _lfg_sampler_chain_default_params()
	params.NoPerf = boolToByte(noPerf)
	return newSampler(_lfg_sampler_chain_init(params))
}

// Add appends a sampler to the chain. The chain takes ownership of the sampler.
func (s *Sampler) Add(child *Sampler) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 || child == nil {
		return
	}
	child.mu.Lock()
	defer child.mu.Unlock()
	if child.c == 0 {
		return
	}
	registerSamplerFuncs()
	_lfg_sampler_chain_add(s.c, child.c)
	child.owned = false
	runtime.SetFinalizer(child, nil)
}

// ChainGet returns the sampler at index i in the chain.
// If i == -1, returns the chain itself (useful to check if it's a chain).
func (s *Sampler) ChainGet(i int) uintptr {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return 0
	}
	registerSamplerFuncs()
	return _lfg_sampler_chain_get(s.c, int32(i))
}

// ChainN returns the total number of samplers in the chain.
func (s *Sampler) ChainN() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return 0
	}
	registerSamplerFuncs()
	return int(_lfg_sampler_chain_n(s.c))
}

// ChainRemove removes and returns the sampler at index i from the chain.
// The caller takes ownership and must call Close().
func (s *Sampler) ChainRemove(i int) *Sampler {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return nil
	}
	registerSamplerFuncs()
	removed := _lfg_sampler_chain_remove(s.c, int32(i))
	if removed == 0 {
		return nil
	}
	return newSampler(removed)
}

// --- Built-in Sampler Constructors ---

// NewGreedySampler creates a greedy sampler (always picks the highest probability token).
func NewGreedySampler() *Sampler {
	registerSamplerFuncs()
	return newSampler(_lfg_sampler_init_greedy())
}

// NewDistSampler creates a distribution-based sampler.
// Use RandomSeed for a random seed.
func NewDistSampler(seed uint32) *Sampler {
	registerSamplerFuncs()
	return newSampler(_lfg_sampler_init_dist(seed))
}

// NewTopKSampler creates a top-K sampler. Setting k <= 0 makes it a noop.
func NewTopKSampler(k int) *Sampler {
	registerSamplerFuncs()
	return newSampler(_lfg_sampler_init_top_k(int32(k)))
}

// NewTopPSampler creates a nucleus (top-p) sampler.
func NewTopPSampler(p float32, minKeep int) *Sampler {
	registerSamplerFuncs()
	return newSampler(_lfg_sampler_init_top_p(p, uintptr(minKeep)))
}

// NewMinPSampler creates a minimum-P sampler.
func NewMinPSampler(p float32, minKeep int) *Sampler {
	registerSamplerFuncs()
	return newSampler(_lfg_sampler_init_min_p(p, uintptr(minKeep)))
}

// NewTypicalSampler creates a locally typical sampler.
func NewTypicalSampler(p float32, minKeep int) *Sampler {
	registerSamplerFuncs()
	return newSampler(_lfg_sampler_init_typical(p, uintptr(minKeep)))
}

// NewTempSampler creates a temperature sampler.
// When t <= 0, the max logit is kept and the rest are set to -inf.
func NewTempSampler(t float32) *Sampler {
	registerSamplerFuncs()
	return newSampler(_lfg_sampler_init_temp(t))
}

// NewTempExtSampler creates a dynamic temperature (entropy) sampler.
func NewTempExtSampler(t, delta, exponent float32) *Sampler {
	registerSamplerFuncs()
	return newSampler(_lfg_sampler_init_temp_ext(t, delta, exponent))
}

// NewXTCSampler creates an XTC sampler.
func NewXTCSampler(p, t float32, minKeep int, seed uint32) *Sampler {
	registerSamplerFuncs()
	return newSampler(_lfg_sampler_init_xtc(p, t, uintptr(minKeep), seed))
}

// NewTopNSigmaSampler creates a Top-n sigma sampler.
func NewTopNSigmaSampler(n float32) *Sampler {
	registerSamplerFuncs()
	return newSampler(_lfg_sampler_init_top_n_sigma(n))
}

// NewMirostatSampler creates a Mirostat 1.0 sampler.
func NewMirostatSampler(nVocab int, seed uint32, tau, eta float32, m int) *Sampler {
	registerSamplerFuncs()
	return newSampler(_lfg_sampler_init_mirostat(int32(nVocab), seed, tau, eta, int32(m)))
}

// NewMirostatV2Sampler creates a Mirostat 2.0 sampler.
func NewMirostatV2Sampler(seed uint32, tau, eta float32) *Sampler {
	registerSamplerFuncs()
	return newSampler(_lfg_sampler_init_mirostat_v2(seed, tau, eta))
}

// NewGrammarSampler creates a GBNF grammar sampler.
// Returns nil if grammar parsing fails.
func NewGrammarSampler(vocab *Vocab, grammarStr, grammarRoot string) *Sampler {
	registerSamplerFuncs()
	strBytes := cString(grammarStr)
	rootBytes := cString(grammarRoot)
	result := _lfg_sampler_init_grammar(vocab.c, cStringPtr(strBytes), cStringPtr(rootBytes))
	runtime.KeepAlive(strBytes)
	runtime.KeepAlive(rootBytes)
	return newSampler(result)
}

// NewPenaltiesSampler creates a repetition/frequency/presence penalty sampler.
func NewPenaltiesSampler(penaltyLastN int, penaltyRepeat, penaltyFreq, penaltyPresent float32) *Sampler {
	registerSamplerFuncs()
	return newSampler(_lfg_sampler_init_penalties(int32(penaltyLastN), penaltyRepeat, penaltyFreq, penaltyPresent))
}

// NewDRYSampler creates a DRY (Don't Repeat Yourself) sampler.
func NewDRYSampler(vocab *Vocab, nCtxTrain int, multiplier, base float32, allowedLength, penaltyLastN int, seqBreakers []string) *Sampler {
	registerSamplerFuncs()

	// Build an array of uintptr (char*) for seqBreakers.
	// With purego, no "Go ptr to Go ptr" rule applies.
	breakerBytes := make([][]byte, len(seqBreakers))
	breakerPtrs := make([]uintptr, len(seqBreakers))
	for i, s := range seqBreakers {
		breakerBytes[i] = cString(s)
		breakerPtrs[i] = cStringPtr(breakerBytes[i])
	}

	var breakersPtr uintptr
	if len(breakerPtrs) > 0 {
		breakersPtr = uintptr(unsafe.Pointer(&breakerPtrs[0]))
	}

	result := _lfg_sampler_init_dry(
		vocab.c,
		int32(nCtxTrain),
		multiplier,
		base,
		int32(allowedLength),
		int32(penaltyLastN),
		breakersPtr,
		uintptr(len(seqBreakers)))
	runtime.KeepAlive(breakerBytes)
	runtime.KeepAlive(breakerPtrs)
	return newSampler(result)
}

// NewAdaptivePSampler creates an adaptive-p sampler.
func NewAdaptivePSampler(target, decay float32, seed uint32) *Sampler {
	registerSamplerFuncs()
	return newSampler(_lfg_sampler_init_adaptive_p(target, decay, seed))
}

// NewLogitBiasSampler creates a logit bias sampler.
func NewLogitBiasSampler(nVocab int, biases []LogitBias) *Sampler {
	registerSamplerFuncs()
	if len(biases) == 0 {
		return newSampler(_lfg_sampler_init_logit_bias(int32(nVocab), 0, 0))
	}
	cBiases := make([]cLogitBias, len(biases))
	for i, b := range biases {
		cBiases[i].Token = int32(b.Token)
		cBiases[i].Bias = b.Bias
	}
	return newSampler(_lfg_sampler_init_logit_bias(
		int32(nVocab), int32(len(biases)), uintptr(unsafe.Pointer(&cBiases[0]))))
}

// NewInfillSampler creates an infill sampler for fill-in-the-middle completion.
func NewInfillSampler(vocab *Vocab) *Sampler {
	registerSamplerFuncs()
	return newSampler(_lfg_sampler_init_infill(vocab.c))
}

// NewPrefixSampler creates a sampler that masks tokens not matching the given prefix.
func NewPrefixSampler(vocab *Vocab, prefix string) *Sampler {
	registerSamplerFuncs()
	prefixBytes := cString(prefix)
	result := _lfg_sampler_init_prefix(vocab.c, cStringPtr(prefixBytes))
	runtime.KeepAlive(prefixBytes)
	return newSampler(result)
}

// PrefixSet updates the prefix for an existing prefix sampler.
func (s *Sampler) PrefixSet(prefix string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return
	}
	registerSamplerFuncs()
	prefixBytes := cString(prefix)
	_lfg_sampler_prefix_set(s.c, cStringPtr(prefixBytes))
	runtime.KeepAlive(prefixBytes)
}

// NewReasoningBudgetSampler creates a sampler that enforces a reasoning token budget.
// When the budget is exceeded, thinking end tokens are forced.
func NewReasoningBudgetSampler(budget int, startTokens, endTokens []Token) *Sampler {
	registerSamplerFuncs()
	return newSampler(_lfg_sampler_init_reasoning_budget(
		int32(budget),
		tokenPtr(startTokens), uintptr(len(startTokens)),
		tokenPtr(endTokens), uintptr(len(endTokens))))
}

// NewGrammarLazyPatternsSampler creates a lazy grammar sampler with pattern triggers.
// The grammar is only activated when a trigger pattern or token is matched.
func NewGrammarLazyPatternsSampler(vocab *Vocab, grammarStr, grammarRoot string, triggerPatterns []string, triggerTokens []Token) *Sampler {
	registerSamplerFuncs()
	strBytes := cString(grammarStr)
	rootBytes := cString(grammarRoot)

	patternBytes := make([][]byte, len(triggerPatterns))
	patternPtrs := make([]uintptr, len(triggerPatterns))
	for i, s := range triggerPatterns {
		patternBytes[i] = cString(s)
		patternPtrs[i] = cStringPtr(patternBytes[i])
	}

	var patternsPtr uintptr
	if len(patternPtrs) > 0 {
		patternsPtr = uintptr(unsafe.Pointer(&patternPtrs[0]))
	}

	result := _lfg_sampler_init_grammar_lazy_patterns(
		vocab.c,
		cStringPtr(strBytes),
		cStringPtr(rootBytes),
		patternsPtr, uintptr(len(triggerPatterns)),
		tokenPtr(triggerTokens), uintptr(len(triggerTokens)))
	runtime.KeepAlive(strBytes)
	runtime.KeepAlive(rootBytes)
	runtime.KeepAlive(patternBytes)
	runtime.KeepAlive(patternPtrs)
	return newSampler(result)
}

// NewReasoningGateSampler creates a sampler that wraps another sampler and only
// applies it outside of reasoning (thinking) blocks delimited by start/end tokens.
// The chain takes ownership of wrappedSampler.
func NewReasoningGateSampler(wrappedSampler *Sampler, startTokens, endTokens []Token) *Sampler {
	registerSamplerFuncs()
	wrappedSampler.mu.Lock()
	cPtr := wrappedSampler.c
	wrappedSampler.owned = false
	runtime.SetFinalizer(wrappedSampler, nil)
	wrappedSampler.mu.Unlock()

	return newSampler(_lfg_sampler_init_reasoning_gate(
		cPtr,
		tokenPtr(startTokens), uintptr(len(startTokens)),
		tokenPtr(endTokens), uintptr(len(endTokens))))
}

// Apply applies the sampler to a token data array, modifying logits and probabilities in-place.
func (s *Sampler) Apply(data []TokenData) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 || len(data) == 0 {
		return
	}
	registerSamplerFuncs()

	cData := make([]cTokenData, len(data))
	for i, d := range data {
		cData[i] = cTokenData{ID: int32(d.ID), Logit: d.Logit, P: d.P}
	}

	arr := cTokenDataArray{
		Data:     uintptr(unsafe.Pointer(&cData[0])),
		Size:     uintptr(len(cData)),
		Selected: -1,
	}
	_lfg_sampler_apply(s.c, uintptr(unsafe.Pointer(&arr)))

	for i := range data {
		data[i].ID = Token(cData[i].ID)
		data[i].Logit = cData[i].Logit
		data[i].P = cData[i].P
	}
}
