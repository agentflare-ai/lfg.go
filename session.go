package lfg

/*
#include "lfg_api.h"
#include <stdlib.h>
*/
import "C"
import (
	"runtime"
	"sync"
	"unsafe"
)

// SamplingConfig mirrors lfg_sampling_config.
type SamplingConfig struct {
	Seed           uint32
	NPrev          int32
	TopK           int32
	TopP           float32
	MinP           float32
	TypP           float32
	Temp           float32
	PenaltyLastN   int32
	PenaltyRepeat  float32
	PenaltyFreq    float32
	PenaltyPresent float32
	Mirostat       int32
	MirostatTau    float32
	MirostatEta    float32
}

// SessionConfig mirrors lfg_session_config.
type SessionConfig struct {
	NThreads                int
	NCtx                    int
	NBatch                  int
	EnableHealing           bool
	StructuredCheckpointing bool
	ReasoningBudget         int   // 0 = disabled. Max tokens allowed for reasoning.
	MaxTokens               int32 // 0 = unlimited. Max tokens to generate per reset cycle.
	Sampling                SamplingConfig
}

// DefaultSamplingConfig returns the default sampling configuration.
func DefaultSamplingConfig() SamplingConfig {
	c := C.lfg_sampling_default_config()
	return SamplingConfig{
		Seed:           uint32(c.seed),
		NPrev:          int32(c.n_prev),
		TopK:           int32(c.top_k),
		TopP:           float32(c.top_p),
		MinP:           float32(c.min_p),
		TypP:           float32(c.typ_p),
		Temp:           float32(c.temp),
		PenaltyLastN:   int32(c.penalty_last_n),
		PenaltyRepeat:  float32(c.penalty_repeat),
		PenaltyFreq:    float32(c.penalty_freq),
		PenaltyPresent: float32(c.penalty_present),
		Mirostat:       int32(c.mirostat),
		MirostatTau:    float32(c.mirostat_tau),
		MirostatEta:    float32(c.mirostat_eta),
	}
}

// DefaultSessionConfig returns the default session configuration.
func DefaultSessionConfig() SessionConfig {
	c := C.lfg_session_default_config()
	return SessionConfig{
		NThreads:                int(c.n_threads),
		NCtx:                    int(c.n_ctx),
		NBatch:                  int(c.n_batch),
		EnableHealing:           bool(c.enable_healing),
		StructuredCheckpointing: bool(c.structured_checkpointing),
		ReasoningBudget:         int(c.reasoning_budget),
		MaxTokens:               int32(c.max_tokens),
		Sampling:                samplingConfigFromC(c.sampling),
	}
}

func samplingConfigFromC(c C.lfg_sampling_config) SamplingConfig {
	return SamplingConfig{
		Seed:           uint32(c.seed),
		NPrev:          int32(c.n_prev),
		TopK:           int32(c.top_k),
		TopP:           float32(c.top_p),
		MinP:           float32(c.min_p),
		TypP:           float32(c.typ_p),
		Temp:           float32(c.temp),
		PenaltyLastN:   int32(c.penalty_last_n),
		PenaltyRepeat:  float32(c.penalty_repeat),
		PenaltyFreq:    float32(c.penalty_freq),
		PenaltyPresent: float32(c.penalty_present),
		Mirostat:       int32(c.mirostat),
		MirostatTau:    float32(c.mirostat_tau),
		MirostatEta:    float32(c.mirostat_eta),
	}
}

func (sc *SamplingConfig) toC() C.lfg_sampling_config {
	return C.lfg_sampling_config{
		seed:            C.uint32_t(sc.Seed),
		n_prev:          C.int32_t(sc.NPrev),
		top_k:           C.int32_t(sc.TopK),
		top_p:           C.float(sc.TopP),
		min_p:           C.float(sc.MinP),
		typ_p:           C.float(sc.TypP),
		temp:            C.float(sc.Temp),
		penalty_last_n:  C.int32_t(sc.PenaltyLastN),
		penalty_repeat:  C.float(sc.PenaltyRepeat),
		penalty_freq:    C.float(sc.PenaltyFreq),
		penalty_present: C.float(sc.PenaltyPresent),
		mirostat:        C.int32_t(sc.Mirostat),
		mirostat_tau:    C.float(sc.MirostatTau),
		mirostat_eta:    C.float(sc.MirostatEta),
	}
}

func (cfg *SessionConfig) toC() C.lfg_session_config {
	return C.lfg_session_config{
		n_threads:                C.int(cfg.NThreads),
		n_ctx:                    C.int(cfg.NCtx),
		n_batch:                  C.int(cfg.NBatch),
		enable_healing:           C.bool(cfg.EnableHealing),
		structured_checkpointing: C.bool(cfg.StructuredCheckpointing),
		reasoning_budget:         C.int(cfg.ReasoningBudget),
		max_tokens:               C.int32_t(cfg.MaxTokens),
		sampling:                 cfg.Sampling.toC(),
	}
}

// SessionOption configures session creation.
type SessionOption func(*SessionConfig)

// WithSessionThreads sets the number of threads.
func WithSessionThreads(n int) SessionOption {
	return func(cfg *SessionConfig) {
		cfg.NThreads = n
	}
}

// WithSessionNCtx sets the context size.
func WithSessionNCtx(n int) SessionOption {
	return func(cfg *SessionConfig) {
		cfg.NCtx = n
	}
}

// WithSessionNBatch sets the batch size.
func WithSessionNBatch(n int) SessionOption {
	return func(cfg *SessionConfig) {
		cfg.NBatch = n
	}
}

// WithSessionHealing enables or disables token healing.
func WithSessionHealing(v bool) SessionOption {
	return func(cfg *SessionConfig) {
		cfg.EnableHealing = v
	}
}

// WithSessionStructuredCheckpointing enables structured decoding checkpointing.
func WithSessionStructuredCheckpointing(v bool) SessionOption {
	return func(cfg *SessionConfig) {
		cfg.StructuredCheckpointing = v
	}
}

// WithSessionReasoningBudget sets the maximum number of tokens allowed for reasoning.
// 0 disables the budget (default). When set, the model will be forced to end reasoning
// after this many tokens, and a soft bias is applied starting at 80% usage.
func WithSessionReasoningBudget(n int) SessionOption {
	return func(cfg *SessionConfig) {
		cfg.ReasoningBudget = n
	}
}

// WithSessionMaxTokens sets the maximum number of tokens to generate per reset cycle.
// 0 means unlimited (default).
func WithSessionMaxTokens(n int32) SessionOption {
	return func(cfg *SessionConfig) {
		cfg.MaxTokens = n
	}
}

// WithSessionSampling sets the sampling configuration.
func WithSessionSampling(sc SamplingConfig) SessionOption {
	return func(cfg *SessionConfig) {
		cfg.Sampling = sc
	}
}

// Session wraps the high-level lfg_session API.
type Session struct {
	mu    sync.Mutex
	c     *C.lfg_session
	model *Model // prevent GC
}

// NewSession creates a new high-level session. Automatically initializes the backend.
func NewSession(model *Model, opts ...SessionOption) (*Session, error) {
	ensureBackend()

	model.mu.RLock()
	if model.c == nil {
		model.mu.RUnlock()
		return nil, &Error{Code: ErrorInvalidArgument, Message: "model is closed"}
	}

	cfg := DefaultSessionConfig()
	for _, opt := range opts {
		opt(&cfg)
	}
	cCfg := cfg.toC()

	cs := C.lfg_session_create(model.c, &cCfg)
	model.mu.RUnlock()

	if cs == nil {
		if err := getLastError(); err != nil {
			return nil, err
		}
		return nil, &Error{Code: ErrorInternal, Message: "failed to create session"}
	}

	s := &Session{c: cs, model: model}
	runtime.SetFinalizer(s, func(s *Session) { s.Close() })
	return s, nil
}

// Close frees session resources. Safe to call multiple times.
func (s *Session) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c != nil {
		C.lfg_session_free(s.c)
		s.c = nil
		runtime.SetFinalizer(s, nil)
	}
	return nil
}

// Reset resets the session state.
func (s *Session) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return
	}
	C.lfg_session_reset(s.c)
}

// ConfigureStructured sets up structured decoding with a grammar or JSON schema.
// If grammarOrSchema starts with '{', it is treated as a JSON schema.
func (s *Session) ConfigureStructured(grammarOrSchema, rootRule string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return &Error{Code: ErrorInvalidArgument, Message: "session is closed"}
	}

	cGrammar := C.CString(grammarOrSchema)
	defer C.free(unsafe.Pointer(cGrammar))

	var cRoot *C.char
	if rootRule != "" {
		cRoot = C.CString(rootRule)
		defer C.free(unsafe.Pointer(cRoot))
	}

	ok := C.lfg_session_configure_structured(s.c, cGrammar, cRoot)
	if !bool(ok) {
		if err := getLastError(); err != nil {
			return err
		}
		return &Error{Code: ErrorInternal, Message: "failed to configure structured decoding"}
	}
	return nil
}

// ConfigureReasoning configures tokens that delimit a reasoning/thinking block.
// Structured constraints are suspended while inside these blocks.
func (s *Session) ConfigureReasoning(startTokens, endTokens []Token) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return
	}

	var startPtr, endPtr *C.lfg_token
	if len(startTokens) > 0 {
		startPtr = (*C.lfg_token)(unsafe.Pointer(&startTokens[0]))
	}
	if len(endTokens) > 0 {
		endPtr = (*C.lfg_token)(unsafe.Pointer(&endTokens[0]))
	}

	C.lfg_session_configure_reasoning(s.c,
		startPtr, C.size_t(len(startTokens)),
		endPtr, C.size_t(len(endTokens)))
}

// ConfigureStopSequences sets stop sequences for generation.
// When any sequence is matched during sampling, generation returns EOS.
// Pass nil or an empty slice to clear stop sequences.
func (s *Session) ConfigureStopSequences(sequences [][]Token) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return &Error{Code: ErrorInvalidArgument, Message: "session is closed"}
	}

	if len(sequences) == 0 {
		ok := C.lfg_session_configure_stop_sequences(s.c, nil, nil, 0)
		if !bool(ok) {
			return &Error{Code: ErrorInvalidArgument, Message: "failed to clear stop sequences"}
		}
		return nil
	}

	n := len(sequences)

	// Allocate the pointer array in C memory so we don't violate CGO's
	// "no Go pointer to Go pointer" rule.
	cPtrs := (*[1 << 30]*C.lfg_token)(C.malloc(C.size_t(n) * C.size_t(unsafe.Sizeof((*C.lfg_token)(nil)))))[:n:n]
	defer C.free(unsafe.Pointer(&cPtrs[0]))

	cLens := make([]C.size_t, n)
	for i, seq := range sequences {
		if len(seq) > 0 {
			cPtrs[i] = (*C.lfg_token)(unsafe.Pointer(&seq[0]))
		}
		cLens[i] = C.size_t(len(seq))
	}

	ok := C.lfg_session_configure_stop_sequences(
		s.c,
		&cPtrs[0],
		&cLens[0],
		C.size_t(n))
	if !bool(ok) {
		return &Error{Code: ErrorInvalidArgument, Message: "failed to configure stop sequences"}
	}
	return nil
}

// IngestTokens feeds tokens into the session.
// If updateSampler is true, the sampler state is updated.
func (s *Session) IngestTokens(tokens []Token, updateSampler bool) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return &Error{Code: ErrorInvalidArgument, Message: "session is closed"}
	}
	if len(tokens) == 0 {
		return nil
	}

	ok := C.lfg_session_ingest_tokens(s.c, (*C.lfg_token)(unsafe.Pointer(&tokens[0])), C.size_t(len(tokens)), C.bool(updateSampler))
	if !bool(ok) {
		if err := getLastError(); err != nil {
			return err
		}
		return &Error{Code: ErrorInternal, Message: "failed to ingest tokens"}
	}
	return nil
}

// Decode runs a decode step.
func (s *Session) Decode() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return &Error{Code: ErrorInvalidArgument, Message: "session is closed"}
	}

	ok := C.lfg_session_decode(s.c)
	if !bool(ok) {
		if err := getLastError(); err != nil {
			return err
		}
		return &Error{Code: ErrorInternal, Message: "decode failed"}
	}
	return nil
}

// Sample samples a single token from the current logits.
func (s *Session) Sample() Token {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return InvalidToken
	}
	return Token(C.lfg_session_sample(s.c))
}

// HealToken performs token healing on the last token.
func (s *Session) HealToken() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return &Error{Code: ErrorInvalidArgument, Message: "session is closed"}
	}

	ok := C.lfg_session_heal_last_token(s.c)
	if !bool(ok) {
		if err := getLastError(); err != nil {
			return err
		}
		return &Error{Code: ErrorInternal, Message: "heal last token failed"}
	}
	return nil
}

// Logits copies logits from the session into the provided buffer.
// If out is nil, returns the required buffer size.
func (s *Session) Logits(out []float32) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return 0
	}
	if len(out) == 0 {
		return int(C.lfg_session_get_logits(s.c, nil, 0))
	}
	return int(C.lfg_session_get_logits(s.c, (*C.float)(unsafe.Pointer(&out[0])), C.int32_t(len(out))))
}

// VocabSize returns the vocabulary size for this session.
func (s *Session) VocabSize() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return 0
	}
	return int(C.lfg_session_get_vocab_size(s.c))
}

// ModelStats holds statistics about a loaded model.
type ModelStats struct {
	ParameterCount uint64
	SizeBytes      uint64
	VocabSize      int32
	ContextSize    int32
}

// Stats returns model statistics using the convenience API.
func (m *Model) Stats() ModelStats {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return ModelStats{}
	}
	cs := C.lfg_model_get_stats(m.c)
	return ModelStats{
		ParameterCount: uint64(cs.n_params),
		SizeBytes:      uint64(cs.size_bytes),
		VocabSize:      int32(cs.n_vocab),
		ContextSize:    int32(cs.n_ctx_train),
	}
}

// Checkpoint wraps an lfg_checkpoint.
type Checkpoint struct {
	c       *C.lfg_checkpoint
	session *Session // prevent GC
}

// CreateCheckpoint creates a snapshot of the current session state.
func (s *Session) CreateCheckpoint() *Checkpoint {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return nil
	}
	cp := C.lfg_session_create_checkpoint(s.c)
	if cp == nil {
		return nil
	}
	c := &Checkpoint{c: cp, session: s}
	runtime.SetFinalizer(c, func(c *Checkpoint) { c.Close() })
	return c
}

// Close frees the checkpoint. Safe to call multiple times.
func (cp *Checkpoint) Close() {
	if cp.c != nil {
		C.lfg_checkpoint_free(cp.c)
		cp.c = nil
		runtime.SetFinalizer(cp, nil)
	}
}

// CheckpointRestoreOptions configures what to restore.
type CheckpointRestoreOptions struct {
	RestoreSamplerState bool
	RestoreGrammar      bool
}

// DefaultCheckpointRestoreOptions returns the default restore options.
func DefaultCheckpointRestoreOptions() CheckpointRestoreOptions {
	c := C.lfg_checkpoint_restore_default_options()
	return CheckpointRestoreOptions{
		RestoreSamplerState: bool(c.restore_sampler_state),
		RestoreGrammar:      bool(c.restore_grammar),
	}
}

// RestoreCheckpoint restores the session to a checkpoint with default options.
func (s *Session) RestoreCheckpoint(cp *Checkpoint) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return &Error{Code: ErrorInvalidArgument, Message: "session is closed"}
	}
	if cp == nil || cp.c == nil {
		return &Error{Code: ErrorInvalidArgument, Message: "checkpoint is nil"}
	}

	ok := C.lfg_session_restore_checkpoint(s.c, cp.c)
	if !bool(ok) {
		if err := getLastError(); err != nil {
			return err
		}
		return &Error{Code: ErrorInternal, Message: "failed to restore checkpoint"}
	}
	return nil
}

// RestoreCheckpointEx restores the session with custom options.
func (s *Session) RestoreCheckpointEx(cp *Checkpoint, opts CheckpointRestoreOptions) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return &Error{Code: ErrorInvalidArgument, Message: "session is closed"}
	}
	if cp == nil || cp.c == nil {
		return &Error{Code: ErrorInvalidArgument, Message: "checkpoint is nil"}
	}

	cOpts := C.lfg_checkpoint_restore_options{
		restore_sampler_state: C.bool(opts.RestoreSamplerState),
		restore_grammar:       C.bool(opts.RestoreGrammar),
	}

	ok := C.lfg_session_restore_checkpoint_ex(s.c, cp.c, &cOpts)
	if !bool(ok) {
		if err := getLastError(); err != nil {
			return err
		}
		return &Error{Code: ErrorInternal, Message: "failed to restore checkpoint"}
	}
	return nil
}

// JSONSchemaToGrammar converts a JSON schema string to a GBNF grammar.
// If forceGBNF is true, always outputs GBNF format.
func JSONSchemaToGrammar(jsonSchema string, forceGBNF bool) (string, error) {
	cSchema := C.CString(jsonSchema)
	defer C.free(unsafe.Pointer(cSchema))

	// First pass: get required size.
	n := C.lfg_json_schema_to_grammar(cSchema, C.bool(forceGBNF), nil, 0)
	if n < 0 {
		if err := getLastError(); err != nil {
			return "", err
		}
		return "", &Error{Code: ErrorInternal, Message: "failed to convert JSON schema to grammar"}
	}
	if n == 0 {
		return "", nil
	}

	buf := make([]byte, int(n)+1)
	n = C.lfg_json_schema_to_grammar(cSchema, C.bool(forceGBNF), (*C.char)(unsafe.Pointer(&buf[0])), C.size_t(len(buf)))
	if n < 0 {
		if err := getLastError(); err != nil {
			return "", err
		}
		return "", &Error{Code: ErrorInternal, Message: "failed to convert JSON schema to grammar"}
	}
	return string(buf[:n]), nil
}

// ToolDesc describes a tool for embedding-based ranking.
type ToolDesc struct {
	Name        string // Tool name.
	Description string // Human-readable description.
	JSONSchema  string // Optional JSON schema for parameters.
}

// RegisterTools registers tools with the session for embedding-based ranking.
// The session computes and caches embeddings for each tool description internally.
// topK controls how many of the highest-ranked tools are injected into context
// on the first decode call. Pass 0 to disable injection.
// Returns the number of tools registered.
func (s *Session) RegisterTools(tools []ToolDesc, topK int32) (int32, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return 0, &Error{Code: ErrorInvalidArgument, Message: "session is closed"}
	}
	if len(tools) == 0 {
		return 0, &Error{Code: ErrorInvalidArgument, Message: "no tools provided"}
	}

	// Build C tool descriptors. Keep CStrings alive until the call returns.
	cDescs := make([]C.lfg_tool_desc, len(tools))
	cStrs := make([]*C.char, 0, len(tools)*3) // for deferred free
	defer func() {
		for _, p := range cStrs {
			C.free(unsafe.Pointer(p))
		}
	}()

	for i, t := range tools {
		cName := C.CString(t.Name)
		cStrs = append(cStrs, cName)
		cDesc := C.CString(t.Description)
		cStrs = append(cStrs, cDesc)
		cDescs[i].name = cName
		cDescs[i].description = cDesc
		if t.JSONSchema != "" {
			cSchema := C.CString(t.JSONSchema)
			cStrs = append(cStrs, cSchema)
			cDescs[i].json_schema = cSchema
		}
	}

	n := C.lfg_session_register_tools(s.c, &cDescs[0], C.int32_t(len(tools)), C.int32_t(topK))
	if n < 0 {
		if err := getLastError(); err != nil {
			return 0, err
		}
		return 0, &Error{Code: ErrorInternal, Message: "failed to register tools"}
	}
	return int32(n), nil
}

// ClearTools removes all registered tools and frees the tool ranking context.
func (s *Session) ClearTools() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return
	}
	C.lfg_session_clear_tools(s.c)
}

// ---------------------------------------------------------------------------
// Entropy Monitor API
// ---------------------------------------------------------------------------

// EntropyEvent represents a single entropy event from the ring buffer.
type EntropyEvent struct {
	Entropy      float32 // Raw Shannon entropy: H = -sum p_i log(p_i)
	Normalized   float32 // entropy / log(n_vocab), range [0,1]
	TopLogprob   float32 // Log probability of the sampled token
	Token        Token   // The sampled token
	NPast        int32   // Token position when event fired
	CheckpointID int32   // Opaque ID for Rewind()
	NEmbedding   int32   // Embedding dimension (for embedding output sizing)
}

// EntropyMonitorConfig configures the entropy monitor.
type EntropyMonitorConfig struct {
	Threshold      float32 // Normalized entropy threshold (0,1]. 0 = disabled.
	CooldownTokens int32   // Min tokens between events.
	RingSize       int32   // Ring buffer slots. 0 = default (4).
}

// DefaultEntropyMonitorConfig returns the default entropy monitor configuration.
func DefaultEntropyMonitorConfig() EntropyMonitorConfig {
	c := C.lfg_entropy_monitor_default_config()
	return EntropyMonitorConfig{
		Threshold:      float32(c.threshold),
		CooldownTokens: int32(c.cooldown_tokens),
		RingSize:       int32(c.ring_size),
	}
}

// ConfigureEntropyMonitor configures the entropy monitor for this session.
// Returns the embedding dimension (n_embd) on success — use this to size
// the embedding buffer passed to EntropyPop.
// Pass nil to disable the entropy monitor (returns 0, nil).
func (s *Session) ConfigureEntropyMonitor(config *EntropyMonitorConfig) (int32, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return 0, &Error{Code: ErrorInvalidArgument, Message: "session is closed"}
	}

	if config == nil {
		C.lfg_session_configure_entropy_monitor(s.c, nil)
		return 0, nil
	}

	cConfig := C.lfg_entropy_monitor_config{
		threshold:       C.float(config.Threshold),
		cooldown_tokens: C.int32_t(config.CooldownTokens),
		ring_size:       C.int32_t(config.RingSize),
	}
	nEmbd := C.lfg_session_configure_entropy_monitor(s.c, &cConfig)
	if nEmbd <= 0 {
		if err := getLastError(); err != nil {
			return 0, err
		}
		return 0, &Error{Code: ErrorInternal, Message: "failed to configure entropy monitor"}
	}
	return int32(nEmbd), nil
}

// EntropyPop pops the next pending entropy event from the ring buffer.
// If embeddingOut is non-nil, the embedding vector is copied into it (must be >= EntropyEvent.NEmbedding floats).
// Pass nil for embeddingOut to skip embedding copy.
// Returns the event and true if an event was available, or a zero event and false if no events are pending.
func (s *Session) EntropyPop(embeddingOut []float32) (EntropyEvent, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return EntropyEvent{}, false
	}

	var cEvent C.lfg_entropy_event
	var embdPtr *C.float
	var embdCap C.int32_t
	if len(embeddingOut) > 0 {
		embdPtr = (*C.float)(unsafe.Pointer(&embeddingOut[0]))
		embdCap = C.int32_t(len(embeddingOut))
	}

	ok := C.lfg_session_entropy_pop(s.c, &cEvent, embdPtr, embdCap)
	if !bool(ok) {
		return EntropyEvent{}, false
	}

	return EntropyEvent{
		Entropy:      float32(cEvent.entropy),
		Normalized:   float32(cEvent.normalized),
		TopLogprob:   float32(cEvent.top_logprob),
		Token:        Token(cEvent.token),
		NPast:        int32(cEvent.n_past),
		CheckpointID: int32(cEvent.checkpoint_id),
		NEmbedding:   int32(cEvent.n_embd),
	}, true
}

// EntropyPending returns the number of pending (unread) entropy events.
func (s *Session) EntropyPending() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return 0
	}
	return int(C.lfg_session_entropy_pending(s.c))
}

// EntropyFlush discards all pending entropy events without reading them.
func (s *Session) EntropyFlush() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return
	}
	C.lfg_session_entropy_flush(s.c)
}

// EntropyCounter returns a pointer to an atomic write counter that is incremented
// each time an entropy event is written to the ring buffer.
// Callers can poll this with sync/atomic.LoadInt32 or use platform-specific wait mechanisms.
// Returns nil if the session is closed or the entropy monitor is not configured.
func (s *Session) EntropyCounter() *int32 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return nil
	}
	p := C.lfg_session_entropy_counter(s.c)
	if p == nil {
		return nil
	}
	return (*int32)(unsafe.Pointer(p))
}

// Rewind rewinds the session to an entropy checkpoint. Truncates the KV cache
// and resets the sampler. The checkpointID comes from EntropyEvent.CheckpointID.
func (s *Session) Rewind(checkpointID int32) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return &Error{Code: ErrorInvalidArgument, Message: "session is closed"}
	}

	ok := C.lfg_session_rewind(s.c, C.int32_t(checkpointID))
	if !bool(ok) {
		if err := getLastError(); err != nil {
			return err
		}
		return &Error{Code: ErrorInternal, Message: "failed to rewind session"}
	}
	return nil
}

// LastEntropy returns the normalized entropy from the last Sample() call.
// Returns -1 if no sample has been performed.
func (s *Session) LastEntropy() float32 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return -1
	}
	return float32(C.lfg_session_get_last_entropy(s.c))
}

// ---------------------------------------------------------------------------
// Embedding API
// ---------------------------------------------------------------------------

// Embed computes a mean-pooled, L2-normalized embedding for the given text.
// Returns the embedding vector on success. Allocates an embedding context on
// the first call (reused across subsequent calls).
func (s *Session) Embed(text string) ([]float32, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return nil, &Error{Code: ErrorInvalidArgument, Message: "session is closed"}
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))
	cLen := C.int32_t(len(text))

	// First call: get embedding dimension.
	nEmbd := C.lfg_session_embed(s.c, cText, cLen, nil, 0)
	if nEmbd <= 0 {
		if err := getLastError(); err != nil {
			return nil, err
		}
		return nil, &Error{Code: ErrorInternal, Message: "failed to compute embedding"}
	}

	out := make([]float32, int(nEmbd))
	n := C.lfg_session_embed(s.c, cText, cLen, (*C.float)(unsafe.Pointer(&out[0])), C.int32_t(len(out)))
	if n <= 0 {
		if err := getLastError(); err != nil {
			return nil, err
		}
		return nil, &Error{Code: ErrorInternal, Message: "failed to compute embedding"}
	}
	return out[:int(n)], nil
}
