//go:build (darwin && arm64) || (linux && amd64) || (linux && arm64)

package lfg

import (
	"runtime"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
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
	ReasoningBudget         int           // 0 = disabled. Max tokens allowed for reasoning.
	MaxTokens               int32         // 0 = unlimited. Max tokens to generate per reset cycle.
	ToolScoreMode           ToolScoreMode // Tool injection gating. 0 = OFF (always inject).
	ToolMinScore            float32       // Threshold value. AUTO: gap above mean. FIXED: absolute minimum.
	Sampling                SamplingConfig
}

// DefaultSamplingConfig returns the default sampling configuration.
func DefaultSamplingConfig() SamplingConfig {
	registerSessionFuncs()
	c := _lfg_sampling_default_config()
	return samplingConfigFromC(c)
}

// DefaultSessionConfig returns the default session configuration.
func DefaultSessionConfig() SessionConfig {
	registerSessionFuncs()
	c := _lfg_session_default_config()
	return SessionConfig{
		NThreads:                int(c.NThreads),
		NCtx:                    int(c.NCtx),
		NBatch:                  int(c.NBatch),
		EnableHealing:           c.EnableHealing != 0,
		StructuredCheckpointing: c.StructuredCheckpointing != 0,
		ReasoningBudget:         int(c.ReasoningBudget),
		MaxTokens:               c.MaxTokens,
		ToolScoreMode:           ToolScoreMode(c.ToolScoreMode),
		ToolMinScore:            c.ToolMinScore,
		Sampling:                samplingConfigFromC(c.Sampling),
	}
}

func samplingConfigFromC(c cSamplingConfig) SamplingConfig {
	return SamplingConfig{
		Seed:           c.Seed,
		NPrev:          c.NPrev,
		TopK:           c.TopK,
		TopP:           c.TopP,
		MinP:           c.MinP,
		TypP:           c.TypP,
		Temp:           c.Temp,
		PenaltyLastN:   c.PenaltyLastN,
		PenaltyRepeat:  c.PenaltyRepeat,
		PenaltyFreq:    c.PenaltyFreq,
		PenaltyPresent: c.PenaltyPresent,
		Mirostat:       c.Mirostat,
		MirostatTau:    c.MirostatTau,
		MirostatEta:    c.MirostatEta,
	}
}

func (sc *SamplingConfig) toC() cSamplingConfig {
	return cSamplingConfig{
		Seed:           sc.Seed,
		NPrev:          sc.NPrev,
		TopK:           sc.TopK,
		TopP:           sc.TopP,
		MinP:           sc.MinP,
		TypP:           sc.TypP,
		Temp:           sc.Temp,
		PenaltyLastN:   sc.PenaltyLastN,
		PenaltyRepeat:  sc.PenaltyRepeat,
		PenaltyFreq:    sc.PenaltyFreq,
		PenaltyPresent: sc.PenaltyPresent,
		Mirostat:       sc.Mirostat,
		MirostatTau:    sc.MirostatTau,
		MirostatEta:    sc.MirostatEta,
	}
}

func (cfg *SessionConfig) toC() cSessionConfig {
	return cSessionConfig{
		NThreads:                int32(cfg.NThreads),
		NCtx:                    int32(cfg.NCtx),
		NBatch:                  int32(cfg.NBatch),
		EnableHealing:           boolToByte(cfg.EnableHealing),
		StructuredCheckpointing: boolToByte(cfg.StructuredCheckpointing),
		ReasoningBudget:         int32(cfg.ReasoningBudget),
		MaxTokens:               cfg.MaxTokens,
		ToolScoreMode:           int32(cfg.ToolScoreMode),
		ToolMinScore:            cfg.ToolMinScore,
		Sampling:                cfg.Sampling.toC(),
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

// WithSessionToolScoreMode sets the tool injection gating mode.
func WithSessionToolScoreMode(mode ToolScoreMode) SessionOption {
	return func(cfg *SessionConfig) {
		cfg.ToolScoreMode = mode
	}
}

// WithSessionToolMinScore sets the tool score threshold.
// For ToolScoreAuto: gap above mean. For ToolScoreFixed: absolute minimum.
func WithSessionToolMinScore(score float32) SessionOption {
	return func(cfg *SessionConfig) {
		cfg.ToolMinScore = score
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
	c     uintptr
	model *Model // prevent GC
}

// NewSession creates a new high-level session. Automatically initializes the backend.
func NewSession(model *Model, opts ...SessionOption) (*Session, error) {
	ensureBackend()
	registerSessionFuncs()

	model.mu.RLock()
	if model.c == 0 {
		model.mu.RUnlock()
		return nil, &Error{Code: ErrorInvalidArgument, Message: "model is closed"}
	}

	cfg := DefaultSessionConfig()
	for _, opt := range opts {
		opt(&cfg)
	}
	cCfg := cfg.toC()

	cs := _lfg_session_create(model.c, uintptr(unsafe.Pointer(&cCfg)))
	model.mu.RUnlock()

	if cs == 0 {
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
	if s.c != 0 {
		_lfg_session_free(s.c)
		s.c = 0
		runtime.SetFinalizer(s, nil)
	}
	return nil
}

// Reset resets the session state.
func (s *Session) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return
	}
	_lfg_session_reset(s.c)
}

// ConfigureStructured sets up structured decoding with a grammar or JSON schema.
// If grammarOrSchema starts with '{', it is treated as a JSON schema.
func (s *Session) ConfigureStructured(grammarOrSchema, rootRule string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return &Error{Code: ErrorInvalidArgument, Message: "session is closed"}
	}

	grammarBytes := cString(grammarOrSchema)
	var rootPtr uintptr
	var rootBytes []byte
	if rootRule != "" {
		rootBytes = cString(rootRule)
		rootPtr = cStringPtr(rootBytes)
	}

	ok := _lfg_session_configure_structured(s.c, cStringPtr(grammarBytes), rootPtr)
	runtime.KeepAlive(grammarBytes)
	runtime.KeepAlive(rootBytes)
	if !ok {
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
	if s.c == 0 {
		return
	}

	_lfg_session_configure_reasoning(s.c,
		tokenPtr(startTokens), uintptr(len(startTokens)),
		tokenPtr(endTokens), uintptr(len(endTokens)))
}

// ConfigureStopSequences sets stop sequences for generation.
// When any sequence is matched during sampling, generation returns EOS.
// Pass nil or an empty slice to clear stop sequences.
func (s *Session) ConfigureStopSequences(sequences [][]Token) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return &Error{Code: ErrorInvalidArgument, Message: "session is closed"}
	}

	if len(sequences) == 0 {
		ok := _lfg_session_configure_stop_sequences(s.c, 0, 0, 0)
		if !ok {
			return &Error{Code: ErrorInvalidArgument, Message: "failed to clear stop sequences"}
		}
		return nil
	}

	n := len(sequences)

	// With purego, no "Go ptr to Go ptr" rule — use Go slices directly.
	ptrs := make([]uintptr, n)
	lens := make([]uintptr, n)
	for i, seq := range sequences {
		if len(seq) > 0 {
			ptrs[i] = uintptr(unsafe.Pointer(&seq[0]))
		}
		lens[i] = uintptr(len(seq))
	}

	ok := _lfg_session_configure_stop_sequences(
		s.c,
		uintptr(unsafe.Pointer(&ptrs[0])),
		uintptr(unsafe.Pointer(&lens[0])),
		uintptr(n))
	runtime.KeepAlive(sequences)
	runtime.KeepAlive(ptrs)
	runtime.KeepAlive(lens)
	if !ok {
		return &Error{Code: ErrorInvalidArgument, Message: "failed to configure stop sequences"}
	}
	return nil
}

// IngestTokens feeds tokens into the session.
// If updateSampler is true, the sampler state is updated.
func (s *Session) IngestTokens(tokens []Token, updateSampler bool) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return &Error{Code: ErrorInvalidArgument, Message: "session is closed"}
	}
	if len(tokens) == 0 {
		return nil
	}

	ok := _lfg_session_ingest_tokens(s.c, tokenPtr(tokens), uintptr(len(tokens)), updateSampler)
	runtime.KeepAlive(tokens)
	if !ok {
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
	if s.c == 0 {
		return &Error{Code: ErrorInvalidArgument, Message: "session is closed"}
	}

	ok := _lfg_session_decode(s.c)
	if !ok {
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
	if s.c == 0 {
		return InvalidToken
	}
	return Token(_lfg_session_sample(s.c))
}

// HealToken performs token healing on the last token.
func (s *Session) HealToken() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return &Error{Code: ErrorInvalidArgument, Message: "session is closed"}
	}

	ok := _lfg_session_heal_last_token(s.c)
	if !ok {
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
	if s.c == 0 {
		return 0
	}
	if len(out) == 0 {
		return int(_lfg_session_get_logits(s.c, 0, 0))
	}
	return int(_lfg_session_get_logits(s.c, uintptr(unsafe.Pointer(&out[0])), int32(len(out))))
}

// VocabSize returns the vocabulary size for this session.
func (s *Session) VocabSize() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return 0
	}
	return int(_lfg_session_get_vocab_size(s.c))
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
	if m.c == 0 {
		return ModelStats{}
	}
	registerSessionFuncs()
	cs := _lfg_model_get_stats(m.c)
	return ModelStats{
		ParameterCount: cs.NParams,
		SizeBytes:      cs.SizeBytes,
		VocabSize:      cs.NVocab,
		ContextSize:    cs.NCtxTrain,
	}
}

// Checkpoint wraps an lfg_checkpoint.
type Checkpoint struct {
	c       uintptr
	session *Session // prevent GC
}

// CreateCheckpoint creates a snapshot of the current session state.
func (s *Session) CreateCheckpoint() *Checkpoint {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return nil
	}
	cp := _lfg_session_create_checkpoint(s.c)
	if cp == 0 {
		return nil
	}
	c := &Checkpoint{c: cp, session: s}
	runtime.SetFinalizer(c, func(c *Checkpoint) { c.Close() })
	return c
}

// Close frees the checkpoint. Safe to call multiple times.
func (cp *Checkpoint) Close() {
	if cp.c != 0 {
		_lfg_checkpoint_free(cp.c)
		cp.c = 0
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
	registerSessionFuncs()
	c := _lfg_checkpoint_restore_default_options()
	return CheckpointRestoreOptions{
		RestoreSamplerState: c.RestoreSamplerState != 0,
		RestoreGrammar:      c.RestoreGrammar != 0,
	}
}

// RestoreCheckpoint restores the session to a checkpoint with default options.
func (s *Session) RestoreCheckpoint(cp *Checkpoint) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return &Error{Code: ErrorInvalidArgument, Message: "session is closed"}
	}
	if cp == nil || cp.c == 0 {
		return &Error{Code: ErrorInvalidArgument, Message: "checkpoint is nil"}
	}

	ok := _lfg_session_restore_checkpoint(s.c, cp.c)
	if !ok {
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
	if s.c == 0 {
		return &Error{Code: ErrorInvalidArgument, Message: "session is closed"}
	}
	if cp == nil || cp.c == 0 {
		return &Error{Code: ErrorInvalidArgument, Message: "checkpoint is nil"}
	}

	cOpts := cCheckpointRestoreOptions{
		RestoreSamplerState: boolToByte(opts.RestoreSamplerState),
		RestoreGrammar:      boolToByte(opts.RestoreGrammar),
	}

	ok := _lfg_session_restore_checkpoint_ex(s.c, cp.c, uintptr(unsafe.Pointer(&cOpts)))
	if !ok {
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
	registerSessionFuncs()
	schemaBytes := cString(jsonSchema)
	schemaPtr := cStringPtr(schemaBytes)

	// First pass: get required size.
	n := _lfg_json_schema_to_grammar(schemaPtr, forceGBNF, 0, 0)
	runtime.KeepAlive(schemaBytes)
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
	n = _lfg_json_schema_to_grammar(schemaPtr, forceGBNF, uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)))
	runtime.KeepAlive(schemaBytes)
	if n < 0 {
		if err := getLastError(); err != nil {
			return "", err
		}
		return "", &Error{Code: ErrorInternal, Message: "failed to convert JSON schema to grammar"}
	}
	return string(buf[:n]), nil
}

// ToolDesc describes a tool for embedding-based ranking and optional auto-execution.
type ToolDesc struct {
	Name        string // Tool name.
	Description string // Human-readable description.
	Parameters  string // Optional JSON schema for parameters.
	Fn          ToolFn // Optional auto-execution function. nil = consumer handles tool calls.
}

// ---------------------------------------------------------------------------
// Tool function trampoline (auto-execution)
// ---------------------------------------------------------------------------

var (
	toolFnMu        sync.Mutex
	toolFnMap       = make(map[uintptr]ToolFn)
	toolFnNextID    uintptr
	toolFnTrampOnce sync.Once
	toolFnTramp     uintptr
)

func initToolFnTrampoline() {
	toolFnTrampOnce.Do(func() {
		registerSessionFuncs() // ensure _malloc is registered

		// C signature: const char * (*)(const char *arguments, void *user_data)
		// Returns a malloc'd C string — the C engine calls free() on it.
		toolFnTramp = purego.NewCallback(func(argsPtr uintptr, userData uintptr) uintptr {
			if userData == 0 {
				return 0
			}
			id := *(*uintptr)(unsafe.Pointer(userData))
			toolFnMu.Lock()
			fn, ok := toolFnMap[id]
			toolFnMu.Unlock()
			if !ok || fn == nil {
				return 0
			}

			goArgs := goString(argsPtr)
			result, err := fn(goArgs)
			if err != nil {
				result = "error: " + err.Error()
			}

			// Allocate C string via malloc — caller (C engine) will free.
			n := len(result)
			cPtr := _malloc(uintptr(n + 1))
			if cPtr == 0 {
				return 0
			}
			dst := unsafe.Slice((*byte)(unsafe.Pointer(cPtr)), n+1)
			copy(dst, result)
			dst[n] = 0
			return cPtr
		})
	})
}

func registerToolFn(fn ToolFn) uintptr {
	toolFnMu.Lock()
	defer toolFnMu.Unlock()
	toolFnNextID++
	id := toolFnNextID
	toolFnMap[id] = fn
	return id
}

func unregisterToolFns(ids []uintptr) {
	toolFnMu.Lock()
	defer toolFnMu.Unlock()
	for _, id := range ids {
		delete(toolFnMap, id)
	}
}

// RegisterTools registers tools with the session for embedding-based ranking.
// The session computes and caches embeddings for each tool description internally.
// topK controls how many of the highest-ranked tools are injected into context
// on the first decode call. Pass 0 to disable injection.
// Returns the number of tools registered.
func (s *Session) RegisterTools(tools []ToolDesc, topK int32) (int32, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return 0, &Error{Code: ErrorInvalidArgument, Message: "session is closed"}
	}
	if len(tools) == 0 {
		return 0, &Error{Code: ErrorInvalidArgument, Message: "no tools provided"}
	}

	// Build C tool descriptors with Go byte slices.
	cDescs := make([]cToolDesc, len(tools))
	keepAlive := make([][]byte, 0, len(tools)*3)
	var fnIDs []uintptr
	var fnIDPtrs []*uintptr // prevent GC

	for i, t := range tools {
		nameBytes := cString(t.Name)
		descBytes := cString(t.Description)
		keepAlive = append(keepAlive, nameBytes, descBytes)
		cDescs[i].Name = cStringPtr(nameBytes)
		cDescs[i].Description = cStringPtr(descBytes)
		if t.Parameters != "" {
			paramBytes := cString(t.Parameters)
			keepAlive = append(keepAlive, paramBytes)
			cDescs[i].Parameters = cStringPtr(paramBytes)
		}
		if t.Fn != nil {
			initToolFnTrampoline()
			id := registerToolFn(t.Fn)
			fnIDs = append(fnIDs, id)
			idPtr := new(uintptr)
			*idPtr = id
			fnIDPtrs = append(fnIDPtrs, idPtr)
			cDescs[i].Fn = toolFnTramp
			cDescs[i].FnUserData = uintptr(unsafe.Pointer(idPtr))
		}
	}

	n := _lfg_session_register_tools(s.c, uintptr(unsafe.Pointer(&cDescs[0])), int32(len(tools)), topK)
	runtime.KeepAlive(keepAlive)
	runtime.KeepAlive(cDescs)
	runtime.KeepAlive(fnIDPtrs)
	if n < 0 {
		unregisterToolFns(fnIDs)
		if err := getLastError(); err != nil {
			return 0, err
		}
		return 0, &Error{Code: ErrorInternal, Message: "failed to register tools"}
	}
	return n, nil
}

// ClearTools removes all registered tools and frees the tool ranking context.
func (s *Session) ClearTools() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return
	}
	_lfg_session_clear_tools(s.c)
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
	Threshold      float32         // Normalized entropy threshold (0,1]. 0 = disabled.
	CooldownTokens int32           // Min tokens between events.
	RingSize       int32           // Ring buffer slots. 0 = default (4).
	GateMode       EntropyGateMode // Gating mode. Off, Fixed (default), or Auto.
}

// DefaultEntropyMonitorConfig returns the default entropy monitor configuration.
func DefaultEntropyMonitorConfig() EntropyMonitorConfig {
	registerSessionFuncs()
	c := _lfg_entropy_monitor_default_config()
	return EntropyMonitorConfig{
		Threshold:      c.Threshold,
		CooldownTokens: c.CooldownTokens,
		RingSize:       c.RingSize,
		GateMode:       EntropyGateMode(c.GateMode),
	}
}

// ConfigureEntropyMonitor configures the entropy monitor for this session.
// Returns the embedding dimension (n_embd) on success — use this to size
// the embedding buffer passed to EntropyPop.
// Pass nil to disable the entropy monitor (returns 0, nil).
func (s *Session) ConfigureEntropyMonitor(config *EntropyMonitorConfig) (int32, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return 0, &Error{Code: ErrorInvalidArgument, Message: "session is closed"}
	}

	if config == nil {
		_lfg_session_configure_entropy_monitor(s.c, 0)
		return 0, nil
	}

	cConfig := cEntropyMonitorConfig{
		Threshold:      config.Threshold,
		CooldownTokens: config.CooldownTokens,
		RingSize:       config.RingSize,
		GateMode:       int32(config.GateMode),
	}
	nEmbd := _lfg_session_configure_entropy_monitor(s.c, uintptr(unsafe.Pointer(&cConfig)))
	if nEmbd <= 0 {
		if err := getLastError(); err != nil {
			return 0, err
		}
		return 0, &Error{Code: ErrorInternal, Message: "failed to configure entropy monitor"}
	}
	return nEmbd, nil
}

// EntropyPop pops the next pending entropy event from the ring buffer.
// If embeddingOut is non-nil, the embedding vector is copied into it (must be >= EntropyEvent.NEmbedding floats).
// Pass nil for embeddingOut to skip embedding copy.
// Returns the event and true if an event was available, or a zero event and false if no events are pending.
func (s *Session) EntropyPop(embeddingOut []float32) (EntropyEvent, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return EntropyEvent{}, false
	}

	var cEvent cEntropyEvent
	var embdPtr uintptr
	var embdCap int32
	if len(embeddingOut) > 0 {
		embdPtr = uintptr(unsafe.Pointer(&embeddingOut[0]))
		embdCap = int32(len(embeddingOut))
	}

	ok := _lfg_session_entropy_pop(s.c, uintptr(unsafe.Pointer(&cEvent)), embdPtr, embdCap)
	if !ok {
		return EntropyEvent{}, false
	}

	return EntropyEvent{
		Entropy:      cEvent.Entropy,
		Normalized:   cEvent.Normalized,
		TopLogprob:   cEvent.TopLogprob,
		Token:        Token(cEvent.Token),
		NPast:        cEvent.NPast,
		CheckpointID: cEvent.CheckpointID,
		NEmbedding:   cEvent.NEmbd,
	}, true
}

// EntropyPending returns the number of pending (unread) entropy events.
func (s *Session) EntropyPending() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return 0
	}
	return int(_lfg_session_entropy_pending(s.c))
}

// EntropyFlush discards all pending entropy events without reading them.
func (s *Session) EntropyFlush() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return
	}
	_lfg_session_entropy_flush(s.c)
}

// EntropyCounter returns a pointer to an atomic write counter that is incremented
// each time an entropy event is written to the ring buffer.
// Callers can poll this with sync/atomic.LoadInt32 or use platform-specific wait mechanisms.
// Returns nil if the session is closed or the entropy monitor is not configured.
func (s *Session) EntropyCounter() *int32 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return nil
	}
	p := _lfg_session_entropy_counter(s.c)
	if p == 0 {
		return nil
	}
	return (*int32)(unsafe.Pointer(p))
}

// Rewind rewinds the session to an entropy checkpoint. Truncates the KV cache
// and resets the sampler. The checkpointID comes from EntropyEvent.CheckpointID.
func (s *Session) Rewind(checkpointID int32) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return &Error{Code: ErrorInvalidArgument, Message: "session is closed"}
	}

	ok := _lfg_session_rewind(s.c, checkpointID)
	if !ok {
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
	if s.c == 0 {
		return -1
	}
	return _lfg_session_get_last_entropy(s.c)
}

// ---------------------------------------------------------------------------
// Confidence Monitor API (inverse entropy — sustained low-entropy span detection)
// ---------------------------------------------------------------------------

// ConfidenceEvent represents a sustained low-entropy span detected by the confidence monitor.
type ConfidenceEvent struct {
	MeanEntropy float32 // Average normalized entropy over the span.
	MinEntropy  float32 // Minimum normalized entropy in the span.
	SpanLength  int32   // Number of consecutive low-entropy tokens.
	StartPos    int32   // Token position at span start.
	EndPos      int32   // Token position at span end.
	NEmbedding  int32   // Embedding dimension (for embedding output sizing).
	SpanText    string  // Detokenized span text. Empty if unavailable.
}

// ConfidenceMonitorConfig configures the confidence monitor.
type ConfidenceMonitorConfig struct {
	Threshold        float32            // Normalized entropy ceiling (0,1]. Tokens below this are "confident".
	MinSpan          int32              // Min consecutive tokens to emit event. 0 = default (5).
	RingSize         int32              // Ring buffer slots. 0 = default (4).
	IncludeReasoning bool               // false (default) = skip reasoning tokens; true = include them.
	GateMode         ConfidenceGateMode // Gating mode. Off, Fixed (default), or Auto.
}

// DefaultConfidenceMonitorConfig returns the default confidence monitor configuration.
func DefaultConfidenceMonitorConfig() ConfidenceMonitorConfig {
	registerSessionFuncs()
	c := _lfg_confidence_monitor_default_config()
	return ConfidenceMonitorConfig{
		Threshold:        c.Threshold,
		MinSpan:          c.MinSpan,
		RingSize:         c.RingSize,
		IncludeReasoning: c.IncludeReasoning != 0,
		GateMode:         ConfidenceGateMode(c.GateMode),
	}
}

// ConfigureConfidenceMonitor configures the confidence monitor for this session.
// Returns the embedding dimension (n_embd) on success — use this to size
// the embedding buffer passed to ConfidencePop.
// Pass nil to disable the confidence monitor (returns 0, nil).
func (s *Session) ConfigureConfidenceMonitor(config *ConfidenceMonitorConfig) (int32, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return 0, &Error{Code: ErrorInvalidArgument, Message: "session is closed"}
	}

	if config == nil {
		_lfg_session_configure_confidence_monitor(s.c, 0)
		return 0, nil
	}

	cConfig := cConfidenceMonitorConfig{
		Threshold:        config.Threshold,
		MinSpan:          config.MinSpan,
		RingSize:         config.RingSize,
		IncludeReasoning: boolToByte(config.IncludeReasoning),
		GateMode:         int32(config.GateMode),
	}
	nEmbd := _lfg_session_configure_confidence_monitor(s.c, uintptr(unsafe.Pointer(&cConfig)))
	if nEmbd <= 0 {
		if err := getLastError(); err != nil {
			return 0, err
		}
		return 0, &Error{Code: ErrorInternal, Message: "failed to configure confidence monitor"}
	}
	return nEmbd, nil
}

// ConfidencePop pops the next pending confidence event from the ring buffer.
// If embeddingOut is non-nil, the embedding vector is copied into it (must be >= ConfidenceEvent.NEmbedding floats).
// Pass nil for embeddingOut to skip embedding copy.
// Returns the event and true if an event was available, or a zero event and false if no events are pending.
func (s *Session) ConfidencePop(embeddingOut []float32) (ConfidenceEvent, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return ConfidenceEvent{}, false
	}

	var cEvent cConfidenceEvent
	var embdPtr uintptr
	var embdCap int32
	if len(embeddingOut) > 0 {
		embdPtr = uintptr(unsafe.Pointer(&embeddingOut[0]))
		embdCap = int32(len(embeddingOut))
	}

	ok := _lfg_session_confidence_pop(s.c, uintptr(unsafe.Pointer(&cEvent)), embdPtr, embdCap)
	if !ok {
		return ConfidenceEvent{}, false
	}

	return ConfidenceEvent{
		MeanEntropy: cEvent.MeanEntropy,
		MinEntropy:  cEvent.MinEntropy,
		SpanLength:  cEvent.SpanLength,
		StartPos:    cEvent.StartPos,
		EndPos:      cEvent.EndPos,
		NEmbedding:  cEvent.NEmbd,
		SpanText:    goStringN(cEvent.SpanText, int(cEvent.SpanTextLen)),
	}, true
}

// ConfidencePending returns the number of pending (unread) confidence events.
func (s *Session) ConfidencePending() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return 0
	}
	return int(_lfg_session_confidence_pending(s.c))
}

// ConfidenceFlush discards all pending confidence events without reading them.
func (s *Session) ConfidenceFlush() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return
	}
	_lfg_session_confidence_flush(s.c)
}

// ConfidenceCounter returns a pointer to an atomic write counter that is incremented
// each time a confidence event is written to the ring buffer.
// Returns nil if the session is closed or the confidence monitor is not configured.
func (s *Session) ConfidenceCounter() *int32 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return nil
	}
	p := _lfg_session_confidence_counter(s.c)
	if p == 0 {
		return nil
	}
	return (*int32)(unsafe.Pointer(p))
}

// ---------------------------------------------------------------------------
// Surprise Monitor API (input novelty — high-surprise span detection during ingestion)
// ---------------------------------------------------------------------------

// SurpriseEvent represents an aggregate surprise result after prompt ingestion.
// Summarizes how surprising the entire input was to the model.
type SurpriseEvent struct {
	MeanSurprise     float32 // Average normalized surprise across above-threshold tokens.
	MaxSurprise      float32 // Maximum normalized surprise.
	NAboveThreshold  int32   // Count of tokens above threshold.
	NTokensEvaluated int32   // Total tokens evaluated (prompt minus BOS).
	NEmbedding       int32   // Embedding dimension (for embedding output sizing).
}

// SurpriseMonitorConfig configures the surprise monitor.
type SurpriseMonitorConfig struct {
	Threshold        float32          // Normalized surprise floor (0,1]. Above = surprising.
	IncludeReasoning bool             // false (default) = skip reasoning tokens; true = include them.
	GateMode         SurpriseGateMode // Gating mode. Off, Fixed (default), or Auto.
}

// DefaultSurpriseMonitorConfig returns the default surprise monitor configuration.
func DefaultSurpriseMonitorConfig() SurpriseMonitorConfig {
	registerSessionFuncs()
	c := _lfg_surprise_monitor_default_config()
	return SurpriseMonitorConfig{
		Threshold:        c.Threshold,
		IncludeReasoning: c.IncludeReasoning != 0,
		GateMode:         SurpriseGateMode(c.GateMode),
	}
}

// ConfigureSurpriseMonitor configures the surprise monitor for this session.
// Returns the embedding dimension (n_embd) on success — use this to size
// the embedding buffer passed to SurprisePop.
// Pass nil to disable the surprise monitor (returns 0, nil).
func (s *Session) ConfigureSurpriseMonitor(config *SurpriseMonitorConfig) (int32, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return 0, &Error{Code: ErrorInvalidArgument, Message: "session is closed"}
	}

	if config == nil {
		_lfg_session_configure_surprise_monitor(s.c, 0)
		return 0, nil
	}

	cConfig := cSurpriseMonitorConfig{
		Threshold:        config.Threshold,
		IncludeReasoning: boolToByte(config.IncludeReasoning),
		GateMode:         int32(config.GateMode),
	}
	nEmbd := _lfg_session_configure_surprise_monitor(s.c, uintptr(unsafe.Pointer(&cConfig)))
	if nEmbd <= 0 {
		if err := getLastError(); err != nil {
			return 0, err
		}
		return 0, &Error{Code: ErrorInternal, Message: "failed to configure surprise monitor"}
	}
	return nEmbd, nil
}

// SurprisePop pops the next pending surprise event from the ring buffer.
// If embeddingOut is non-nil, the embedding vector is copied into it (must be >= SurpriseEvent.NEmbedding floats).
// Pass nil for embeddingOut to skip embedding copy.
// Returns the event and true if an event was available, or a zero event and false if no events are pending.
func (s *Session) SurprisePop(embeddingOut []float32) (SurpriseEvent, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return SurpriseEvent{}, false
	}

	var cEvent cSurpriseEvent
	var embdPtr uintptr
	var embdCap int32
	if len(embeddingOut) > 0 {
		embdPtr = uintptr(unsafe.Pointer(&embeddingOut[0]))
		embdCap = int32(len(embeddingOut))
	}

	ok := _lfg_session_surprise_pop(s.c, uintptr(unsafe.Pointer(&cEvent)), embdPtr, embdCap)
	if !ok {
		return SurpriseEvent{}, false
	}

	return SurpriseEvent{
		MeanSurprise:     cEvent.MeanSurprise,
		MaxSurprise:      cEvent.MaxSurprise,
		NAboveThreshold:  cEvent.NAboveThreshold,
		NTokensEvaluated: cEvent.NTokensEvaluated,
		NEmbedding:       cEvent.NEmbd,
	}, true
}

// ConfigureStopStrings sets text-based stop strings for generation.
// Unlike token-level stop sequences, text stops are encoding-independent
// (same text always matches regardless of how the tokenizer splits it).
// Pass nil or an empty slice to clear stop strings.
func (s *Session) ConfigureStopStrings(strings []string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return &Error{Code: ErrorInvalidArgument, Message: "session is closed"}
	}

	if len(strings) == 0 {
		ok := _lfg_session_configure_stop_strings(s.c, 0, 0)
		if !ok {
			return &Error{Code: ErrorInvalidArgument, Message: "failed to clear stop strings"}
		}
		return nil
	}

	n := len(strings)

	// With purego, no "Go ptr to Go ptr" rule — use Go slices directly.
	strBytes := make([][]byte, n)
	strPtrs := make([]uintptr, n)
	for i, s := range strings {
		strBytes[i] = cString(s)
		strPtrs[i] = cStringPtr(strBytes[i])
	}

	ok := _lfg_session_configure_stop_strings(s.c, uintptr(unsafe.Pointer(&strPtrs[0])), int32(n))
	runtime.KeepAlive(strBytes)
	runtime.KeepAlive(strPtrs)
	if !ok {
		return &Error{Code: ErrorInvalidArgument, Message: "failed to configure stop strings"}
	}
	return nil
}

// ---------------------------------------------------------------------------
// Convenience Model Loader (lfg_api.h)
// ---------------------------------------------------------------------------

// ModelLoadConfig configures the convenience model loader.
type ModelLoadConfig struct {
	ModelPath string
	UseMmap   bool
	UseMlock  bool
	GPULayers int
}

// DefaultModelLoadConfig returns the default model loading configuration.
func DefaultModelLoadConfig() ModelLoadConfig {
	registerSessionFuncs()
	c := _lfg_model_load_default_config()
	return ModelLoadConfig{
		UseMmap:   c.UseMmap != 0,
		UseMlock:  c.UseMlock != 0,
		GPULayers: int(c.NGPULayers),
	}
}

// LoadModelSimple loads a model using the simplified API from lfg_api.h.
// This is a convenience alternative to LoadModel that takes fewer parameters
// and reduces the setup to a single C call.
func LoadModelSimple(path string, opts ...ModelOption) (*Model, error) {
	ensureBackend()
	registerSessionFuncs()

	defaults := DefaultModelLoadConfig()
	cfg := ModelConfig{
		Mmap:      defaults.UseMmap,
		Mlock:     defaults.UseMlock,
		GPULayers: defaults.GPULayers,
	}
	for _, opt := range opts {
		opt(&cfg)
	}

	pathBytes := cString(path)

	cCfg := cModelLoadConfig{
		ModelPath:  cStringPtr(pathBytes),
		UseMmap:    boolToByte(cfg.Mmap),
		UseMlock:   boolToByte(cfg.Mlock),
		NGPULayers: int32(cfg.GPULayers),
	}

	cModel := _lfg_load_model(uintptr(unsafe.Pointer(&cCfg)))
	runtime.KeepAlive(pathBytes)
	if cModel == 0 {
		if err := getLastError(); err != nil {
			return nil, err
		}
		return nil, &Error{Code: ErrorInternal, Message: "failed to load model"}
	}

	m := &Model{c: cModel}
	runtime.SetFinalizer(m, func(m *Model) { m.Close() })
	return m, nil
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
	if s.c == 0 {
		return nil, &Error{Code: ErrorInvalidArgument, Message: "session is closed"}
	}

	// Get output embedding dimension from the model. The C API requires a
	// valid output buffer on every call (it does not support a nil/0 query pattern).
	registerModelFuncs()
	nEmbd := int(_lfg_model_n_embd_out(s.model.c))
	if nEmbd <= 0 {
		return nil, &Error{Code: ErrorInternal, Message: "model has no embedding dimension"}
	}

	textBytes := cString(text)
	textPtr := cStringPtr(textBytes)
	cLen := int32(len(text))

	out := make([]float32, nEmbd)
	n := _lfg_session_embed(s.c, textPtr, cLen, uintptr(unsafe.Pointer(&out[0])), int32(nEmbd))
	runtime.KeepAlive(textBytes)
	if n <= 0 {
		if err := getLastError(); err != nil {
			return nil, err
		}
		return nil, &Error{Code: ErrorInternal, Message: "failed to compute embedding"}
	}
	return out[:int(n)], nil
}

// EmbedTokens computes per-token, L2-normalized embeddings for the given text.
// Returns a flat slice of n_tok * n_embd floats where each consecutive n_embd
// floats represent one token's embedding. Use [Model.OutputEmbeddingSize] to
// get n_embd. Allocates a per-token embedding context on the first call
// (reused across subsequent calls).
func (s *Session) EmbedTokens(text string) (embeddings []float32, nTokens int, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return nil, 0, &Error{Code: ErrorInvalidArgument, Message: "session is closed"}
	}

	registerModelFuncs()
	nEmbd := int(_lfg_model_n_embd_out(s.model.c))
	if nEmbd <= 0 {
		return nil, 0, &Error{Code: ErrorInternal, Message: "model has no embedding dimension"}
	}

	textBytes := cString(text)
	textPtr := cStringPtr(textBytes)
	cLen := int32(len(text))

	// Allocate for up to cLen+16 tokens (same heuristic as C side).
	tokCap := int(cLen) + 16
	outCap := tokCap * nEmbd
	out := make([]float32, outCap)

	nTok := _lfg_session_embed_tokens(s.c, textPtr, cLen, uintptr(unsafe.Pointer(&out[0])), int32(outCap))
	runtime.KeepAlive(textBytes)
	if nTok <= 0 {
		if e := getLastError(); e != nil {
			return nil, 0, e
		}
		return nil, 0, &Error{Code: ErrorInternal, Message: "failed to compute per-token embeddings"}
	}
	return out[:int(nTok)*nEmbd], int(nTok), nil
}

// ---------------------------------------------------------------------------
// Tool Call Introspection & Parsing
// ---------------------------------------------------------------------------

// RankTools ranks registered tools against a query string.
// Returns a JSON string with the ranking results.
func (s *Session) RankTools(query string) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return "", &Error{Code: ErrorInvalidArgument, Message: "session is closed"}
	}

	queryBytes := cString(query)
	queryPtr := cStringPtr(queryBytes)
	cLen := int32(len(query))

	// First pass: get required size.
	n := _lfg_session_rank_tools(s.c, queryPtr, cLen, 0, 0)
	runtime.KeepAlive(queryBytes)
	if n < 0 {
		if err := getLastError(); err != nil {
			return "", err
		}
		return "", &Error{Code: ErrorInternal, Message: "failed to rank tools"}
	}
	if n == 0 {
		return "", nil
	}

	buf := make([]byte, int(n)+1)
	n = _lfg_session_rank_tools(s.c, queryPtr, cLen, uintptr(unsafe.Pointer(&buf[0])), int32(len(buf)))
	runtime.KeepAlive(queryBytes)
	if n < 0 {
		if err := getLastError(); err != nil {
			return "", err
		}
		return "", &Error{Code: ErrorInternal, Message: "failed to rank tools"}
	}
	return string(buf[:n]), nil
}

// LastPrompt returns the fully-formatted prompt from the last chat_generate call.
// The returned string is owned by the session and valid until the next generate/reset/free.
func (s *Session) LastPrompt() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return ""
	}

	var lenOut int32
	ptr := _lfg_session_get_last_prompt(s.c, uintptr(unsafe.Pointer(&lenOut)))
	return goStringN(ptr, int(lenOut))
}

// ToolCalls returns parsed tool calls from the last generation.
// Valid after a generate call that stopped with StopReasonToolCall.
// The returned data is owned by the session and valid until the next generate/reset/free.
func (s *Session) ToolCalls() []ToolCall {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return nil
	}

	var nOut int32
	ptr := _lfg_session_get_tool_calls(s.c, uintptr(unsafe.Pointer(&nOut)))
	if ptr == 0 || nOut <= 0 {
		return nil
	}

	cCalls := unsafe.Slice((*cToolCall)(unsafe.Pointer(ptr)), int(nOut))
	calls := make([]ToolCall, int(nOut))
	for i := range calls {
		calls[i] = ToolCall{
			ID:        goString(cCalls[i].ID),
			Name:      goString(cCalls[i].Name),
			Arguments: goString(cCalls[i].Arguments),
		}
	}
	return calls
}

// LastOutput returns the raw text output from the last generation.
// The returned string is owned by the session and valid until the next generate/reset/free.
func (s *Session) LastOutput() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return ""
	}

	var lenOut int32
	ptr := _lfg_session_get_last_output(s.c, uintptr(unsafe.Pointer(&lenOut)))
	return goStringN(ptr, int(lenOut))
}

// SetToolCallFormat sets the format used for tool call parsing.
func (s *Session) SetToolCallFormat(format ToolCallFormat) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return
	}
	_lfg_session_set_tool_call_format(s.c, int32(format))
}

// ParsePythonicToolCalls parses pythonic-format tool calls from text.
// Returns a slice of parsed tool calls. The id/name/arguments strings
// in each returned ToolCall are allocated and owned by Go.
func ParsePythonicToolCalls(text string) []ToolCall {
	registerSessionFuncs()

	textBytes := cString(text)
	textPtr := cStringPtr(textBytes)
	cLen := int32(len(text))

	// First pass: get count.
	n := _lfg_parse_pythonic_tool_calls(textPtr, cLen, 0, 0)
	runtime.KeepAlive(textBytes)
	if n <= 0 {
		return nil
	}

	// Allocate output array and parse.
	cCalls := make([]cToolCall, int(n))
	n = _lfg_parse_pythonic_tool_calls(textPtr, cLen, uintptr(unsafe.Pointer(&cCalls[0])), n)
	runtime.KeepAlive(textBytes)
	if n <= 0 {
		return nil
	}

	calls := make([]ToolCall, int(n))
	for i := range calls {
		calls[i] = ToolCall{
			ID:        goString(cCalls[i].ID),
			Name:      goString(cCalls[i].Name),
			Arguments: goString(cCalls[i].Arguments),
		}
		// Free C-allocated strings (parse allocates via malloc).
		if cCalls[i].ID != 0 {
			_free(cCalls[i].ID)
		}
		if cCalls[i].Name != 0 {
			_free(cCalls[i].Name)
		}
		if cCalls[i].Arguments != 0 {
			_free(cCalls[i].Arguments)
		}
	}
	return calls
}
