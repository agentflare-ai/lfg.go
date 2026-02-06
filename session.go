package lfg

/*
typedef struct lfm_model lfm_model;
typedef struct lfm_context lfm_context;
typedef struct lfm_vocab lfm_vocab;
typedef struct lfm_sampler lfm_sampler;
#include "lfm_api.h"
#include <stdlib.h>
*/
import "C"
import (
	"runtime"
	"sync"
	"unsafe"
)

// SamplingConfig mirrors lfm_sampling_config.
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

// SessionConfig mirrors lfm_session_config.
type SessionConfig struct {
	NThreads                int
	NCtx                    int
	NBatch                  int
	EnableHealing           bool
	StructuredCheckpointing bool
	Sampling                SamplingConfig
}

// DefaultSamplingConfig returns the default sampling configuration.
func DefaultSamplingConfig() SamplingConfig {
	c := C.lfm_sampling_default_config()
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
	c := C.lfm_session_default_config()
	return SessionConfig{
		NThreads:                int(c.n_threads),
		NCtx:                    int(c.n_ctx),
		NBatch:                  int(c.n_batch),
		EnableHealing:           bool(c.enable_healing),
		StructuredCheckpointing: bool(c.structured_checkpointing),
		Sampling:                samplingConfigFromC(c.sampling),
	}
}

func samplingConfigFromC(c C.lfm_sampling_config) SamplingConfig {
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

func (sc *SamplingConfig) toC() C.lfm_sampling_config {
	return C.lfm_sampling_config{
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

func (cfg *SessionConfig) toC() C.lfm_session_config {
	return C.lfm_session_config{
		n_threads:                C.int(cfg.NThreads),
		n_ctx:                    C.int(cfg.NCtx),
		n_batch:                  C.int(cfg.NBatch),
		enable_healing:           C.bool(cfg.EnableHealing),
		structured_checkpointing: C.bool(cfg.StructuredCheckpointing),
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

// WithSessionSampling sets the sampling configuration.
func WithSessionSampling(sc SamplingConfig) SessionOption {
	return func(cfg *SessionConfig) {
		cfg.Sampling = sc
	}
}

// Session wraps the high-level lfm_session API.
type Session struct {
	mu    sync.Mutex
	c     *C.lfm_session
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

	cs := C.lfm_session_create(model.c, &cCfg)
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
		C.lfm_session_free(s.c)
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
	C.lfm_session_reset(s.c)
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

	ok := C.lfm_session_configure_structured(s.c, cGrammar, cRoot)
	if !bool(ok) {
		if err := getLastError(); err != nil {
			return err
		}
		return &Error{Code: ErrorInternal, Message: "failed to configure structured decoding"}
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

	ok := C.lfm_session_ingest_tokens(s.c, (*C.lfm_token)(&tokens[0]), C.size_t(len(tokens)), C.bool(updateSampler))
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

	ok := C.lfm_session_decode(s.c)
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
		return TokenNull
	}
	return Token(C.lfm_session_sample(s.c))
}

// HealLastToken performs token healing on the last token.
func (s *Session) HealLastToken() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return &Error{Code: ErrorInvalidArgument, Message: "session is closed"}
	}

	ok := C.lfm_session_heal_last_token(s.c)
	if !bool(ok) {
		if err := getLastError(); err != nil {
			return err
		}
		return &Error{Code: ErrorInternal, Message: "heal last token failed"}
	}
	return nil
}

// GetLogits copies logits from the session into the provided buffer.
// If out is nil, returns the required buffer size.
func (s *Session) GetLogits(out []float32) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return 0
	}
	if len(out) == 0 {
		return int(C.lfm_session_get_logits(s.c, nil, 0))
	}
	return int(C.lfm_session_get_logits(s.c, (*C.float)(unsafe.Pointer(&out[0])), C.int32_t(len(out))))
}

// GetVocabSize returns the vocabulary size for this session.
func (s *Session) GetVocabSize() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return 0
	}
	return int(C.lfm_session_get_vocab_size(s.c))
}

// Checkpoint wraps an lfm_checkpoint.
type Checkpoint struct {
	c       *C.lfm_checkpoint
	session *Session // prevent GC
}

// CreateCheckpoint creates a snapshot of the current session state.
func (s *Session) CreateCheckpoint() *Checkpoint {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return nil
	}
	cp := C.lfm_session_create_checkpoint(s.c)
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
		C.lfm_checkpoint_free(cp.c)
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
	c := C.lfm_checkpoint_restore_default_options()
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

	ok := C.lfm_session_restore_checkpoint(s.c, cp.c)
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

	cOpts := C.lfm_checkpoint_restore_options{
		restore_sampler_state: C.bool(opts.RestoreSamplerState),
		restore_grammar:       C.bool(opts.RestoreGrammar),
	}

	ok := C.lfm_session_restore_checkpoint_ex(s.c, cp.c, &cOpts)
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
	n := C.lfm_json_schema_to_grammar(cSchema, C.bool(forceGBNF), nil, 0)
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
	n = C.lfm_json_schema_to_grammar(cSchema, C.bool(forceGBNF), (*C.char)(unsafe.Pointer(&buf[0])), C.size_t(len(buf)))
	if n < 0 {
		if err := getLastError(); err != nil {
			return "", err
		}
		return "", &Error{Code: ErrorInternal, Message: "failed to convert JSON schema to grammar"}
	}
	return string(buf[:n]), nil
}
