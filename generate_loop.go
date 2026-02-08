package lfg

/*
#include "lfg_api.h"
#include <stdlib.h>

// Forward-declare Go-exported callback trampolines.
// Types must match exactly what cgo generates (no const qualifiers).
extern int goGenerateTokenCB(int32_t token, char *piece, int32_t piece_len, void *user_data);
extern char * goGenerateEntropyCB(lfg_entropy_event *event, float *embedding, void *user_data);
extern void goGenerateConfidenceCB(lfg_confidence_event *event, float *embedding, void *user_data);
extern void goGenerateSurpriseCB(lfg_surprise_event *event, float *embedding, void *user_data);

// C trampolines matching the callback typedefs.
static lfg_generate_action c_token_trampoline(lfg_token token, const char *piece, int32_t piece_len, void *user_data) {
    return (lfg_generate_action)goGenerateTokenCB((int32_t)token, (char *)piece, piece_len, user_data);
}

static const char * c_entropy_trampoline(const lfg_entropy_event *event, const float *embedding, void *user_data) {
    return (const char *)goGenerateEntropyCB((lfg_entropy_event *)event, (float *)embedding, user_data);
}

static void c_confidence_trampoline(const lfg_confidence_event *event, const float *embedding, void *user_data) {
    goGenerateConfidenceCB((lfg_confidence_event *)event, (float *)embedding, user_data);
}

static void c_surprise_trampoline(const lfg_surprise_event *event, const float *embedding, void *user_data) {
    goGenerateSurpriseCB((lfg_surprise_event *)event, (float *)embedding, user_data);
}

static lfg_generate_token_cb get_token_trampoline(void) {
    return c_token_trampoline;
}

static lfg_generate_entropy_cb get_entropy_trampoline(void) {
    return c_entropy_trampoline;
}

static lfg_generate_confidence_cb get_confidence_trampoline(void) {
    return c_confidence_trampoline;
}

static lfg_generate_surprise_cb get_surprise_trampoline(void) {
    return c_surprise_trampoline;
}
*/
import "C"
import (
	"sync"
	"unsafe"
)

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

// StopReason indicates why the C-side generate loop stopped.
type StopReason int

const (
	StopReasonEOS       StopReason = C.LFG_STOP_EOS        // End-of-generation token.
	StopReasonMaxTokens StopReason = C.LFG_STOP_MAX_TOKENS // Hit max_tokens limit.
	StopReasonCallback  StopReason = C.LFG_STOP_CALLBACK   // Token callback returned GenerateStop.
)

// GenerateAction controls whether the C-side generate loop continues or stops.
type GenerateAction int

const (
	GenerateContinue GenerateAction = C.LFG_GENERATE_CONTINUE
	GenerateStop     GenerateAction = C.LFG_GENERATE_STOP
)

// TokenCallback is called for each generated token during a C-side generate loop.
// token is the generated token ID, piece is its text representation.
// Return GenerateContinue to keep generating, or GenerateStop to halt.
type TokenCallback func(token Token, piece string) GenerateAction

// EntropyCallback is called when entropy exceeds the configured threshold.
// event contains entropy metrics, embedding is the token's embedding vector (may be nil).
// Return a non-empty string to inject (the C loop handles rewind + tokenize + ingest),
// or empty string to skip this event and continue generating.
type EntropyCallback func(event EntropyEvent, embedding []float32) string

// ConfidenceCallback is called when a sustained low-entropy span ends.
// event contains span metrics, embedding is the mean-pooled embedding (may be nil).
// Informational only — no return value needed (unlike EntropyCallback, no rewind).
type ConfidenceCallback func(event ConfidenceEvent, embedding []float32)

// SurpriseCallback is called when a sustained high-surprise span is detected during prompt ingestion.
// event contains surprise metrics, embedding is the mean-pooled embedding (may be nil).
// Informational only — no return value needed.
type SurpriseCallback func(event SurpriseEvent, embedding []float32)

// GenerateConfig configures a C-side generate loop.
type GenerateConfig struct {
	MaxTokens          int32              // Hard token limit. 0 = use session config.
	TokenCallback      TokenCallback      // Per-token callback (optional).
	EntropyCallback    EntropyCallback    // Entropy threshold callback (optional).
	ConfidenceCallback ConfidenceCallback // Confidence span callback (optional).
	SurpriseCallback   SurpriseCallback   // Surprise span callback (optional).
}

// GenerateLoopResult holds the result from a C-side generate loop.
type GenerateLoopResult struct {
	TokenCount      int        // Number of tokens generated.
	Retrievals      int        // Number of entropy-triggered rewind+inject cycles.
	ConfidenceSpans int        // Number of confidence events fired.
	SurpriseEvents  int        // Number of surprise events from prompt ingestion (0 or 1).
	StopReason      StopReason // Why generation stopped.
}

// DefaultGenerateConfig returns a zero-valued generate configuration.
func DefaultGenerateConfig() GenerateConfig {
	return GenerateConfig{}
}

// ---------------------------------------------------------------------------
// Callback registry
// ---------------------------------------------------------------------------

type generateHandle struct {
	tokenCB      TokenCallback
	entropyCB    EntropyCallback
	confidenceCB ConfidenceCallback
	surpriseCB   SurpriseCallback
	cStrings     []unsafe.Pointer // CStrings allocated by entropy callback, freed after generate.
}

var (
	genCBMu     sync.Mutex
	genCBMap    = make(map[uintptr]*generateHandle)
	genCBNextID uintptr
)

func registerGenerateHandle(h *generateHandle) uintptr {
	genCBMu.Lock()
	defer genCBMu.Unlock()
	genCBNextID++
	id := genCBNextID
	genCBMap[id] = h
	return id
}

func unregisterGenerateHandle(id uintptr) {
	genCBMu.Lock()
	defer genCBMu.Unlock()
	delete(genCBMap, id)
}

// ---------------------------------------------------------------------------
// Exported callbacks (called from C trampolines)
// ---------------------------------------------------------------------------

//export goGenerateTokenCB
func goGenerateTokenCB(token C.int32_t, piece *C.char, pieceLen C.int32_t, userData unsafe.Pointer) C.int {
	if userData == nil {
		return C.int(GenerateContinue)
	}
	id := *(*uintptr)(userData)
	genCBMu.Lock()
	h, ok := genCBMap[id]
	genCBMu.Unlock()
	if !ok || h.tokenCB == nil {
		return C.int(GenerateContinue)
	}

	var goPiece string
	if piece != nil && pieceLen > 0 {
		goPiece = C.GoStringN(piece, C.int(pieceLen))
	}

	action := h.tokenCB(Token(token), goPiece)
	return C.int(action)
}

//export goGenerateEntropyCB
func goGenerateEntropyCB(event *C.lfg_entropy_event, embedding *C.float, userData unsafe.Pointer) *C.char {
	if userData == nil {
		return nil
	}
	id := *(*uintptr)(userData)
	genCBMu.Lock()
	h, ok := genCBMap[id]
	genCBMu.Unlock()
	if !ok || h.entropyCB == nil {
		return nil
	}

	goEvent := EntropyEvent{
		Entropy:      float32(event.entropy),
		Normalized:   float32(event.normalized),
		TopLogprob:   float32(event.top_logprob),
		Token:        Token(event.token),
		NPast:        int32(event.n_past),
		CheckpointID: int32(event.checkpoint_id),
		NEmbedding:   int32(event.n_embd),
	}

	var goEmbed []float32
	if embedding != nil && goEvent.NEmbedding > 0 {
		goEmbed = unsafe.Slice((*float32)(unsafe.Pointer(embedding)), goEvent.NEmbedding)
	}

	result := h.entropyCB(goEvent, goEmbed)
	if result == "" {
		return nil
	}

	cs := C.CString(result)
	// Track for cleanup after generate returns.
	genCBMu.Lock()
	h.cStrings = append(h.cStrings, unsafe.Pointer(cs))
	genCBMu.Unlock()
	return cs
}

//export goGenerateConfidenceCB
func goGenerateConfidenceCB(event *C.lfg_confidence_event, embedding *C.float, userData unsafe.Pointer) {
	if userData == nil {
		return
	}
	id := *(*uintptr)(userData)
	genCBMu.Lock()
	h, ok := genCBMap[id]
	genCBMu.Unlock()
	if !ok || h.confidenceCB == nil {
		return
	}

	goEvent := ConfidenceEvent{
		MeanEntropy: float32(event.mean_entropy),
		MinEntropy:  float32(event.min_entropy),
		SpanLength:  int32(event.span_length),
		StartPos:    int32(event.start_pos),
		EndPos:      int32(event.end_pos),
		NEmbedding:  int32(event.n_embd),
	}

	var goEmbed []float32
	if embedding != nil && goEvent.NEmbedding > 0 {
		goEmbed = unsafe.Slice((*float32)(unsafe.Pointer(embedding)), goEvent.NEmbedding)
	}

	h.confidenceCB(goEvent, goEmbed)
}

//export goGenerateSurpriseCB
func goGenerateSurpriseCB(event *C.lfg_surprise_event, embedding *C.float, userData unsafe.Pointer) {
	if userData == nil {
		return
	}
	id := *(*uintptr)(userData)
	genCBMu.Lock()
	h, ok := genCBMap[id]
	genCBMu.Unlock()
	if !ok || h.surpriseCB == nil {
		return
	}

	goEvent := SurpriseEvent{
		MeanSurprise:     float32(event.mean_surprise),
		MaxSurprise:      float32(event.max_surprise),
		NAboveThreshold:  int32(event.n_above_threshold),
		NTokensEvaluated: int32(event.n_tokens_evaluated),
		NEmbedding:       int32(event.n_embd),
	}

	var goEmbed []float32
	if embedding != nil && goEvent.NEmbedding > 0 {
		goEmbed = unsafe.Slice((*float32)(unsafe.Pointer(embedding)), goEvent.NEmbedding)
	}

	h.surpriseCB(goEvent, goEmbed)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// prepareGenerate sets up the callback handle and C config.
// The config is passed by value to C, so embedded Go pointers in user_data
// fields are safe (CGo only checks pointer arguments, not value arguments).
func prepareGenerate(config *GenerateConfig) (func(), C.lfg_generate_config) {
	cCfg := C.lfg_generate_default_config()

	if config == nil {
		return func() {}, cCfg
	}

	cCfg.max_tokens = C.int32_t(config.MaxTokens)

	if config.TokenCallback == nil && config.EntropyCallback == nil && config.ConfidenceCallback == nil && config.SurpriseCallback == nil {
		return func() {}, cCfg
	}

	h := &generateHandle{
		tokenCB:      config.TokenCallback,
		entropyCB:    config.EntropyCallback,
		confidenceCB: config.ConfidenceCallback,
		surpriseCB:   config.SurpriseCallback,
	}
	id := registerGenerateHandle(h)

	idPtr := new(uintptr)
	*idPtr = id

	if config.TokenCallback != nil {
		cCfg.token_cb = C.get_token_trampoline()
		cCfg.token_cb_data = unsafe.Pointer(idPtr)
	}
	if config.EntropyCallback != nil {
		cCfg.entropy_cb = C.get_entropy_trampoline()
		cCfg.entropy_cb_data = unsafe.Pointer(idPtr)
	}
	if config.ConfidenceCallback != nil {
		cCfg.confidence_cb = C.get_confidence_trampoline()
		cCfg.confidence_cb_data = unsafe.Pointer(idPtr)
	}
	if config.SurpriseCallback != nil {
		cCfg.surprise_cb = C.get_surprise_trampoline()
		cCfg.surprise_cb_data = unsafe.Pointer(idPtr)
	}

	cleanup := func() {
		for _, p := range h.cStrings {
			C.free(p)
		}
		unregisterGenerateHandle(id)
	}

	return cleanup, cCfg
}

func resultFromC(r C.lfg_generate_result) GenerateLoopResult {
	return GenerateLoopResult{
		TokenCount:      int(r.n_tokens),
		Retrievals:      int(r.n_retrievals),
		ConfidenceSpans: int(r.n_confidence_spans),
		SurpriseEvents:  int(r.n_surprise_events),
		StopReason:      StopReason(r.stop_reason),
	}
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// GenerateFromState runs the C-side decode+sample loop from the current session state.
// The prompt must already be ingested via IngestTokens. This runs the entire
// generation loop in C, dramatically reducing CGo crossings compared to calling
// Decode/Sample/IngestTokens per token from Go.
func (s *Session) GenerateFromState(config GenerateConfig) (GenerateLoopResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return GenerateLoopResult{}, &Error{Code: ErrorInvalidArgument, Message: "session is closed"}
	}

	cleanup, cCfg := prepareGenerate(&config)
	defer cleanup()

	r := C.lfg_session_generate(s.c, cCfg)
	return resultFromC(r), nil
}

// PromptGenerate tokenizes the prompt, ingests it, and generates in a single C call.
// This is the most efficient way to do instruction/completion-style generation,
// reducing CGo crossings to a single round-trip.
// addBOS controls whether a BOS token is prepended during tokenization.
func (s *Session) PromptGenerate(prompt string, addBOS bool, config GenerateConfig) (GenerateLoopResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return GenerateLoopResult{}, &Error{Code: ErrorInvalidArgument, Message: "session is closed"}
	}

	cPrompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cPrompt))
	cLen := C.int32_t(len(prompt))

	cleanup, cCfg := prepareGenerate(&config)
	defer cleanup()

	r := C.lfg_session_prompt_generate(s.c, cPrompt, cLen, C.bool(addBOS), cCfg)
	return resultFromC(r), nil
}

// ChatGenerate formats messages with the model's chat template, tokenizes, ingests,
// and generates in a single C call. This is the most efficient way to do
// chat-style generation, reducing the entire pipeline to one CGo round-trip.
func (s *Session) ChatGenerate(messages []ChatMessage, config GenerateConfig) (GenerateLoopResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return GenerateLoopResult{}, &Error{Code: ErrorInvalidArgument, Message: "session is closed"}
	}
	if len(messages) == 0 {
		return GenerateLoopResult{}, &Error{Code: ErrorInvalidArgument, Message: "no messages provided"}
	}

	// Convert Go ChatMessages to C lfg_chat_message array.
	cMessages := make([]C.struct_lfg_chat_message, len(messages))
	cStrs := make([]*C.char, 0, len(messages)*2)
	defer func() {
		for _, p := range cStrs {
			C.free(unsafe.Pointer(p))
		}
	}()

	for i, msg := range messages {
		cRole := C.CString(msg.Role)
		cContent := C.CString(msg.Content)
		cStrs = append(cStrs, cRole, cContent)
		cMessages[i].role = cRole
		cMessages[i].content = cContent
	}

	cleanup, cCfg := prepareGenerate(&config)
	defer cleanup()

	r := C.lfg_session_chat_generate(s.c, &cMessages[0], C.size_t(len(messages)), cCfg)
	return resultFromC(r), nil
}
