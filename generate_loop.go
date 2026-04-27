//go:build (darwin && arm64) || (linux && amd64) || (linux && arm64)

package lfg

import (
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/ebitengine/purego"
)

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

// StopReason indicates why the C-side generate loop stopped.
type StopReason int

const (
	StopReasonEOS       StopReason = 0 // End-of-generation token.
	StopReasonMaxTokens StopReason = 1 // Hit max_tokens limit.
	StopReasonCallback  StopReason = 2 // Token callback returned GenerateStop.
	StopReasonToolCall  StopReason = 3 // Model emitted tool call end token.
)

// GenerateAction controls whether the C-side generate loop continues or stops.
type GenerateAction int

const (
	GenerateContinue GenerateAction = 0
	GenerateStop     GenerateAction = 1
)

// TokenCallback is called for each generated token during a C-side generate loop.
// token is the generated token ID, piece is its text representation.
// Return GenerateContinue to keep generating, or GenerateStop to halt.
// Do not call Session methods from this callback.
type TokenCallback func(token Token, piece string) GenerateAction

// EntropyCallback is called during the C-side generate loop when the native
// entropy monitor fires. Return text to rewind to the checkpoint and inject it
// before generation continues. Return an empty string to continue unchanged.
// Do not call Session methods from this callback.
type EntropyCallback func(event EntropyEvent, embedding []float32) string

// GenerateConfig configures a C-side generate loop.
type GenerateConfig struct {
	MaxTokens               int32            // Hard token limit. 0 = use session config.
	IncludeHistoryReasoning bool             // Include <think> blocks in chat history (default false).
	IncludeOutputEmbeddings bool             // Compute per-token embeddings for generated output (default false).
	TokenCallback           TokenCallback    // Per-token callback (optional).
	EntropyCallback         EntropyCallback  // Live entropy rewind/injection callback (optional).
	ToolCallCallback        ToolCallCallback // Observation callback for auto-executed tool calls (optional).
	MaxToolRounds           int32            // Max auto-execution rounds. 0 = default (5).
}

// GenerateLoopResult holds the result from a C-side generate loop.
type GenerateLoopResult struct {
	TokenCount      int        // Number of tokens generated.
	Retrievals      int        // Live entropy rewinds/injections performed during generation.
	ConfidenceSpans int        // Reserved by lfg.cpp (currently always 0).
	SurpriseEvents  int        // Reserved by lfg.cpp (currently always 0).
	ToolCallCount   int        // Number of parsed tool calls.
	ToolRounds      int        // Auto-execution rounds completed.
	StopReason      StopReason // Why generation stopped.

	// OutputEmbeddings is a flat slice of OutputEmbeddingTokens * OutputEmbeddingSize
	// floats when GenerateConfig.IncludeOutputEmbeddings is true.
	OutputEmbeddings      []float32
	OutputEmbeddingTokens int
	OutputEmbeddingSize   int
}

// DefaultGenerateConfig returns a zero-valued generate configuration.
func DefaultGenerateConfig() GenerateConfig {
	return GenerateConfig{}
}

// ---------------------------------------------------------------------------
// Callback registry
// ---------------------------------------------------------------------------

type generateHandle struct {
	tokenCB    TokenCallback
	entropyCB  EntropyCallback
	toolCallCB ToolCallCallback
	session    *Session
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
// purego callback trampolines (created once)
// ---------------------------------------------------------------------------

var (
	genTrampolineOnce  sync.Once
	tokenTrampoline    uintptr
	entropyTrampoline  uintptr
	toolCallTrampoline uintptr
)

func initGenerateTrampolines() {
	genTrampolineOnce.Do(func() {
		registerSessionFuncs()

		// Token callback: int (*)(lfg_token token, const char *piece, int32_t piece_len, void *user_data)
		// Returns lfg_generate_action (int).
		tokenTrampoline = purego.NewCallback(func(token int32, piece uintptr, pieceLen int32, userData uintptr) int32 {
			if userData == 0 {
				return int32(GenerateContinue)
			}
			id := *(*uintptr)(unsafe.Pointer(userData))
			genCBMu.Lock()
			h, ok := genCBMap[id]
			genCBMu.Unlock()
			if !ok || h.tokenCB == nil {
				return int32(GenerateContinue)
			}
			goPiece := goStringN(piece, int(pieceLen))
			if h.session != nil {
				atomic.AddInt32(&h.session.callbackDepth, 1)
				defer atomic.AddInt32(&h.session.callbackDepth, -1)
			}
			action := h.tokenCB(Token(token), goPiece)
			return int32(action)
		})

		// Entropy callback: const char *(*)(const lfg_entropy_event *event, const float *embedding, void *user_data)
		// Returns a malloc'd C string — the C engine consumes it during the callback.
		entropyTrampoline = purego.NewCallback(func(eventPtr uintptr, embeddingPtr uintptr, userData uintptr) uintptr {
			if userData == 0 {
				return 0
			}
			id := *(*uintptr)(unsafe.Pointer(userData))
			genCBMu.Lock()
			h, ok := genCBMap[id]
			genCBMu.Unlock()
			if !ok || h.entropyCB == nil || eventPtr == 0 {
				return 0
			}

			cEvent := (*cEntropyEvent)(unsafe.Pointer(eventPtr))
			event := EntropyEvent{
				Entropy:      cEvent.Entropy,
				Normalized:   cEvent.Normalized,
				TopLogprob:   cEvent.TopLogprob,
				Token:        Token(cEvent.Token),
				NPast:        cEvent.NPast,
				CheckpointID: cEvent.CheckpointID,
				NEmbedding:   cEvent.NEmbd,
			}

			var embedding []float32
			if embeddingPtr != 0 && cEvent.NEmbd > 0 {
				embedding = unsafe.Slice((*float32)(unsafe.Pointer(embeddingPtr)), int(cEvent.NEmbd))
			}

			if h.session != nil {
				atomic.AddInt32(&h.session.callbackDepth, 1)
				defer atomic.AddInt32(&h.session.callbackDepth, -1)
			}

			inject := h.entropyCB(event, embedding)
			if inject == "" {
				return 0
			}
			return mallocCString(inject)
		})

		// Tool call callback: void (*)(const lfg_tool_call *call, const char *result, int32_t result_len, int32_t round, void *user_data)
		toolCallTrampoline = purego.NewCallback(func(callPtr uintptr, resultPtr uintptr, resultLen int32, round int32, userData uintptr) {
			if userData == 0 {
				return
			}
			id := *(*uintptr)(unsafe.Pointer(userData))
			genCBMu.Lock()
			h, ok := genCBMap[id]
			genCBMu.Unlock()
			if !ok || h.toolCallCB == nil {
				return
			}

			cCall := (*cToolCall)(unsafe.Pointer(callPtr))
			goCall := ToolCall{
				ID:        goString(cCall.ID),
				Name:      goString(cCall.Name),
				Arguments: goString(cCall.Arguments),
			}
			goResult := goStringN(resultPtr, int(resultLen))
			if h.session != nil {
				atomic.AddInt32(&h.session.callbackDepth, 1)
				defer atomic.AddInt32(&h.session.callbackDepth, -1)
			}
			h.toolCallCB(goCall, goResult, int(round))
		})
	})
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// prepareGenerate sets up the callback handle and C config.
func prepareGenerate(session *Session, config *GenerateConfig) (func(), cGenerateConfig, error) {
	registerGenerateFuncs()
	cCfg := _lfg_generate_default_config()

	if config == nil {
		return func() {}, cCfg, nil
	}

	cCfg.MaxTokens = config.MaxTokens
	cCfg.IncludeHistoryReasoning = boolToByte(config.IncludeHistoryReasoning)
	cCfg.IncludeOutputEmbeddings = boolToByte(config.IncludeOutputEmbeddings)
	cCfg.MaxToolRounds = config.MaxToolRounds

	if config.TokenCallback == nil && config.EntropyCallback == nil && config.ToolCallCallback == nil {
		return func() {}, cCfg, nil
	}

	initGenerateTrampolines()

	h := &generateHandle{
		tokenCB:    config.TokenCallback,
		entropyCB:  config.EntropyCallback,
		toolCallCB: config.ToolCallCallback,
		session:    session,
	}
	id := registerGenerateHandle(h)

	idPtr := new(uintptr)
	*idPtr = id

	if config.TokenCallback != nil {
		cCfg.TokenCB = tokenTrampoline
		cCfg.TokenCBData = uintptr(unsafe.Pointer(idPtr))
	}
	if config.EntropyCallback != nil {
		cCfg.EntropyCB = entropyTrampoline
		cCfg.EntropyCBData = uintptr(unsafe.Pointer(idPtr))
	}
	if config.ToolCallCallback != nil {
		cCfg.ToolCallCB = toolCallTrampoline
		cCfg.ToolCallCBData = uintptr(unsafe.Pointer(idPtr))
	}

	cleanup := func() {
		unregisterGenerateHandle(id)
		runtime.KeepAlive(idPtr)
		runtime.KeepAlive(h)
	}

	return cleanup, cCfg, nil
}

func resultFromC(r cGenerateResult, includeOutputEmbeddings bool) GenerateLoopResult {
	result := GenerateLoopResult{
		TokenCount:      int(r.NTokens),
		Retrievals:      int(r.NRetrievals),
		ConfidenceSpans: int(r.NConfidenceSpans),
		SurpriseEvents:  int(r.NSurpriseEvents),
		ToolCallCount:   int(r.NToolCalls),
		ToolRounds:      int(r.NToolRounds),
		StopReason:      StopReason(r.StopReason),
	}
	if includeOutputEmbeddings && r.OutputEmbeddings != 0 && r.NOutputEmbeddingFloats > 0 {
		src := unsafe.Slice((*float32)(unsafe.Pointer(r.OutputEmbeddings)), int(r.NOutputEmbeddingFloats))
		result.OutputEmbeddings = append([]float32(nil), src...)
		result.OutputEmbeddingTokens = int(r.NOutputEmbeddingTokens)
		result.OutputEmbeddingSize = int(r.OutputEmbeddingSize)
	}
	return result
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// GenerateFromState runs the C-side decode+sample loop from the current session state.
// The prompt must already be ingested via IngestTokens. This runs the entire
// generation loop in C, dramatically reducing CGo crossings compared to calling
// Decode/Sample/IngestTokens per token from Go.
func (s *Session) GenerateFromState(config GenerateConfig) (GenerateLoopResult, error) {
	if s.inGenerateCallback() {
		return GenerateLoopResult{}, callbackReentryError()
	}
	s.generateMu.Lock()
	defer s.generateMu.Unlock()
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.c == 0 {
		return GenerateLoopResult{}, &Error{Code: ErrorInvalidArgument, Message: "session is closed"}
	}

	cleanup, cCfg, err := prepareGenerate(s, &config)
	if err != nil {
		return GenerateLoopResult{}, err
	}
	defer cleanup()

	r := _lfg_session_generate(s.c, cCfg)
	return resultFromC(r, config.IncludeOutputEmbeddings), nil
}

// PromptGenerate tokenizes the prompt, ingests it, and generates in a single C call.
// This is the most efficient way to do instruction/completion-style generation,
// reducing CGo crossings to a single round-trip.
// addBOS controls whether a BOS token is prepended during tokenization.
func (s *Session) PromptGenerate(prompt string, addBOS bool, config GenerateConfig) (GenerateLoopResult, error) {
	if s.inGenerateCallback() {
		return GenerateLoopResult{}, callbackReentryError()
	}
	s.generateMu.Lock()
	defer s.generateMu.Unlock()
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.c == 0 {
		return GenerateLoopResult{}, &Error{Code: ErrorInvalidArgument, Message: "session is closed"}
	}

	promptBytes := cString(prompt)
	promptPtr := cStringPtr(promptBytes)
	cLen := int32(len(prompt))

	cleanup, cCfg, err := prepareGenerate(s, &config)
	if err != nil {
		return GenerateLoopResult{}, err
	}
	defer cleanup()

	r := _lfg_session_prompt_generate(s.c, promptPtr, cLen, addBOS, cCfg)
	runtime.KeepAlive(promptBytes)
	return resultFromC(r, config.IncludeOutputEmbeddings), nil
}

// ChatGenerate formats messages with the model's chat template, tokenizes, ingests,
// and generates in a single C call. This is the most efficient way to do
// chat-style generation, reducing the entire pipeline to one CGo round-trip.
func (s *Session) ChatGenerate(messages []ChatMessage, config GenerateConfig) (GenerateLoopResult, error) {
	if s.inGenerateCallback() {
		return GenerateLoopResult{}, callbackReentryError()
	}
	s.generateMu.Lock()
	defer s.generateMu.Unlock()
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.c == 0 {
		return GenerateLoopResult{}, &Error{Code: ErrorInvalidArgument, Message: "session is closed"}
	}
	if len(messages) == 0 {
		return GenerateLoopResult{}, &Error{Code: ErrorInvalidArgument, Message: "no messages provided"}
	}

	// Convert Go ChatMessages to C lfg_chat_message array.
	cMessages := make([]cChatMessage, len(messages))
	keepAlive := make([][]byte, 0, len(messages)*2)

	for i, msg := range messages {
		roleBytes := cString(msg.Role)
		contentBytes := cString(msg.Content)
		keepAlive = append(keepAlive, roleBytes, contentBytes)
		cMessages[i].Role = cStringPtr(roleBytes)
		cMessages[i].Content = cStringPtr(contentBytes)
	}

	cleanup, cCfg, err := prepareGenerate(s, &config)
	if err != nil {
		return GenerateLoopResult{}, err
	}
	defer cleanup()

	r := _lfg_session_chat_generate(s.c, uintptr(unsafe.Pointer(&cMessages[0])), uintptr(len(messages)), cCfg)
	runtime.KeepAlive(keepAlive)
	runtime.KeepAlive(cMessages)
	return resultFromC(r, config.IncludeOutputEmbeddings), nil
}
