//go:build (darwin && arm64) || (linux && amd64) || (linux && arm64)

package lfg

import (
	"runtime"
	"sync"
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
	MaxTokens                int32              // Hard token limit. 0 = use session config.
	IncludeHistoryReasoning  bool               // Include <think> blocks in chat history (default false).
	TokenCallback            TokenCallback      // Per-token callback (optional).
	EntropyCallback          EntropyCallback    // Entropy threshold callback (optional).
	ConfidenceCallback       ConfidenceCallback // Confidence span callback (optional).
	SurpriseCallback         SurpriseCallback   // Surprise span callback (optional).
	ToolCallCallback         ToolCallCallback   // Observation callback for auto-executed tool calls (optional).
	MaxToolRounds            int32              // Max auto-execution rounds. 0 = default (5).
}

// GenerateLoopResult holds the result from a C-side generate loop.
type GenerateLoopResult struct {
	TokenCount      int        // Number of tokens generated.
	Retrievals      int        // Number of entropy-triggered rewind+inject cycles.
	ConfidenceSpans int        // Number of confidence events fired.
	SurpriseEvents  int        // Number of surprise events from prompt ingestion (0 or 1).
	ToolCallCount   int        // Number of parsed tool calls.
	ToolRounds      int        // Auto-execution rounds completed.
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
	toolCallCB   ToolCallCallback
	pinnedStrings [][]byte // Pinned Go strings returned by entropy callback; GC'd after generate.
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
	genTrampolineOnce     sync.Once
	tokenTrampoline       uintptr
	entropyTrampoline     uintptr
	confidenceTrampoline  uintptr
	surpriseTrampoline    uintptr
	toolCallTrampoline    uintptr
)

func initGenerateTrampolines() {
	genTrampolineOnce.Do(func() {
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
			action := h.tokenCB(Token(token), goPiece)
			return int32(action)
		})

		// Entropy callback: const char * (*)(const lfg_entropy_event *event, const float *embedding, void *user_data)
		// Returns a pointer to a null-terminated C string (or 0/nil to skip).
		entropyTrampoline = purego.NewCallback(func(eventPtr uintptr, embeddingPtr uintptr, userData uintptr) uintptr {
			if userData == 0 {
				return 0
			}
			id := *(*uintptr)(unsafe.Pointer(userData))
			genCBMu.Lock()
			h, ok := genCBMap[id]
			genCBMu.Unlock()
			if !ok || h.entropyCB == nil {
				return 0
			}

			event := (*cEntropyEvent)(unsafe.Pointer(eventPtr))
			goEvent := EntropyEvent{
				Entropy:      event.Entropy,
				Normalized:   event.Normalized,
				TopLogprob:   event.TopLogprob,
				Token:        Token(event.Token),
				NPast:        event.NPast,
				CheckpointID: event.CheckpointID,
				NEmbedding:   event.NEmbd,
			}

			var goEmbed []float32
			if embeddingPtr != 0 && goEvent.NEmbedding > 0 {
				goEmbed = unsafe.Slice((*float32)(unsafe.Pointer(embeddingPtr)), goEvent.NEmbedding)
			}

			result := h.entropyCB(goEvent, goEmbed)
			if result == "" {
				return 0
			}

			// Pin the string in Go memory. GC will clean it up after generate returns.
			pinned := cString(result)
			genCBMu.Lock()
			h.pinnedStrings = append(h.pinnedStrings, pinned)
			genCBMu.Unlock()
			return cStringPtr(pinned)
		})

		// Confidence callback: void (*)(const lfg_confidence_event *event, const float *embedding, void *user_data)
		confidenceTrampoline = purego.NewCallback(func(eventPtr uintptr, embeddingPtr uintptr, userData uintptr) {
			if userData == 0 {
				return
			}
			id := *(*uintptr)(unsafe.Pointer(userData))
			genCBMu.Lock()
			h, ok := genCBMap[id]
			genCBMu.Unlock()
			if !ok || h.confidenceCB == nil {
				return
			}

			event := (*cConfidenceEvent)(unsafe.Pointer(eventPtr))
			goEvent := ConfidenceEvent{
				MeanEntropy: event.MeanEntropy,
				MinEntropy:  event.MinEntropy,
				SpanLength:  event.SpanLength,
				StartPos:    event.StartPos,
				EndPos:      event.EndPos,
				NEmbedding:  event.NEmbd,
				SpanText:    goStringN(event.SpanText, int(event.SpanTextLen)),
			}

			var goEmbed []float32
			if embeddingPtr != 0 && goEvent.NEmbedding > 0 {
				goEmbed = unsafe.Slice((*float32)(unsafe.Pointer(embeddingPtr)), goEvent.NEmbedding)
			}

			h.confidenceCB(goEvent, goEmbed)
		})

		// Surprise callback: void (*)(const lfg_surprise_event *event, const float *embedding, void *user_data)
		surpriseTrampoline = purego.NewCallback(func(eventPtr uintptr, embeddingPtr uintptr, userData uintptr) {
			if userData == 0 {
				return
			}
			id := *(*uintptr)(unsafe.Pointer(userData))
			genCBMu.Lock()
			h, ok := genCBMap[id]
			genCBMu.Unlock()
			if !ok || h.surpriseCB == nil {
				return
			}

			event := (*cSurpriseEvent)(unsafe.Pointer(eventPtr))
			goEvent := SurpriseEvent{
				MeanSurprise:     event.MeanSurprise,
				MaxSurprise:      event.MaxSurprise,
				NAboveThreshold:  event.NAboveThreshold,
				NTokensEvaluated: event.NTokensEvaluated,
				NEmbedding:       event.NEmbd,
			}

			var goEmbed []float32
			if embeddingPtr != 0 && goEvent.NEmbedding > 0 {
				goEmbed = unsafe.Slice((*float32)(unsafe.Pointer(embeddingPtr)), goEvent.NEmbedding)
			}

			h.surpriseCB(goEvent, goEmbed)
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
			h.toolCallCB(goCall, goResult, int(round))
		})
	})
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// prepareGenerate sets up the callback handle and C config.
func prepareGenerate(config *GenerateConfig) (func(), cGenerateConfig) {
	registerGenerateFuncs()
	cCfg := _lfg_generate_default_config()

	if config == nil {
		return func() {}, cCfg
	}

	cCfg.MaxTokens = config.MaxTokens
	cCfg.IncludeHistoryReasoning = boolToByte(config.IncludeHistoryReasoning)
	cCfg.MaxToolRounds = config.MaxToolRounds

	if config.TokenCallback == nil && config.EntropyCallback == nil && config.ConfidenceCallback == nil && config.SurpriseCallback == nil && config.ToolCallCallback == nil {
		return func() {}, cCfg
	}

	initGenerateTrampolines()

	h := &generateHandle{
		tokenCB:      config.TokenCallback,
		entropyCB:    config.EntropyCallback,
		confidenceCB: config.ConfidenceCallback,
		surpriseCB:   config.SurpriseCallback,
		toolCallCB:   config.ToolCallCallback,
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
	if config.ConfidenceCallback != nil {
		cCfg.ConfidenceCB = confidenceTrampoline
		cCfg.ConfidenceCBData = uintptr(unsafe.Pointer(idPtr))
	}
	if config.SurpriseCallback != nil {
		cCfg.SurpriseCB = surpriseTrampoline
		cCfg.SurpriseCBData = uintptr(unsafe.Pointer(idPtr))
	}
	if config.ToolCallCallback != nil {
		cCfg.ToolCallCB = toolCallTrampoline
		cCfg.ToolCallCBData = uintptr(unsafe.Pointer(idPtr))
	}

	cleanup := func() {
		// pinnedStrings are GC'd automatically — no C.free needed with purego.
		unregisterGenerateHandle(id)
		runtime.KeepAlive(idPtr)
		runtime.KeepAlive(h)
	}

	return cleanup, cCfg
}

func resultFromC(r cGenerateResult) GenerateLoopResult {
	return GenerateLoopResult{
		TokenCount:      int(r.NTokens),
		Retrievals:      int(r.NRetrievals),
		ConfidenceSpans: int(r.NConfidenceSpans),
		SurpriseEvents:  int(r.NSurpriseEvents),
		ToolCallCount:   int(r.NToolCalls),
		ToolRounds:      int(r.NToolRounds),
		StopReason:      StopReason(r.StopReason),
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
	if s.c == 0 {
		return GenerateLoopResult{}, &Error{Code: ErrorInvalidArgument, Message: "session is closed"}
	}

	cleanup, cCfg := prepareGenerate(&config)
	defer cleanup()

	r := _lfg_session_generate(s.c, cCfg)
	return resultFromC(r), nil
}

// PromptGenerate tokenizes the prompt, ingests it, and generates in a single C call.
// This is the most efficient way to do instruction/completion-style generation,
// reducing CGo crossings to a single round-trip.
// addBOS controls whether a BOS token is prepended during tokenization.
func (s *Session) PromptGenerate(prompt string, addBOS bool, config GenerateConfig) (GenerateLoopResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return GenerateLoopResult{}, &Error{Code: ErrorInvalidArgument, Message: "session is closed"}
	}

	promptBytes := cString(prompt)
	promptPtr := cStringPtr(promptBytes)
	cLen := int32(len(prompt))

	cleanup, cCfg := prepareGenerate(&config)
	defer cleanup()

	r := _lfg_session_prompt_generate(s.c, promptPtr, cLen, addBOS, cCfg)
	runtime.KeepAlive(promptBytes)
	return resultFromC(r), nil
}

// ChatGenerate formats messages with the model's chat template, tokenizes, ingests,
// and generates in a single C call. This is the most efficient way to do
// chat-style generation, reducing the entire pipeline to one CGo round-trip.
func (s *Session) ChatGenerate(messages []ChatMessage, config GenerateConfig) (GenerateLoopResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
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

	cleanup, cCfg := prepareGenerate(&config)
	defer cleanup()

	r := _lfg_session_chat_generate(s.c, uintptr(unsafe.Pointer(&cMessages[0])), uintptr(len(messages)), cCfg)
	runtime.KeepAlive(keepAlive)
	runtime.KeepAlive(cMessages)
	return resultFromC(r), nil
}
