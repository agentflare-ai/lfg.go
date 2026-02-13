//go:build (darwin && arm64) || (linux && amd64) || (linux && arm64)

package lfg

// C struct mirrors for purego. Field types are chosen to match C layout exactly:
// - C bool in struct fields → byte (1 byte)
// - C enum → int32 (4 bytes, C default)
// - C pointers, size_t → uintptr
// - C int32_t → int32
// - C uint32_t → uint32
// - C int64_t → int64
// - C float → float32
// - C double → float64
// - C function pointers → uintptr

// cBatch mirrors lfg_batch.
type cBatch struct {
	NTokens int32
	_       [4]byte // padding on 64-bit
	Token   uintptr // *lfg_token
	Embd    uintptr // *float
	Pos     uintptr // *lfg_pos
	NSeqID  uintptr // *int32_t
	SeqID   uintptr // **lfg_seq_id
	Logits  uintptr // *int8_t
}

// cModelParams mirrors struct lfg_model_params.
type cModelParams struct {
	Devices              uintptr // ggml_backend_dev_t *
	TensorBuftOverrides  uintptr // const struct lfg_model_tensor_buft_override *
	NGPULayers           int32
	SplitMode            int32 // enum lfg_split_mode
	MainGPU              int32
	_                    [4]byte // padding
	TensorSplit          uintptr // const float *
	ProgressCallback     uintptr // lfg_progress_callback
	ProgressCallbackData uintptr // void *
	KVOverrides          uintptr // const struct lfg_model_kv_override *
	VocabOnly            byte    // bool
	UseMmap              byte    // bool
	UseDirectIO          byte    // bool
	UseMlock             byte    // bool
	CheckTensors         byte    // bool
	UseExtraBuffts       byte    // bool
	NoHost               byte    // bool
	NoAlloc              byte    // bool
	TypeK                int32   // enum ggml_type
	TypeV                int32   // enum ggml_type
	SWAFull              byte    // bool
	_                    [3]byte // padding to next alignment
}

// cContextParams mirrors struct lfg_context_params.
type cContextParams struct {
	NCtx            uint32
	NBatch          uint32
	NUBatch         uint32
	NSeqMax         uint32
	NThreads        int32
	NThreadsBatch   int32
	RopeScalingType int32 // enum lfg_rope_scaling_type
	PoolingType     int32 // enum lfg_pooling_type
	AttentionType   int32 // enum lfg_attention_type
	FlashAttnType   int32 // enum lfg_flash_attn_type
	RopeFreqBase    float32
	RopeFreqScale   float32
	YarnExtFactor   float32
	YarnAttnFactor  float32
	YarnBetaFast    float32
	YarnBetaSlow    float32
	YarnOrigCtx     uint32
	DefragThold     float32
	CbEval          uintptr // ggml_backend_sched_eval_callback
	CbEvalUserData  uintptr // void *
	TypeK           int32   // enum ggml_type
	TypeV           int32   // enum ggml_type
	AbortCallback   uintptr // ggml_abort_callback
	AbortCallbackData uintptr // void *
	Embeddings      byte // bool
	OffloadKQV      byte // bool
	NoPerf          byte // bool
	OpOffload       byte // bool
	SWAFull         byte // bool
	KVUnified       byte // bool
	_               [2]byte // padding
	Samplers        uintptr // struct lfg_sampler_seq_config *
	NSamplers       uintptr // size_t
}

// cSamplerChainParams mirrors lfg_sampler_chain_params.
type cSamplerChainParams struct {
	NoPerf byte // bool
}

// cLogitBias mirrors lfg_logit_bias.
type cLogitBias struct {
	Token int32
	Bias  float32
}

// cChatMessage mirrors lfg_chat_message.
type cChatMessage struct {
	Role       uintptr // const char *
	Content    uintptr // const char *
	ToolCalls  uintptr // const struct lfg_tool_call * (nullable)
	NToolCalls int32   // int32_t (0 = none)
	_          [4]byte // padding to 8-byte alignment
	ToolCallID uintptr // const char * (nullable, for role="tool")
}

// cPerfContextData mirrors lfg_perf_context_data.
type cPerfContextData struct {
	TStartMs float64
	TLoadMs  float64
	TPEvalMs float64
	TEvalMs  float64
	NPEval   int32
	NEval    int32
	NReused  int32
	_        [4]byte // padding to 8-byte alignment
}

// cPerfSamplerData mirrors lfg_perf_sampler_data.
type cPerfSamplerData struct {
	TSampleMs float64
	NSample   int32
	_         [4]byte // padding
}

// cSamplingConfig mirrors lfg_sampling_config.
type cSamplingConfig struct {
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

// cSessionConfig mirrors lfg_session_config.
type cSessionConfig struct {
	NThreads                int32   // int
	NCtx                    int32   // int
	NBatch                  int32   // int
	EnableHealing           byte    // bool
	StructuredCheckpointing byte    // bool
	_                       [2]byte // padding
	ReasoningBudget         int32   // int
	MaxTokens               int32   // int32_t
	ToolScoreMode           int32   // lfg_tool_score_mode (enum)
	ToolMinScore            float32 // float
	Sampling                cSamplingConfig
}

// cEntropyEvent mirrors lfg_entropy_event.
type cEntropyEvent struct {
	Entropy      float32
	Normalized   float32
	TopLogprob   float32
	Token        int32
	NPast        int32
	CheckpointID int32
	NEmbd        int32
}

// cEntropyMonitorConfig mirrors lfg_entropy_monitor_config.
type cEntropyMonitorConfig struct {
	Threshold      float32
	CooldownTokens int32
	RingSize       int32
	GateMode       int32 // lfg_entropy_gate_mode
}

// cConfidenceEvent mirrors lfg_confidence_event.
type cConfidenceEvent struct {
	MeanEntropy float32
	MinEntropy  float32
	SpanLength  int32
	StartPos    int32
	EndPos      int32
	NEmbd       int32
	SpanText    uintptr // const char *
	SpanTextLen int32
	_           [4]byte // padding to 8-byte alignment
}

// cConfidenceMonitorConfig mirrors lfg_confidence_monitor_config.
type cConfidenceMonitorConfig struct {
	Threshold        float32
	MinSpan          int32
	RingSize         int32
	IncludeReasoning byte // bool
	_                [3]byte
	GateMode         int32 // lfg_confidence_gate_mode
}

// cSurpriseEvent mirrors lfg_surprise_event.
type cSurpriseEvent struct {
	MeanSurprise     float32
	MaxSurprise      float32
	NAboveThreshold  int32
	NTokensEvaluated int32
	NEmbd            int32
}

// cSurpriseMonitorConfig mirrors lfg_surprise_monitor_config.
type cSurpriseMonitorConfig struct {
	Threshold        float32
	IncludeReasoning byte // bool
	_                [3]byte
	GateMode         int32 // lfg_surprise_gate_mode
}

// cToolCall mirrors lfg_tool_call.
type cToolCall struct {
	ID        uintptr // const char *
	Name      uintptr // const char *
	Arguments uintptr // const char *
}

// cToolDesc mirrors lfg_tool_desc.
type cToolDesc struct {
	Name        uintptr // const char *
	Description uintptr // const char *
	Parameters  uintptr // const char *
	Fn          uintptr // lfg_tool_fn
	FnUserData  uintptr // void *
}

// cGenerateConfig mirrors lfg_generate_config.
type cGenerateConfig struct {
	MaxTokens               int32
	IncludeHistoryReasoning byte    // bool
	_                       [3]byte // padding to 8
	TokenCB                 uintptr // lfg_generate_token_cb
	TokenCBData             uintptr // void *
	EntropyCB               uintptr // lfg_generate_entropy_cb
	EntropyCBData           uintptr // void *
	ConfidenceCB            uintptr // lfg_generate_confidence_cb
	ConfidenceCBData        uintptr // void *
	SurpriseCB              uintptr // lfg_generate_surprise_cb
	SurpriseCBData          uintptr // void *
	ToolCallCB              uintptr // lfg_tool_call_cb
	ToolCallCBData          uintptr // void *
	MaxToolRounds           int32   // 0 = default (5)
	_pad                    [4]byte // padding to 8-byte alignment
}

// cGenerateResult mirrors lfg_generate_result.
type cGenerateResult struct {
	NTokens          int32
	NRetrievals      int32
	NConfidenceSpans int32
	NSurpriseEvents  int32
	NToolCalls       int32
	NToolRounds      int32
	StopReason       int32 // lfg_stop_reason
}

// cCheckpointRestoreOptions mirrors lfg_checkpoint_restore_options.
type cCheckpointRestoreOptions struct {
	RestoreSamplerState byte // bool
	RestoreGrammar      byte // bool
}

// cModelLoadConfig mirrors lfg_model_load_config.
type cModelLoadConfig struct {
	ModelPath  uintptr // const char *
	UseMmap    byte    // bool
	UseMlock   byte    // bool
	_          [2]byte // padding
	NGPULayers int32   // int
}

// cModelStats mirrors lfg_model_stats (from lfg_api.h).
type cModelStats struct {
	NParams   uint64
	SizeBytes uint64
	NVocab    int32
	NCtxTrain int32
}

// cTokenData mirrors lfg_token_data.
type cTokenData struct {
	ID    int32   // lfg_token
	Logit float32 // float
	P     float32 // float
}

// cTokenDataArray mirrors lfg_token_data_array.
type cTokenDataArray struct {
	Data     uintptr // lfg_token_data *
	Size     uintptr // size_t
	Selected int64   // int64_t
	Sorted   byte    // bool
	_        [7]byte // padding to 8-byte alignment
}
