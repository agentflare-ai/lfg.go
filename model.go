package lfg

/*
#include "lfg_inference.h"
#include <stdlib.h>
*/
import "C"
import (
	"runtime"
	"sync"
	"unsafe"
)

// ModelConfig holds Go-native model loading parameters.
type ModelConfig struct {
	GPULayers    int
	SplitMode    SplitMode
	MainGPU      int
	VocabOnly    bool
	Mmap         bool
	Mlock        bool
	CheckTensors bool
	Progress     ProgressCallback
}

// ModelOption configures model loading parameters.
type ModelOption func(*ModelConfig)

// WithGPULayers sets the number of layers to offload to GPU.
// A negative value offloads all layers.
func WithGPULayers(n int) ModelOption {
	return func(cfg *ModelConfig) {
		cfg.GPULayers = n
	}
}

// WithSplitMode sets how the model is split across GPUs.
func WithSplitMode(mode SplitMode) ModelOption {
	return func(cfg *ModelConfig) {
		cfg.SplitMode = mode
	}
}

// WithMainGPU sets the main GPU index for single-GPU mode.
func WithMainGPU(gpu int) ModelOption {
	return func(cfg *ModelConfig) {
		cfg.MainGPU = gpu
	}
}

// WithVocabOnly loads only the vocabulary, no weights.
func WithVocabOnly(v bool) ModelOption {
	return func(cfg *ModelConfig) {
		cfg.VocabOnly = v
	}
}

// WithMmap enables or disables memory-mapped file loading.
func WithMmap(v bool) ModelOption {
	return func(cfg *ModelConfig) {
		cfg.Mmap = v
	}
}

// WithMlock forces the system to keep the model in RAM.
func WithMlock(v bool) ModelOption {
	return func(cfg *ModelConfig) {
		cfg.Mlock = v
	}
}

// WithCheckTensors enables validation of model tensor data.
func WithCheckTensors(v bool) ModelOption {
	return func(cfg *ModelConfig) {
		cfg.CheckTensors = v
	}
}

// WithProgressCallback sets a callback for model loading progress.
func WithProgressCallback(cb ProgressCallback) ModelOption {
	return func(cfg *ModelConfig) {
		cfg.Progress = cb
	}
}

// Model wraps an lfg_model pointer with thread-safe access.
type Model struct {
	mu sync.RWMutex
	c  *C.struct_lfg_model
}

// LoadModel loads a model from a GGUF file. Automatically initializes the backend.
func LoadModel(path string, opts ...ModelOption) (*Model, error) {
	ensureBackend()

	// Build Go config from defaults + options.
	defaults := C.lfg_model_default_params()
	cfg := ModelConfig{
		GPULayers: int(defaults.n_gpu_layers),
		SplitMode: SplitMode(defaults.split_mode),
		MainGPU:   int(defaults.main_gpu),
		VocabOnly: bool(defaults.vocab_only),
		Mmap:      bool(defaults.use_mmap),
		Mlock:     bool(defaults.use_mlock),
		CheckTensors: bool(defaults.check_tensors),
	}
	for _, opt := range opts {
		opt(&cfg)
	}

	// Convert to C params.
	params := defaults
	params.n_gpu_layers = C.int32_t(cfg.GPULayers)
	params.split_mode = C.enum_lfg_split_mode(cfg.SplitMode)
	params.main_gpu = C.int32_t(cfg.MainGPU)
	params.vocab_only = C.bool(cfg.VocabOnly)
	params.use_mmap = C.bool(cfg.Mmap)
	params.use_mlock = C.bool(cfg.Mlock)
	params.check_tensors = C.bool(cfg.CheckTensors)

	var cbIDPtr *uintptr
	if cfg.Progress != nil {
		id := registerProgressCallback(cfg.Progress)
		cbIDPtr = new(uintptr)
		*cbIDPtr = id
		params.progress_callback = getProgressTrampoline()
		params.progress_callback_user_data = unsafe.Pointer(cbIDPtr)
	}

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	cModel := C.lfg_model_load_from_file(cPath, params)
	if cbIDPtr != nil {
		unregisterProgressCallback(*cbIDPtr)
	}
	if cModel == nil {
		if err := getLastError(); err != nil {
			return nil, err
		}
		return nil, &Error{Code: ErrorInternal, Message: "failed to load model"}
	}

	m := &Model{c: cModel}
	runtime.SetFinalizer(m, func(m *Model) { m.Close() })
	return m, nil
}

// Close frees the model resources. Safe to call multiple times.
func (m *Model) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.c != nil {
		C.lfg_model_free(m.c)
		m.c = nil
		runtime.SetFinalizer(m, nil)
	}
	return nil
}

// Vocab returns the vocabulary associated with this model.
func (m *Model) Vocab() *Vocab {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return nil
	}
	cv := C.lfg_model_get_vocab(m.c)
	return &Vocab{c: cv, model: m}
}

// TrainingContextSize returns the context size the model was trained with.
func (m *Model) TrainingContextSize() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return 0
	}
	return int(C.lfg_model_n_ctx_train(m.c))
}

// EmbeddingSize returns the embedding dimension.
func (m *Model) EmbeddingSize() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return 0
	}
	return int(C.lfg_model_n_embd(m.c))
}

// InputEmbeddingSize returns the input embedding dimension.
func (m *Model) InputEmbeddingSize() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return 0
	}
	return int(C.lfg_model_n_embd_inp(m.c))
}

// OutputEmbeddingSize returns the output embedding dimension.
func (m *Model) OutputEmbeddingSize() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return 0
	}
	return int(C.lfg_model_n_embd_out(m.c))
}

// LayerCount returns the number of layers.
func (m *Model) LayerCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return 0
	}
	return int(C.lfg_model_n_layer(m.c))
}

// HeadCount returns the number of attention heads.
func (m *Model) HeadCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return 0
	}
	return int(C.lfg_model_n_head(m.c))
}

// KVHeadCount returns the number of key/value attention heads.
func (m *Model) KVHeadCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return 0
	}
	return int(C.lfg_model_n_head_kv(m.c))
}

// SlidingWindowSize returns the sliding window attention size.
func (m *Model) SlidingWindowSize() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return 0
	}
	return int(C.lfg_model_n_swa(m.c))
}

// RopeFreqScaleTrain returns the RoPE frequency scaling factor from training.
func (m *Model) RopeFreqScaleTrain() float32 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return 0
	}
	return float32(C.lfg_model_rope_freq_scale_train(m.c))
}

// RopeType returns the RoPE type used by the model.
func (m *Model) RopeType() RopeType {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return RopeTypeNone
	}
	return RopeType(C.lfg_model_rope_type(m.c))
}

// Size returns the total size of all tensors in bytes.
func (m *Model) Size() uint64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return 0
	}
	return uint64(C.lfg_model_size(m.c))
}

// ParameterCount returns the total number of model parameters.
func (m *Model) ParameterCount() uint64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return 0
	}
	return uint64(C.lfg_model_n_params(m.c))
}

// Description returns a string describing the model type.
func (m *Model) Description() string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return ""
	}
	var buf [256]C.char
	n := C.lfg_model_desc(m.c, &buf[0], 256)
	if n < 0 {
		return ""
	}
	return C.GoStringN(&buf[0], n)
}

// HasEncoder returns true if the model contains an encoder.
func (m *Model) HasEncoder() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return false
	}
	return bool(C.lfg_model_has_encoder(m.c))
}

// HasDecoder returns true if the model contains a decoder.
func (m *Model) HasDecoder() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return false
	}
	return bool(C.lfg_model_has_decoder(m.c))
}

// DecoderStartToken returns the token that starts decoder generation.
// Returns InvalidToken for non-encoder-decoder models.
func (m *Model) DecoderStartToken() Token {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return InvalidToken
	}
	return Token(C.lfg_model_decoder_start_token(m.c))
}

// IsRecurrent returns true if the model is recurrent (e.g., Mamba, RWKV).
func (m *Model) IsRecurrent() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return false
	}
	return bool(C.lfg_model_is_recurrent(m.c))
}

// IsHybrid returns true if the model is a hybrid architecture.
func (m *Model) IsHybrid() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return false
	}
	return bool(C.lfg_model_is_hybrid(m.c))
}

// IsDiffusion returns true if the model is diffusion-based.
func (m *Model) IsDiffusion() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return false
	}
	return bool(C.lfg_model_is_diffusion(m.c))
}

// Metadata returns a metadata value by key name.
func (m *Model) Metadata(key string) (string, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return "", false
	}
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))

	var buf [512]C.char
	n := C.lfg_model_meta_val_str(m.c, cKey, &buf[0], 512)
	if n < 0 {
		return "", false
	}
	return C.GoStringN(&buf[0], n), true
}

// MetadataCount returns the number of metadata key/value pairs.
func (m *Model) MetadataCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return 0
	}
	return int(C.lfg_model_meta_count(m.c))
}

// MetadataKeyAt returns the metadata key at the given index.
func (m *Model) MetadataKeyAt(i int) (string, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return "", false
	}
	var buf [256]C.char
	n := C.lfg_model_meta_key_by_index(m.c, C.int32_t(i), &buf[0], 256)
	if n < 0 {
		return "", false
	}
	return C.GoStringN(&buf[0], n), true
}

// MetadataValueAt returns the metadata value at the given index.
func (m *Model) MetadataValueAt(i int) (string, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return "", false
	}
	var buf [512]C.char
	n := C.lfg_model_meta_val_str_by_index(m.c, C.int32_t(i), &buf[0], 512)
	if n < 0 {
		return "", false
	}
	return C.GoStringN(&buf[0], n), true
}

// ChatTemplate returns the model's chat template. If name is empty, the default template is returned.
// Returns empty string and false if not available.
func (m *Model) ChatTemplate(name string) (string, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return "", false
	}
	var cName *C.char
	if name != "" {
		cName = C.CString(name)
		defer C.free(unsafe.Pointer(cName))
	}
	result := C.lfg_model_chat_template(m.c, cName)
	if result == nil {
		return "", false
	}
	return C.GoString(result), true
}

// ClassifierOutputCount returns the number of classifier outputs (only for classifier models).
func (m *Model) ClassifierOutputCount() uint32 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return 0
	}
	return uint32(C.lfg_model_n_cls_out(m.c))
}

// ClassifierLabel returns the classifier label at the given index.
func (m *Model) ClassifierLabel(i uint32) string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return ""
	}
	result := C.lfg_model_cls_label(m.c, C.uint32_t(i))
	if result == nil {
		return ""
	}
	return C.GoString(result)
}
