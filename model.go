package lfg

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
	c  uintptr
}

// boolToByte converts a Go bool to a byte (0 or 1) for C struct fields.
func boolToByte(b bool) byte {
	if b {
		return 1
	}
	return 0
}

// LoadModel loads a model from a GGUF file. Automatically initializes the backend.
func LoadModel(path string, opts ...ModelOption) (*Model, error) {
	ensureBackend()
	registerModelFuncs()

	// Build Go config from defaults + options.
	defaults := _lfg_model_default_params()
	cfg := ModelConfig{
		GPULayers:    int(defaults.NGPULayers),
		SplitMode:    SplitMode(defaults.SplitMode),
		MainGPU:      int(defaults.MainGPU),
		VocabOnly:    defaults.VocabOnly != 0,
		Mmap:         defaults.UseMmap != 0,
		Mlock:        defaults.UseMlock != 0,
		CheckTensors: defaults.CheckTensors != 0,
	}
	for _, opt := range opts {
		opt(&cfg)
	}

	// Convert to C params.
	params := defaults
	params.NGPULayers = int32(cfg.GPULayers)
	params.SplitMode = int32(cfg.SplitMode)
	params.MainGPU = int32(cfg.MainGPU)
	params.VocabOnly = boolToByte(cfg.VocabOnly)
	params.UseMmap = boolToByte(cfg.Mmap)
	params.UseMlock = boolToByte(cfg.Mlock)
	params.CheckTensors = boolToByte(cfg.CheckTensors)

	var cbIDPtr *uintptr
	if cfg.Progress != nil {
		id := registerProgressCallback(cfg.Progress)
		cbIDPtr = new(uintptr)
		*cbIDPtr = id
		params.ProgressCallback = getProgressTrampoline()
		params.ProgressCallbackData = uintptr(unsafe.Pointer(cbIDPtr))
	}

	pathBytes := cString(path)
	pathPtr := cStringPtr(pathBytes)

	cModel := _lfg_model_load_from_file(pathPtr, params)
	runtime.KeepAlive(pathBytes)
	runtime.KeepAlive(cbIDPtr)

	if cbIDPtr != nil {
		unregisterProgressCallback(*cbIDPtr)
	}
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

// Close frees the model resources. Safe to call multiple times.
func (m *Model) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.c != 0 {
		_lfg_model_free(m.c)
		m.c = 0
		runtime.SetFinalizer(m, nil)
	}
	return nil
}

// Vocab returns the vocabulary associated with this model.
func (m *Model) Vocab() *Vocab {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == 0 {
		return nil
	}
	cv := _lfg_model_get_vocab(m.c)
	return &Vocab{c: cv, model: m}
}

// TrainingContextSize returns the context size the model was trained with.
func (m *Model) TrainingContextSize() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == 0 {
		return 0
	}
	return int(_lfg_model_n_ctx_train(m.c))
}

// EmbeddingSize returns the embedding dimension.
func (m *Model) EmbeddingSize() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == 0 {
		return 0
	}
	return int(_lfg_model_n_embd(m.c))
}

// InputEmbeddingSize returns the input embedding dimension.
func (m *Model) InputEmbeddingSize() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == 0 {
		return 0
	}
	return int(_lfg_model_n_embd_inp(m.c))
}

// OutputEmbeddingSize returns the output embedding dimension.
func (m *Model) OutputEmbeddingSize() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == 0 {
		return 0
	}
	return int(_lfg_model_n_embd_out(m.c))
}

// LayerCount returns the number of layers.
func (m *Model) LayerCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == 0 {
		return 0
	}
	return int(_lfg_model_n_layer(m.c))
}

// HeadCount returns the number of attention heads.
func (m *Model) HeadCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == 0 {
		return 0
	}
	return int(_lfg_model_n_head(m.c))
}

// KVHeadCount returns the number of key/value attention heads.
func (m *Model) KVHeadCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == 0 {
		return 0
	}
	return int(_lfg_model_n_head_kv(m.c))
}

// SlidingWindowSize returns the sliding window attention size.
func (m *Model) SlidingWindowSize() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == 0 {
		return 0
	}
	return int(_lfg_model_n_swa(m.c))
}

// RopeFreqScaleTrain returns the RoPE frequency scaling factor from training.
func (m *Model) RopeFreqScaleTrain() float32 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == 0 {
		return 0
	}
	return _lfg_model_rope_freq_scale_train(m.c)
}

// RopeType returns the RoPE type used by the model.
func (m *Model) RopeType() RopeType {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == 0 {
		return RopeTypeNone
	}
	return RopeType(_lfg_model_rope_type(m.c))
}

// Size returns the total size of all tensors in bytes.
func (m *Model) Size() uint64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == 0 {
		return 0
	}
	return _lfg_model_size(m.c)
}

// ParameterCount returns the total number of model parameters.
func (m *Model) ParameterCount() uint64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == 0 {
		return 0
	}
	return _lfg_model_n_params(m.c)
}

// Description returns a string describing the model type.
func (m *Model) Description() string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == 0 {
		return ""
	}
	var buf [256]byte
	n := _lfg_model_desc(m.c, uintptr(unsafe.Pointer(&buf[0])), uintptr(256))
	if n < 0 {
		return ""
	}
	return goStringN(uintptr(unsafe.Pointer(&buf[0])), int(n))
}

// HasEncoder returns true if the model contains an encoder.
func (m *Model) HasEncoder() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == 0 {
		return false
	}
	return _lfg_model_has_encoder(m.c)
}

// HasDecoder returns true if the model contains a decoder.
func (m *Model) HasDecoder() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == 0 {
		return false
	}
	return _lfg_model_has_decoder(m.c)
}

// DecoderStartToken returns the token that starts decoder generation.
// Returns InvalidToken for non-encoder-decoder models.
func (m *Model) DecoderStartToken() Token {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == 0 {
		return InvalidToken
	}
	return Token(_lfg_model_decoder_start_token(m.c))
}

// IsRecurrent returns true if the model is recurrent (e.g., Mamba, RWKV).
func (m *Model) IsRecurrent() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == 0 {
		return false
	}
	return _lfg_model_is_recurrent(m.c)
}

// IsHybrid returns true if the model is a hybrid architecture.
func (m *Model) IsHybrid() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == 0 {
		return false
	}
	return _lfg_model_is_hybrid(m.c)
}

// IsDiffusion returns true if the model is diffusion-based.
func (m *Model) IsDiffusion() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == 0 {
		return false
	}
	return _lfg_model_is_diffusion(m.c)
}

// Metadata returns a metadata value by key name.
func (m *Model) Metadata(key string) (string, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == 0 {
		return "", false
	}
	keyBytes := cString(key)
	keyPtr := cStringPtr(keyBytes)

	var buf [512]byte
	n := _lfg_model_meta_val_str(m.c, keyPtr, uintptr(unsafe.Pointer(&buf[0])), uintptr(512))
	runtime.KeepAlive(keyBytes)
	if n < 0 {
		return "", false
	}
	return goStringN(uintptr(unsafe.Pointer(&buf[0])), int(n)), true
}

// MetadataCount returns the number of metadata key/value pairs.
func (m *Model) MetadataCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == 0 {
		return 0
	}
	return int(_lfg_model_meta_count(m.c))
}

// MetadataKeyAt returns the metadata key at the given index.
func (m *Model) MetadataKeyAt(i int) (string, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == 0 {
		return "", false
	}
	var buf [256]byte
	n := _lfg_model_meta_key_by_index(m.c, int32(i), uintptr(unsafe.Pointer(&buf[0])), uintptr(256))
	if n < 0 {
		return "", false
	}
	return goStringN(uintptr(unsafe.Pointer(&buf[0])), int(n)), true
}

// MetadataValueAt returns the metadata value at the given index.
func (m *Model) MetadataValueAt(i int) (string, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == 0 {
		return "", false
	}
	var buf [512]byte
	n := _lfg_model_meta_val_str_by_index(m.c, int32(i), uintptr(unsafe.Pointer(&buf[0])), uintptr(512))
	if n < 0 {
		return "", false
	}
	return goStringN(uintptr(unsafe.Pointer(&buf[0])), int(n)), true
}

// ChatTemplate returns the model's chat template. If name is empty, the default template is returned.
// Returns empty string and false if not available.
func (m *Model) ChatTemplate(name string) (string, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == 0 {
		return "", false
	}
	var namePtr uintptr
	var nameBytes []byte
	if name != "" {
		nameBytes = cString(name)
		namePtr = cStringPtr(nameBytes)
	}
	result := _lfg_model_chat_template(m.c, namePtr)
	runtime.KeepAlive(nameBytes)
	if result == 0 {
		return "", false
	}
	return goString(result), true
}

// ClassifierOutputCount returns the number of classifier outputs (only for classifier models).
func (m *Model) ClassifierOutputCount() uint32 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == 0 {
		return 0
	}
	return _lfg_model_n_cls_out(m.c)
}

// ClassifierLabel returns the classifier label at the given index.
func (m *Model) ClassifierLabel(i uint32) string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == 0 {
		return ""
	}
	result := _lfg_model_cls_label(m.c, i)
	if result == 0 {
		return ""
	}
	return goString(result)
}

// LoadModelFromSplits loads a model from multiple split GGUF files.
// Automatically initializes the backend.
func LoadModelFromSplits(paths []string, opts ...ModelOption) (*Model, error) {
	ensureBackend()
	registerModelFuncs()

	defaults := _lfg_model_default_params()
	cfg := ModelConfig{
		GPULayers:    int(defaults.NGPULayers),
		SplitMode:    SplitMode(defaults.SplitMode),
		MainGPU:      int(defaults.MainGPU),
		VocabOnly:    defaults.VocabOnly != 0,
		Mmap:         defaults.UseMmap != 0,
		Mlock:        defaults.UseMlock != 0,
		CheckTensors: defaults.CheckTensors != 0,
	}
	for _, opt := range opts {
		opt(&cfg)
	}

	params := defaults
	params.NGPULayers = int32(cfg.GPULayers)
	params.SplitMode = int32(cfg.SplitMode)
	params.MainGPU = int32(cfg.MainGPU)
	params.VocabOnly = boolToByte(cfg.VocabOnly)
	params.UseMmap = boolToByte(cfg.Mmap)
	params.UseMlock = boolToByte(cfg.Mlock)
	params.CheckTensors = boolToByte(cfg.CheckTensors)

	var cbIDPtr *uintptr
	if cfg.Progress != nil {
		id := registerProgressCallback(cfg.Progress)
		cbIDPtr = new(uintptr)
		*cbIDPtr = id
		params.ProgressCallback = getProgressTrampoline()
		params.ProgressCallbackData = uintptr(unsafe.Pointer(cbIDPtr))
	}

	pathBytes := make([][]byte, len(paths))
	pathPtrs := make([]uintptr, len(paths))
	for i, p := range paths {
		pathBytes[i] = cString(p)
		pathPtrs[i] = cStringPtr(pathBytes[i])
	}

	var pathsPtr uintptr
	if len(pathPtrs) > 0 {
		pathsPtr = uintptr(unsafe.Pointer(&pathPtrs[0]))
	}

	cModel := _lfg_model_load_from_splits(pathsPtr, uintptr(len(paths)), params)
	runtime.KeepAlive(pathBytes)
	runtime.KeepAlive(pathPtrs)
	runtime.KeepAlive(cbIDPtr)

	if cbIDPtr != nil {
		unregisterProgressCallback(*cbIDPtr)
	}
	if cModel == 0 {
		if err := getLastError(); err != nil {
			return nil, err
		}
		return nil, &Error{Code: ErrorInternal, Message: "failed to load model from splits"}
	}

	m := &Model{c: cModel}
	runtime.SetFinalizer(m, func(m *Model) { m.Close() })
	return m, nil
}

// SaveToFile saves the model to a GGUF file.
func (m *Model) SaveToFile(path string) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == 0 {
		return
	}
	registerModelFuncs()
	pathBytes := cString(path)
	_lfg_model_save_to_file(m.c, cStringPtr(pathBytes))
	runtime.KeepAlive(pathBytes)
}

// MetaKeyString returns the C string name for a model metadata key enum value.
func MetaKeyString(key ModelMetaKey) string {
	registerModelFuncs()
	ptr := _lfg_model_meta_key_str(int32(key))
	if ptr == 0 {
		return ""
	}
	return goString(ptr)
}
