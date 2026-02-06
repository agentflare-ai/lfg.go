package lfg

/*
typedef struct lfm_model lfm_model;
typedef struct lfm_context lfm_context;
typedef struct lfm_vocab lfm_vocab;
typedef struct lfm_sampler lfm_sampler;
#include "lfm_inference.h"
#include <stdlib.h>
*/
import "C"
import (
	"runtime"
	"sync"
	"unsafe"
)

// ModelOption configures model loading parameters.
type ModelOption func(*C.struct_lfm_model_params)

// WithGPULayers sets the number of layers to offload to GPU.
// A negative value offloads all layers.
func WithGPULayers(n int) ModelOption {
	return func(p *C.struct_lfm_model_params) {
		p.n_gpu_layers = C.int32_t(n)
	}
}

// WithSplitMode sets how the model is split across GPUs.
func WithSplitMode(mode SplitMode) ModelOption {
	return func(p *C.struct_lfm_model_params) {
		p.split_mode = C.enum_lfm_split_mode(mode)
	}
}

// WithMainGPU sets the main GPU index for single-GPU mode.
func WithMainGPU(gpu int) ModelOption {
	return func(p *C.struct_lfm_model_params) {
		p.main_gpu = C.int32_t(gpu)
	}
}

// WithVocabOnly loads only the vocabulary, no weights.
func WithVocabOnly(v bool) ModelOption {
	return func(p *C.struct_lfm_model_params) {
		p.vocab_only = C.bool(v)
	}
}

// WithMmap enables or disables memory-mapped file loading.
func WithMmap(v bool) ModelOption {
	return func(p *C.struct_lfm_model_params) {
		p.use_mmap = C.bool(v)
	}
}

// WithMlock forces the system to keep the model in RAM.
func WithMlock(v bool) ModelOption {
	return func(p *C.struct_lfm_model_params) {
		p.use_mlock = C.bool(v)
	}
}

// WithCheckTensors enables validation of model tensor data.
func WithCheckTensors(v bool) ModelOption {
	return func(p *C.struct_lfm_model_params) {
		p.check_tensors = C.bool(v)
	}
}

// WithProgressCallback sets a callback for model loading progress.
func WithProgressCallback(cb ProgressCallback) ModelOption {
	return func(p *C.struct_lfm_model_params) {
		id := registerProgressCallback(cb)
		idPtr := new(uintptr)
		*idPtr = id
		p.progress_callback = getProgressTrampoline()
		p.progress_callback_user_data = unsafe.Pointer(idPtr)
	}
}

// Model wraps an lfm_model pointer with thread-safe access.
type Model struct {
	mu sync.RWMutex
	c  *C.struct_lfm_model
}

// LoadModel loads a model from a GGUF file. Automatically initializes the backend.
func LoadModel(path string, opts ...ModelOption) (*Model, error) {
	ensureBackend()

	params := C.lfm_model_default_params()
	var cbIDPtr *uintptr
	for _, opt := range opts {
		opt(&params)
	}
	if params.progress_callback_user_data != nil {
		cbIDPtr = (*uintptr)(params.progress_callback_user_data)
	}

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	cModel := C.lfm_model_load_from_file(cPath, params)
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
		C.lfm_model_free(m.c)
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
	cv := C.lfm_model_get_vocab(m.c)
	return &Vocab{c: cv, model: m}
}

// NCtxTrain returns the context size the model was trained with.
func (m *Model) NCtxTrain() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return 0
	}
	return int(C.lfm_model_n_ctx_train(m.c))
}

// NEmbd returns the embedding dimension.
func (m *Model) NEmbd() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return 0
	}
	return int(C.lfm_model_n_embd(m.c))
}

// NEmbdInp returns the input embedding dimension.
func (m *Model) NEmbdInp() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return 0
	}
	return int(C.lfm_model_n_embd_inp(m.c))
}

// NEmbdOut returns the output embedding dimension.
func (m *Model) NEmbdOut() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return 0
	}
	return int(C.lfm_model_n_embd_out(m.c))
}

// NLayer returns the number of layers.
func (m *Model) NLayer() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return 0
	}
	return int(C.lfm_model_n_layer(m.c))
}

// NHead returns the number of attention heads.
func (m *Model) NHead() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return 0
	}
	return int(C.lfm_model_n_head(m.c))
}

// NHeadKV returns the number of key/value attention heads.
func (m *Model) NHeadKV() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return 0
	}
	return int(C.lfm_model_n_head_kv(m.c))
}

// NSWA returns the sliding window attention size.
func (m *Model) NSWA() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return 0
	}
	return int(C.lfm_model_n_swa(m.c))
}

// RopeFreqScaleTrain returns the RoPE frequency scaling factor from training.
func (m *Model) RopeFreqScaleTrain() float32 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return 0
	}
	return float32(C.lfm_model_rope_freq_scale_train(m.c))
}

// RopeType returns the RoPE type used by the model.
func (m *Model) RopeType() RopeType {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return RopeTypeNone
	}
	return RopeType(C.lfm_model_rope_type(m.c))
}

// Size returns the total size of all tensors in bytes.
func (m *Model) Size() uint64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return 0
	}
	return uint64(C.lfm_model_size(m.c))
}

// NParams returns the total number of model parameters.
func (m *Model) NParams() uint64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return 0
	}
	return uint64(C.lfm_model_n_params(m.c))
}

// Desc returns a string describing the model type.
func (m *Model) Desc() string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return ""
	}
	var buf [256]C.char
	n := C.lfm_model_desc(m.c, &buf[0], 256)
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
	return bool(C.lfm_model_has_encoder(m.c))
}

// HasDecoder returns true if the model contains a decoder.
func (m *Model) HasDecoder() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return false
	}
	return bool(C.lfm_model_has_decoder(m.c))
}

// DecoderStartToken returns the token that starts decoder generation.
// Returns TokenNull for non-encoder-decoder models.
func (m *Model) DecoderStartToken() Token {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return TokenNull
	}
	return Token(C.lfm_model_decoder_start_token(m.c))
}

// IsRecurrent returns true if the model is recurrent (e.g., Mamba, RWKV).
func (m *Model) IsRecurrent() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return false
	}
	return bool(C.lfm_model_is_recurrent(m.c))
}

// IsHybrid returns true if the model is a hybrid architecture.
func (m *Model) IsHybrid() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return false
	}
	return bool(C.lfm_model_is_hybrid(m.c))
}

// IsDiffusion returns true if the model is diffusion-based.
func (m *Model) IsDiffusion() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return false
	}
	return bool(C.lfm_model_is_diffusion(m.c))
}

// MetaValStr returns a metadata value by key name.
func (m *Model) MetaValStr(key string) (string, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return "", false
	}
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))

	var buf [512]C.char
	n := C.lfm_model_meta_val_str(m.c, cKey, &buf[0], 512)
	if n < 0 {
		return "", false
	}
	return C.GoStringN(&buf[0], n), true
}

// MetaCount returns the number of metadata key/value pairs.
func (m *Model) MetaCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return 0
	}
	return int(C.lfm_model_meta_count(m.c))
}

// MetaKeyByIndex returns the metadata key at the given index.
func (m *Model) MetaKeyByIndex(i int) (string, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return "", false
	}
	var buf [256]C.char
	n := C.lfm_model_meta_key_by_index(m.c, C.int32_t(i), &buf[0], 256)
	if n < 0 {
		return "", false
	}
	return C.GoStringN(&buf[0], n), true
}

// MetaValStrByIndex returns the metadata value at the given index.
func (m *Model) MetaValStrByIndex(i int) (string, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return "", false
	}
	var buf [512]C.char
	n := C.lfm_model_meta_val_str_by_index(m.c, C.int32_t(i), &buf[0], 512)
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
	result := C.lfm_model_chat_template(m.c, cName)
	if result == nil {
		return "", false
	}
	return C.GoString(result), true
}

// NClsOut returns the number of classifier outputs (only for classifier models).
func (m *Model) NClsOut() uint32 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return 0
	}
	return uint32(C.lfm_model_n_cls_out(m.c))
}

// ClsLabel returns the classifier label at the given index.
func (m *Model) ClsLabel(i uint32) string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.c == nil {
		return ""
	}
	result := C.lfm_model_cls_label(m.c, C.uint32_t(i))
	if result == nil {
		return ""
	}
	return C.GoString(result)
}
