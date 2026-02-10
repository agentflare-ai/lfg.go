//go:build (darwin && arm64) || (linux && amd64) || (linux && arm64)

package lfg

import (
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// ---------------------------------------------------------------------------
// Backend functions
// ---------------------------------------------------------------------------

var (
	backendFuncsOnce sync.Once

	_lfg_backend_init       func()
	_lfg_backend_free       func()
	_lfg_api_version_string func() uintptr
	_lfg_api_version        func(major, minor, patch *uint32)
	_lfg_abi_version        func() uint32
	_lfg_print_system_info  func() uintptr
	_lfg_time_us            func() int64
	_lfg_max_devices        func() uintptr
	_lfg_max_parallel_sequences func() uintptr
	_lfg_supports_mmap      func() bool
	_lfg_supports_mlock     func() bool
	_lfg_supports_gpu_offload func() bool
	_lfg_supports_rpc       func() bool
	_lfg_error_string       func(code int32) uintptr
	_lfg_get_last_error     func(buf uintptr, bufSize uintptr) int32
	_lfg_clear_last_error   func()
)

func registerBackendFuncs() {
	backendFuncsOnce.Do(func() {
		lib, err := loadLibrary()
		if err != nil {
			panic("lfg: failed to load library: " + err.Error())
		}
		purego.RegisterLibFunc(&_lfg_backend_init, lib, "lfg_backend_init")
		purego.RegisterLibFunc(&_lfg_backend_free, lib, "lfg_backend_free")
		purego.RegisterLibFunc(&_lfg_api_version_string, lib, "lfg_api_version_string")
		purego.RegisterLibFunc(&_lfg_api_version, lib, "lfg_api_version")
		purego.RegisterLibFunc(&_lfg_abi_version, lib, "lfg_abi_version")
		purego.RegisterLibFunc(&_lfg_print_system_info, lib, "lfg_print_system_info")
		purego.RegisterLibFunc(&_lfg_time_us, lib, "lfg_time_us")
		purego.RegisterLibFunc(&_lfg_max_devices, lib, "lfg_max_devices")
		purego.RegisterLibFunc(&_lfg_max_parallel_sequences, lib, "lfg_max_parallel_sequences")
		purego.RegisterLibFunc(&_lfg_supports_mmap, lib, "lfg_supports_mmap")
		purego.RegisterLibFunc(&_lfg_supports_mlock, lib, "lfg_supports_mlock")
		purego.RegisterLibFunc(&_lfg_supports_gpu_offload, lib, "lfg_supports_gpu_offload")
		purego.RegisterLibFunc(&_lfg_supports_rpc, lib, "lfg_supports_rpc")
		purego.RegisterLibFunc(&_lfg_error_string, lib, "lfg_error_string")
		purego.RegisterLibFunc(&_lfg_get_last_error, lib, "lfg_get_last_error")
		purego.RegisterLibFunc(&_lfg_clear_last_error, lib, "lfg_clear_last_error")
	})
}

// ---------------------------------------------------------------------------
// Model functions
// ---------------------------------------------------------------------------

var (
	modelFuncsOnce sync.Once

	_lfg_model_default_params         func() cModelParams
	_lfg_model_load_from_file         func(path uintptr, params cModelParams) uintptr
	_lfg_model_free                   func(model uintptr)
	_lfg_model_get_vocab              func(model uintptr) uintptr
	_lfg_model_n_ctx_train            func(model uintptr) int32
	_lfg_model_n_embd                 func(model uintptr) int32
	_lfg_model_n_embd_inp             func(model uintptr) int32
	_lfg_model_n_embd_out             func(model uintptr) int32
	_lfg_model_n_layer                func(model uintptr) int32
	_lfg_model_n_head                 func(model uintptr) int32
	_lfg_model_n_head_kv              func(model uintptr) int32
	_lfg_model_n_swa                  func(model uintptr) int32
	_lfg_model_rope_freq_scale_train  func(model uintptr) float32
	_lfg_model_rope_type              func(model uintptr) int32
	_lfg_model_size                   func(model uintptr) uint64
	_lfg_model_n_params               func(model uintptr) uint64
	_lfg_model_desc                   func(model uintptr, buf uintptr, bufSize uintptr) int32
	_lfg_model_meta_val_str           func(model uintptr, key uintptr, buf uintptr, bufSize uintptr) int32
	_lfg_model_meta_count             func(model uintptr) int32
	_lfg_model_meta_key_by_index      func(model uintptr, i int32, buf uintptr, bufSize uintptr) int32
	_lfg_model_meta_val_str_by_index  func(model uintptr, i int32, buf uintptr, bufSize uintptr) int32
	_lfg_model_has_encoder            func(model uintptr) bool
	_lfg_model_has_decoder            func(model uintptr) bool
	_lfg_model_decoder_start_token    func(model uintptr) int32
	_lfg_model_is_recurrent           func(model uintptr) bool
	_lfg_model_is_hybrid              func(model uintptr) bool
	_lfg_model_is_diffusion           func(model uintptr) bool
	_lfg_model_chat_template          func(model uintptr, name uintptr) uintptr
	_lfg_model_n_cls_out              func(model uintptr) uint32
	_lfg_model_cls_label              func(model uintptr, i uint32) uintptr
)

func registerModelFuncs() {
	modelFuncsOnce.Do(func() {
		lib, err := loadLibrary()
		if err != nil {
			panic("lfg: failed to load library: " + err.Error())
		}
		purego.RegisterLibFunc(&_lfg_model_default_params, lib, "lfg_model_default_params")
		purego.RegisterLibFunc(&_lfg_model_load_from_file, lib, "lfg_model_load_from_file")
		purego.RegisterLibFunc(&_lfg_model_free, lib, "lfg_model_free")
		purego.RegisterLibFunc(&_lfg_model_get_vocab, lib, "lfg_model_get_vocab")
		purego.RegisterLibFunc(&_lfg_model_n_ctx_train, lib, "lfg_model_n_ctx_train")
		purego.RegisterLibFunc(&_lfg_model_n_embd, lib, "lfg_model_n_embd")
		purego.RegisterLibFunc(&_lfg_model_n_embd_inp, lib, "lfg_model_n_embd_inp")
		purego.RegisterLibFunc(&_lfg_model_n_embd_out, lib, "lfg_model_n_embd_out")
		purego.RegisterLibFunc(&_lfg_model_n_layer, lib, "lfg_model_n_layer")
		purego.RegisterLibFunc(&_lfg_model_n_head, lib, "lfg_model_n_head")
		purego.RegisterLibFunc(&_lfg_model_n_head_kv, lib, "lfg_model_n_head_kv")
		purego.RegisterLibFunc(&_lfg_model_n_swa, lib, "lfg_model_n_swa")
		purego.RegisterLibFunc(&_lfg_model_rope_freq_scale_train, lib, "lfg_model_rope_freq_scale_train")
		purego.RegisterLibFunc(&_lfg_model_rope_type, lib, "lfg_model_rope_type")
		purego.RegisterLibFunc(&_lfg_model_size, lib, "lfg_model_size")
		purego.RegisterLibFunc(&_lfg_model_n_params, lib, "lfg_model_n_params")
		purego.RegisterLibFunc(&_lfg_model_desc, lib, "lfg_model_desc")
		purego.RegisterLibFunc(&_lfg_model_meta_val_str, lib, "lfg_model_meta_val_str")
		purego.RegisterLibFunc(&_lfg_model_meta_count, lib, "lfg_model_meta_count")
		purego.RegisterLibFunc(&_lfg_model_meta_key_by_index, lib, "lfg_model_meta_key_by_index")
		purego.RegisterLibFunc(&_lfg_model_meta_val_str_by_index, lib, "lfg_model_meta_val_str_by_index")
		purego.RegisterLibFunc(&_lfg_model_has_encoder, lib, "lfg_model_has_encoder")
		purego.RegisterLibFunc(&_lfg_model_has_decoder, lib, "lfg_model_has_decoder")
		purego.RegisterLibFunc(&_lfg_model_decoder_start_token, lib, "lfg_model_decoder_start_token")
		purego.RegisterLibFunc(&_lfg_model_is_recurrent, lib, "lfg_model_is_recurrent")
		purego.RegisterLibFunc(&_lfg_model_is_hybrid, lib, "lfg_model_is_hybrid")
		purego.RegisterLibFunc(&_lfg_model_is_diffusion, lib, "lfg_model_is_diffusion")
		purego.RegisterLibFunc(&_lfg_model_chat_template, lib, "lfg_model_chat_template")
		purego.RegisterLibFunc(&_lfg_model_n_cls_out, lib, "lfg_model_n_cls_out")
		purego.RegisterLibFunc(&_lfg_model_cls_label, lib, "lfg_model_cls_label")
	})
}

// ---------------------------------------------------------------------------
// Vocab functions
// ---------------------------------------------------------------------------

var (
	vocabFuncsOnce sync.Once

	_lfg_vocab_type        func(vocab uintptr) int32
	_lfg_vocab_n_tokens    func(vocab uintptr) int32
	_lfg_vocab_get_text    func(vocab uintptr, token int32) uintptr
	_lfg_vocab_get_score   func(vocab uintptr, token int32) float32
	_lfg_vocab_get_attr    func(vocab uintptr, token int32) int32
	_lfg_vocab_is_eog      func(vocab uintptr, token int32) bool
	_lfg_vocab_is_control  func(vocab uintptr, token int32) bool
	_lfg_vocab_bos         func(vocab uintptr) int32
	_lfg_vocab_eos         func(vocab uintptr) int32
	_lfg_vocab_eot         func(vocab uintptr) int32
	_lfg_vocab_sep         func(vocab uintptr) int32
	_lfg_vocab_nl          func(vocab uintptr) int32
	_lfg_vocab_pad         func(vocab uintptr) int32
	_lfg_vocab_mask        func(vocab uintptr) int32
	_lfg_vocab_get_add_bos func(vocab uintptr) bool
	_lfg_vocab_get_add_eos func(vocab uintptr) bool
	_lfg_vocab_get_add_sep func(vocab uintptr) bool
	_lfg_vocab_fim_pre     func(vocab uintptr) int32
	_lfg_vocab_fim_suf     func(vocab uintptr) int32
	_lfg_vocab_fim_mid     func(vocab uintptr) int32
	_lfg_vocab_fim_pad     func(vocab uintptr) int32
	_lfg_vocab_fim_rep     func(vocab uintptr) int32
	_lfg_vocab_fim_sep     func(vocab uintptr) int32
	_lfg_tokenize          func(vocab uintptr, text uintptr, textLen int32, tokens uintptr, nTokensMax int32, addSpecial bool, parseSpecial bool) int32
	_lfg_detokenize        func(vocab uintptr, tokens uintptr, nTokens int32, text uintptr, textLenMax int32, removeSpecial bool, unparseSpecial bool) int32
	_lfg_token_to_piece    func(vocab uintptr, token int32, buf uintptr, length int32, lstrip int32, special bool) int32
)

func registerVocabFuncs() {
	vocabFuncsOnce.Do(func() {
		lib, err := loadLibrary()
		if err != nil {
			panic("lfg: failed to load library: " + err.Error())
		}
		purego.RegisterLibFunc(&_lfg_vocab_type, lib, "lfg_vocab_type")
		purego.RegisterLibFunc(&_lfg_vocab_n_tokens, lib, "lfg_vocab_n_tokens")
		purego.RegisterLibFunc(&_lfg_vocab_get_text, lib, "lfg_vocab_get_text")
		purego.RegisterLibFunc(&_lfg_vocab_get_score, lib, "lfg_vocab_get_score")
		purego.RegisterLibFunc(&_lfg_vocab_get_attr, lib, "lfg_vocab_get_attr")
		purego.RegisterLibFunc(&_lfg_vocab_is_eog, lib, "lfg_vocab_is_eog")
		purego.RegisterLibFunc(&_lfg_vocab_is_control, lib, "lfg_vocab_is_control")
		purego.RegisterLibFunc(&_lfg_vocab_bos, lib, "lfg_vocab_bos")
		purego.RegisterLibFunc(&_lfg_vocab_eos, lib, "lfg_vocab_eos")
		purego.RegisterLibFunc(&_lfg_vocab_eot, lib, "lfg_vocab_eot")
		purego.RegisterLibFunc(&_lfg_vocab_sep, lib, "lfg_vocab_sep")
		purego.RegisterLibFunc(&_lfg_vocab_nl, lib, "lfg_vocab_nl")
		purego.RegisterLibFunc(&_lfg_vocab_pad, lib, "lfg_vocab_pad")
		purego.RegisterLibFunc(&_lfg_vocab_mask, lib, "lfg_vocab_mask")
		purego.RegisterLibFunc(&_lfg_vocab_get_add_bos, lib, "lfg_vocab_get_add_bos")
		purego.RegisterLibFunc(&_lfg_vocab_get_add_eos, lib, "lfg_vocab_get_add_eos")
		purego.RegisterLibFunc(&_lfg_vocab_get_add_sep, lib, "lfg_vocab_get_add_sep")
		purego.RegisterLibFunc(&_lfg_vocab_fim_pre, lib, "lfg_vocab_fim_pre")
		purego.RegisterLibFunc(&_lfg_vocab_fim_suf, lib, "lfg_vocab_fim_suf")
		purego.RegisterLibFunc(&_lfg_vocab_fim_mid, lib, "lfg_vocab_fim_mid")
		purego.RegisterLibFunc(&_lfg_vocab_fim_pad, lib, "lfg_vocab_fim_pad")
		purego.RegisterLibFunc(&_lfg_vocab_fim_rep, lib, "lfg_vocab_fim_rep")
		purego.RegisterLibFunc(&_lfg_vocab_fim_sep, lib, "lfg_vocab_fim_sep")
		purego.RegisterLibFunc(&_lfg_tokenize, lib, "lfg_tokenize")
		purego.RegisterLibFunc(&_lfg_detokenize, lib, "lfg_detokenize")
		purego.RegisterLibFunc(&_lfg_token_to_piece, lib, "lfg_token_to_piece")
	})
}

// ---------------------------------------------------------------------------
// Context functions
// ---------------------------------------------------------------------------

var (
	contextFuncsOnce sync.Once

	_lfg_context_default_params func() cContextParams
	_lfg_init_from_model        func(model uintptr, params cContextParams) uintptr
	_lfg_free                   func(ctx uintptr)
	_lfg_get_model              func(ctx uintptr) uintptr
	_lfg_get_memory             func(ctx uintptr) uintptr
	_lfg_pooling_type           func(ctx uintptr) int32
	_lfg_n_ctx                  func(ctx uintptr) uint32
	_lfg_n_ctx_seq              func(ctx uintptr) uint32
	_lfg_n_batch                func(ctx uintptr) uint32
	_lfg_n_ubatch               func(ctx uintptr) uint32
	_lfg_n_seq_max              func(ctx uintptr) uint32
	_lfg_set_n_threads          func(ctx uintptr, nThreads int32, nThreadsBatch int32)
	_lfg_n_threads              func(ctx uintptr) int32
	_lfg_n_threads_batch        func(ctx uintptr) int32
	_lfg_set_embeddings         func(ctx uintptr, embeddings bool)
	_lfg_set_causal_attn        func(ctx uintptr, causalAttn bool)
	_lfg_set_warmup             func(ctx uintptr, warmup bool)
	_lfg_synchronize            func(ctx uintptr)
	_lfg_get_logits             func(ctx uintptr) uintptr
	_lfg_get_logits_ith         func(ctx uintptr, i int32) uintptr
	_lfg_get_embeddings         func(ctx uintptr) uintptr
	_lfg_get_embeddings_ith     func(ctx uintptr, i int32) uintptr
	_lfg_get_embeddings_seq     func(ctx uintptr, seqID int32) uintptr
)

func registerContextFuncs() {
	contextFuncsOnce.Do(func() {
		lib, err := loadLibrary()
		if err != nil {
			panic("lfg: failed to load library: " + err.Error())
		}
		purego.RegisterLibFunc(&_lfg_context_default_params, lib, "lfg_context_default_params")
		purego.RegisterLibFunc(&_lfg_init_from_model, lib, "lfg_init_from_model")
		purego.RegisterLibFunc(&_lfg_free, lib, "lfg_free")
		purego.RegisterLibFunc(&_lfg_get_model, lib, "lfg_get_model")
		purego.RegisterLibFunc(&_lfg_get_memory, lib, "lfg_get_memory")
		purego.RegisterLibFunc(&_lfg_pooling_type, lib, "lfg_pooling_type")
		purego.RegisterLibFunc(&_lfg_n_ctx, lib, "lfg_n_ctx")
		purego.RegisterLibFunc(&_lfg_n_ctx_seq, lib, "lfg_n_ctx_seq")
		purego.RegisterLibFunc(&_lfg_n_batch, lib, "lfg_n_batch")
		purego.RegisterLibFunc(&_lfg_n_ubatch, lib, "lfg_n_ubatch")
		purego.RegisterLibFunc(&_lfg_n_seq_max, lib, "lfg_n_seq_max")
		purego.RegisterLibFunc(&_lfg_set_n_threads, lib, "lfg_set_n_threads")
		purego.RegisterLibFunc(&_lfg_n_threads, lib, "lfg_n_threads")
		purego.RegisterLibFunc(&_lfg_n_threads_batch, lib, "lfg_n_threads_batch")
		purego.RegisterLibFunc(&_lfg_set_embeddings, lib, "lfg_set_embeddings")
		purego.RegisterLibFunc(&_lfg_set_causal_attn, lib, "lfg_set_causal_attn")
		purego.RegisterLibFunc(&_lfg_set_warmup, lib, "lfg_set_warmup")
		purego.RegisterLibFunc(&_lfg_synchronize, lib, "lfg_synchronize")
		purego.RegisterLibFunc(&_lfg_get_logits, lib, "lfg_get_logits")
		purego.RegisterLibFunc(&_lfg_get_logits_ith, lib, "lfg_get_logits_ith")
		purego.RegisterLibFunc(&_lfg_get_embeddings, lib, "lfg_get_embeddings")
		purego.RegisterLibFunc(&_lfg_get_embeddings_ith, lib, "lfg_get_embeddings_ith")
		purego.RegisterLibFunc(&_lfg_get_embeddings_seq, lib, "lfg_get_embeddings_seq")
	})
}

// ---------------------------------------------------------------------------
// Batch functions
// ---------------------------------------------------------------------------

var (
	batchFuncsOnce sync.Once

	_lfg_batch_get_one func(tokens uintptr, nTokens int32) cBatch
	_lfg_batch_init    func(nTokens int32, embd int32, nSeqMax int32) cBatch
	_lfg_batch_free    func(batch cBatch)
	_lfg_decode        func(ctx uintptr, batch cBatch) int32
	_lfg_encode        func(ctx uintptr, batch cBatch) int32
)

func registerBatchFuncs() {
	batchFuncsOnce.Do(func() {
		lib, err := loadLibrary()
		if err != nil {
			panic("lfg: failed to load library: " + err.Error())
		}
		purego.RegisterLibFunc(&_lfg_batch_get_one, lib, "lfg_batch_get_one")
		purego.RegisterLibFunc(&_lfg_batch_init, lib, "lfg_batch_init")
		purego.RegisterLibFunc(&_lfg_batch_free, lib, "lfg_batch_free")
		purego.RegisterLibFunc(&_lfg_decode, lib, "lfg_decode")
		purego.RegisterLibFunc(&_lfg_encode, lib, "lfg_encode")
	})
}

// ---------------------------------------------------------------------------
// Memory functions
// ---------------------------------------------------------------------------

var (
	memoryFuncsOnce sync.Once

	_lfg_memory_clear       func(mem uintptr, data bool)
	_lfg_memory_seq_rm      func(mem uintptr, seqID int32, p0 int32, p1 int32) bool
	_lfg_memory_seq_cp      func(mem uintptr, seqSrc int32, seqDst int32, p0 int32, p1 int32)
	_lfg_memory_seq_keep    func(mem uintptr, seqID int32)
	_lfg_memory_seq_add     func(mem uintptr, seqID int32, p0 int32, p1 int32, delta int32)
	_lfg_memory_seq_div     func(mem uintptr, seqID int32, p0 int32, p1 int32, d int32)
	_lfg_memory_seq_pos_min func(mem uintptr, seqID int32) int32
	_lfg_memory_seq_pos_max func(mem uintptr, seqID int32) int32
	_lfg_memory_can_shift   func(mem uintptr) bool
)

func registerMemoryFuncs() {
	memoryFuncsOnce.Do(func() {
		lib, err := loadLibrary()
		if err != nil {
			panic("lfg: failed to load library: " + err.Error())
		}
		purego.RegisterLibFunc(&_lfg_memory_clear, lib, "lfg_memory_clear")
		purego.RegisterLibFunc(&_lfg_memory_seq_rm, lib, "lfg_memory_seq_rm")
		purego.RegisterLibFunc(&_lfg_memory_seq_cp, lib, "lfg_memory_seq_cp")
		purego.RegisterLibFunc(&_lfg_memory_seq_keep, lib, "lfg_memory_seq_keep")
		purego.RegisterLibFunc(&_lfg_memory_seq_add, lib, "lfg_memory_seq_add")
		purego.RegisterLibFunc(&_lfg_memory_seq_div, lib, "lfg_memory_seq_div")
		purego.RegisterLibFunc(&_lfg_memory_seq_pos_min, lib, "lfg_memory_seq_pos_min")
		purego.RegisterLibFunc(&_lfg_memory_seq_pos_max, lib, "lfg_memory_seq_pos_max")
		purego.RegisterLibFunc(&_lfg_memory_can_shift, lib, "lfg_memory_can_shift")
	})
}

// ---------------------------------------------------------------------------
// Sampler functions
// ---------------------------------------------------------------------------

var (
	samplerFuncsOnce sync.Once

	_lfg_sampler_chain_default_params func() cSamplerChainParams
	_lfg_sampler_chain_init           func(params cSamplerChainParams) uintptr
	_lfg_sampler_chain_add            func(chain uintptr, smpl uintptr)
	_lfg_sampler_chain_get            func(chain uintptr, i int32) uintptr
	_lfg_sampler_chain_n              func(chain uintptr) int32
	_lfg_sampler_chain_remove         func(chain uintptr, i int32) uintptr
	_lfg_sampler_free                 func(smpl uintptr)
	_lfg_sampler_name                 func(smpl uintptr) uintptr
	_lfg_sampler_accept               func(smpl uintptr, token int32)
	_lfg_sampler_reset                func(smpl uintptr)
	_lfg_sampler_clone                func(smpl uintptr) uintptr
	_lfg_sampler_get_seed             func(smpl uintptr) uint32
	_lfg_sampler_sample               func(smpl uintptr, ctx uintptr, idx int32) int32
	_lfg_sampler_init_greedy          func() uintptr
	_lfg_sampler_init_dist            func(seed uint32) uintptr
	_lfg_sampler_init_top_k           func(k int32) uintptr
	_lfg_sampler_init_top_p           func(p float32, minKeep uintptr) uintptr
	_lfg_sampler_init_min_p           func(p float32, minKeep uintptr) uintptr
	_lfg_sampler_init_typical         func(p float32, minKeep uintptr) uintptr
	_lfg_sampler_init_temp            func(t float32) uintptr
	_lfg_sampler_init_temp_ext        func(t float32, delta float32, exponent float32) uintptr
	_lfg_sampler_init_xtc             func(p float32, t float32, minKeep uintptr, seed uint32) uintptr
	_lfg_sampler_init_top_n_sigma     func(n float32) uintptr
	_lfg_sampler_init_mirostat        func(nVocab int32, seed uint32, tau float32, eta float32, m int32) uintptr
	_lfg_sampler_init_mirostat_v2     func(seed uint32, tau float32, eta float32) uintptr
	_lfg_sampler_init_grammar         func(vocab uintptr, grammarStr uintptr, grammarRoot uintptr) uintptr
	_lfg_sampler_init_penalties       func(penaltyLastN int32, penaltyRepeat float32, penaltyFreq float32, penaltyPresent float32) uintptr
	_lfg_sampler_init_dry             func(vocab uintptr, nCtxTrain int32, multiplier float32, base float32, allowedLength int32, penaltyLastN int32, seqBreakers uintptr, numBreakers uintptr) uintptr
	_lfg_sampler_init_adaptive_p      func(target float32, decay float32, seed uint32) uintptr
	_lfg_sampler_init_logit_bias      func(nVocab int32, nLogitBias int32, logitBias uintptr) uintptr
	_lfg_sampler_init_infill          func(vocab uintptr) uintptr
	_lfg_sampler_init_prefix          func(vocab uintptr, prefix uintptr) uintptr
	_lfg_sampler_prefix_set           func(smpl uintptr, prefix uintptr)
)

func registerSamplerFuncs() {
	samplerFuncsOnce.Do(func() {
		lib, err := loadLibrary()
		if err != nil {
			panic("lfg: failed to load library: " + err.Error())
		}
		purego.RegisterLibFunc(&_lfg_sampler_chain_default_params, lib, "lfg_sampler_chain_default_params")
		purego.RegisterLibFunc(&_lfg_sampler_chain_init, lib, "lfg_sampler_chain_init")
		purego.RegisterLibFunc(&_lfg_sampler_chain_add, lib, "lfg_sampler_chain_add")
		purego.RegisterLibFunc(&_lfg_sampler_chain_get, lib, "lfg_sampler_chain_get")
		purego.RegisterLibFunc(&_lfg_sampler_chain_n, lib, "lfg_sampler_chain_n")
		purego.RegisterLibFunc(&_lfg_sampler_chain_remove, lib, "lfg_sampler_chain_remove")
		purego.RegisterLibFunc(&_lfg_sampler_free, lib, "lfg_sampler_free")
		purego.RegisterLibFunc(&_lfg_sampler_name, lib, "lfg_sampler_name")
		purego.RegisterLibFunc(&_lfg_sampler_accept, lib, "lfg_sampler_accept")
		purego.RegisterLibFunc(&_lfg_sampler_reset, lib, "lfg_sampler_reset")
		purego.RegisterLibFunc(&_lfg_sampler_clone, lib, "lfg_sampler_clone")
		purego.RegisterLibFunc(&_lfg_sampler_get_seed, lib, "lfg_sampler_get_seed")
		purego.RegisterLibFunc(&_lfg_sampler_sample, lib, "lfg_sampler_sample")
		purego.RegisterLibFunc(&_lfg_sampler_init_greedy, lib, "lfg_sampler_init_greedy")
		purego.RegisterLibFunc(&_lfg_sampler_init_dist, lib, "lfg_sampler_init_dist")
		purego.RegisterLibFunc(&_lfg_sampler_init_top_k, lib, "lfg_sampler_init_top_k")
		purego.RegisterLibFunc(&_lfg_sampler_init_top_p, lib, "lfg_sampler_init_top_p")
		purego.RegisterLibFunc(&_lfg_sampler_init_min_p, lib, "lfg_sampler_init_min_p")
		purego.RegisterLibFunc(&_lfg_sampler_init_typical, lib, "lfg_sampler_init_typical")
		purego.RegisterLibFunc(&_lfg_sampler_init_temp, lib, "lfg_sampler_init_temp")
		purego.RegisterLibFunc(&_lfg_sampler_init_temp_ext, lib, "lfg_sampler_init_temp_ext")
		purego.RegisterLibFunc(&_lfg_sampler_init_xtc, lib, "lfg_sampler_init_xtc")
		purego.RegisterLibFunc(&_lfg_sampler_init_top_n_sigma, lib, "lfg_sampler_init_top_n_sigma")
		purego.RegisterLibFunc(&_lfg_sampler_init_mirostat, lib, "lfg_sampler_init_mirostat")
		purego.RegisterLibFunc(&_lfg_sampler_init_mirostat_v2, lib, "lfg_sampler_init_mirostat_v2")
		purego.RegisterLibFunc(&_lfg_sampler_init_grammar, lib, "lfg_sampler_init_grammar")
		purego.RegisterLibFunc(&_lfg_sampler_init_penalties, lib, "lfg_sampler_init_penalties")
		purego.RegisterLibFunc(&_lfg_sampler_init_dry, lib, "lfg_sampler_init_dry")
		purego.RegisterLibFunc(&_lfg_sampler_init_adaptive_p, lib, "lfg_sampler_init_adaptive_p")
		purego.RegisterLibFunc(&_lfg_sampler_init_logit_bias, lib, "lfg_sampler_init_logit_bias")
		purego.RegisterLibFunc(&_lfg_sampler_init_infill, lib, "lfg_sampler_init_infill")
		purego.RegisterLibFunc(&_lfg_sampler_init_prefix, lib, "lfg_sampler_init_prefix")
		purego.RegisterLibFunc(&_lfg_sampler_prefix_set, lib, "lfg_sampler_prefix_set")
	})
}

// ---------------------------------------------------------------------------
// Adapter functions
// ---------------------------------------------------------------------------

var (
	adapterFuncsOnce sync.Once

	_lfg_adapter_lora_init                      func(model uintptr, path uintptr) uintptr
	_lfg_adapter_meta_val_str                   func(adapter uintptr, key uintptr, buf uintptr, bufSize uintptr) int32
	_lfg_adapter_meta_count                     func(adapter uintptr) int32
	_lfg_adapter_meta_key_by_index              func(adapter uintptr, i int32, buf uintptr, bufSize uintptr) int32
	_lfg_adapter_meta_val_str_by_index          func(adapter uintptr, i int32, buf uintptr, bufSize uintptr) int32
	_lfg_adapter_get_alora_n_invocation_tokens  func(adapter uintptr) uint64
	_lfg_adapter_get_alora_invocation_tokens    func(adapter uintptr) uintptr
	_lfg_set_adapter_lora                       func(ctx uintptr, adapter uintptr, scale float32) int32
	_lfg_rm_adapter_lora                        func(ctx uintptr, adapter uintptr) int32
	_lfg_clear_adapter_lora                     func(ctx uintptr)
	_lfg_apply_adapter_cvec                     func(ctx uintptr, data uintptr, dataLen uintptr, nEmbd int32, ilStart int32, ilEnd int32) int32
)

func registerAdapterFuncs() {
	adapterFuncsOnce.Do(func() {
		lib, err := loadLibrary()
		if err != nil {
			panic("lfg: failed to load library: " + err.Error())
		}
		purego.RegisterLibFunc(&_lfg_adapter_lora_init, lib, "lfg_adapter_lora_init")
		purego.RegisterLibFunc(&_lfg_adapter_meta_val_str, lib, "lfg_adapter_meta_val_str")
		purego.RegisterLibFunc(&_lfg_adapter_meta_count, lib, "lfg_adapter_meta_count")
		purego.RegisterLibFunc(&_lfg_adapter_meta_key_by_index, lib, "lfg_adapter_meta_key_by_index")
		purego.RegisterLibFunc(&_lfg_adapter_meta_val_str_by_index, lib, "lfg_adapter_meta_val_str_by_index")
		purego.RegisterLibFunc(&_lfg_adapter_get_alora_n_invocation_tokens, lib, "lfg_adapter_get_alora_n_invocation_tokens")
		purego.RegisterLibFunc(&_lfg_adapter_get_alora_invocation_tokens, lib, "lfg_adapter_get_alora_invocation_tokens")
		purego.RegisterLibFunc(&_lfg_set_adapter_lora, lib, "lfg_set_adapter_lora")
		purego.RegisterLibFunc(&_lfg_rm_adapter_lora, lib, "lfg_rm_adapter_lora")
		purego.RegisterLibFunc(&_lfg_clear_adapter_lora, lib, "lfg_clear_adapter_lora")
		purego.RegisterLibFunc(&_lfg_apply_adapter_cvec, lib, "lfg_apply_adapter_cvec")
	})
}

// ---------------------------------------------------------------------------
// State functions
// ---------------------------------------------------------------------------

var (
	stateFuncsOnce sync.Once

	_lfg_state_get_size      func(ctx uintptr) uintptr
	_lfg_state_get_data      func(ctx uintptr, dst uintptr, size uintptr) uintptr
	_lfg_state_set_data      func(ctx uintptr, src uintptr, size uintptr) uintptr
	_lfg_state_load_file     func(ctx uintptr, path uintptr, tokens uintptr, nTokenCap uintptr, nTokenCountOut uintptr) bool
	_lfg_state_save_file     func(ctx uintptr, path uintptr, tokens uintptr, nTokenCount uintptr) bool
	_lfg_state_seq_get_size  func(ctx uintptr, seqID int32) uintptr
	_lfg_state_seq_get_data  func(ctx uintptr, dst uintptr, size uintptr, seqID int32) uintptr
	_lfg_state_seq_set_data  func(ctx uintptr, src uintptr, size uintptr, destSeqID int32) uintptr
	_lfg_state_seq_save_file func(ctx uintptr, filepath uintptr, seqID int32, tokens uintptr, nTokenCount uintptr) uintptr
	_lfg_state_seq_load_file func(ctx uintptr, filepath uintptr, destSeqID int32, tokens uintptr, nTokenCap uintptr, nTokenCountOut uintptr) uintptr
)

func registerStateFuncs() {
	stateFuncsOnce.Do(func() {
		lib, err := loadLibrary()
		if err != nil {
			panic("lfg: failed to load library: " + err.Error())
		}
		purego.RegisterLibFunc(&_lfg_state_get_size, lib, "lfg_state_get_size")
		purego.RegisterLibFunc(&_lfg_state_get_data, lib, "lfg_state_get_data")
		purego.RegisterLibFunc(&_lfg_state_set_data, lib, "lfg_state_set_data")
		purego.RegisterLibFunc(&_lfg_state_load_file, lib, "lfg_state_load_file")
		purego.RegisterLibFunc(&_lfg_state_save_file, lib, "lfg_state_save_file")
		purego.RegisterLibFunc(&_lfg_state_seq_get_size, lib, "lfg_state_seq_get_size")
		purego.RegisterLibFunc(&_lfg_state_seq_get_data, lib, "lfg_state_seq_get_data")
		purego.RegisterLibFunc(&_lfg_state_seq_set_data, lib, "lfg_state_seq_set_data")
		purego.RegisterLibFunc(&_lfg_state_seq_save_file, lib, "lfg_state_seq_save_file")
		purego.RegisterLibFunc(&_lfg_state_seq_load_file, lib, "lfg_state_seq_load_file")
	})
}

// ---------------------------------------------------------------------------
// Chat functions
// ---------------------------------------------------------------------------

var (
	chatFuncsOnce sync.Once

	_lfg_chat_apply_template    func(tmpl uintptr, chat uintptr, nMsg uintptr, addAss bool, buf uintptr, length int32) int32
	_lfg_chat_builtin_templates func(output uintptr, length uintptr) int32
)

func registerChatFuncs() {
	chatFuncsOnce.Do(func() {
		lib, err := loadLibrary()
		if err != nil {
			panic("lfg: failed to load library: " + err.Error())
		}
		purego.RegisterLibFunc(&_lfg_chat_apply_template, lib, "lfg_chat_apply_template")
		purego.RegisterLibFunc(&_lfg_chat_builtin_templates, lib, "lfg_chat_builtin_templates")
	})
}

// ---------------------------------------------------------------------------
// Performance functions
// ---------------------------------------------------------------------------

var (
	perfFuncsOnce sync.Once

	_lfg_perf_context       func(ctx uintptr) cPerfContextData
	_lfg_perf_context_print func(ctx uintptr)
	_lfg_perf_context_reset func(ctx uintptr)
	_lfg_perf_sampler       func(chain uintptr) cPerfSamplerData
	_lfg_perf_sampler_print func(chain uintptr)
	_lfg_perf_sampler_reset func(chain uintptr)
	_lfg_memory_breakdown_print func(ctx uintptr)
)

func registerPerfFuncs() {
	perfFuncsOnce.Do(func() {
		lib, err := loadLibrary()
		if err != nil {
			panic("lfg: failed to load library: " + err.Error())
		}
		purego.RegisterLibFunc(&_lfg_perf_context, lib, "lfg_perf_context")
		purego.RegisterLibFunc(&_lfg_perf_context_print, lib, "lfg_perf_context_print")
		purego.RegisterLibFunc(&_lfg_perf_context_reset, lib, "lfg_perf_context_reset")
		purego.RegisterLibFunc(&_lfg_perf_sampler, lib, "lfg_perf_sampler")
		purego.RegisterLibFunc(&_lfg_perf_sampler_print, lib, "lfg_perf_sampler_print")
		purego.RegisterLibFunc(&_lfg_perf_sampler_reset, lib, "lfg_perf_sampler_reset")
		purego.RegisterLibFunc(&_lfg_memory_breakdown_print, lib, "lfg_memory_breakdown_print")
	})
}

// ---------------------------------------------------------------------------
// Session functions
// ---------------------------------------------------------------------------

var (
	sessionFuncsOnce sync.Once

	_lfg_sampling_default_config              func() cSamplingConfig
	_lfg_session_default_config               func() cSessionConfig
	_lfg_session_create                       func(model uintptr, config uintptr) uintptr
	_lfg_session_free                         func(session uintptr)
	_lfg_session_reset                        func(session uintptr)
	_lfg_session_configure_structured         func(session uintptr, grammarOrSchema uintptr, rootRule uintptr) bool
	_lfg_session_configure_reasoning          func(session uintptr, startTokens uintptr, nStart uintptr, endTokens uintptr, nEnd uintptr)
	_lfg_session_configure_stop_sequences     func(session uintptr, sequences uintptr, seqLengths uintptr, nSequences uintptr) bool
	_lfg_session_configure_stop_strings       func(session uintptr, strings uintptr, nStrings int32) bool
	_lfg_session_ingest_tokens                func(session uintptr, tokens uintptr, nTokens uintptr, updateSampler bool) bool
	_lfg_session_decode                       func(session uintptr) bool
	_lfg_session_sample                       func(session uintptr) int32
	_lfg_session_heal_last_token              func(session uintptr) bool
	_lfg_session_get_logits                   func(session uintptr, out uintptr, maxOut int32) int32
	_lfg_session_get_vocab_size               func(session uintptr) int32
	_lfg_session_create_checkpoint            func(session uintptr) uintptr
	_lfg_checkpoint_restore_default_options   func() cCheckpointRestoreOptions
	_lfg_session_restore_checkpoint           func(session uintptr, checkpoint uintptr) bool
	_lfg_session_restore_checkpoint_ex        func(session uintptr, checkpoint uintptr, options uintptr) bool
	_lfg_checkpoint_free                      func(checkpoint uintptr)
	_lfg_session_register_tools               func(session uintptr, tools uintptr, nTools int32, topK int32) int32
	_lfg_session_clear_tools                  func(session uintptr)
	_lfg_entropy_monitor_default_config       func() cEntropyMonitorConfig
	_lfg_session_configure_entropy_monitor    func(session uintptr, config uintptr) int32
	_lfg_session_entropy_pop                  func(session uintptr, eventOut uintptr, embdOut uintptr, embdCap int32) bool
	_lfg_session_entropy_pending              func(session uintptr) int32
	_lfg_session_entropy_flush                func(session uintptr)
	_lfg_session_entropy_counter              func(session uintptr) uintptr
	_lfg_session_rewind                       func(session uintptr, checkpointID int32) bool
	_lfg_session_get_last_entropy             func(session uintptr) float32
	_lfg_confidence_monitor_default_config    func() cConfidenceMonitorConfig
	_lfg_session_configure_confidence_monitor func(session uintptr, config uintptr) int32
	_lfg_session_confidence_pop               func(session uintptr, eventOut uintptr, embdOut uintptr, embdCap int32) bool
	_lfg_session_confidence_pending           func(session uintptr) int32
	_lfg_session_confidence_flush             func(session uintptr)
	_lfg_session_confidence_counter           func(session uintptr) uintptr
	_lfg_surprise_monitor_default_config      func() cSurpriseMonitorConfig
	_lfg_session_configure_surprise_monitor   func(session uintptr, config uintptr) int32
	_lfg_session_surprise_pop                 func(session uintptr, eventOut uintptr, embdOut uintptr, embdCap int32) bool
	_lfg_session_embed                        func(session uintptr, text uintptr, textLen int32, out uintptr, outCap int32) int32
	_lfg_session_rank_tools                  func(session uintptr, query uintptr, queryLen int32, buf uintptr, bufSize int32) int32
	_lfg_session_get_last_prompt             func(session uintptr, lenOut uintptr) uintptr
	_lfg_session_get_tool_calls              func(session uintptr, nOut uintptr) uintptr
	_lfg_session_get_last_output             func(session uintptr, lenOut uintptr) uintptr
	_lfg_session_set_tool_call_format        func(session uintptr, format int32)
	_lfg_parse_pythonic_tool_calls           func(text uintptr, textLen int32, out uintptr, outCap int32) int32
	_lfg_json_schema_to_grammar              func(jsonSchema uintptr, forceGBNF bool, buf uintptr, bufSize uintptr) int32
	_lfg_model_load_default_config           func() cModelLoadConfig
	_lfg_load_model                          func(config uintptr) uintptr
	_lfg_model_get_stats                     func(model uintptr) cModelStats
	_free                                    func(ptr uintptr)
	_malloc                                  func(size uintptr) uintptr
)

func registerSessionFuncs() {
	sessionFuncsOnce.Do(func() {
		lib, err := loadLibrary()
		if err != nil {
			panic("lfg: failed to load library: " + err.Error())
		}
		purego.RegisterLibFunc(&_lfg_sampling_default_config, lib, "lfg_sampling_default_config")
		purego.RegisterLibFunc(&_lfg_session_default_config, lib, "lfg_session_default_config")
		purego.RegisterLibFunc(&_lfg_session_create, lib, "lfg_session_create")
		purego.RegisterLibFunc(&_lfg_session_free, lib, "lfg_session_free")
		purego.RegisterLibFunc(&_lfg_session_reset, lib, "lfg_session_reset")
		purego.RegisterLibFunc(&_lfg_session_configure_structured, lib, "lfg_session_configure_structured")
		purego.RegisterLibFunc(&_lfg_session_configure_reasoning, lib, "lfg_session_configure_reasoning")
		purego.RegisterLibFunc(&_lfg_session_configure_stop_sequences, lib, "lfg_session_configure_stop_sequences")
		purego.RegisterLibFunc(&_lfg_session_configure_stop_strings, lib, "lfg_session_configure_stop_strings")
		purego.RegisterLibFunc(&_lfg_session_ingest_tokens, lib, "lfg_session_ingest_tokens")
		purego.RegisterLibFunc(&_lfg_session_decode, lib, "lfg_session_decode")
		purego.RegisterLibFunc(&_lfg_session_sample, lib, "lfg_session_sample")
		purego.RegisterLibFunc(&_lfg_session_heal_last_token, lib, "lfg_session_heal_last_token")
		purego.RegisterLibFunc(&_lfg_session_get_logits, lib, "lfg_session_get_logits")
		purego.RegisterLibFunc(&_lfg_session_get_vocab_size, lib, "lfg_session_get_vocab_size")
		purego.RegisterLibFunc(&_lfg_session_create_checkpoint, lib, "lfg_session_create_checkpoint")
		purego.RegisterLibFunc(&_lfg_checkpoint_restore_default_options, lib, "lfg_checkpoint_restore_default_options")
		purego.RegisterLibFunc(&_lfg_session_restore_checkpoint, lib, "lfg_session_restore_checkpoint")
		purego.RegisterLibFunc(&_lfg_session_restore_checkpoint_ex, lib, "lfg_session_restore_checkpoint_ex")
		purego.RegisterLibFunc(&_lfg_checkpoint_free, lib, "lfg_checkpoint_free")
		purego.RegisterLibFunc(&_lfg_session_register_tools, lib, "lfg_session_register_tools")
		purego.RegisterLibFunc(&_lfg_session_clear_tools, lib, "lfg_session_clear_tools")
		purego.RegisterLibFunc(&_lfg_session_rank_tools, lib, "lfg_session_rank_tools")
		purego.RegisterLibFunc(&_lfg_session_get_last_prompt, lib, "lfg_session_get_last_prompt")
		purego.RegisterLibFunc(&_lfg_session_get_tool_calls, lib, "lfg_session_get_tool_calls")
		purego.RegisterLibFunc(&_lfg_session_get_last_output, lib, "lfg_session_get_last_output")
		purego.RegisterLibFunc(&_lfg_session_set_tool_call_format, lib, "lfg_session_set_tool_call_format")
		purego.RegisterLibFunc(&_lfg_parse_pythonic_tool_calls, lib, "lfg_parse_pythonic_tool_calls")
		purego.RegisterLibFunc(&_lfg_entropy_monitor_default_config, lib, "lfg_entropy_monitor_default_config")
		purego.RegisterLibFunc(&_lfg_session_configure_entropy_monitor, lib, "lfg_session_configure_entropy_monitor")
		purego.RegisterLibFunc(&_lfg_session_entropy_pop, lib, "lfg_session_entropy_pop")
		purego.RegisterLibFunc(&_lfg_session_entropy_pending, lib, "lfg_session_entropy_pending")
		purego.RegisterLibFunc(&_lfg_session_entropy_flush, lib, "lfg_session_entropy_flush")
		purego.RegisterLibFunc(&_lfg_session_entropy_counter, lib, "lfg_session_entropy_counter")
		purego.RegisterLibFunc(&_lfg_session_rewind, lib, "lfg_session_rewind")
		purego.RegisterLibFunc(&_lfg_session_get_last_entropy, lib, "lfg_session_get_last_entropy")
		purego.RegisterLibFunc(&_lfg_confidence_monitor_default_config, lib, "lfg_confidence_monitor_default_config")
		purego.RegisterLibFunc(&_lfg_session_configure_confidence_monitor, lib, "lfg_session_configure_confidence_monitor")
		purego.RegisterLibFunc(&_lfg_session_confidence_pop, lib, "lfg_session_confidence_pop")
		purego.RegisterLibFunc(&_lfg_session_confidence_pending, lib, "lfg_session_confidence_pending")
		purego.RegisterLibFunc(&_lfg_session_confidence_flush, lib, "lfg_session_confidence_flush")
		purego.RegisterLibFunc(&_lfg_session_confidence_counter, lib, "lfg_session_confidence_counter")
		purego.RegisterLibFunc(&_lfg_surprise_monitor_default_config, lib, "lfg_surprise_monitor_default_config")
		purego.RegisterLibFunc(&_lfg_session_configure_surprise_monitor, lib, "lfg_session_configure_surprise_monitor")
		purego.RegisterLibFunc(&_lfg_session_surprise_pop, lib, "lfg_session_surprise_pop")
		purego.RegisterLibFunc(&_lfg_session_embed, lib, "lfg_session_embed")
		purego.RegisterLibFunc(&_lfg_json_schema_to_grammar, lib, "lfg_json_schema_to_grammar")
		purego.RegisterLibFunc(&_lfg_model_load_default_config, lib, "lfg_model_load_default_config")
		purego.RegisterLibFunc(&_lfg_load_model, lib, "lfg_load_model")
		purego.RegisterLibFunc(&_lfg_model_get_stats, lib, "lfg_model_get_stats")
		purego.RegisterLibFunc(&_free, lib, "free")
		purego.RegisterLibFunc(&_malloc, lib, "malloc")
	})
}

// ---------------------------------------------------------------------------
// Generate loop functions
// ---------------------------------------------------------------------------

var (
	generateFuncsOnce sync.Once

	_lfg_generate_default_config      func() cGenerateConfig
	_lfg_session_generate             func(session uintptr, config cGenerateConfig) cGenerateResult
	_lfg_session_prompt_generate      func(session uintptr, prompt uintptr, promptLen int32, addBOS bool, config cGenerateConfig) cGenerateResult
	_lfg_session_chat_generate        func(session uintptr, messages uintptr, nMessages uintptr, config cGenerateConfig) cGenerateResult
)

func registerGenerateFuncs() {
	generateFuncsOnce.Do(func() {
		lib, err := loadLibrary()
		if err != nil {
			panic("lfg: failed to load library: " + err.Error())
		}
		purego.RegisterLibFunc(&_lfg_generate_default_config, lib, "lfg_generate_default_config")
		purego.RegisterLibFunc(&_lfg_session_generate, lib, "lfg_session_generate")
		purego.RegisterLibFunc(&_lfg_session_prompt_generate, lib, "lfg_session_prompt_generate")
		purego.RegisterLibFunc(&_lfg_session_chat_generate, lib, "lfg_session_chat_generate")
	})
}

// ---------------------------------------------------------------------------
// Helper: token slice to uintptr
// ---------------------------------------------------------------------------

func tokenPtr(tokens []Token) uintptr {
	if len(tokens) == 0 {
		return 0
	}
	return uintptr(unsafe.Pointer(&tokens[0]))
}
