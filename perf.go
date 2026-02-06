package lfg

/*
typedef struct lfm_model lfm_model;
typedef struct lfm_context lfm_context;
typedef struct lfm_vocab lfm_vocab;
typedef struct lfm_sampler lfm_sampler;
#include "lfm_inference.h"
*/
import "C"

// PerfContextData holds performance metrics for a context.
type PerfContextData struct {
	TStartMs float64 // absolute start time in ms
	TLoadMs  float64 // time for loading the model in ms
	TPEvalMs float64 // time for processing the prompt in ms
	TEvalMs  float64 // time for generating tokens in ms
	NPEval   int     // number of prompt tokens
	NEval    int     // number of generated tokens
	NReused  int     // number of reused compute graphs
}

// PerfSamplerData holds performance metrics for a sampler chain.
type PerfSamplerData struct {
	TSampleMs float64 // time for sampling in ms
	NSample   int     // number of sampled tokens
}

// PerfContext returns performance data for the context.
func (ctx *Context) PerfContext() PerfContextData {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == nil {
		return PerfContextData{}
	}
	data := C.lfm_perf_context(ctx.c)
	return PerfContextData{
		TStartMs: float64(data.t_start_ms),
		TLoadMs:  float64(data.t_load_ms),
		TPEvalMs: float64(data.t_p_eval_ms),
		TEvalMs:  float64(data.t_eval_ms),
		NPEval:   int(data.n_p_eval),
		NEval:    int(data.n_eval),
		NReused:  int(data.n_reused),
	}
}

// PerfContextPrint prints performance data to the log.
func (ctx *Context) PerfContextPrint() {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == nil {
		return
	}
	C.lfm_perf_context_print(ctx.c)
}

// PerfContextReset resets performance counters.
func (ctx *Context) PerfContextReset() {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == nil {
		return
	}
	C.lfm_perf_context_reset(ctx.c)
}

// PerfSampler returns performance data for a sampler chain.
func (s *Sampler) PerfSampler() PerfSamplerData {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return PerfSamplerData{}
	}
	data := C.lfm_perf_sampler(s.c)
	return PerfSamplerData{
		TSampleMs: float64(data.t_sample_ms),
		NSample:   int(data.n_sample),
	}
}

// PerfSamplerPrint prints sampler performance data to the log.
func (s *Sampler) PerfSamplerPrint() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return
	}
	C.lfm_perf_sampler_print(s.c)
}

// PerfSamplerReset resets sampler performance counters.
func (s *Sampler) PerfSamplerReset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return
	}
	C.lfm_perf_sampler_reset(s.c)
}

// MemoryBreakdownPrint prints per-device memory usage via the log.
func (ctx *Context) MemoryBreakdownPrint() {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == nil {
		return
	}
	C.lfm_memory_breakdown_print(ctx.c)
}
