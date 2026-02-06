package lfg

/*
#include "lfg_inference.h"
*/
import "C"

// ContextPerf holds performance metrics for a context.
type ContextPerf struct {
	TStartMs float64 // absolute start time in ms
	TLoadMs  float64 // time for loading the model in ms
	TPEvalMs float64 // time for processing the prompt in ms
	TEvalMs  float64 // time for generating tokens in ms
	NPEval   int     // number of prompt tokens
	NEval    int     // number of generated tokens
	NReused  int     // number of reused compute graphs
}

// SamplerPerf holds performance metrics for a sampler chain.
type SamplerPerf struct {
	TSampleMs float64 // time for sampling in ms
	NSample   int     // number of sampled tokens
}

// Performance returns performance data for the context.
func (ctx *Context) Performance() ContextPerf {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == nil {
		return ContextPerf{}
	}
	data := C.lfg_perf_context(ctx.c)
	return ContextPerf{
		TStartMs: float64(data.t_start_ms),
		TLoadMs:  float64(data.t_load_ms),
		TPEvalMs: float64(data.t_p_eval_ms),
		TEvalMs:  float64(data.t_eval_ms),
		NPEval:   int(data.n_p_eval),
		NEval:    int(data.n_eval),
		NReused:  int(data.n_reused),
	}
}

// PrintPerformance prints performance data to the log.
func (ctx *Context) PrintPerformance() {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == nil {
		return
	}
	C.lfg_perf_context_print(ctx.c)
}

// ResetPerformance resets performance counters.
func (ctx *Context) ResetPerformance() {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == nil {
		return
	}
	C.lfg_perf_context_reset(ctx.c)
}

// Performance returns performance data for a sampler chain.
func (s *Sampler) Performance() SamplerPerf {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return SamplerPerf{}
	}
	data := C.lfg_perf_sampler(s.c)
	return SamplerPerf{
		TSampleMs: float64(data.t_sample_ms),
		NSample:   int(data.n_sample),
	}
}

// PrintPerformance prints sampler performance data to the log.
func (s *Sampler) PrintPerformance() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return
	}
	C.lfg_perf_sampler_print(s.c)
}

// ResetPerformance resets sampler performance counters.
func (s *Sampler) ResetPerformance() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == nil {
		return
	}
	C.lfg_perf_sampler_reset(s.c)
}

// MemoryBreakdownPrint prints per-device memory usage via the log.
func (ctx *Context) MemoryBreakdownPrint() {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == nil {
		return
	}
	C.lfg_memory_breakdown_print(ctx.c)
}
