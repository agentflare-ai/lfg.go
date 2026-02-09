//go:build (darwin && arm64) || (linux && amd64) || (linux && arm64)

package lfg

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
	if ctx.c == 0 {
		return ContextPerf{}
	}
	registerPerfFuncs()
	data := _lfg_perf_context(ctx.c)
	return ContextPerf{
		TStartMs: data.TStartMs,
		TLoadMs:  data.TLoadMs,
		TPEvalMs: data.TPEvalMs,
		TEvalMs:  data.TEvalMs,
		NPEval:   int(data.NPEval),
		NEval:    int(data.NEval),
		NReused:  int(data.NReused),
	}
}

// PrintPerformance prints performance data to the log.
func (ctx *Context) PrintPerformance() {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == 0 {
		return
	}
	registerPerfFuncs()
	_lfg_perf_context_print(ctx.c)
}

// ResetPerformance resets performance counters.
func (ctx *Context) ResetPerformance() {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.c == 0 {
		return
	}
	registerPerfFuncs()
	_lfg_perf_context_reset(ctx.c)
}

// Performance returns performance data for a sampler chain.
func (s *Sampler) Performance() SamplerPerf {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return SamplerPerf{}
	}
	registerPerfFuncs()
	data := _lfg_perf_sampler(s.c)
	return SamplerPerf{
		TSampleMs: data.TSampleMs,
		NSample:   int(data.NSample),
	}
}

// PrintPerformance prints sampler performance data to the log.
func (s *Sampler) PrintPerformance() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return
	}
	registerPerfFuncs()
	_lfg_perf_sampler_print(s.c)
}

// ResetPerformance resets sampler performance counters.
func (s *Sampler) ResetPerformance() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.c == 0 {
		return
	}
	registerPerfFuncs()
	_lfg_perf_sampler_reset(s.c)
}

// MemoryBreakdownPrint prints per-device memory usage via the log.
func (ctx *Context) MemoryBreakdownPrint() {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == 0 {
		return
	}
	registerPerfFuncs()
	_lfg_memory_breakdown_print(ctx.c)
}
