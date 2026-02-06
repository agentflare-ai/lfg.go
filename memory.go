package lfg

/*
typedef struct lfm_model lfm_model;
typedef struct lfm_context lfm_context;
typedef struct lfm_vocab lfm_vocab;
typedef struct lfm_sampler lfm_sampler;
#include "lfm_inference.h"
*/
import "C"

// Memory wraps the KV cache / memory interface of a context.
// It borrows the parent Context's lock.
type Memory struct {
	c   C.lfm_memory_t
	ctx *Context // prevent GC of parent
}

// GetMemory returns the memory (KV cache) handle for this context.
func (ctx *Context) GetMemory() *Memory {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == nil {
		return nil
	}
	mem := C.lfm_get_memory(ctx.c)
	return &Memory{c: mem, ctx: ctx}
}

// Clear clears the memory contents.
// If data is true, data buffers are also cleared.
func (m *Memory) Clear(data bool) {
	m.ctx.mu.Lock()
	defer m.ctx.mu.Unlock()
	if m.ctx.c == nil {
		return
	}
	C.lfm_memory_clear(m.c, C.bool(data))
}

// SeqRm removes tokens in the range [p0, p1) for the given sequence.
// seqID < 0 matches any sequence. p0 < 0 means [0, p1]. p1 < 0 means [p0, inf).
// Returns false if a partial sequence cannot be removed.
func (m *Memory) SeqRm(seqID SeqID, p0, p1 Pos) bool {
	m.ctx.mu.Lock()
	defer m.ctx.mu.Unlock()
	if m.ctx.c == nil {
		return false
	}
	return bool(C.lfm_memory_seq_rm(m.c, C.lfm_seq_id(seqID), C.lfm_pos(p0), C.lfm_pos(p1)))
}

// SeqCp copies tokens from seqSrc to seqDst in the range [p0, p1).
func (m *Memory) SeqCp(seqSrc, seqDst SeqID, p0, p1 Pos) {
	m.ctx.mu.Lock()
	defer m.ctx.mu.Unlock()
	if m.ctx.c == nil {
		return
	}
	C.lfm_memory_seq_cp(m.c, C.lfm_seq_id(seqSrc), C.lfm_seq_id(seqDst), C.lfm_pos(p0), C.lfm_pos(p1))
}

// SeqKeep removes all tokens that don't belong to the specified sequence.
func (m *Memory) SeqKeep(seqID SeqID) {
	m.ctx.mu.Lock()
	defer m.ctx.mu.Unlock()
	if m.ctx.c == nil {
		return
	}
	C.lfm_memory_seq_keep(m.c, C.lfm_seq_id(seqID))
}

// SeqAdd adds a relative position delta to tokens in [p0, p1) for the given sequence.
func (m *Memory) SeqAdd(seqID SeqID, p0, p1 Pos, delta Pos) {
	m.ctx.mu.Lock()
	defer m.ctx.mu.Unlock()
	if m.ctx.c == nil {
		return
	}
	C.lfm_memory_seq_add(m.c, C.lfm_seq_id(seqID), C.lfm_pos(p0), C.lfm_pos(p1), C.lfm_pos(delta))
}

// SeqDiv divides positions by factor d for tokens in [p0, p1).
func (m *Memory) SeqDiv(seqID SeqID, p0, p1 Pos, d int) {
	m.ctx.mu.Lock()
	defer m.ctx.mu.Unlock()
	if m.ctx.c == nil {
		return
	}
	C.lfm_memory_seq_div(m.c, C.lfm_seq_id(seqID), C.lfm_pos(p0), C.lfm_pos(p1), C.int(d))
}

// SeqPosMin returns the smallest position present for the given sequence.
// Returns -1 if the sequence is empty.
func (m *Memory) SeqPosMin(seqID SeqID) Pos {
	m.ctx.mu.RLock()
	defer m.ctx.mu.RUnlock()
	if m.ctx.c == nil {
		return -1
	}
	return Pos(C.lfm_memory_seq_pos_min(m.c, C.lfm_seq_id(seqID)))
}

// SeqPosMax returns the largest position present for the given sequence.
// Returns -1 if the sequence is empty.
func (m *Memory) SeqPosMax(seqID SeqID) Pos {
	m.ctx.mu.RLock()
	defer m.ctx.mu.RUnlock()
	if m.ctx.c == nil {
		return -1
	}
	return Pos(C.lfm_memory_seq_pos_max(m.c, C.lfm_seq_id(seqID)))
}

// CanShift returns whether the memory supports shifting.
func (m *Memory) CanShift() bool {
	m.ctx.mu.RLock()
	defer m.ctx.mu.RUnlock()
	if m.ctx.c == nil {
		return false
	}
	return bool(C.lfm_memory_can_shift(m.c))
}
