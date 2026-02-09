//go:build (darwin && arm64) || (linux && amd64) || (linux && arm64)

package lfg

// Memory wraps the KV cache / memory interface of a context.
// It borrows the parent Context's lock.
type Memory struct {
	c   uintptr
	ctx *Context // prevent GC of parent
}

// Memory returns the memory (KV cache) handle for this context.
func (ctx *Context) Memory() *Memory {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()
	if ctx.c == 0 {
		return nil
	}
	registerContextFuncs()
	mem := _lfg_get_memory(ctx.c)
	return &Memory{c: mem, ctx: ctx}
}

// Clear clears the memory contents.
// If data is true, data buffers are also cleared.
func (m *Memory) Clear(data bool) {
	m.ctx.mu.Lock()
	defer m.ctx.mu.Unlock()
	if m.ctx.c == 0 {
		return
	}
	registerMemoryFuncs()
	_lfg_memory_clear(m.c, data)
}

// SeqRm removes tokens in the range [p0, p1) for the given sequence.
// seqID < 0 matches any sequence. p0 < 0 means [0, p1]. p1 < 0 means [p0, inf).
// Returns false if a partial sequence cannot be removed.
func (m *Memory) SeqRm(seqID SequenceID, p0, p1 Position) bool {
	m.ctx.mu.Lock()
	defer m.ctx.mu.Unlock()
	if m.ctx.c == 0 {
		return false
	}
	registerMemoryFuncs()
	return _lfg_memory_seq_rm(m.c, int32(seqID), int32(p0), int32(p1))
}

// SeqCp copies tokens from seqSrc to seqDst in the range [p0, p1).
func (m *Memory) SeqCp(seqSrc, seqDst SequenceID, p0, p1 Position) {
	m.ctx.mu.Lock()
	defer m.ctx.mu.Unlock()
	if m.ctx.c == 0 {
		return
	}
	registerMemoryFuncs()
	_lfg_memory_seq_cp(m.c, int32(seqSrc), int32(seqDst), int32(p0), int32(p1))
}

// SeqKeep removes all tokens that don't belong to the specified sequence.
func (m *Memory) SeqKeep(seqID SequenceID) {
	m.ctx.mu.Lock()
	defer m.ctx.mu.Unlock()
	if m.ctx.c == 0 {
		return
	}
	registerMemoryFuncs()
	_lfg_memory_seq_keep(m.c, int32(seqID))
}

// SeqAdd adds a relative position delta to tokens in [p0, p1) for the given sequence.
func (m *Memory) SeqAdd(seqID SequenceID, p0, p1 Position, delta Position) {
	m.ctx.mu.Lock()
	defer m.ctx.mu.Unlock()
	if m.ctx.c == 0 {
		return
	}
	registerMemoryFuncs()
	_lfg_memory_seq_add(m.c, int32(seqID), int32(p0), int32(p1), int32(delta))
}

// SeqDiv divides positions by factor d for tokens in [p0, p1).
func (m *Memory) SeqDiv(seqID SequenceID, p0, p1 Position, d int) {
	m.ctx.mu.Lock()
	defer m.ctx.mu.Unlock()
	if m.ctx.c == 0 {
		return
	}
	registerMemoryFuncs()
	_lfg_memory_seq_div(m.c, int32(seqID), int32(p0), int32(p1), int32(d))
}

// SeqPosMin returns the smallest position present for the given sequence.
// Returns -1 if the sequence is empty.
func (m *Memory) SeqPosMin(seqID SequenceID) Position {
	m.ctx.mu.RLock()
	defer m.ctx.mu.RUnlock()
	if m.ctx.c == 0 {
		return -1
	}
	registerMemoryFuncs()
	return Position(_lfg_memory_seq_pos_min(m.c, int32(seqID)))
}

// SeqPosMax returns the largest position present for the given sequence.
// Returns -1 if the sequence is empty.
func (m *Memory) SeqPosMax(seqID SequenceID) Position {
	m.ctx.mu.RLock()
	defer m.ctx.mu.RUnlock()
	if m.ctx.c == 0 {
		return -1
	}
	registerMemoryFuncs()
	return Position(_lfg_memory_seq_pos_max(m.c, int32(seqID)))
}

// CanShift returns whether the memory supports shifting.
func (m *Memory) CanShift() bool {
	m.ctx.mu.RLock()
	defer m.ctx.mu.RUnlock()
	if m.ctx.c == 0 {
		return false
	}
	registerMemoryFuncs()
	return _lfg_memory_can_shift(m.c)
}
