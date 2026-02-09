//go:build (darwin && arm64) || (linux && amd64) || (linux && arm64)

package lfg

import (
	"runtime"
	"unsafe"
)

// Vocab provides access to the model's vocabulary. It borrows its parent
// Model's lock — the Model must remain alive while Vocab is in use.
type Vocab struct {
	c     uintptr // *lfg_vocab (C pointer as uintptr)
	model *Model  // prevent GC of parent
}

// TokenCount returns the total number of tokens in the vocabulary.
func (v *Vocab) TokenCount() int {
	v.model.mu.RLock()
	defer v.model.mu.RUnlock()
	if v.c == 0 {
		return 0
	}
	registerVocabFuncs()
	return int(_lfg_vocab_n_tokens(v.c))
}

// Type returns the vocabulary type.
func (v *Vocab) Type() VocabType {
	v.model.mu.RLock()
	defer v.model.mu.RUnlock()
	if v.c == 0 {
		return VocabTypeNone
	}
	registerVocabFuncs()
	return VocabType(_lfg_vocab_type(v.c))
}

// Text returns the raw text of a token.
func (v *Vocab) Text(token Token) string {
	v.model.mu.RLock()
	defer v.model.mu.RUnlock()
	if v.c == 0 {
		return ""
	}
	registerVocabFuncs()
	return goString(_lfg_vocab_get_text(v.c, int32(token)))
}

// Score returns the score (log probability) of a token.
func (v *Vocab) Score(token Token) float32 {
	v.model.mu.RLock()
	defer v.model.mu.RUnlock()
	if v.c == 0 {
		return 0
	}
	registerVocabFuncs()
	return _lfg_vocab_get_score(v.c, int32(token))
}

// Attributes returns the attributes of a token.
func (v *Vocab) Attributes(token Token) TokenAttr {
	v.model.mu.RLock()
	defer v.model.mu.RUnlock()
	if v.c == 0 {
		return TokenAttrUndefined
	}
	registerVocabFuncs()
	return TokenAttr(_lfg_vocab_get_attr(v.c, int32(token)))
}

// IsEOG returns true if the token is an end-of-generation token.
func (v *Vocab) IsEOG(token Token) bool {
	v.model.mu.RLock()
	defer v.model.mu.RUnlock()
	if v.c == 0 {
		return false
	}
	registerVocabFuncs()
	return _lfg_vocab_is_eog(v.c, int32(token))
}

// IsControl returns true if the token is a control token.
func (v *Vocab) IsControl(token Token) bool {
	v.model.mu.RLock()
	defer v.model.mu.RUnlock()
	if v.c == 0 {
		return false
	}
	registerVocabFuncs()
	return _lfg_vocab_is_control(v.c, int32(token))
}

// BOS returns the beginning-of-sentence token.
func (v *Vocab) BOS() Token {
	registerVocabFuncs()
	return Token(_lfg_vocab_bos(v.c))
}

// EOS returns the end-of-sentence token.
func (v *Vocab) EOS() Token {
	registerVocabFuncs()
	return Token(_lfg_vocab_eos(v.c))
}

// EOT returns the end-of-turn token.
func (v *Vocab) EOT() Token {
	registerVocabFuncs()
	return Token(_lfg_vocab_eot(v.c))
}

// SEP returns the sentence separator token.
func (v *Vocab) SEP() Token {
	registerVocabFuncs()
	return Token(_lfg_vocab_sep(v.c))
}

// NL returns the newline token.
func (v *Vocab) NL() Token {
	registerVocabFuncs()
	return Token(_lfg_vocab_nl(v.c))
}

// PAD returns the padding token.
func (v *Vocab) PAD() Token {
	registerVocabFuncs()
	return Token(_lfg_vocab_pad(v.c))
}

// MASK returns the mask token.
func (v *Vocab) MASK() Token {
	registerVocabFuncs()
	return Token(_lfg_vocab_mask(v.c))
}

// AddBOS returns whether BOS should be added automatically.
func (v *Vocab) AddBOS() bool {
	registerVocabFuncs()
	return _lfg_vocab_get_add_bos(v.c)
}

// AddEOS returns whether EOS should be added automatically.
func (v *Vocab) AddEOS() bool {
	registerVocabFuncs()
	return _lfg_vocab_get_add_eos(v.c)
}

// AddSEP returns whether SEP should be added automatically.
func (v *Vocab) AddSEP() bool {
	registerVocabFuncs()
	return _lfg_vocab_get_add_sep(v.c)
}

// FIMPre returns the fill-in-middle prefix token.
func (v *Vocab) FIMPre() Token {
	registerVocabFuncs()
	return Token(_lfg_vocab_fim_pre(v.c))
}

// FIMSuf returns the fill-in-middle suffix token.
func (v *Vocab) FIMSuf() Token {
	registerVocabFuncs()
	return Token(_lfg_vocab_fim_suf(v.c))
}

// FIMMid returns the fill-in-middle middle token.
func (v *Vocab) FIMMid() Token {
	registerVocabFuncs()
	return Token(_lfg_vocab_fim_mid(v.c))
}

// FIMPad returns the fill-in-middle padding token.
func (v *Vocab) FIMPad() Token {
	registerVocabFuncs()
	return Token(_lfg_vocab_fim_pad(v.c))
}

// FIMRep returns the fill-in-middle repeat token.
func (v *Vocab) FIMRep() Token {
	registerVocabFuncs()
	return Token(_lfg_vocab_fim_rep(v.c))
}

// FIMSep returns the fill-in-middle separator token.
func (v *Vocab) FIMSep() Token {
	registerVocabFuncs()
	return Token(_lfg_vocab_fim_sep(v.c))
}

// Tokenize converts text into tokens using the two-pass pattern.
// If addSpecial is true, BOS/EOS tokens are added per model config.
// If parseSpecial is true, special tokens in the text are parsed.
func (v *Vocab) Tokenize(text string, addSpecial, parseSpecial bool) ([]Token, error) {
	v.model.mu.RLock()
	defer v.model.mu.RUnlock()
	if v.c == 0 {
		return nil, &Error{Code: ErrorInvalidArgument, Message: "vocab is nil"}
	}

	registerVocabFuncs()

	cText := cString(text)
	textPtr := cStringPtr(cText)
	textLen := int32(len(text))

	// First pass: determine required size.
	n := _lfg_tokenize(v.c, textPtr, textLen, 0, 0, addSpecial, parseSpecial)
	runtime.KeepAlive(cText)
	if n == 0 {
		return nil, nil
	}
	nTokens := n
	if nTokens < 0 {
		nTokens = -nTokens
	}

	tokens := make([]Token, nTokens)
	n = _lfg_tokenize(v.c, textPtr, textLen, tokenPtr(tokens), nTokens, addSpecial, parseSpecial)
	runtime.KeepAlive(cText)
	runtime.KeepAlive(tokens)
	if n < 0 {
		return nil, &Error{Code: ErrorInternal, Message: "tokenization failed"}
	}
	return tokens[:n], nil
}

// Detokenize converts tokens back into text.
// If removeSpecial is true, BOS/EOS tokens are removed.
// If unparseSpecial is true, special tokens are rendered.
func (v *Vocab) Detokenize(tokens []Token, removeSpecial, unparseSpecial bool) (string, error) {
	v.model.mu.RLock()
	defer v.model.mu.RUnlock()
	if v.c == 0 {
		return "", &Error{Code: ErrorInvalidArgument, Message: "vocab is nil"}
	}
	if len(tokens) == 0 {
		return "", nil
	}

	registerVocabFuncs()

	tokPtr := tokenPtr(tokens)
	nTok := int32(len(tokens))

	// First pass: determine required size.
	n := _lfg_detokenize(v.c, tokPtr, nTok, 0, 0, removeSpecial, unparseSpecial)
	runtime.KeepAlive(tokens)
	if n == 0 {
		return "", nil
	}
	nChars := n
	if nChars < 0 {
		nChars = -nChars
	}

	buf := make([]byte, nChars)
	n = _lfg_detokenize(v.c, tokPtr, nTok, uintptr(unsafe.Pointer(&buf[0])), nChars, removeSpecial, unparseSpecial)
	runtime.KeepAlive(tokens)
	runtime.KeepAlive(buf)
	if n < 0 {
		return "", &Error{Code: ErrorInternal, Message: "detokenization failed"}
	}
	return string(buf[:n]), nil
}

// TokenText converts a single token to its string representation.
// Uses a stack-allocated buffer for the common case to avoid heap allocation.
func (v *Vocab) TokenText(token Token, special bool) string {
	v.model.mu.RLock()
	defer v.model.mu.RUnlock()
	if v.c == 0 {
		return ""
	}

	registerVocabFuncs()

	// Try stack buffer first (handles most tokens).
	var buf [128]byte
	bufPtr := uintptr(unsafe.Pointer(&buf[0]))
	n := _lfg_token_to_piece(v.c, int32(token), bufPtr, 128, 0, special)
	if n >= 0 && n <= 128 {
		return goStringN(bufPtr, int(n))
	}

	// Rare case: token piece is larger than 128 bytes.
	if n < 0 {
		n = -n
	}
	bigBuf := make([]byte, n)
	bigPtr := uintptr(unsafe.Pointer(&bigBuf[0]))
	n = _lfg_token_to_piece(v.c, int32(token), bigPtr, n, 0, special)
	runtime.KeepAlive(bigBuf)
	if n < 0 {
		return ""
	}
	return string(bigBuf[:n])
}
