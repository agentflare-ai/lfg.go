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
import "unsafe"

// Vocab provides access to the model's vocabulary. It borrows its parent
// Model's lock — the Model must remain alive while Vocab is in use.
type Vocab struct {
	c     *C.struct_lfm_vocab
	model *Model // prevent GC of parent
}

// NTokens returns the total number of tokens in the vocabulary.
func (v *Vocab) NTokens() int {
	v.model.mu.RLock()
	defer v.model.mu.RUnlock()
	if v.c == nil {
		return 0
	}
	return int(C.lfm_vocab_n_tokens(v.c))
}

// Type returns the vocabulary type.
func (v *Vocab) Type() VocabType {
	v.model.mu.RLock()
	defer v.model.mu.RUnlock()
	if v.c == nil {
		return VocabTypeNone
	}
	return VocabType(C.lfm_vocab_type(v.c))
}

// GetText returns the raw text of a token.
func (v *Vocab) GetText(token Token) string {
	v.model.mu.RLock()
	defer v.model.mu.RUnlock()
	if v.c == nil {
		return ""
	}
	return C.GoString(C.lfm_vocab_get_text(v.c, C.lfm_token(token)))
}

// GetScore returns the score (log probability) of a token.
func (v *Vocab) GetScore(token Token) float32 {
	v.model.mu.RLock()
	defer v.model.mu.RUnlock()
	if v.c == nil {
		return 0
	}
	return float32(C.lfm_vocab_get_score(v.c, C.lfm_token(token)))
}

// GetAttr returns the attributes of a token.
func (v *Vocab) GetAttr(token Token) TokenAttr {
	v.model.mu.RLock()
	defer v.model.mu.RUnlock()
	if v.c == nil {
		return TokenAttrUndefined
	}
	return TokenAttr(C.lfm_vocab_get_attr(v.c, C.lfm_token(token)))
}

// IsEOG returns true if the token is an end-of-generation token.
func (v *Vocab) IsEOG(token Token) bool {
	v.model.mu.RLock()
	defer v.model.mu.RUnlock()
	if v.c == nil {
		return false
	}
	return bool(C.lfm_vocab_is_eog(v.c, C.lfm_token(token)))
}

// IsControl returns true if the token is a control token.
func (v *Vocab) IsControl(token Token) bool {
	v.model.mu.RLock()
	defer v.model.mu.RUnlock()
	if v.c == nil {
		return false
	}
	return bool(C.lfm_vocab_is_control(v.c, C.lfm_token(token)))
}

// BOS returns the beginning-of-sentence token.
func (v *Vocab) BOS() Token { return Token(C.lfm_vocab_bos(v.c)) }

// EOS returns the end-of-sentence token.
func (v *Vocab) EOS() Token { return Token(C.lfm_vocab_eos(v.c)) }

// EOT returns the end-of-turn token.
func (v *Vocab) EOT() Token { return Token(C.lfm_vocab_eot(v.c)) }

// SEP returns the sentence separator token.
func (v *Vocab) SEP() Token { return Token(C.lfm_vocab_sep(v.c)) }

// NL returns the newline token.
func (v *Vocab) NL() Token { return Token(C.lfm_vocab_nl(v.c)) }

// PAD returns the padding token.
func (v *Vocab) PAD() Token { return Token(C.lfm_vocab_pad(v.c)) }

// MASK returns the mask token.
func (v *Vocab) MASK() Token { return Token(C.lfm_vocab_mask(v.c)) }

// GetAddBOS returns whether BOS should be added automatically.
func (v *Vocab) GetAddBOS() bool { return bool(C.lfm_vocab_get_add_bos(v.c)) }

// GetAddEOS returns whether EOS should be added automatically.
func (v *Vocab) GetAddEOS() bool { return bool(C.lfm_vocab_get_add_eos(v.c)) }

// GetAddSEP returns whether SEP should be added automatically.
func (v *Vocab) GetAddSEP() bool { return bool(C.lfm_vocab_get_add_sep(v.c)) }

// FIMPre returns the fill-in-middle prefix token.
func (v *Vocab) FIMPre() Token { return Token(C.lfm_vocab_fim_pre(v.c)) }

// FIMSuf returns the fill-in-middle suffix token.
func (v *Vocab) FIMSuf() Token { return Token(C.lfm_vocab_fim_suf(v.c)) }

// FIMMid returns the fill-in-middle middle token.
func (v *Vocab) FIMMid() Token { return Token(C.lfm_vocab_fim_mid(v.c)) }

// FIMPad returns the fill-in-middle padding token.
func (v *Vocab) FIMPad() Token { return Token(C.lfm_vocab_fim_pad(v.c)) }

// FIMRep returns the fill-in-middle repeat token.
func (v *Vocab) FIMRep() Token { return Token(C.lfm_vocab_fim_rep(v.c)) }

// FIMSep returns the fill-in-middle separator token.
func (v *Vocab) FIMSep() Token { return Token(C.lfm_vocab_fim_sep(v.c)) }

// Tokenize converts text into tokens using the two-pass pattern.
// If addSpecial is true, BOS/EOS tokens are added per model config.
// If parseSpecial is true, special tokens in the text are parsed.
func (v *Vocab) Tokenize(text string, addSpecial, parseSpecial bool) ([]Token, error) {
	v.model.mu.RLock()
	defer v.model.mu.RUnlock()
	if v.c == nil {
		return nil, &Error{Code: ErrorInvalidArgument, Message: "vocab is nil"}
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))
	textLen := C.int32_t(len(text))

	// First pass: determine required size.
	n := C.lfm_tokenize(v.c, cText, textLen, nil, 0, C.bool(addSpecial), C.bool(parseSpecial))
	if n == 0 {
		return nil, nil
	}
	nTokens := n
	if nTokens < 0 {
		nTokens = -nTokens
	}

	tokens := make([]Token, nTokens)
	n = C.lfm_tokenize(v.c, cText, textLen, (*C.lfm_token)(&tokens[0]), C.int32_t(nTokens), C.bool(addSpecial), C.bool(parseSpecial))
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
	if v.c == nil {
		return "", &Error{Code: ErrorInvalidArgument, Message: "vocab is nil"}
	}
	if len(tokens) == 0 {
		return "", nil
	}

	// First pass: determine required size.
	n := C.lfm_detokenize(v.c, (*C.lfm_token)(&tokens[0]), C.int32_t(len(tokens)), nil, 0, C.bool(removeSpecial), C.bool(unparseSpecial))
	if n == 0 {
		return "", nil
	}
	nChars := n
	if nChars < 0 {
		nChars = -nChars
	}

	buf := make([]byte, nChars)
	n = C.lfm_detokenize(v.c, (*C.lfm_token)(&tokens[0]), C.int32_t(len(tokens)), (*C.char)(unsafe.Pointer(&buf[0])), C.int32_t(nChars), C.bool(removeSpecial), C.bool(unparseSpecial))
	if n < 0 {
		return "", &Error{Code: ErrorInternal, Message: "detokenization failed"}
	}
	return string(buf[:n]), nil
}

// TokenToPiece converts a single token to its string representation.
// Uses a stack-allocated buffer for the common case to avoid heap allocation.
func (v *Vocab) TokenToPiece(token Token, special bool) string {
	v.model.mu.RLock()
	defer v.model.mu.RUnlock()
	if v.c == nil {
		return ""
	}

	// Try stack buffer first (handles most tokens).
	var buf [128]C.char
	n := C.lfm_token_to_piece(v.c, C.lfm_token(token), &buf[0], 128, 0, C.bool(special))
	if n >= 0 && n <= 128 {
		return C.GoStringN(&buf[0], n)
	}

	// Rare case: token piece is larger than 128 bytes.
	if n < 0 {
		n = -n
	}
	bigBuf := make([]byte, n)
	n = C.lfm_token_to_piece(v.c, C.lfm_token(token), (*C.char)(unsafe.Pointer(&bigBuf[0])), n, 0, C.bool(special))
	if n < 0 {
		return ""
	}
	return string(bigBuf[:n])
}
