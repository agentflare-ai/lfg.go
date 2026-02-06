package lfg

/*
#include "lfg_inference.h"
*/
import "C"

// Token represents a token ID in the vocabulary.
type Token int32

// Position represents a position in a sequence.
type Position int32

// SequenceID represents a sequence identifier.
type SequenceID int32

// InvalidToken is the sentinel value for an invalid token.
const InvalidToken Token = C.LFG_TOKEN_NULL

// RandomSeed is the default seed value that signals random seed selection.
const RandomSeed uint32 = C.LFG_DEFAULT_SEED

// VocabType represents the type of vocabulary used by the model.
type VocabType int

const (
	VocabTypeNone   VocabType = C.LFG_VOCAB_TYPE_NONE
	VocabTypeSPM    VocabType = C.LFG_VOCAB_TYPE_SPM
	VocabTypeBPE    VocabType = C.LFG_VOCAB_TYPE_BPE
	VocabTypeWPM    VocabType = C.LFG_VOCAB_TYPE_WPM
	VocabTypeUGM    VocabType = C.LFG_VOCAB_TYPE_UGM
	VocabTypeRWKV   VocabType = C.LFG_VOCAB_TYPE_RWKV
	VocabTypePLaMo2 VocabType = C.LFG_VOCAB_TYPE_PLAMO2
)

// RopeType represents the type of RoPE (Rotary Position Embedding).
type RopeType int

const (
	RopeTypeNone RopeType = C.LFG_ROPE_TYPE_NONE
	RopeTypeNorm RopeType = C.LFG_ROPE_TYPE_NORM
)

// TokenType represents the type of a token.
type TokenType int

const (
	TokenTypeUndefined   TokenType = C.LFG_TOKEN_TYPE_UNDEFINED
	TokenTypeNormal      TokenType = C.LFG_TOKEN_TYPE_NORMAL
	TokenTypeUnknown     TokenType = C.LFG_TOKEN_TYPE_UNKNOWN
	TokenTypeControl     TokenType = C.LFG_TOKEN_TYPE_CONTROL
	TokenTypeUserDefined TokenType = C.LFG_TOKEN_TYPE_USER_DEFINED
	TokenTypeUnused      TokenType = C.LFG_TOKEN_TYPE_UNUSED
	TokenTypeByte        TokenType = C.LFG_TOKEN_TYPE_BYTE
)

// TokenAttr represents token attributes (bitfield).
type TokenAttr int

const (
	TokenAttrUndefined   TokenAttr = C.LFG_TOKEN_ATTR_UNDEFINED
	TokenAttrUnknown     TokenAttr = C.LFG_TOKEN_ATTR_UNKNOWN
	TokenAttrUnused      TokenAttr = C.LFG_TOKEN_ATTR_UNUSED
	TokenAttrNormal      TokenAttr = C.LFG_TOKEN_ATTR_NORMAL
	TokenAttrControl     TokenAttr = C.LFG_TOKEN_ATTR_CONTROL
	TokenAttrUserDefined TokenAttr = C.LFG_TOKEN_ATTR_USER_DEFINED
	TokenAttrByte        TokenAttr = C.LFG_TOKEN_ATTR_BYTE
	TokenAttrNormalized  TokenAttr = C.LFG_TOKEN_ATTR_NORMALIZED
	TokenAttrLstrip      TokenAttr = C.LFG_TOKEN_ATTR_LSTRIP
	TokenAttrRstrip      TokenAttr = C.LFG_TOKEN_ATTR_RSTRIP
	TokenAttrSingleWord  TokenAttr = C.LFG_TOKEN_ATTR_SINGLE_WORD
)

// FType represents model file quantization types.
type FType int

const (
	FTypeAllF32         FType = C.LFG_FTYPE_ALL_F32
	FTypeMostlyF16      FType = C.LFG_FTYPE_MOSTLY_F16
	FTypeMostlyQ4_0     FType = C.LFG_FTYPE_MOSTLY_Q4_0
	FTypeMostlyQ4_1     FType = C.LFG_FTYPE_MOSTLY_Q4_1
	FTypeMostlyQ8_0     FType = C.LFG_FTYPE_MOSTLY_Q8_0
	FTypeMostlyQ5_0     FType = C.LFG_FTYPE_MOSTLY_Q5_0
	FTypeMostlyQ5_1     FType = C.LFG_FTYPE_MOSTLY_Q5_1
	FTypeMostlyQ2_K     FType = C.LFG_FTYPE_MOSTLY_Q2_K
	FTypeMostlyQ3_K_S   FType = C.LFG_FTYPE_MOSTLY_Q3_K_S
	FTypeMostlyQ3_K_M   FType = C.LFG_FTYPE_MOSTLY_Q3_K_M
	FTypeMostlyQ3_K_L   FType = C.LFG_FTYPE_MOSTLY_Q3_K_L
	FTypeMostlyQ4_K_S   FType = C.LFG_FTYPE_MOSTLY_Q4_K_S
	FTypeMostlyQ4_K_M   FType = C.LFG_FTYPE_MOSTLY_Q4_K_M
	FTypeMostlyQ5_K_S   FType = C.LFG_FTYPE_MOSTLY_Q5_K_S
	FTypeMostlyQ5_K_M   FType = C.LFG_FTYPE_MOSTLY_Q5_K_M
	FTypeMostlyQ6_K     FType = C.LFG_FTYPE_MOSTLY_Q6_K
	FTypeMostlyBF16     FType = C.LFG_FTYPE_MOSTLY_BF16
	FTypeMostlyTQ1_0    FType = C.LFG_FTYPE_MOSTLY_TQ1_0
	FTypeMostlyTQ2_0    FType = C.LFG_FTYPE_MOSTLY_TQ2_0
	FTypeGuessed        FType = C.LFG_FTYPE_GUESSED
)

// RopeScalingType represents the RoPE scaling type.
type RopeScalingType int

const (
	RopeScalingTypeUnspecified RopeScalingType = C.LFG_ROPE_SCALING_TYPE_UNSPECIFIED
	RopeScalingTypeNone       RopeScalingType = C.LFG_ROPE_SCALING_TYPE_NONE
	RopeScalingTypeLinear     RopeScalingType = C.LFG_ROPE_SCALING_TYPE_LINEAR
	RopeScalingTypeYarn       RopeScalingType = C.LFG_ROPE_SCALING_TYPE_YARN
	RopeScalingTypeLongRope   RopeScalingType = C.LFG_ROPE_SCALING_TYPE_LONGROPE
)

// PoolingType represents the pooling type for embeddings.
type PoolingType int

const (
	PoolingTypeUnspecified PoolingType = C.LFG_POOLING_TYPE_UNSPECIFIED
	PoolingTypeNone       PoolingType = C.LFG_POOLING_TYPE_NONE
	PoolingTypeMean       PoolingType = C.LFG_POOLING_TYPE_MEAN
	PoolingTypeCLS        PoolingType = C.LFG_POOLING_TYPE_CLS
	PoolingTypeLast       PoolingType = C.LFG_POOLING_TYPE_LAST
	PoolingTypeRank       PoolingType = C.LFG_POOLING_TYPE_RANK
)

// AttentionType represents the attention type.
type AttentionType int

const (
	AttentionTypeUnspecified AttentionType = C.LFG_ATTENTION_TYPE_UNSPECIFIED
	AttentionTypeCausal     AttentionType = C.LFG_ATTENTION_TYPE_CAUSAL
	AttentionTypeNonCausal  AttentionType = C.LFG_ATTENTION_TYPE_NON_CAUSAL
)

// FlashAttnType represents flash attention configuration.
type FlashAttnType int

const (
	FlashAttnTypeAuto     FlashAttnType = C.LFG_FLASH_ATTN_TYPE_AUTO
	FlashAttnTypeDisabled FlashAttnType = C.LFG_FLASH_ATTN_TYPE_DISABLED
	FlashAttnTypeEnabled  FlashAttnType = C.LFG_FLASH_ATTN_TYPE_ENABLED
)

// SplitMode represents how the model is split across GPUs.
type SplitMode int

const (
	SplitModeNone  SplitMode = C.LFG_SPLIT_MODE_NONE
	SplitModeLayer SplitMode = C.LFG_SPLIT_MODE_LAYER
	SplitModeRow   SplitMode = C.LFG_SPLIT_MODE_ROW
)

// TokenData holds token ID, logit, and probability.
type TokenData struct {
	ID    Token
	Logit float32
	P     float32
}

// LogitBias represents a bias applied to a specific token's logit.
type LogitBias struct {
	Token Token
	Bias  float32
}
