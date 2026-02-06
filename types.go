package lfg

/*
typedef struct lfm_model lfm_model;
typedef struct lfm_context lfm_context;
typedef struct lfm_vocab lfm_vocab;
typedef struct lfm_sampler lfm_sampler;
#include "lfm_inference.h"
*/
import "C"

// Token represents a token ID in the vocabulary.
type Token = C.lfm_token

// Pos represents a position in a sequence.
type Pos = C.lfm_pos

// SeqID represents a sequence identifier.
type SeqID = C.lfm_seq_id

// TokenNull is the sentinel value for an invalid token.
const TokenNull Token = C.LFM_TOKEN_NULL

// DefaultSeed is the default seed value that signals random seed selection.
const DefaultSeed = C.LFM_DEFAULT_SEED

// VocabType represents the type of vocabulary used by the model.
type VocabType int

const (
	VocabTypeNone   VocabType = C.LFM_VOCAB_TYPE_NONE
	VocabTypeSPM    VocabType = C.LFM_VOCAB_TYPE_SPM
	VocabTypeBPE    VocabType = C.LFM_VOCAB_TYPE_BPE
	VocabTypeWPM    VocabType = C.LFM_VOCAB_TYPE_WPM
	VocabTypeUGM    VocabType = C.LFM_VOCAB_TYPE_UGM
	VocabTypeRWKV   VocabType = C.LFM_VOCAB_TYPE_RWKV
	VocabTypePLaMo2 VocabType = C.LFM_VOCAB_TYPE_PLAMO2
)

// RopeType represents the type of RoPE (Rotary Position Embedding).
type RopeType int

const (
	RopeTypeNone RopeType = C.LFM_ROPE_TYPE_NONE
	RopeTypeNorm RopeType = C.LFM_ROPE_TYPE_NORM
)

// TokenType represents the type of a token.
type TokenType int

const (
	TokenTypeUndefined   TokenType = C.LFM_TOKEN_TYPE_UNDEFINED
	TokenTypeNormal      TokenType = C.LFM_TOKEN_TYPE_NORMAL
	TokenTypeUnknown     TokenType = C.LFM_TOKEN_TYPE_UNKNOWN
	TokenTypeControl     TokenType = C.LFM_TOKEN_TYPE_CONTROL
	TokenTypeUserDefined TokenType = C.LFM_TOKEN_TYPE_USER_DEFINED
	TokenTypeUnused      TokenType = C.LFM_TOKEN_TYPE_UNUSED
	TokenTypeByte        TokenType = C.LFM_TOKEN_TYPE_BYTE
)

// TokenAttr represents token attributes (bitfield).
type TokenAttr int

const (
	TokenAttrUndefined   TokenAttr = C.LFM_TOKEN_ATTR_UNDEFINED
	TokenAttrUnknown     TokenAttr = C.LFM_TOKEN_ATTR_UNKNOWN
	TokenAttrUnused      TokenAttr = C.LFM_TOKEN_ATTR_UNUSED
	TokenAttrNormal      TokenAttr = C.LFM_TOKEN_ATTR_NORMAL
	TokenAttrControl     TokenAttr = C.LFM_TOKEN_ATTR_CONTROL
	TokenAttrUserDefined TokenAttr = C.LFM_TOKEN_ATTR_USER_DEFINED
	TokenAttrByte        TokenAttr = C.LFM_TOKEN_ATTR_BYTE
	TokenAttrNormalized  TokenAttr = C.LFM_TOKEN_ATTR_NORMALIZED
	TokenAttrLstrip      TokenAttr = C.LFM_TOKEN_ATTR_LSTRIP
	TokenAttrRstrip      TokenAttr = C.LFM_TOKEN_ATTR_RSTRIP
	TokenAttrSingleWord  TokenAttr = C.LFM_TOKEN_ATTR_SINGLE_WORD
)

// FType represents model file quantization types.
type FType int

const (
	FTypeAllF32         FType = C.LFM_FTYPE_ALL_F32
	FTypeMostlyF16      FType = C.LFM_FTYPE_MOSTLY_F16
	FTypeMostlyQ4_0     FType = C.LFM_FTYPE_MOSTLY_Q4_0
	FTypeMostlyQ4_1     FType = C.LFM_FTYPE_MOSTLY_Q4_1
	FTypeMostlyQ8_0     FType = C.LFM_FTYPE_MOSTLY_Q8_0
	FTypeMostlyQ5_0     FType = C.LFM_FTYPE_MOSTLY_Q5_0
	FTypeMostlyQ5_1     FType = C.LFM_FTYPE_MOSTLY_Q5_1
	FTypeMostlyQ2_K     FType = C.LFM_FTYPE_MOSTLY_Q2_K
	FTypeMostlyQ3_K_S   FType = C.LFM_FTYPE_MOSTLY_Q3_K_S
	FTypeMostlyQ3_K_M   FType = C.LFM_FTYPE_MOSTLY_Q3_K_M
	FTypeMostlyQ3_K_L   FType = C.LFM_FTYPE_MOSTLY_Q3_K_L
	FTypeMostlyQ4_K_S   FType = C.LFM_FTYPE_MOSTLY_Q4_K_S
	FTypeMostlyQ4_K_M   FType = C.LFM_FTYPE_MOSTLY_Q4_K_M
	FTypeMostlyQ5_K_S   FType = C.LFM_FTYPE_MOSTLY_Q5_K_S
	FTypeMostlyQ5_K_M   FType = C.LFM_FTYPE_MOSTLY_Q5_K_M
	FTypeMostlyQ6_K     FType = C.LFM_FTYPE_MOSTLY_Q6_K
	FTypeMostlyBF16     FType = C.LFM_FTYPE_MOSTLY_BF16
	FTypeMostlyTQ1_0    FType = C.LFM_FTYPE_MOSTLY_TQ1_0
	FTypeMostlyTQ2_0    FType = C.LFM_FTYPE_MOSTLY_TQ2_0
	FTypeGuessed        FType = C.LFM_FTYPE_GUESSED
)

// RopeScalingType represents the RoPE scaling type.
type RopeScalingType int

const (
	RopeScalingTypeUnspecified RopeScalingType = C.LFM_ROPE_SCALING_TYPE_UNSPECIFIED
	RopeScalingTypeNone       RopeScalingType = C.LFM_ROPE_SCALING_TYPE_NONE
	RopeScalingTypeLinear     RopeScalingType = C.LFM_ROPE_SCALING_TYPE_LINEAR
	RopeScalingTypeYarn       RopeScalingType = C.LFM_ROPE_SCALING_TYPE_YARN
	RopeScalingTypeLongRope   RopeScalingType = C.LFM_ROPE_SCALING_TYPE_LONGROPE
)

// PoolingType represents the pooling type for embeddings.
type PoolingType int

const (
	PoolingTypeUnspecified PoolingType = C.LFM_POOLING_TYPE_UNSPECIFIED
	PoolingTypeNone       PoolingType = C.LFM_POOLING_TYPE_NONE
	PoolingTypeMean       PoolingType = C.LFM_POOLING_TYPE_MEAN
	PoolingTypeCLS        PoolingType = C.LFM_POOLING_TYPE_CLS
	PoolingTypeLast       PoolingType = C.LFM_POOLING_TYPE_LAST
	PoolingTypeRank       PoolingType = C.LFM_POOLING_TYPE_RANK
)

// AttentionType represents the attention type.
type AttentionType int

const (
	AttentionTypeUnspecified AttentionType = C.LFM_ATTENTION_TYPE_UNSPECIFIED
	AttentionTypeCausal     AttentionType = C.LFM_ATTENTION_TYPE_CAUSAL
	AttentionTypeNonCausal  AttentionType = C.LFM_ATTENTION_TYPE_NON_CAUSAL
)

// FlashAttnType represents flash attention configuration.
type FlashAttnType int

const (
	FlashAttnTypeAuto     FlashAttnType = C.LFM_FLASH_ATTN_TYPE_AUTO
	FlashAttnTypeDisabled FlashAttnType = C.LFM_FLASH_ATTN_TYPE_DISABLED
	FlashAttnTypeEnabled  FlashAttnType = C.LFM_FLASH_ATTN_TYPE_ENABLED
)

// SplitMode represents how the model is split across GPUs.
type SplitMode int

const (
	SplitModeNone  SplitMode = C.LFM_SPLIT_MODE_NONE
	SplitModeLayer SplitMode = C.LFM_SPLIT_MODE_LAYER
	SplitModeRow   SplitMode = C.LFM_SPLIT_MODE_ROW
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
