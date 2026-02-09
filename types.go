package lfg

// Token represents a token ID in the vocabulary.
type Token int32

// Position represents a position in a sequence.
type Position int32

// SequenceID represents a sequence identifier.
type SequenceID int32

// InvalidToken is the sentinel value for an invalid token.
const InvalidToken Token = -1

// RandomSeed is the default seed value that signals random seed selection.
const RandomSeed uint32 = 0xFFFFFFFF

// VocabType represents the type of vocabulary used by the model.
type VocabType int

const (
	VocabTypeNone   VocabType = 0
	VocabTypeSPM    VocabType = 1
	VocabTypeBPE    VocabType = 2
	VocabTypeWPM    VocabType = 3
	VocabTypeUGM    VocabType = 4
	VocabTypeRWKV   VocabType = 5
	VocabTypePLaMo2 VocabType = 6
)

// RopeType represents the type of RoPE (Rotary Position Embedding).
type RopeType int

const (
	RopeTypeNone RopeType = -1
	RopeTypeNorm RopeType = 0
)

// TokenType represents the type of a token.
type TokenType int

const (
	TokenTypeUndefined   TokenType = 0
	TokenTypeNormal      TokenType = 1
	TokenTypeUnknown     TokenType = 2
	TokenTypeControl     TokenType = 3
	TokenTypeUserDefined TokenType = 4
	TokenTypeUnused      TokenType = 5
	TokenTypeByte        TokenType = 6
)

// TokenAttr represents token attributes (bitfield).
type TokenAttr int

const (
	TokenAttrUndefined   TokenAttr = 0
	TokenAttrUnknown     TokenAttr = 1 << 0
	TokenAttrUnused      TokenAttr = 1 << 1
	TokenAttrNormal      TokenAttr = 1 << 2
	TokenAttrControl     TokenAttr = 1 << 3
	TokenAttrUserDefined TokenAttr = 1 << 4
	TokenAttrByte        TokenAttr = 1 << 5
	TokenAttrNormalized  TokenAttr = 1 << 6
	TokenAttrLstrip      TokenAttr = 1 << 7
	TokenAttrRstrip      TokenAttr = 1 << 8
	TokenAttrSingleWord  TokenAttr = 1 << 9
)

// FType represents model file quantization types.
type FType int

const (
	FTypeAllF32         FType = 0
	FTypeMostlyF16      FType = 1
	FTypeMostlyQ4_0     FType = 2
	FTypeMostlyQ4_1     FType = 3
	FTypeMostlyQ8_0     FType = 7
	FTypeMostlyQ5_0     FType = 8
	FTypeMostlyQ5_1     FType = 9
	FTypeMostlyQ2_K     FType = 10
	FTypeMostlyQ3_K_S   FType = 11
	FTypeMostlyQ3_K_M   FType = 12
	FTypeMostlyQ3_K_L   FType = 13
	FTypeMostlyQ4_K_S   FType = 14
	FTypeMostlyQ4_K_M   FType = 15
	FTypeMostlyQ5_K_S   FType = 16
	FTypeMostlyQ5_K_M   FType = 17
	FTypeMostlyQ6_K     FType = 18
	FTypeMostlyBF16     FType = 32
	FTypeMostlyTQ1_0    FType = 36
	FTypeMostlyTQ2_0    FType = 37
	FTypeGuessed        FType = 1024
)

// RopeScalingType represents the RoPE scaling type.
type RopeScalingType int

const (
	RopeScalingTypeUnspecified RopeScalingType = -1
	RopeScalingTypeNone       RopeScalingType = 0
	RopeScalingTypeLinear     RopeScalingType = 1
	RopeScalingTypeYarn       RopeScalingType = 2
	RopeScalingTypeLongRope   RopeScalingType = 3
)

// PoolingType represents the pooling type for embeddings.
type PoolingType int

const (
	PoolingTypeUnspecified PoolingType = -1
	PoolingTypeNone       PoolingType = 0
	PoolingTypeMean       PoolingType = 1
	PoolingTypeCLS        PoolingType = 2
	PoolingTypeLast       PoolingType = 3
	PoolingTypeRank       PoolingType = 4
)

// AttentionType represents the attention type.
type AttentionType int

const (
	AttentionTypeUnspecified AttentionType = -1
	AttentionTypeCausal     AttentionType = 0
	AttentionTypeNonCausal  AttentionType = 1
)

// FlashAttnType represents flash attention configuration.
type FlashAttnType int

const (
	FlashAttnTypeAuto     FlashAttnType = -1
	FlashAttnTypeDisabled FlashAttnType = 0
	FlashAttnTypeEnabled  FlashAttnType = 1
)

// SplitMode represents how the model is split across GPUs.
type SplitMode int

const (
	SplitModeNone  SplitMode = 0
	SplitModeLayer SplitMode = 1
	SplitModeRow   SplitMode = 2
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
