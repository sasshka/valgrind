#ifdef AVX_512
#ifndef __VEX_INTEL_AVX512ER_H_
#define __VEX_INTEL_AVX512ER_H_

#include "libvex_basictypes.h"
typedef union {
   UInt u;
   Int s;
   Float f;
} type32;
typedef union {
   ULong u;
   Long s;
   Double f;
} type64;

#define FP32_SIGN_BIT                     0x80000000
#define FP32_EXP_MASK                     0x7f800000
#define FP32_SIGNIF_MASK                  0x007fffff
#define FP32_QNAN_BIT                     0x00400000
#define FP32_DEFAULT_NAN_AS_UINT32        0xffc00000

#define FP64_SIGN_BIT                     0x8000000000000000ull
#define FP64_EXP_MASK                     0x7ff0000000000000ull
#define FP64_SIGNIF_MASK                  0x000fffffffffffffull
#define FP64_QNAN_BIT                     0x0008000000000000ull
#define FP64_DEFAULT_NAN_AS_UINT64        0xfff8000000000000ull

extern void RCP14S (UInt mxcsr, type32 *dst, type32 src);
extern void RCP14D (UInt mxcsr, type64 *dst, type64 src);
extern void RSQRT14S (UInt mxcsr, type32 *dst, type32 src);
extern void RSQRT14D (UInt mxcsr, type64 *dst, type64 src);
extern UInt RCP28S (type32 *dst, type32 src);
extern UInt RCP28D (type64 *dst, type64 src);
extern UInt RSQRT28S (type32 *dst, type32 src);
extern UInt RSQRT28D (type64 *dst, type64 src);
extern UInt EXP2S (type32 *dst, type32 src);
extern UInt EXP2D (type64 *dst, type64 src);

Long floor(Double);
Long floorf(Float);
Double fabs(Double);
Float  fabsf(Float);
Int isnan(Double);
Int isnanf(Float);
Int isinf(Double);
Int isinff(Float);
Int isnormal(Double);
Int isnormalf(Float);

/* order matches the specifier for vfpclasspd instruction */
enum FP_Type {
   FP_QNAN,
   FP_ZERO,
   FP_NEG_ZERO,
   FP_INFINITE,
   FP_NEG_INFINITE,
   FP_SUBNORMAL,
   FP_NEGATIVE,
   FP_SNAN,
   FP_NORMAL,
};
enum FP_Type fpclassify(ULong);
enum FP_Type fpclassifyf(UInt);

#endif  /* ndef __VEX_INTEL-AVX512ER_H_ */
#endif /* ndef AVX_512 */
