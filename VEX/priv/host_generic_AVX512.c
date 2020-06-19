/*---------------------------------------------------------------*/
/*--- begin                             host_generic_AVX512.c ---*/
/*---------------------------------------------------------------*/

/*
   This file is part of Valgrind, a dynamic binary instrumentation
   framework.

   Copyright (C) 2012-2017 OpenWorks GbR
      info@open-works.net

   This program is free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public License as
   published by the Free Software Foundation; either version 2 of the
   License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
   02110-1301, USA.

   The GNU General Public License is contained in the file COPYING.
*/

/* Generic helper functions for doing 512-bit SIMD arithmetic in cases
   where the instruction selectors cannot generate code in-line.
   These are purely back-end entities and cannot be seen/referenced
   from IR. */

#ifdef AVX_512

#include "libvex_basictypes.h"
#include "host_generic_AVX512.h"
#include "host_generic_simd64.h"
#include <host_generic_AVX512_ER.h>
#include <float.h>
#include <limits.h>

/* Work with the layout of a floating point number. */
typedef union {
   Short half_val;
   struct {
      UInt mantissa: 10;
      UInt exponent: 5;
      UInt sign: 1;
   } parts; /* IEEE754 half-precision (16-bit) value */
} half_cast;

typedef union {
   Float float_val;
   UInt  uint_val;
   struct {
      UInt mantissa: 23;
      UInt exponent: 8;
      UInt sign: 1;
   } parts;
} float_cast;

typedef union {
   Double double_val;
   ULong  ulong_val;
   struct {
      Long mantissa: 52;
      UInt exponent: 11;
      UInt sign: 1;
   } parts;
} double_cast;

VEX_REGPARM(3) void h_Iop_Max64Sx2 ( V128* res, V128* argL, V128* argR ) {
   res->w64[0] = ((Long)argL->w64[0] > (Long)argR->w64[0]) ? argL->w64[0] : argR->w64[0];
   res->w64[1] = ((Long)argL->w64[1] > (Long)argR->w64[1]) ? argL->w64[1] : argR->w64[1];
}

VEX_REGPARM(3) void h_Iop_Min64Sx2 ( V128* res, V128* argL, V128* argR ) {
   res->w64[0] = ((Long)argL->w64[0] < (Long)argR->w64[0]) ? argL->w64[0] : argR->w64[0];
   res->w64[1] = ((Long)argL->w64[1] < (Long)argR->w64[1]) ? argL->w64[1] : argR->w64[1];
}

VEX_REGPARM(3) void h_Iop_Max64Ux2 ( V128* res, V128* argL, V128* argR ) {
   res->w64[0] = (argL->w64[0] > argR->w64[0]) ? argL->w64[0] : argR->w64[0];
   res->w64[1] = (argL->w64[1] > argR->w64[1]) ? argL->w64[1] : argR->w64[1];
}

VEX_REGPARM(3) void h_Iop_Min64Ux2 ( V128* res, V128* argL, V128* argR ) {
   res->w64[0] = (argL->w64[0] < argR->w64[0]) ? argL->w64[0] : argR->w64[0];
   res->w64[1] = (argL->w64[1] < argR->w64[1]) ? argL->w64[1] : argR->w64[1];
}

VEX_REGPARM(4) void h_Iop_Test128 ( ULong* res, V128* argL, V128* argR, UInt count) {
   ULong result = 0;
   UInt width = 128/count;
   for (Int i = 0; i < count; i++)
      switch (width) {
         case 8:  if (argL->w8[i]  & argR->w8[i])  result |= 1ULL << i; break;
         case 16: if (argL->w16[i] & argR->w16[i]) result |= 1ULL << i; break;
         case 32: if (argL->w32[i] & argR->w32[i]) result |= 1ULL << i; break;
         case 64: if (argL->w64[i] & argR->w64[i]) result |= 1ULL << i; break;
      }
   *res = result;
}
VEX_REGPARM(4) void h_Iop_Test256 ( ULong* res, V256* argL, V256* argR, UInt count) {
   ULong result = 0;
   UInt width = 256/count;
   for (Int i = 0; i < count; i++)
      switch (width) {
         case 8:  if (argL->w8[i]  & argR->w8[i])  result |= 1ULL << i; break;
         case 16: if (argL->w16[i] & argR->w16[i]) result |= 1ULL << i; break;
         case 32: if (argL->w32[i] & argR->w32[i]) result |= 1ULL << i; break;
         case 64: if (argL->w64[i] & argR->w64[i]) result |= 1ULL << i; break;
      }
   *res = result;
}
VEX_REGPARM(4) void h_Iop_Test512 ( ULong* res, V512* argL, V512* argR, UInt count) {
   ULong result = 0;
   UInt width = 512/count;
   for (Int i = 0; i < count; i++)
      switch (width) {
         case 8:  if (argL->w8[i]  & argR->w8[i])  result |= 1ULL << i; break;
         case 16: if (argL->w16[i] & argR->w16[i]) result |= 1ULL << i; break;
         case 32: if (argL->w32[i] & argR->w32[i]) result |= 1ULL << i; break;
         case 64: if (argL->w64[i] & argR->w64[i]) result |= 1ULL << i; break;
      }
   *res = result;
}
VEX_REGPARM(4) void h_Iop_TestN128 ( ULong* res, V128* argL, V128* argR, UInt count) {
   ULong result = 0;
   UInt width = 128/count;
   for (Int i = 0; i < count; i++)
      switch (width) {
         case 8:  if (!(argL->w8[i]  & argR->w8[i]))  result |= 1ULL << i; break;
         case 16: if (!(argL->w16[i] & argR->w16[i])) result |= 1ULL << i; break;
         case 32: if (!(argL->w32[i] & argR->w32[i])) result |= 1ULL << i; break;
         case 64: if (!(argL->w64[i] & argR->w64[i])) result |= 1ULL << i; break;
      }
   *res = result;
}
VEX_REGPARM(4) void h_Iop_TestN256 ( ULong* res, V256* argL, V256* argR, UInt count) {
   ULong result = 0;
   UInt width = 256/count;
   for (Int i = 0; i < count; i++)
      switch (width) {
         case 8:  if (!(argL->w8[i]  & argR->w8[i]))  result |= 1ULL << i; break;
         case 16: if (!(argL->w16[i] & argR->w16[i])) result |= 1ULL << i; break;
         case 32: if (!(argL->w32[i] & argR->w32[i])) result |= 1ULL << i; break;
         case 64: if (!(argL->w64[i] & argR->w64[i])) result |= 1ULL << i; break;
      }
   *res = result;
}
VEX_REGPARM(4) void h_Iop_TestN512 ( ULong* res, V512* argL, V512* argR, UInt count) {
   ULong result = 0;
   UInt width = 512/count;
   for (Int i = 0; i < count; i++)
      switch (width) {
         case 8:  if (!(argL->w8[i]  & argR->w8[i]))  result |= 1ULL << i; break;
         case 16: if (!(argL->w16[i] & argR->w16[i])) result |= 1ULL << i; break;
         case 32: if (!(argL->w32[i] & argR->w32[i])) result |= 1ULL << i; break;
         case 64: if (!(argL->w64[i] & argR->w64[i])) result |= 1ULL << i; break;
      }
   *res = result;
}

#define GET_BIT(x, i) ((x>>i) & 1)
VEX_REGPARM(4) void h_Iop_Ternlog32 (UInt* src_dst, UInt src2, UInt src3, UInt imm8) {
   Int res = 0;
   for (Int j = 0; j < 32; j++) {
      Int temp = 4 * GET_BIT(*src_dst, j) +
                 2 * GET_BIT(src2, j) +
                     GET_BIT(src3, j);
      res += GET_BIT(imm8, temp) << j;
   }
   *src_dst = res;
}
VEX_REGPARM(4) void h_Iop_Ternlog64 (ULong* src_dst, ULong src2, ULong src3, UInt imm8) {
   Long res = 0;
   for (Int j = 0; j < 64; j++) {
      Long temp = 4 * GET_BIT(*src_dst, j) +
                 2 * GET_BIT(src2, j) +
                     GET_BIT(src3, j);
      res += (Long)(GET_BIT(imm8, temp)) << j;
   }
   *src_dst = res;
}

static Long round_with_rmode64(Double in, int rmode) {
   switch (rmode) {
      case 0x0: { // Round to nearest even
         Long rounded = (Long)in;
         if (in > 0) {
            Double part = in - rounded;
            if (part < 0.5) return rounded;
            if (part > 0.5) return rounded+1;
            // if part == 0.5, return even
            return rounded + (rounded % 2);
         } else {
            Double part = rounded - in;
            if (part < 0.5) return rounded;
            if (part > 0.5) return rounded-1;
            return rounded - (rounded % 2);
         }
      }
      case 0x1: // Round down
         return floor(in);
      case 0x2: // Round up
         return -floor(-in);
      case 0x3: // Truncate
         return (Long)in;
   }
   return -1; // Invalid input
}

static Int round_with_rmode32(Float in, Int rmode) {
   switch (rmode) {
      case 0x0: { // Round to nearest even
         Int rounded = (Int)in;
         if (in > 0) {
            Float part = in - rounded;
            if (part < 0.5) return rounded;
            if (part > 0.5) return rounded+1;
            return rounded + (rounded % 2);
         } else {
            Float part = rounded - in;
            if (part < 0.5) return rounded;
            if (part > 0.5) return rounded-1;
            return rounded - (rounded % 2);
         }
      }
      case 0x1: // Round down
         return floorf(in);
      case 0x2: // Round up
         return -floorf(-in);
      case 0x3: // Truncate
         return (Int)in;
   }
   return -1; // Invalid input
}


static UInt QNAN_32(UInt x) {
   return x|FP32_QNAN_BIT;
}
static ULong QNAN_64(ULong x) {
   return x|FP64_QNAN_BIT;
}
type64 INFINITY_64 = {.u = FP64_EXP_MASK};
type32 INFINITY_32 = {.u = FP32_EXP_MASK};
type64 QNAN_INDEF_64 = {.u = FP64_DEFAULT_NAN_AS_UINT64 };
type32 QNAN_INDEF_32 = {.u = FP32_DEFAULT_NAN_AS_UINT32 };

VEX_REGPARM(2) void h_Iop_ExtractExp32(UInt* dst_u, UInt src_u) {
   float_cast f = {.uint_val = src_u};
   switch (fpclassifyf(src_u)) {
      case FP_QNAN:
      case FP_SNAN:
         *dst_u = QNAN_32(src_u);
         return;
      case FP_INFINITE:
      case FP_NEG_INFINITE:
         *dst_u = INFINITY_32.u;
         return;
      case FP_ZERO:
         *dst_u = -INFINITY_32.u;
         return;
      case FP_SUBNORMAL: { // Nomalize FP input
         UShort jbit = 0x0;
         f.parts.exponent = 0x1;
         while (!(jbit & 0x1)) {
            jbit = f.parts.mantissa >> 22; // MSB of mantissa
            f.parts.mantissa <<= 1;
            f.parts.exponent--;
         }
         f.parts.sign = 0x1;
         break;
      }
      case FP_NORMAL:
         f.parts.sign = 0x0;
         break;
   }
   f.parts.mantissa = 0x0;
   Int tmp = (f.parts.sign << 31) | (f.parts.exponent << 23) | f.parts.mantissa;
   Float out = (Float)((tmp >>= 23) - 127);
   type32 dst = {.f = out};
   *dst_u = dst.u;
}

VEX_REGPARM(2) void h_Iop_ExtractExp64(ULong* dst_u, ULong src_u) {
   double_cast d = {.ulong_val = src_u};
   switch (fpclassify(src_u)) {
      case FP_QNAN:
      case FP_SNAN:
         *dst_u = QNAN_64(src_u);
         return;
      case FP_INFINITE:
      case FP_NEG_INFINITE:
         *dst_u = INFINITY_64.u;
         return;
      case FP_ZERO:
         *dst_u = -INFINITY_64.u;
         return;
      case FP_SUBNORMAL: { // Nomalize
         UShort jbit = 0x0;
         d.parts.exponent = 0x1;
         while (!(jbit & 0x1)) {
            jbit = d.parts.mantissa >> 51;
            d.parts.mantissa <<= 1;
            d.parts.exponent--;
         }
         d.parts.sign = 0x1;
         break;
      }
      case FP_NORMAL:
         d.parts.sign = 0x0;
         break;
   }
   d.parts.mantissa = 0x0;
   Long tmp = ((Long)d.parts.sign << 63) | ((Long)d.parts.exponent << 52) | d.parts.mantissa;
   Double out = (Double)((tmp >>= 52) - 1023);
   type64 dst = {.f = out};
   *dst_u = dst.u;
}

VEX_REGPARM(3) void h_Iop_ExtractMant32(UInt* dst_u, UInt src_u, UInt imm8)
{
   /* Sign Control (SC) = imm8[3:2] = 00b : sign(SRC)
                        = imm8[3:2] = 01b : 0
                        = imm8[3]   = 1b  : qNan_Indefinite if sign(SRC) != 0
      Normalization Interval = imm8[1:0]
                             = 00b : Interval [1,   2)
                             = 01b : Interval [1/2, 2)
                             = 10b : Interval [1/2, 1)
                             = 11b : Interval [3/4, 3/2)
   */
   Int sc = (imm8 >> 2) & 0x3;
   Int interv = imm8 & 0x3;

   float_cast f = {.uint_val = src_u};
   f.parts.sign = (sc & 0x1) ? 0x0 : f.parts.sign;
   if (f.parts.sign && (sc > 1)) {
      *dst_u = QNAN_INDEF_32.u;
      return;
   }
   switch (fpclassifyf(src_u)) {
      case FP_QNAN:
      case FP_SNAN:
         *dst_u = QNAN_32(src_u);
         return;
      case FP_INFINITE:
      case FP_NEG_INFINITE:
         *dst_u = INFINITY_32.u;
         return;
      case FP_ZERO:
         *dst_u = -INFINITY_32.u;
         return;
      case FP_SUBNORMAL: { // Normalize
         UShort jbit = 0x0;
         f.parts.exponent = 0x7F;
         while (!(jbit & 0x1)) {
            jbit = f.parts.mantissa >> 22;
            f.parts.mantissa <<= 1;
            f.parts.exponent--;
         }
         break;
      }
      case FP_NORMAL:
         break;
   }
   switch (interv) {
      case 0x0:
         f.parts.exponent = 0x7F;
         break;
      case 0x1:
         f.parts.exponent = ((f.parts.exponent - 0x7F) & 0x1) ? 0x7E : 0x7F;
         break;
      case 0x2:
         f.parts.exponent = 0x7E;
         break;
      case 0x3:
         f.parts.exponent = (f.parts.mantissa >> 22) ? 0x7E : 0x7F;
         break;
   }
   *dst_u = f.uint_val;
}

VEX_REGPARM(3) void h_Iop_ExtractMant64(ULong* dst_u, ULong src_u, UInt imm8) {
   /* Sign Control and Normalization Interval are identical to h_Iop_ExtractMant32 */
   Int sc = (imm8 >> 2) & 0x3;
   Int interv = imm8 & 0x3;

   double_cast d = {.ulong_val = src_u};
   d.parts.sign = (sc & 0x1) ? 0x0 : d.parts.sign;
   if (d.parts.sign && (sc > 1)) {
      *dst_u = QNAN_INDEF_64.u;
      return;
   }
   switch (fpclassify(src_u)) {
      case FP_QNAN:
      case FP_SNAN:
         *dst_u = QNAN_64(src_u);
         return;
      case FP_INFINITE:
      case FP_NEG_INFINITE:
         *dst_u = INFINITY_64.u;
         return;
      case FP_ZERO:
         *dst_u = -INFINITY_64.u;
         return;
      case FP_SUBNORMAL: { // Normalize
         UShort jbit = 0x0;
         d.parts.exponent = 0x3FF;
         while (!(jbit & 0x1)) {
            jbit = d.parts.mantissa >> 51;
            d.parts.mantissa <<= 1;
            d.parts.exponent--;
         }
         break;
      }
      case FP_NORMAL:
         break;
   }
   switch (interv) {
      case 0x0:
         d.parts.exponent = 0x3FF;
         break;
      case 0x1:
         d.parts.exponent = ((d.parts.exponent - 0x3FF) & 0x1) ? 0x3FE : 0x3FF;
         break;
      case 0x2:
         d.parts.exponent = 0x3FE;
         break;
      case 0x3:
         d.parts.exponent = (d.parts.mantissa >> 51) ? 0x3FE : 0x3FF;
         break;
   }
   *dst_u = d.ulong_val;
}

// Limitation: do not set #PE flag in case of presicion loss
VEX_REGPARM(3) void h_Iop_RoundScaleF32(UInt* dst_u, UInt src_u, UInt imm8) {
   switch (fpclassifyf(src_u)) {
      case FP_QNAN:
      case FP_SNAN:
         *dst_u = QNAN_32(src_u);
         return;
      case FP_INFINITE:
      case FP_NEG_INFINITE:
      case FP_ZERO:
         *dst_u = src_u;
         return;
   }
   type32 src = {.u = src_u};
   Float in = src.f, out = 0;
   if (in <= -0x1.0p23 || in >= 0x1.0p23) { // Overflow
      *dst_u = src_u;
      return;
   }
   Int M = (imm8 >> 4);
   Float t1 = (1 << M) * in;
   if ((INT_MIN >= t1) || (t1 >= INT_MAX)) {
      out = t1 / (1 << M);
      type32 dst = {.f = out};
      *dst_u = dst.u;
      return;
   }
   Float tmp = (Float) round_with_rmode32(t1, imm8 & 0x3);
   if ((tmp == 0) && (t1 < 0))
      tmp = -0.0f;
   out = tmp / (1 << M);
   type32 dst = {.f = out};
   *dst_u = dst.u;
}

VEX_REGPARM(3) void h_Iop_RoundScaleF64(ULong* dst_u, ULong src_u, UInt imm8) {
   switch (fpclassify(src_u)) {
      case FP_QNAN:
      case FP_SNAN:
         *dst_u = QNAN_64(src_u);
         return;
      case FP_INFINITE:
      case FP_NEG_INFINITE:
      case FP_ZERO:
         *dst_u = src_u;
         return;
      default: break;
   }
   type64 src = {.u = src_u};
   Double in = src.f, out = 0;
   if (in <= -0x1.0p52 || in >= 0x1.0p52) { // Overflow
      *dst_u = src_u;
      return;
   }
   Int M = imm8 >> 4;
   Double t1 = (1 << M) * in;
   Double tmp;
   if ((LONG_MIN >= t1) || (t1 >= LONG_MAX)) {
      tmp = t1;
      out = tmp / (1 << M);
      type64 dst = {.f = out};
      *dst_u = dst.u;
      return;
   }
   tmp = (Double) round_with_rmode64(t1, imm8 & 0x3);
   if ((tmp == 0) && (t1 < 0))
      tmp = -0.0;
   out = tmp / (1 << M);
   type64 dst = {.f = out};
   *dst_u = dst.u;
}

VEX_REGPARM(2) void h_Iop_16Fto32F(UInt* dst_u, UInt in_u) {
   half_cast h = {.half_val = in_u};
   float_cast f = {.uint_val = 0};
   f.parts.sign = h.parts.sign;

   Bool ZeroOperand = ((h.parts.exponent == 0x0) & (h.parts.mantissa == 0x0));
   // Bool DenormOperand = ((h.parts.exponent == 0x0) & (h.parts.mantissa != 0x0));
   Bool InfOperand  = ((h.parts.exponent == 0x1F) & (h.parts.mantissa == 0x0));
   Bool NaNOperand  = ((h.parts.exponent == 0x1F) & (h.parts.mantissa != 0x0));

   if (ZeroOperand) {
      f.parts.exponent = 0x0;
      f.parts.mantissa = 0x0;
      *dst_u = f.uint_val;
      return f.float_val;
   }
   if (InfOperand) {
      f.parts.exponent = 0xFF;
      f.parts.mantissa = 0x0;
      *dst_u = f.uint_val;
      return f.float_val;
   }
   if (NaNOperand) {
      // Return QNAN(SRC)
      f.parts.exponent = 0xFF;
      f.parts.mantissa = (h.parts.mantissa << 13) | 0x400000ULL;
      *dst_u = f.uint_val;
      return f.float_val;
   }
   f.parts.exponent = h.parts.exponent + 112;
   f.parts.mantissa = h.parts.mantissa << 13;
   *dst_u = f.uint_val;
}

VEX_REGPARM(3) void h_Iop_32Fto16F(UInt* dst_u, UInt src_u, UInt rmode) {
   float_cast f = {.uint_val = src_u};
   half_cast h;

   h.parts.sign = f.parts.sign;

   Bool ZeroOperand   = ((f.parts.exponent == 0x0) & (f.parts.mantissa == 0x0));
   // Bool DenormOperand = ((f.parts.exponent == 0x0) & (f.parts.mantissa != 0x0));
   Bool InfOperand    = ((f.parts.exponent == 0xFF) & (f.parts.mantissa == 0x0));
   Bool NaNOperand    = ((f.parts.exponent == 0xFF) & (f.parts.mantissa != 0x0));
   if (ZeroOperand) {
      h.parts.exponent = 0x0;
      h.parts.mantissa = 0x0;
      return h.half_val;
   }
   if (InfOperand) {
      h.parts.exponent = 0x1F;
      h.parts.mantissa = 0x0;
      return h.half_val;
   }
   if (NaNOperand) {
      h.parts.exponent = 0x1F;
      h.parts.mantissa = f.parts.mantissa >> 13;
      return h.half_val;
   }

   switch (rmode & 0x3) {
      case 0x0: // Round nearest even
         if ((-65504 >= f.float_val) || (f.float_val >= 65504)) {
            // Out of bound
            h.parts.exponent = 0x1F;
            h.parts.mantissa = 0x0;
            return h.half_val;
         }
         if ((-5.96046e-8 <= f.float_val) && (f.float_val <= 5.96046e-8)) {
            // Maximum decimal digits it can handle. Value has to be 0.0.
            h.parts.exponent = 0x0;
            h.parts.mantissa = 0x0;
            return h.half_val;
         }
         if ((-14 > (f.parts.exponent-127)) || ((f.parts.exponent-127) > 15))
            h.parts.exponent = 0x0;
         else
            h.parts.exponent = (f.parts.exponent - 112) & 0x1F;
         // Supada's note: This only truncates, have to round up to nearest even
         h.parts.mantissa = f.parts.mantissa >> 13;
         break;
      case 0x1: // Round down
         if ((-65504 >= f.float_val) || (f.float_val >= 65504)) {
            // Out of bound
            h.parts.exponent = 0x1E;
            h.parts.mantissa = 0x3FF;
            // Rounding down sets negative number to infinity.
            if (h.parts.sign == 0x1) h.half_val += 1;
            return h.half_val;
         }
         if ((-5.96046e-8 <= f.float_val) && (f.float_val <= 5.96046e-8)) {
            // Maximum decimal digits it can handle. Value has to be 0.0.
            h.parts.exponent = 0x0;
            h.parts.mantissa = 0x0;
            // For negative number, rounding down needs to add one to mantissa.
            if (h.parts.sign == 0x1) h.parts.mantissa += 1;
            return h.half_val;
         }
         if ((-14 > (f.parts.exponent-127)) || ((f.parts.exponent-127) > 15))
            h.parts.exponent = 0x0;
         else
            h.parts.exponent = (f.parts.exponent - 112) & 0x1F;
         h.parts.mantissa = f.parts.mantissa >> 13;
         // For negative number, rounding down needs to add one to mantissa.
         if (h.parts.sign == 0x1) h.parts.mantissa += 1;
         break;
      case 0x2: // Round up
         if ((-65504 >= f.float_val) || (f.float_val >= 65504)) {
            // Out of bound
            h.parts.exponent = 0x1E;
            h.parts.mantissa = 0x3FF;
            // Rounding up sets possitive number to infinity.
            if (h.parts.sign == 0x0) h.half_val += 1;
            return h.half_val;
         }
         if ((-5.96046e-8 <= f.float_val) && (f.float_val <= 5.96046e-8)) {
            // Maximum decimal digits it can handle. Value has to be 0.0.
            h.parts.exponent = 0x0;
            h.parts.mantissa = 0x0;
            // For positive number, rounding up needs to add one to mantissa.
            if (h.parts.sign == 0x0) h.parts.mantissa += 1;
            return h.half_val;
         }
         if ((-14 > (f.parts.exponent-127)) || ((f.parts.exponent-127) > 15))
            // Exponent overflow before rounding up
            h.parts.exponent = 0x0;
         else
            h.parts.exponent = (f.parts.exponent - 112) & 0x1F;
         h.parts.mantissa = (f.parts.mantissa >> 13);
         // For positive number, rounding up needs to add one to mantissa.
         if (h.parts.sign == 0x0) h.parts.mantissa += 1;
         break;
      case 0x3: // Truncate or round to zero
         if ((-65504 >= f.float_val) || (f.float_val >= 65504)) {
            // Out of bound
            h.parts.exponent = 0x1E;
            h.parts.mantissa = 0x3FF;
            return h.half_val;
         }
         if ((-5.96046e-8 <= f.float_val) && (f.float_val <= 5.96046e-8)) {
            // Maximum decimal digits it can handle. Value has to be 0.0.
            h.parts.exponent = 0x0;
            h.parts.mantissa = 0x0;
            return h.half_val;
         }
         if ((-14 > (f.parts.exponent-127)) || ((f.parts.exponent-127) > 15))
            // Exponent overflow
            h.parts.exponent = 0x0;
         else
            h.parts.exponent = f.parts.exponent - 112;
         h.parts.mantissa = f.parts.mantissa >> 13;
         break;
   }
   return h.half_val;
}

// NOTE: keep in sync with operand_width from guest_amd64_toIR512.c
enum dstWidth {
   W_8bits = 0,
   W_16bits,
   W_32bits,
   W_64bits,
};
union I64 {
   UChar  w8[8];
   UShort w16[4];
   UInt   w32[2];
   Int   ws32[2];
   ULong  w64;
   Long  ws64;
};

#define GET_BIT_TO_INT(i) (((mask>>i)&1) ? (-1) : 0 )

VEX_REGPARM(3) void h_Iop_ExpandBitsToInt ( ULong* out, ULong mask, UInt elem_width) {
   Int i=0;
   union I64 res;
   switch (elem_width) {
      case W_8bits:  for (i=0;i<8;i++) res.w8[i]  = GET_BIT_TO_INT(i); break;
      case W_16bits: for (i=0;i<4;i++) res.w16[i] = GET_BIT_TO_INT(i); break;
      case W_32bits: for (i=0;i<2;i++) res.w32[i] = GET_BIT_TO_INT(i); break;
      case W_64bits:                   res.w64    = GET_BIT_TO_INT(0); break;
      default: vpanic("h_Iop_ExpandBitsToInt - not implemented\n");
   }
   *out = res.w64;
}
VEX_REGPARM(3) void h_Iop_ExpandBitsToV128 ( V128* out, ULong mask, UInt elem_width) {
   Int i = 0;
   switch (elem_width) {
      case W_8bits:  for (i=0;i<16;i++) out->w8[i]  = GET_BIT_TO_INT(i); break;
      case W_16bits: for (i=0;i<8; i++) out->w16[i] = GET_BIT_TO_INT(i); break;
      case W_32bits: for (i=0;i<4; i++) out->w32[i] = GET_BIT_TO_INT(i); break;
      case W_64bits: for (i=0;i<2; i++) out->w64[i] = GET_BIT_TO_INT(i); break;
      default: vpanic("h_Iop_ExpandBitsToV128 - not implemented\n");
   }
}
VEX_REGPARM(3) void h_Iop_ExpandBitsToV256 ( V256* out, ULong mask, UInt elem_width) {
   Int i = 0;
   switch (elem_width) {
      case W_8bits:  for (i=0;i<32;i++) out->w8[i]  = GET_BIT_TO_INT(i); break;
      case W_16bits: for (i=0;i<16;i++) out->w16[i] = GET_BIT_TO_INT(i); break;
      case W_32bits: for (i=0;i<8; i++) out->w32[i] = GET_BIT_TO_INT(i); break;
      case W_64bits: for (i=0;i<4; i++) out->w64[i] = GET_BIT_TO_INT(i); break;
      default: vpanic("h_Iop_ExpandBitsToV256 - not implemented\n");
   }
}
VEX_REGPARM(3) void h_Iop_ExpandBitsToV512 ( V512* out, ULong mask, UInt elem_width ) {
   Int i = 0;
   switch (elem_width) {
      case W_8bits:  for (i=0;i<64;i++) out->w8[i]  = GET_BIT_TO_INT(i); break;
      case W_16bits: for (i=0;i<32;i++) out->w16[i] = GET_BIT_TO_INT(i); break;
      case W_32bits: for (i=0;i<16;i++) out->w32[i] = GET_BIT_TO_INT(i); break;
      case W_64bits: for (i=0;i<8; i++) out->w64[i] = GET_BIT_TO_INT(i); break;
      default: vpanic("h_Iop_ExpandBitsToV512 - not implemented\n");
   }
}
#undef GET_BIT_TO_INT

/* Count leading zeros */
VEX_REGPARM(2) void h_Iop_Clz32( UInt* dst, UInt src) {
   *dst = (src == 0) ? 32 : __builtin_clz(src);
}

/* Conflict detection */
VEX_REGPARM(2) void h_Iop_CfD32x4( V128* dst, V128* src) {
   for (Int i = 0; i < 4; i++) {
      dst->w32[i] = 0;
      for (Int j = 0; j < i; j++)
         if (src->w32[i] == src->w32[j])
            dst->w32[i] |= 1 << j;
   }
}
VEX_REGPARM(2) void h_Iop_CfD32x8( V256* dst, V256* src) {
   for (Int i = 0; i < 8; i++) {
      dst->w32[i] = 0;
      for (Int j = 0; j < i; j++)
         if (src->w32[i] == src->w32[j])
            dst->w32[i] |= 1 << j;
   }
}
VEX_REGPARM(2) void h_Iop_CfD32x16( V512* dst, V512* src) {
   for (Int i = 0; i < 16; i++) {
      dst->w32[i] = 0;
      for (Int j = 0; j < i; j++)
         if (src->w32[i] == src->w32[j])
            dst->w32[i] |= 1 << j;
   }
}
VEX_REGPARM(2) void h_Iop_CfD64x2( V128* dst, V128* src) {
   for (Int i = 0; i < 2; i++) {
      dst->w64[i] = 0;
      for (Int j = 0; j < i; j++)
         if (src->w64[i] == src->w64[j])
            dst->w64[i] |= 1 << j;
   }
}
VEX_REGPARM(2) void h_Iop_CfD64x4( V256* dst, V256* src) {
   for (Int i = 0; i < 4; i++) {
      dst->w64[i] = 0;
      for (Int j = 0; j < i; j++)
         if (src->w64[i] == src->w64[j])
            dst->w64[i] |= 1 << j;
   }
}
VEX_REGPARM(2) void h_Iop_CfD64x8( V512* dst, V512* src) {
   for (Int i = 0; i < 8; i++) {
      dst->w64[i] = 0;
      for (Int j = 0; j < i; j++)
         if (src->w64[i] == src->w64[j])
            dst->w64[i] |= 1 << j;
   }
}

VEX_REGPARM(2) void h_Iop_Recip14_32( UInt* dst_u, UInt src_u) {
   type32 dst, src = {src_u};
   RCP14S(0, &dst, src );
   *dst_u = dst.u;
}
VEX_REGPARM(2) void h_Iop_Recip14_64( ULong* dst_u, ULong src_u) {
   type64 dst, src = {src_u};
   RCP14D(0, &dst, src );
   *dst_u = dst.u;
}

VEX_REGPARM(2) void h_Iop_RSqrt14_32( UInt* dst_u, UInt src_u) {
   type32 dst, src = {src_u};
   RSQRT14S(0, &dst, src );
   *dst_u = dst.u;
}
VEX_REGPARM(2) void h_Iop_RSqrt14_64( ULong* dst_u, ULong src_u) {
   type64 dst, src = {src_u};
   RSQRT14D(0, &dst, src );
   *dst_u = dst.u;
}


# define M_PI_2         1.57079632679489661923  /* pi/2 */
# define M_PI_2l        1.570796326794896619231321691639751442L /* pi/2 */

Float fixup_value (Float src, Float dst, ULong lookup_entry)
{
   Int category = -1;
   type64 d = {.f = src};
   switch (fpclassify(d.u)) {
      case FP_QNAN:         category = 0; break;
      case FP_SNAN:         category = 1; break;
      case FP_ZERO:         category = 2; break;
      case FP_NEG_INFINITE: category = 4; break;
      case FP_INFINITE:     category = 5; break;
      case FP_SUBNORMAL: //fallthrough
      case FP_NORMAL: {
         if (src < 0.0)       category = 6;
         else if (src == 1.0) category = 3;
         else                 category = 7;
         break;
      }
   }
   Int lookup_val = (lookup_entry >> (category * 4)) & 0xF;
   switch (lookup_val) {
      case 0x0: return dst;
      case 0x1: return src;
      case 0x2: { // Return qNaN with payload = source
         d.u |= FP64_EXP_MASK;
         d.u |= FP64_QNAN_BIT;
         return d.f;
      }
      case 0x3: { // Return indefinite qNaN
         type64 d = {FP64_DEFAULT_NAN_AS_UINT64};
         return d.f;
      }
      case 0x4: { // +INF
         type64 d = {FP64_SIGN_BIT | FP64_EXP_MASK};
         return d.f;
      }
      case 0x5: { // -INF
         type64 d = {FP64_EXP_MASK};
         return d.f;
      }
      case 0x6: { // INF
         type64 d = {FP64_EXP_MASK};
         if (src < 0.0) d.u |= FP64_SIGN_BIT;
         return d.f;
      }
      case 0x7: return -0.0;
      case 0x8: return +0.0;
      case 0x9: return -1.0;
      case 0xA: return +1.0;
      case 0xB: return  0.5;
      case 0xC: return 90.0;
      case 0xD: return M_PI_2l;
      case 0xE: return DBL_MAX;
      case 0xF: return -DBL_MAX;
   }
}


Float fixup_value32 (Float src, Float dst, UInt lookup_entry)
{
   Int category = -1;
   type32 f = {.f = src};
   switch (fpclassifyf(f.u)) {
      case FP_QNAN:         category = 0; break;
      case FP_SNAN:         category = 1; break;
      case FP_ZERO:         category = 2; break;
      case FP_NEG_INFINITE: category = 4; break;
      case FP_INFINITE:     category = 5; break;
      case FP_SUBNORMAL: //fallthrough
      case FP_NORMAL: {
         if (src < 0.0)       category = 6;
         else if (src == 1.0) category = 3;
         else                 category = 7;
         break;
      }
   }
   Int lookup_val = (lookup_entry >> (category * 4)) & 0xF;
   switch (lookup_val) {
      case 0x0: return dst;
      case 0x1: return src;
      case 0x2: { // Return qNaN with payload = source
         f.u |= FP32_EXP_MASK;
         f.u |= FP32_QNAN_BIT;
         return f.f;
      }
      case 0x3: { // Return indefinite qNaN
         type32 f = {FP32_DEFAULT_NAN_AS_UINT32};
         return f.f;
      }
      case 0x4: { // +INF
         type32 f = {FP32_SIGN_BIT | FP32_EXP_MASK};
         return f.f;
      }
      case 0x5: { // -INF
         type32 f = {FP32_EXP_MASK};
         return f.f;
      }
      case 0x6: { // INF
         type32 f = {FP32_EXP_MASK};
         if (src < 0.0) f.u |= FP32_SIGN_BIT;
         return f.f;
      }
      case 0x7: return -0.f;
      case 0x8: return +0.f;
      case 0x9: return -1.f;
      case 0xA: return +1.f;
      case 0xB: return 0.5f;
      case 0xC: return 90.f;
      case 0xD: return M_PI_2;
      case 0xE: return FLT_MAX;
      case 0xF: return -FLT_MAX;
   }
}

VEX_REGPARM(4) void h_Iop_FixupImm64 (ULong* dst_u,
      ULong src_u, ULong lookup_table, UInt imm8 )
{
   type64 dst = {*dst_u}, src = {src_u};
   dst.f = fixup_value(src.f, dst.f, lookup_table);
   *dst_u = dst.u;
}

VEX_REGPARM(4) void h_Iop_FixupImm32 (UInt* dst_u,
      UInt src_u, UInt lookup_table, UInt imm8 ) {
   type32 dst = {*dst_u}, src = {src_u};
   dst.f = fixup_value32(src.f, dst.f, lookup_table);
   *dst_u = dst.u;
}

VEX_REGPARM(3) void h_Iop_Perm8x16 ( V128* res, V128* index, V128* value ) {
   for (Int i = 0; i < 16; i++)
      res->w8[i] = value->w8[ index->w8[i] & 0xF ];
}
VEX_REGPARM(3) void h_Iop_Perm8x32 ( V256* res, V256* index, V256* value ) {
   for (Int i = 0; i < 32; i++)
      res->w8[i] = value->w8[ index->w8[i] & 0x1F ];
}
VEX_REGPARM(3) void h_Iop_Perm8x64 ( V512* res, V512* index, V512* value ) {
   for (Int i = 0; i < 64; i++)
      res->w8[i] = value->w8[ index->w8[i] & 0x3F ];
}
VEX_REGPARM(3) void h_Iop_Perm16x8 ( V128* res, V128* index, V128* value ) {
   for (Int i = 0; i < 8; i++)
      res->w16[i] = value->w16[ index->w16[i] & 0x7 ];
}
VEX_REGPARM(3) void h_Iop_Perm16x16 ( V256* res, V256* index, V256* value ) {
   for (Int i = 0; i < 16; i++)
      res->w16[i] = value->w16[ index->w16[i] & 0xF ];
}
VEX_REGPARM(3) void h_Iop_Perm16x32 ( V512* res, V512* index, V512* value ) {
   for (Int i = 0; i < 32; i++)
      res->w16[i] = value->w16[ index->w16[i] & 0x1F ];
}
VEX_REGPARM(3) void h_Iop_Perm32x16 ( V512* res, V512* index, V512* value ) {
   for (Int i = 0; i < 16; i++)
      res->w32[i] = value->w32[ index->w32[i] & 0xF ];
}
VEX_REGPARM(3) void h_Iop_Perm64x4 ( V256* res, V256* index, V256* value ) {
   for (Int i = 0; i < 4; i++)
      res->w64[i] = value->w64[ index->w64[i] & 0x3 ];
}
VEX_REGPARM(3) void h_Iop_Perm64x8 ( V512* res, V512* index, V512* value ) {
   for (Int i = 0; i < 8; i++)
      res->w64[i] = value->w64[ index->w64[i] & 0x7 ];
}

VEX_REGPARM(3) void h_Iop_PermI8x16 (V128* dst, V128* a, V128* b ) {
   for (Int i = 0; i < 16; i++) {
      Int offset = dst->w8[i] & 0xF;
      dst->w8[i] = (dst->w8[i] & (1<<4)) ? b->w8[offset] : a->w8[offset];
   }
}
VEX_REGPARM(3) void h_Iop_PermI8x32 (V256* dst, V256* a, V256* b ) {
   for (Int i = 0; i < 32; i++) {
      Int offset = dst->w8[i] & 0x1F;
      dst->w8[i] = (dst->w8[i] & (1<<5)) ? b->w8[offset] : a->w8[offset];
   }
}
VEX_REGPARM(3) void h_Iop_PermI8x64 (V512* dst, V512* a, V512* b ) {
   for (Int i = 0; i < 64; i++) {
      Int offset = dst->w8[i] & 0x3F;
      dst->w8[i] = (dst->w8[i] & (1<<6)) ? b->w8[offset] : a->w8[offset];
   }
}
VEX_REGPARM(3) void h_Iop_PermI16x8 (V128* dst, V128* a, V128* b ) {
   for (Int i = 0; i < 8; i++) {
      Int offset = dst->w16[i] & 0x7;
      dst->w16[i] = (dst->w16[i] & (1<<3)) ? b->w16[offset] : a->w16[offset];
   }
}
VEX_REGPARM(3) void h_Iop_PermI16x16 (V256* dst, V256* a, V256* b ) {
   for (Int i = 0; i < 16; i++) {
      Int offset = dst->w16[i] & 0xF;
      dst->w16[i] = (dst->w16[i] & (1<<4)) ? b->w16[offset] : a->w16[offset];
   }
}
VEX_REGPARM(3) void h_Iop_PermI16x32 (V512* dst, V512* a, V512* b ) {
   for (Int i = 0; i < 32; i++) {
      Int offset = dst->w16[i] & 0x1F;
      dst->w16[i] = (dst->w16[i] & (1<<5)) ? b->w16[offset] : a->w16[offset];
   }
}
VEX_REGPARM(3) void h_Iop_PermI32x4 (V128* dst, V128* a, V128* b ) {
   for (Int i = 0; i < 4; i++) {
      Int offset = dst->w32[i] & 0x3;
      dst->w32[i] = (dst->w32[i] & (1<<2)) ? b->w32[offset] : a->w32[offset];
   }
}
VEX_REGPARM(3) void h_Iop_PermI32x8 (V256* dst, V256* a, V256* b ) {
   for (Int i = 0; i < 8; i++) {
      Int offset = dst->w32[i] & 0x7;
      dst->w32[i] = (dst->w32[i] & (1<<3)) ? b->w32[offset] : a->w32[offset];
   }
}
VEX_REGPARM(3) void h_Iop_PermI32x16 (V512* dst, V512* a, V512* b ) {
   for (Int i = 0; i < 16; i++) {
      Int offset = dst->w32[i] & 0xF;
      dst->w32[i] = (dst->w32[i] & (1<<4)) ? b->w32[offset] : a->w32[offset];
   }
}
VEX_REGPARM(3) void h_Iop_PermI64x2 (V128* dst, V128* a, V128* b ) {
   for (Int i = 0; i < 2; i++) {
      Int offset = dst->w64[i] & 0x1;
      dst->w64[i] = (dst->w64[i] & 1<<1) ? b->w64[offset] : a->w64[offset];
   }
}
VEX_REGPARM(3) void h_Iop_PermI64x4 (V256* dst, V256* a, V256* b ) {
   for (Int i = 0; i < 4; i++) {
      Int offset = dst->w64[i] & 0x3;
      dst->w64[i] = (dst->w64[i] & (1<<2)) ? b->w64[offset] : a->w64[offset];
   }
}
VEX_REGPARM(3) void h_Iop_PermI64x8 (V512* dst, V512* a, V512* b ) {
   for (Int i = 0; i < 8; i++) {
      Int offset = dst->w64[i] & 0x7;
      dst->w64[i] = (dst->w64[i] & (1<<3)) ? b->w64[offset] : a->w64[offset];
   }
}

#define CMP_WITH_PREDICATE(x, y, result, predicate) \
   switch (predicate) { \
      case 0x00: result = ((x) == (y)); break; \
      case 0x01: result = ((x) <  (y)); break; \
      case 0x02: result = ((x) <= (y)); break; \
      default:   break; \
   }

// Limitation for all comparisons: ignore the signaling
VEX_REGPARM(4) void h_Iop_Cmp64Sx2( ULong* res, V128* argL, V128* argR, UInt imm8) {
   // Basic comparison
   ULong result = 0x0;
   for (Int i = 0; i < 2; i++) {
      Bool bit = False;
      CMP_WITH_PREDICATE( (Long)argL->w64[i], (Long)argR->w64[i], bit, (imm8 & 0x3) )
      if (bit) result |= (1ULL << i);
   }
   // Invert the result
   if (imm8 & 0x4) result = (~result) & 0x3;
   *res = result;
}
VEX_REGPARM(4) void h_Iop_Cmp64Sx4( ULong* res, V256* argL, V256* argR, UInt imm8) {
   ULong result = 0x0;
   for (Int i = 0; i < 4; i++) {
      Bool bit = False;
      CMP_WITH_PREDICATE( (Long)argL->w64[i], (Long)argR->w64[i], bit, (imm8 & 0x3) )
      if (bit) result |= (1ULL << i);
   }
   if (imm8 & 0x4) result = (~result) & 0xF;
   *res = result;
}
VEX_REGPARM(4) void h_Iop_Cmp64Sx8( ULong* res, V512* argL, V512* argR, UInt imm8) {
   ULong result = 0x0;
   for (Int i = 0; i < 8; i++) {
      Bool bit = False;
      CMP_WITH_PREDICATE( (Long)argL->w64[i], (Long)argR->w64[i], bit, (imm8 & 0x3) )
      if (bit) result |= (1ULL << i);
   }
   if (imm8 & 0x4) result = (~result) & 0xFF;
   *res = result;
}

VEX_REGPARM(4) void h_Iop_Cmp32Sx4 ( ULong* res, V128* argL, V128* argR, UInt imm8) {
   ULong result = 0x0;
   for (Int i = 0; i < 4; i++) {
      Bool bit = False;
      CMP_WITH_PREDICATE( (Int)argL->w32[i], (Int)argR->w32[i], bit, (imm8 & 0x3));
      if (bit) result |= (1ULL << i);
   }
   if (imm8 & 0x4) result = (~result) & 0xF;
   *res = result;
}
VEX_REGPARM(4) void h_Iop_Cmp32Sx8 ( ULong* res, V256* argL, V256* argR, UInt imm8) {
   ULong result = 0x0;
   for (Int i = 0; i < 8; i++) {
      Bool bit = False;
      CMP_WITH_PREDICATE( (Int)argL->w32[i], (Int)argR->w32[i], bit, (imm8 & 0x3));
      if (bit) result |= (1ULL << i);
   }
   if (imm8 & 0x4) result = (~result) & 0xFF;
   *res = result;
}
VEX_REGPARM(4) void h_Iop_Cmp32Sx16 ( ULong* res, V512* argL, V512* argR, UInt imm8)
{
   ULong result = 0x0;
   for (Int i = 0; i < 16; i++) {
      Bool bit = False;
      CMP_WITH_PREDICATE( (Int)argL->w32[i], (Int)argR->w32[i], bit, (imm8 & 0x3));
      if (bit) result |= (1ULL << i);
   }
   if (imm8 & 0x4) result = (~result) & 0xFFFF;
   *res = result;
}
VEX_REGPARM(4) void h_Iop_Cmp16Sx8 ( ULong* res, V128* argL, V128* argR, UInt imm8) {
   ULong result = 0x0;
   for (Int i = 0; i < 8; i++) {
      Bool bit = False;
      CMP_WITH_PREDICATE( (Short)argL->w16[i], (Short)argR->w16[i], bit, (imm8 & 0x3));
      if (bit) result |= (1ULL << i);
   }
   if (imm8 & 0x4) result = (~result) & 0xF;
   *res = result;
}
VEX_REGPARM(4) void h_Iop_Cmp16Sx16 ( ULong* res, V256* argL, V256* argR, UInt imm8) {
   ULong result = 0x0;
   for (Int i = 0; i < 16; i++) {
      Bool bit = False;
      CMP_WITH_PREDICATE( (Short)argL->w16[i], (Short)argR->w16[i], bit, (imm8 & 0x3));
      if (bit) result |= (1ULL << i);
   }
   if (imm8 & 0x4) result = (~result) & 0xFF;
   *res = result;
}
VEX_REGPARM(4) void h_Iop_Cmp16Sx32 ( ULong* res, V512* argL, V512* argR, UInt imm8)
{
   ULong result = 0x0;
   for (Int i = 0; i < 32; i++) {
      Bool bit = False;
      CMP_WITH_PREDICATE( (Short)argL->w16[i], (Short)argR->w16[i], bit, (imm8 & 0x3));
      if (bit) result |= (1ULL << i);
   }
   if (imm8 & 0x4) result = (~result) & 0xFFFF;
   *res = result;
}
VEX_REGPARM(4) void h_Iop_Cmp8Sx16 ( ULong* res, V128* argL, V128* argR, UInt imm8) {
   ULong result = 0x0;
   for (Int i = 0; i < 16; i++) {
      Bool bit = False;
      CMP_WITH_PREDICATE( (Char)argL->w8[i], (Char)argR->w8[i], bit, (imm8 & 0x3));
      if (bit) result |= (1ULL << i);
   }
   if (imm8 & 0x4) result = (~result) & 0xF;
   *res = result;
}
VEX_REGPARM(4) void h_Iop_Cmp8Sx32 ( ULong* res, V256* argL, V256* argR, UInt imm8) {
   ULong result = 0x0;
   for (Int i = 0; i < 32; i++) {
      Bool bit = False;
      CMP_WITH_PREDICATE( (Char)argL->w8[i], (Char)argR->w8[i], bit, (imm8 & 0x3));
      if (bit) result |= (1ULL << i);
   }
   if (imm8 & 0x4) result = (~result) & 0xFF;
   *res = result;
}
VEX_REGPARM(4) void h_Iop_Cmp8Sx64 ( ULong* res, V512* argL, V512* argR, UInt imm8)
{
   ULong result = 0x0;
   for (Int i = 0; i < 64; i++) {
      Bool bit = False;
      CMP_WITH_PREDICATE( (Char)argL->w8[i], (Char)argR->w8[i], bit, (imm8 & 0x3));
      if (bit) result |= (1ULL << i);
   }
   if (imm8 & 0x4) result = (~result) & 0xFFFF;
   *res = result;
}

VEX_REGPARM(4) void h_Iop_Cmp64Ux2( ULong* res, V128* argL, V128* argR, UInt imm8) {
   ULong result = 0x0;
   for (Int i = 0; i < 2; i++) {
      Bool bit = False;
      CMP_WITH_PREDICATE( argL->w64[i], argR->w64[i], bit, (imm8 & 0x3) );
      if (bit) result |= (1ULL << i);
   }
   if (imm8 & 0x4) result = (~result) & 0x3;
   *res = result;
}
VEX_REGPARM(4) void h_Iop_Cmp64Ux4( ULong* res, V256* argL, V256* argR, UInt imm8) {
   ULong result = 0x0;
   for (Int i = 0; i < 4; i++) {
      Bool bit = False;
      CMP_WITH_PREDICATE( argL->w64[i], argR->w64[i], bit, (imm8 & 0x3) );
      if (bit) result |= (1ULL << i);
   }
   if (imm8 & 0x4) result = (~result) & 0xF;
   *res = result;
}
VEX_REGPARM(4) void h_Iop_Cmp64Ux8( ULong* res, V512* argL, V512* argR, UInt imm8) {
   ULong result = 0x0;
   for (Int i = 0; i < 8; i++) {
      Bool bit = False;
      CMP_WITH_PREDICATE( argL->w64[i], argR->w64[i], bit, (imm8 & 0x3) );
      if (bit) result |= (1ULL << i);
   }
   if (imm8 & 0x4) result = (~result) & 0xFF;
   *res = result;
}

VEX_REGPARM(4) void h_Iop_Cmp32Ux4 ( ULong* res, V128* argL, V128* argR, UInt imm8) {
   ULong result = 0x0;
   for (Int i = 0; i < 4; i++) {
      Bool bit = False;
      CMP_WITH_PREDICATE( argL->w32[i], argR->w32[i], bit, (imm8 & 0x3));
      if (bit) result |= (1ULL << i);
   }
   if (imm8 & 0x4) result = (~result) & 0xF;
   *res = result;
}
VEX_REGPARM(4) void h_Iop_Cmp32Ux8 ( ULong* res, V256* argL, V256* argR, UInt imm8){
   ULong result = 0x0;
   for (Int i = 0; i < 8; i++) {
      Bool bit = False;
      CMP_WITH_PREDICATE( argL->w32[i], argR->w32[i], bit, (imm8 & 0x3));
      if (bit) result |= (1ULL << i);
   }
   if (imm8 & 0x4) result = (~result) & 0xFF;
   *res = result;
}
VEX_REGPARM(4) void h_Iop_Cmp32Ux16 ( ULong* res, V512* argL, V512* argR, UInt imm8)
{
   ULong result = 0x0;
   for (Int i = 0; i < 16; i++) {
      Bool bit = False;
      CMP_WITH_PREDICATE( argL->w32[i], argR->w32[i], bit, (imm8 & 0x3));
      if (bit) result |= (1ULL << i);
   }
   if (imm8 & 0x4) result = (~result) & 0xFFFF;
   *res = result;
}
VEX_REGPARM(4) void h_Iop_Cmp16Ux8 ( ULong* res, V128* argL, V128* argR, UInt imm8) {
   ULong result = 0x0;
   for (Int i = 0; i < 8; i++) {
      Bool bit = False;
      CMP_WITH_PREDICATE( argL->w16[i], argR->w16[i], bit, (imm8 & 0x3));
      if (bit) result |= (1UL << i);
   }
   if (imm8 & 0x4) result = (~result) & 0xF;
   *res = result;
}
VEX_REGPARM(4) void h_Iop_Cmp16Ux16 ( ULong* res, V256* argL, V256* argR, UInt imm8) {
   ULong result = 0x0;
   for (Int i = 0; i < 16; i++) {
      Bool bit = False;
      CMP_WITH_PREDICATE( argL->w16[i], argR->w16[i], bit, (imm8 & 0x3));
      if (bit) result |= (1UL << i);
   }
   if (imm8 & 0x4) result = (~result) & 0xFF;
   *res = result;
}
VEX_REGPARM(4) void h_Iop_Cmp16Ux32 ( ULong* res, V512* argL, V512* argR, UInt imm8)
{
   ULong result = 0x0;
   for (Int i = 0; i < 32; i++) {
      Bool bit = False;
      CMP_WITH_PREDICATE( argL->w16[i], argR->w16[i], bit, (imm8 & 0x3));
      if (bit) result |= (1UL << i);
   }
   if (imm8 & 0x4) result = (~result) & 0xFFFF;
   *res = result;
}
VEX_REGPARM(4) void h_Iop_Cmp8Ux16 ( ULong* res, V128* argL, V128* argR, UInt imm8) {
   ULong result = 0x0;
   for (Int i = 0; i < 16; i++) {
      Bool bit = False;
      CMP_WITH_PREDICATE( argL->w8[i], argR->w8[i], bit, (imm8 & 0x3));
      if (bit) result |= (1UL << i);
   }
   if (imm8 & 0x4) result = (~result) & 0xF;
   *res = result;
}
VEX_REGPARM(4) void h_Iop_Cmp8Ux32 ( ULong* res, V256* argL, V256* argR, UInt imm8) {
   ULong result = 0x0;
   for (Int i = 0; i < 32; i++) {
      Bool bit = False;
      CMP_WITH_PREDICATE( argL->w8[i], argR->w8[i], bit, (imm8 & 0x3));
      if (bit) result |= (1UL << i);
   }
   if (imm8 & 0x4) result = (~result) & 0xFF;
   *res = result;
}
VEX_REGPARM(4) void h_Iop_Cmp8Ux64 ( ULong* res, V512* argL, V512* argR, UInt imm8)
{
   ULong result = 0x0;
   for (Int i = 0; i < 64; i++) {
      Bool bit = False;
      CMP_WITH_PREDICATE( argL->w8[i], argR->w8[i], bit, (imm8 & 0x3));
      if (bit) result |= (1UL << i);
   }
   if (imm8 & 0x4) result = (~result) & 0xFFFF;
   *res = result;
}

Bool Cmp64F(ULong argL_u, ULong argR_u, UInt imm8) {
   type64 left = {.u = argL_u}, right = {.u = argR_u};
   Bool bit = False;
   CMP_WITH_PREDICATE(left.f, right.f, bit, (imm8 & 0x3));
   // Bool is actually UChar, so simple ~ for inversion doe not work
   if (imm8 & 0x4)
      bit = bit ? 0 : 1;
   // Update unordered comparisons
   Bool NaNs = isnan(left.f) || isnan(right.f);
   switch (imm8) {
      case 0x00: case 0x01: case 0x02:
      case 0x07:
      case 0x0B: case 0x0C: case 0x0D: case 0x0E:
      case 0x10: case 0x11: case 0x12:
      case 0x17:
      case 0x1B: case 0x1C: case 0x1D: case 0x1E:
         bit &= ~NaNs; break;
      default:
         bit |= NaNs; break;
   }
   return bit;
}

VEX_REGPARM(4) void h_Iop_Cmp64F( ULong* res, ULong argL, ULong argR, UInt imm8) {
   *res = Cmp64F(argL, argR, imm8);
}
VEX_REGPARM(4) void h_Iop_Cmp64Fx2( ULong* res, V128* argL, V128* argR, UInt imm8) {
   ULong result = 0x0;
   for (Int i = 0; i < 2; i++) {
      Bool bit = Cmp64F(argL->w64[i], argR->w64[i], imm8);
      if (bit) result |= (1ULL << i);
   }
   *res = result;
}
VEX_REGPARM(4) void h_Iop_Cmp64Fx4( ULong* res, V256* argL, V256* argR, UInt imm8) {
   ULong result = 0x0;
   for (Int i = 0; i < 4; i++) {
      Bool bit = Cmp64F(argL->w64[i], argR->w64[i], imm8);
      if (bit) result |= (1ULL << i);
   }
   *res = result;
}
VEX_REGPARM(4) void h_Iop_Cmp64Fx8( ULong* res, V512* argL, V512* argR, UInt imm8) {
   ULong result = 0x0;
   for (Int i = 0; i < 8; i++) {
      Bool bit = Cmp64F(argL->w64[i], argR->w64[i], imm8);
      if (bit) result |= (1ULL << i);
   }
   *res = result;
}

Bool Cmp32F(UInt argL_u, UInt argR_u, UInt imm8) {
   type32 left = {.u = argL_u}, right = {.u = argR_u};
   Bool bit = False;
   CMP_WITH_PREDICATE(left.f, right.f, bit, (imm8 & 0x3));
   // Invert the result
   if (imm8 & 0x4)
      bit = bit ? 0 : 1;
   // Update unordered comparisons
   Bool NaNs = isnanf(left.f) || isnanf(right.f);
   switch (imm8) {
      case 0x00: case 0x01: case 0x02:
      case 0x07:
      case 0x0B: case 0x0C: case 0x0D: case 0x0E:
      case 0x10: case 0x11: case 0x12:
      case 0x17:
      case 0x1B: case 0x1C: case 0x1D: case 0x1E:
         bit &= ~NaNs; break;
      default:
         bit |= NaNs; break;
   }
   return bit;
}
VEX_REGPARM(4) void h_Iop_Cmp32F( ULong* res, UInt argL, UInt argR, UInt imm8) {
   *res = Cmp32F((Float)argL, (Float)argR, imm8);
}
VEX_REGPARM(4) void h_Iop_Cmp32Fx4 ( ULong* res, V128* argL, V128* argR, UInt imm8) {
   ULong result = 0x0;
   for (Int i = 0; i < 4; i++) {
      Bool bit = Cmp32F((Float)argL->w32[i], (Float)argR->w32[i], imm8);
      if (bit) result |= (1ULL << i);
   }
   *res = result;
}
VEX_REGPARM(4) void h_Iop_Cmp32Fx8 ( ULong* res, V256* argL, V256* argR, UInt imm8) {
   ULong result = 0x0;
   for (Int i = 0; i < 8; i++) {
      Bool bit = Cmp32F((Float)argL->w32[i], (Float)argR->w32[i], imm8);
      if (bit) result |= (1ULL << i);
   }
   *res = result;
}
VEX_REGPARM(4) void h_Iop_Cmp32Fx16 ( ULong* res, V512* argL, V512* argR, UInt imm8) {
   ULong result = 0x0;
   for (Int i = 0; i < 16; i++) {
      Bool bit = Cmp32F((Float)argL->w32[i], (Float)argR->w32[i], imm8);
      if (bit) result |= (1ULL << i);
   }
   *res = result;
}
#undef CMP_WITH_PREDICATE

V512 tmp_array;

VEX_REGPARM(3) void h_Iop_Align32x4 ( V128* in_out, V128* orig, UInt imm8) {
   Int i;
   imm8 = imm8 & 0x3;
   for (i = 0; i < 4-imm8; i++)
      tmp_array.w32[i] = orig->w32[i+imm8];
   for (i; i < 4; i++)
      tmp_array.w32[i] = in_out->w32[i+imm8-4];
   for (i = 0; i < 4; i++)
      in_out->w32[i] = tmp_array.w32[i];
}
VEX_REGPARM(3) void h_Iop_Align32x8 ( V256* in_out, V256* orig, UInt imm8) {
   Int i;
   imm8 = imm8 & 0x7;
   for (i = 0; i < 8-imm8; i++)
      tmp_array.w32[i] = orig->w32[i+imm8];
   for (i; i < 8; i++)
      tmp_array.w32[i] = in_out->w32[i+imm8-8];
   for (i = 0; i < 8; i++)
      in_out->w32[i] = tmp_array.w32[i];
}
VEX_REGPARM(3) void h_Iop_Align32x16 ( V512* in_out, V512* orig, UInt imm8) {
   Int i;
   imm8 = imm8 & 0xF;
   for (i = 0; i < 16-imm8; i++)
      tmp_array.w32[i] = orig->w32[i+imm8];
   for (i; i < 16; i++)
      tmp_array.w32[i] = in_out->w32[i+imm8-16];
   for (i = 0; i < 16; i++)
      in_out->w32[i] = tmp_array.w32[i];
}

VEX_REGPARM(3) void h_Iop_Align64x2 ( V128* in_out, V128* orig, UInt imm8) {
   Int i;
   imm8 = imm8 & 0x1;
   for (i = 0; i < 2-imm8; i++)
      tmp_array.w64[i] = orig->w64[i+imm8];
   for (i; i < 2; i++)
      tmp_array.w64[i] = in_out->w64[i+imm8-2];
   for (i = 0; i < 2; i++)
      in_out->w64[i] = tmp_array.w64[i];
}
VEX_REGPARM(3) void h_Iop_Align64x4 ( V256* in_out, V256* orig, UInt imm8) {
   Int i;
   imm8 = imm8 & 0x3;
   for (i = 0; i < 4-imm8; i++)
      tmp_array.w64[i] = orig->w64[i+imm8];
   for (Int i = 4-imm8; i < 4; i++)
      tmp_array.w64[i] = in_out->w64[i+imm8-4];
   for (i = 0; i < 4; i++)
      in_out->w64[i] = tmp_array.w64[i];
}
VEX_REGPARM(3) void h_Iop_Align64x8 ( V512* in_out, V512* orig, UInt imm8) {
   Int i;
   imm8 = imm8 & 0x7;
   for (i = 0; i < 8-imm8; i++)
      tmp_array.w64[i] = orig->w64[i+imm8];
   for (i; i < 8; i++)
      tmp_array.w64[i] = in_out->w64[i+imm8-8];
   for (i = 0; i < 8; i++)
      in_out->w64[i] = tmp_array.w64[i];
}


VEX_REGPARM(3) void h_Iop_Expand32x4( V128* src_dst, V128* vec, ULong mask) {
   UInt m = 0;
   if (mask == 0) mask = -1;
   for (Int i = 0; i < 4; i++)
      if GET_BIT(mask, i) src_dst->w32[i] = vec->w32[m++];
}
VEX_REGPARM(3) void h_Iop_Expand32x8( V256* src_dst, V256* vec, ULong mask) {
   UInt m = 0;
   if (mask == 0) mask = -1;
   for (Int i = 0; i < 8; i++)
      if GET_BIT(mask, i) src_dst->w32[i] = vec->w32[m++];
}
VEX_REGPARM(3) void h_Iop_Expand32x16( V512* src_dst, V512* vec, ULong mask) {
   UInt m = 0;
   if (mask == 0) mask = -1;
   for (Int i = 0; i < 16; i++)
      if GET_BIT(mask, i) src_dst->w32[i] = vec->w32[m++];
}

VEX_REGPARM(3) void h_Iop_Expand64x2( V128* src_dst, V128* vec, ULong mask) {
   UInt m = 0;
   if (mask == 0) mask = -1;
   for (Int i = 0; i < 2; i++)
      if GET_BIT(mask, i) src_dst->w64[i] = vec->w64[m++];
}
VEX_REGPARM(3) void h_Iop_Expand64x4( V256* src_dst, V256* vec, ULong mask) {
   UInt m = 0;
   if (mask == 0) mask = -1;
   for (Int i = 0; i < 4; i++)
      if GET_BIT(mask, i) src_dst->w64[i] = vec->w64[m++];
}
VEX_REGPARM(3) void h_Iop_Expand64x8( V512* src_dst, V512* vec, ULong mask) {
   UInt m = 0;
   if (mask == 0) mask = -1;
   for (Int i = 0; i < 8; i++)
      if GET_BIT(mask, i) src_dst->w64[i] = vec->w64[m++];
}

VEX_REGPARM(4) void h_Iop_Compress32x4( V128* src_dst, V128* vec, ULong mask, UInt zero) {
   Int m = 0;
   if (mask == 0) mask = -1;
   for (Int i = 0; i < 4; i++)
      if (GET_BIT(mask, i)) src_dst->w32[m++] = vec->w32[i];
   if (zero)
      for (Int i = m; i < 4; i++)
         src_dst->w32[i] = 0;
}
VEX_REGPARM(4) void h_Iop_Compress32x8( V256* src_dst, V256* vec, ULong mask, UInt zero) {
   Int m = 0;
   if (mask == 0) mask = -1;
   for (Int i = 0; i < 8; i++)
      if (GET_BIT(mask, i)) src_dst->w32[m++] = vec->w32[i];
   if (zero)
      for (Int i = m; i < 8; i++)
         src_dst->w32[i] = 0;
}
VEX_REGPARM(4) void h_Iop_Compress32x16( V512* src_dst, V512* vec, ULong mask, UInt zero) {
   Int m = 0;
   if (mask == 0) mask = -1;
   for (Int i = 0; i < 16; i++)
      if (GET_BIT(mask, i)) src_dst->w32[m++] = vec->w32[i];
   if (zero)
      for (Int i = m; i < 16; i++)
         src_dst->w32[i] = 0;
}
VEX_REGPARM(4) void h_Iop_Compress64x2( V128* src_dst, V128* vec, ULong mask, UInt zero) {
   Int m = 0;
   if (mask == 0) mask = -1;
   for (Int i = 0; i < 2; i++)
      if (GET_BIT(mask, i)) src_dst->w64[m++] = vec->w64[i];
   if (zero)
      for (Int i = m; i < 2; i++)
         src_dst->w64[i] = 0;
}
VEX_REGPARM(4) void h_Iop_Compress64x4( V256* src_dst, V256* vec, ULong mask, UInt zero) {
   Int m = 0;
   if (mask == 0) mask = -1;
   for (Int i = 0; i < 4; i++)
      if (GET_BIT(mask, i)) src_dst->w64[m++] = vec->w64[i];
   if (zero)
      for (Int i = m; i < 4; i++)
         src_dst->w64[i] = 0;
}
VEX_REGPARM(4) void h_Iop_Compress64x8( V512* src_dst, V512* vec, ULong mask, UInt zero) {
   Int m = 0;
   if (mask == 0) mask = -1;
   for (Int i = 0; i < 8; i++)
      if (GET_BIT(mask, i)) src_dst->w64[m++] = vec->w64[i];
   if (zero)
      for (Int i = m; i < 8; i++)
         src_dst->w64[i] = 0;
}

#undef GET_BIT

VEX_REGPARM(3) void h_Iop_Classify32 ( ULong* res, UInt src, UInt category) {
   ULong result = (fpclassifyf(src) == category);
   *res = result;
}
VEX_REGPARM(3) void h_Iop_Classify32x4 ( ULong* res, V128* src, UInt category) {
   ULong result = 0;
   for (Int i = 0; i < 4; i++)
      if (fpclassifyf( src->w32[i] ) == category) result |= 1ULL << i;
   *res = result;
}
VEX_REGPARM(3) void h_Iop_Classify32x8 ( ULong* res, V256* src, UInt category) {
   ULong result = 0;
   for (Int i = 0; i < 8; i++)
      if (fpclassifyf( src->w32[i] ) == category) result |= 1ULL << i;
   *res = result;
}
VEX_REGPARM(3) void h_Iop_Classify32x16 ( ULong* res, V512* src, UInt category) {
   ULong result = 0;
   for (Int i = 0; i < 16; i++)
      if (fpclassifyf( src->w32[i] ) == category) result |= 1ULL << i;
   *res = result;
}
VEX_REGPARM(3) void h_Iop_Classify64 ( ULong* res, ULong src, UInt category) {
   ULong result = (fpclassify(src) == category);
   *res = result;
}
VEX_REGPARM(3) void h_Iop_Classify64x2 ( ULong* res, V128* src, UInt category) {
   ULong result = 0;
   for (Int i = 0; i < 2; i++)
      if (fpclassify( src->w64[i] ) == category) result |= 1ULL << i;
   *res = result;
}
VEX_REGPARM(3) void h_Iop_Classify64x4 ( ULong* res, V256* src, UInt category) {
   ULong result = 0;
   for (Int i = 0; i < 4; i++)
      if (fpclassify( src->w64[i] ) == category) result |= 1ULL << i;
   *res = result;
}
VEX_REGPARM(3) void h_Iop_Classify64x8 ( ULong* res, V512* src, UInt category) {
   ULong result = 0;
   for (Int i = 0; i < 8; i++)
      if (fpclassify( src->w64[i] ) == category) result |= 1ULL << i;
   *res = result;
}

VEX_REGPARM(3) void h_Iop_Reduce64 ( ULong* dst_u, ULong src_u, UInt imm8) {
   switch (fpclassify(src_u)) {
      case FP_QNAN:
      case FP_SNAN:
         *dst_u = QNAN_64(src_u);
         return;
      case FP_INFINITE:
      case FP_NEG_INFINITE:
         *dst_u = 0.0;
         return;
      case FP_ZERO:
         *dst_u = -0.0;
         return;
   }
   type64 src = {.u = src_u};
   UInt M = (imm8 >> 4) & 0xF;
   Double res = (1 << M) * src.f;
   if ((LONG_MIN >= res) || (res >= LONG_MAX)) {
      *dst_u = 0;
      return;
   }
   res = (Double) round_with_rmode64(res, imm8 & 0x3);
   res = res / (1 << M);
   res = src.f - res;
   type64 dst = {.f = res};
   *dst_u = dst.u;
}
VEX_REGPARM(3) void h_Iop_Reduce32 ( UInt* dst_u, UInt src_u, UInt imm8) {
   switch (fpclassifyf(src_u)) {
      case FP_QNAN:
      case FP_SNAN:
         *dst_u = QNAN_32(src_u);
         return;
      case FP_INFINITE:
      case FP_NEG_INFINITE:
         *dst_u = 0.0;
         return;
      case FP_ZERO:
         *dst_u = -0.0;
         return;
   }
   type32 src = {.u = src_u};
   UInt M = (imm8 >> 4) & 0xF;
   Float res = (1 << M) * src.f;
   if ((INT_MIN >= res) || (res >= INT_MAX)) {
      *dst_u = 0;
      return;
   }
   res = (Float) round_with_rmode32(res, imm8 & 0x3);
   res = res / (1 << M);
   res = src.f - res;
   type32 dst = {.f = res};
   *dst_u = dst.u;
}

#define ABS(x)  (((x) < 0) ? -(x) : (x))
VEX_REGPARM(3) void h_Iop_VDBPSADBW ( V128* argR_dst, V128* argL, UInt imm8 ) {
   // Shuffle 2nd source into temporary
   for (int i=0; i<4; i++)
      tmp_array.w32[i] = argL->w32[((imm8 >> (i-1)*2)) & 0x3];
   // Sum absolute differences into, temporarily, src2
   for (int j4=0; j4<8; j4 += 4 ) {
      argL->w16[0+j4] = ABS(tmp_array.w8[0+j4*2] - argR_dst->w8[0+j4*2]) +
                        ABS(tmp_array.w8[1+j4*2] - argR_dst->w8[1+j4*2]) +
                        ABS(tmp_array.w8[2+j4*2] - argR_dst->w8[2+j4*2]) +
                        ABS(tmp_array.w8[3+j4*2] - argR_dst->w8[3+j4*2]);
      argL->w16[1+j4] = ABS(tmp_array.w8[0+j4*2] - argR_dst->w8[1+j4*2]) +
                        ABS(tmp_array.w8[1+j4*2] - argR_dst->w8[2+j4*2]) +
                        ABS(tmp_array.w8[2+j4*2] - argR_dst->w8[3+j4*2]) +
                        ABS(tmp_array.w8[3+j4*2] - argR_dst->w8[4+j4*2]);
      argL->w16[2+j4] = ABS(tmp_array.w8[4+j4*2] - argR_dst->w8[2+j4*2]) +
                        ABS(tmp_array.w8[5+j4*2] - argR_dst->w8[3+j4*2]) +
                        ABS(tmp_array.w8[6+j4*2] - argR_dst->w8[4+j4*2]) +
                        ABS(tmp_array.w8[7+j4*2] - argR_dst->w8[5+j4*2]);
      argL->w16[3+j4] = ABS(tmp_array.w8[4+j4*2] - argR_dst->w8[3+j4*2]) +
                        ABS(tmp_array.w8[5+j4*2] - argR_dst->w8[4+j4*2]) +
                        ABS(tmp_array.w8[6+j4*2] - argR_dst->w8[5+j4*2]) +
                        ABS(tmp_array.w8[7+j4*2] - argR_dst->w8[6+j4*2]);
   }
   // Move result into destinataion
   for (int i=0; i<2; i++)
      argR_dst->w64[i] = argL->w64[i];
}

VEX_REGPARM(3) void h_Iop_Range32 ( UInt* src_dst_u, UInt src2_u, UInt imm8 ) {
   UInt CompareOp = imm8 & 0x3;
   UInt SignControl = (imm8 >> 2) & 0x3;
   type32 src1 = {.u = *src_dst_u};
   type32 src2 = {.u = src2_u};

   if (isnanf(src1.f)){
      *src_dst_u= QNAN_32(src1.u);
      return;
   }
   if (isnanf(src2.f)){
      *src_dst_u = QNAN_32(src2.u);
      return;
   }

   // Denormal input is not handled yet

   if ((ABS(src1.f) == 0) && (ABS(src2.f) == 0)) {
      *src_dst_u = (imm8 & 0x1) ? 0 : -0;
      return;
   }
   if ((ABS(src1.f) == ABS(src2.f)) && (src1.f != src2.f)) {
      *src_dst_u = (imm8 & 0x1) ?  ABS(src1.u) : -ABS(src1.u);
      return;
   }

   float_cast res = {.uint_val = 0};
   switch (CompareOp) {
      case 0x0: res.float_val = (src1.f <= src2.f) ? src1.f : src2.f; break;
      case 0x1: res.float_val = (src1.f <= src2.f) ? src2.f : src1.f; break;
      case 0x2: res.float_val = (ABS(src1.f) <= ABS(src2.f)) ? src1.f : src2.f; break;
      case 0x3: res.float_val = (ABS(src1.f) <= ABS(src2.f)) ? src1.f : src2.f; break;
   }
   switch (SignControl) {
      case 0x0: res.parts.sign = (src1.f > 0) ? 0:1; break;
      case 0x1: break;
      case 0x2: res.parts.sign = 0; break;
      case 0x3: res.parts.sign = 1; break;
   }
   *src_dst_u = res.uint_val;
}
VEX_REGPARM(3) void h_Iop_Range64 ( ULong* src_dst_u, ULong src2_u, UInt imm8 ) {
   UInt CompareOp = imm8 & 0x3;
   UInt SignControl = (imm8 >> 2) & 0x3;
   type64 src1 = {.u = *src_dst_u};
   type64 src2 = {.u = src2_u};

   if (isnan(src1.f)){
      *src_dst_u = QNAN_64(src1.u);
      return;
   }
   if (isnan(src2.f)){
      *src_dst_u = QNAN_64(src2.u);
      return;
   }

   // Denormal input is not handled yet

   if ((ABS(src1.f) == 0) && (ABS(src2.f) == 0)) {
      *src_dst_u = (imm8 & 0x1) ? 0 : -0;
      return;
   }
   if ((ABS(src1.f) == ABS(src2.f)) && (src1.f != src2.f)) {
      *src_dst_u = (imm8 & 0x1) ?  ABS(src1.u) : -ABS(src1.u);
      return;
   }

   double_cast res = {.ulong_val= 0};
   switch (CompareOp) {
      case 0x0: res.double_val = (src1.f <= src2.f) ? src1.f : src2.f; break;
      case 0x1: res.double_val = (src1.f <= src2.f) ? src2.f : src1.f; break;
      case 0x2: res.double_val = (ABS(src1.f) <= ABS(src2.f)) ? src1.f : src2.f; break;
      case 0x3: res.double_val = (ABS(src1.f) <= ABS(src2.f)) ? src1.f : src2.f; break;
   }
   switch (SignControl) {
      case 0x0: res.parts.sign = (src1.f > 0) ? 0:1; break;
      case 0x1: break;
      case 0x2: res.parts.sign = 0; break;
      case 0x3: res.parts.sign = 1; break;
   }
   *src_dst_u = res.ulong_val;
}
#undef ABS

VEX_REGPARM(3) void h_Iop_F32toI32U (UInt* dst, UInt rmode, UInt src_u) {
   type32 src = {.u = src_u};
   if (src.f <= INT_MIN ||src.f > UINT_MAX) {
      *dst = -1;
      return;
   }
   *dst = (UInt) floorf(src.f);
}
VEX_REGPARM(3) void h_Iop_F32TtoI32U (UInt* dst, UInt rmode, UInt src_u) {
   type32 src = {.u = src_u};
   *dst = (src.f <= INT_MIN || src.f > INT_MAX) ? 0x80000000 : (UInt) src.f;
}
VEX_REGPARM(3) void h_Iop_F32TtoI32S (UInt* dst, UInt rmode, UInt src_u) {
   type32 src = {.u = src_u};
   *dst = (src.f <= INT_MIN || src.f > INT_MAX) ? 0x80000000 : (Int) src.f;
}
VEX_REGPARM(3) void h_Iop_F64toI64U (ULong* dst, UInt rmode, ULong src_u) {
   type64 src = {.u = src_u};
   if (src.f <= LONG_MIN || src.f > LONG_MAX) {
      *dst = -1;
      return;
   }
   *dst = (ULong) floor(src.f);
}
VEX_REGPARM(3) void h_Iop_F64TtoI64U (ULong* dst, UInt rmode, ULong src_u) {
   type64 src = {.u = src_u};
   *dst = (src.f <= LONG_MIN || src.f > LONG_MAX) ? 0x8000000000000000 : (ULong) src.f;
}
VEX_REGPARM(3) void h_Iop_F64TtoI64S (ULong* dst, UInt rmode, ULong src_u) {
   type64 src = {.u = src_u};
   *dst = (src.f <= LONG_MIN || src.f > LONG_MAX) ? 0x8000000000000000 : (Long) src.f;
}

VEX_REGPARM(3) void h_Iop_I32UtoF32 (UInt* dst_u, UInt rmode_fake, UInt src) {
   type32 dst = {.f = (Float)src};
   *dst_u = dst.u;
}
VEX_REGPARM(3) void h_Iop_I64UtoF64 (ULong* dst_u, UInt rmode_fake, ULong src) {
   type64 dst = {.f = (Double)src};
   *dst_u = dst.u;
}

#endif /* ndef AVX_512 */
/*---------------------------------------------------------------*/
/*--- end                               host_generic_AVX512.c ---*/
/*---------------------------------------------------------------*/
