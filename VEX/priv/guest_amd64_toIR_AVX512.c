/*--------------------------------------------------------------------*/
/*--- begin                                     guest_amd64_toIR.c ---*/
/*--------------------------------------------------------------------*/

/*
   This file is part of Valgrind, a dynamic binary instrumentation
   framework.

   Copyright (C) 2004-2017 OpenWorks LLP
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
   along with this program; if not, see <http://www.gnu.org/licenses/>.

   The GNU General Public License is contained in the file COPYING.

   Neither the names of the U.S. Department of Energy nor the
   University of California nor the names of its contributors may be
   used to endorse or promote products derived from this software
   without prior written permission.
*/

/* Translates AVX-512 AMD64 code to IR. */
#ifdef AVX_512

#include "guest_amd64_toIR_AVX512.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DIP(format, args...)           \
   if (vex_traceflags & VEX_TRACE_FE)  \
      vex_printf(format, ## args)

#define DIS(buf, format, args...)      \
   if (vex_traceflags & VEX_TRACE_FE)  \
      vex_sprintf(buf, format, ## args)

typedef UInt Prefix_EVEX;
// Have to implement it as a global variable: some VEX functions have to refer
// to EVEX prefix, and changing their signatures requires too many changes
Prefix_EVEX evex = 0;

#define EVEX_R1     (1<<0)    /* R1 bit */
#define EVEX_VEXnV4 (1<<1)    /* EVEX vvvv[4] bit */
#define EVEX_EVEXb  (1<<2)    /* EVEX b bit */
#define EVEX_WIG    (1<<3)    /* Ignore element width */
#define EVEX_MASKZ  (1<<4)    /* 0 for merging, 1 for zeroing */
#define EVEX_MASKR_OFFSET  (5)  /* 3 bits, mask register number */
#define EVEX_DSTW_OFFSET   (8)  /* 3 bits, see enum operand_width */
#define EVEX_EVEXL_OFFSET  (11) /* 2 bits, source vector length. 0 for 128, ..., 2 for 512 */
#define EVEX_EVEXTT_OFFSET (13) /* 4 bits, EVEX tuple type */
#define EVEX_VL_OVERRIDE   (1<<17)
#define PFX_NA (0)

static void setZeroMode( Bool mode_z ) {
   if (mode_z) evex |= EVEX_MASKZ;
   else evex &= ~EVEX_MASKZ;
}
static Bool getZeroMode( void ) {
   return toBool(evex & EVEX_MASKZ);
}
static void setEvexMask (UInt mask) {
   evex &= ~(0x7 << EVEX_MASKR_OFFSET);
   evex |= (mask << EVEX_MASKR_OFFSET);
}
static UInt getEvexMask( void ) {
   return (evex >> EVEX_MASKR_OFFSET) & 0x7;
}
UInt getEVexNvvvv ( Prefix pfx ) {
   UInt r = (UInt)pfx;
   r /= (UInt)PFX_VEXnV0; /* pray this turns into a shift */
   r &= 0xF;
   if (evex & EVEX_VEXnV4) r += 16;
   return r;
}
static Int getEvexb( void ) {
   return (evex & EVEX_EVEXb) ? 1 : 0;
}
static void setWIG( void ) {
   evex |= EVEX_WIG;
}
static Int getWIG( void ) {
   return ((evex & EVEX_WIG) ? 1 : 0);
}

// NOTE: keep in sync with dstWidth from host_generic_simd512.c
enum operand_width { W_8=0, W_16=1, W_32=2, W_64=3, WIG, W0, W1, W_1 };

static void setDstW ( enum operand_width width ) {
   evex &= ~(0x7 << EVEX_DSTW_OFFSET);
   evex |= ((UInt)width << EVEX_DSTW_OFFSET);
}
static Int getDstW( void ) {
   return ((evex>>EVEX_DSTW_OFFSET) & 0x7);
}

static void setEVexL ( UInt vl ) {
   evex &= ~(0x3 << EVEX_EVEXL_OFFSET);
   evex |= ((UInt)vl << EVEX_EVEXL_OFFSET);
}
static Int getEVexL ( void ) {
   Int vl = (evex >> EVEX_EVEXL_OFFSET) & 0x3;
   if (vl == 3) {
      if (getEvexb())
         DIP("VL OVERRIDE!\n");
      vl = 2;
   }
   return vl;
}

UInt gregOfRexRM32 ( Prefix pfx, UChar mod_reg_rm ) {
   Int reg = (Int)((mod_reg_rm >> 3) & 7);
   reg += (pfx & PFX_REXR) ? 8 : 0;
   reg += (evex & EVEX_R1) ? 16 : 0;
   return reg;
}
UInt eregOfRexRM32 ( Prefix pfx, UChar mod_reg_rm )  {
   vassert(epartIsReg(mod_reg_rm));
   Int reg = (Int)((mod_reg_rm) & 7);
   reg += (pfx & PFX_REXB) ? 8 : 0;
   reg += (pfx & PFX_REXX) ? 16 : 0;
   return reg;
}
UChar get_reg_16_32(UChar index_r) {
   return (evex & EVEX_VEXnV4) ? (index_r+16) : (index_r);
}

void setTupleType ( enum TupleType type ) {
   evex &= ~(0xF << EVEX_EVEXTT_OFFSET);
   evex |= (((UInt)type & 0xF) << EVEX_EVEXTT_OFFSET);
}
static Int getTupleType ( void ) {
   return ((evex >> EVEX_EVEXTT_OFFSET) & 0xF);
}

static Int pow(int x, int n) {
   int res = 1;
   for (int i = 0; i < n; ++i)
      res *= x;
   return res;
}

Int getEVEXMult( Prefix pfx ) {
   if (evex == 0)
      return 1;

   Int type = getTupleType();
   if (getEvexb()) { // use embedded broadcast
      DIP("use embedded broadcast\n");
      switch (type) {
         case FullVector: return (getDstW() == W_64) ? 8 : 4;
         case HalfVector: return 4;
      }
      vpanic("unsupported tuple embedded broadcast\n");
   } else {
      DIP(" no embedded broadcast\n");
      switch (type) {
         case NoTupleType:   return 1;
         case FullVector:
         case FullVectorMem: return pow(2, getEVexL()+4);
         case HalfVector:
         case HalfMem:       return pow(2, getEVexL()+3);
         case QuarterMem:    return pow(2, getEVexL()+2);
         case OctMem:        return pow(2, getEVexL()+1);
         case Tuple1Fixed:   return (getDstW() == W_64) ? 8 : 4;
         case Tuple2:        return (getDstW() == W_64) ? 16 : 8;
         case Tuple4:        return (getDstW() == W_64) ? 32 : 16;
         case Tuple8:        return 32;
         case Mem128:        return 16;
         // Tuple1Scalar relies on src width, but src does not have 8- and 16-bit cases
         case Tuple1Scalar:  return getWIG() ? pow(2, getDstW()) : pow(2, getRexW(pfx)+2);
         case MOVDDUP: {
            switch (getEVexL()) {
               case 0: return 8;
               case 1: return 32;
               case 2: return 64;
               default: vpanic("invalid vector length\n");
            }
            break;
         }
      }
   }
   vpanic("invalid tuple type\n");
}

const HChar* nameXMMRegEVEX (Int xmmreg) {
   static const HChar* xmm_names[16]
      = { "%xmm16", "%xmm17", "%xmm18", "%xmm19",
         "%xmm20", "%xmm21", "%xmm22", "%xmm23",
         "%xmm24", "%xmm25", "%xmm26", "%xmm27",
         "%xmm28", "%xmm29", "%xmm30", "%xmm31" };
   if (xmmreg < 16 || xmmreg > 31) {
      vpanic("nameXMMReg(avx512)");
   }
   return xmm_names[xmmreg-16];
}

const HChar* nameYMMRegEVEX (Int ymmreg) {
   static const HChar* ymm_names[16]
      = { "%ymm16", "%ymm17", "%ymm18", "%ymm19",
         "%ymm20", "%ymm21", "%ymm22", "%ymm23",
         "%ymm24", "%ymm25", "%ymm26", "%ymm27",
         "%ymm28", "%ymm29", "%ymm30", "%ymm31" };
   if (ymmreg < 16 || ymmreg > 31) {
      vpanic("nameYMMReg(avx512)");
   }
   return ymm_names[ymmreg-16];
}

const HChar* nameZMMReg ( Int zmmreg ) {
   static const HChar* zmm_names[32]
      = { "%zmm0",  "%zmm1",  "%zmm2",  "%zmm3",
         "%zmm4",  "%zmm5",  "%zmm6",  "%zmm7",
         "%zmm8",  "%zmm9",  "%zmm10", "%zmm11",
         "%zmm12", "%zmm13", "%zmm14", "%zmm15",
         "%zmm16", "%zmm17", "%zmm18", "%zmm19",
         "%zmm20", "%zmm21", "%zmm22", "%zmm23",
         "%zmm24", "%zmm25", "%zmm26", "%zmm27",
         "%zmm28", "%zmm29", "%zmm30", "%zmm31", };
   if (zmmreg < 0 || zmmreg > 31) vpanic("namezmmReg(amd64)");
   return zmm_names[zmmreg];
}

static const HChar* nameKReg ( Int kreg ) {
   static const HChar* k_names[8]
      = { "%k0", "%k1", "%k2", "%k3",
         "%k4", "%k5", "%k6", "%k7", };
   if (kreg < 0 || kreg > 7) {
      vpanic("namekReg(amd64)");
   }
   return k_names[kreg];
}

Int ymmGuestRegOffsetEVEX ( Int ymmreg )
{
   switch (ymmreg) {
      case 16: return OFFB_YMM16;
      case 17: return OFFB_YMM17;
      case 18: return OFFB_YMM18;
      case 19: return OFFB_YMM19;
      case 20: return OFFB_YMM20;
      case 21: return OFFB_YMM21;
      case 22: return OFFB_YMM22;
      case 23: return OFFB_YMM23;
      case 24: return OFFB_YMM24;
      case 25: return OFFB_YMM25;
      case 26: return OFFB_YMM26;
      case 27: return OFFB_YMM27;
      case 28: return OFFB_YMM28;
      case 29: return OFFB_YMM29;
      case 30: return OFFB_YMM30;
      case 31: return OFFB_YMM32;
      default: {
         vpanic("ymmGuestRegOffset(avx512)");
      }
   }
}

static Int zmmGuestRegOffset ( UInt zmmreg ) {
   switch (zmmreg) {
      case 0: return OFFB_ZMM0;
      case 1: return OFFB_ZMM1;
      case 2: return OFFB_ZMM2;
      case 3: return OFFB_ZMM3;
      case 4: return OFFB_ZMM4;
      case 5: return OFFB_ZMM5;
      case 6: return OFFB_ZMM6;
      case 7: return OFFB_ZMM7;
      case 8: return OFFB_ZMM8;
      case 9: return OFFB_ZMM9;
      case 10: return OFFB_ZMM10;
      case 11: return OFFB_ZMM11;
      case 12: return OFFB_ZMM12;
      case 13: return OFFB_ZMM13;
      case 14: return OFFB_ZMM14;
      case 15: return OFFB_ZMM15;
      case 16: return OFFB_ZMM16;
      case 17: return OFFB_ZMM17;
      case 18: return OFFB_ZMM18;
      case 19: return OFFB_ZMM19;
      case 20: return OFFB_ZMM20;
      case 21: return OFFB_ZMM21;
      case 22: return OFFB_ZMM22;
      case 23: return OFFB_ZMM23;
      case 24: return OFFB_ZMM24;
      case 25: return OFFB_ZMM25;
      case 26: return OFFB_ZMM26;
      case 27: return OFFB_ZMM27;
      case 28: return OFFB_ZMM28;
      case 29: return OFFB_ZMM29;
      case 30: return OFFB_ZMM30;
      case 31: return OFFB_ZMM31;
      default: vex_printf("%x\n", zmmreg);
               vpanic("zmmGuestRegOffset(amd64)");
   }
}
static Int kGuestRegOffset ( UInt kreg ) {
   switch (kreg) {
      case 0: return OFFB_K0;
      case 1: return OFFB_K1;
      case 2: return OFFB_K2;
      case 3: return OFFB_K3;
      case 4: return OFFB_K4;
      case 5: return OFFB_K5;
      case 6: return OFFB_K6;
      case 7: return OFFB_K7;
      default: vpanic("kGuestRegOffset(amd64)");
   }
}

static IRExpr* qop ( IROp op, IRExpr* a1, IRExpr* a2, IRExpr* a3, IRExpr* a4) {
   return IRExpr_Qop(op, a1, a2, a3, a4);
}


static Int zmmGuestRegLane32offset ( UInt zmmreg, Int laneno ) {
   /* Correct for little-endian host only. */
   vassert(host_endness == VexEndnessLE);
   vassert(laneno >= 0 && laneno < 16);
   return zmmGuestRegOffset( zmmreg ) + 4 * laneno;
}


static Int zmmGuestRegLane64offset ( UInt zmmreg, Int laneno ) {
   /* Correct for little-endian host only. */
   vassert(host_endness == VexEndnessLE);
   vassert(laneno >= 0 && laneno < 8);
   return zmmGuestRegOffset( zmmreg ) + 8 * laneno;
}



static Int zmmGuestRegLane128offset ( UInt zmmreg, Int laneno ) {
   /* Correct for little-endian host only. */
   vassert(host_endness == VexEndnessLE);
   vassert(laneno >= 0 && laneno < 4);
   return zmmGuestRegOffset( zmmreg ) + 16 * laneno;
}

static Int zmmGuestRegLane256offset ( UInt zmmreg, Int laneno ) {
   /* Correct for little-endian host only. */
   vassert(host_endness == VexEndnessLE);
   vassert(laneno >= 0 && laneno < 2);
   return zmmGuestRegOffset( zmmreg ) + 32 * laneno;
}

static Int kGuestRegLane32Offset( UInt kreg, Int laneno ) {
   /* Correct for little-endian host only. */
   vassert(host_endness == VexEndnessLE);
   vassert(laneno >= 0 && laneno < 2);
   return kGuestRegOffset( kreg ) + 4 * laneno;
}

static Int kGuestRegLane16Offset( UInt kreg, Int laneno ) {
   /* Correct for little-endian host only. */
   vassert(host_endness == VexEndnessLE);
   vassert(laneno >= 0 && laneno < 4);
   return kGuestRegOffset( kreg ) + 2 * laneno;
}

static Int kGuestRegLane8Offset( UInt kreg, Int laneno ) {
   /* Correct for little-endian host only. */
   vassert(host_endness == VexEndnessLE);
   vassert(laneno >= 0 && laneno < 8);
   return kGuestRegOffset( kreg ) + laneno;
}

IRExpr* mkV256 ( UInt val ) {
   return IRExpr_Const(IRConst_V256(val));
}

static IRExpr* get_evex_roundingmode () {
   if (getEvexb())
      return mkU32(getEVexL() + 0x08);
   else
      return get_sse_roundingmode();
}

IRExpr* getZMMReg ( UInt zmmreg ) {
   return IRExpr_Get( zmmGuestRegOffset(zmmreg), Ity_V512 );
}

IRExpr* getZMMRegLane256 ( UInt zmmreg, Int laneno ) {
   return IRExpr_Get( zmmGuestRegLane256offset(zmmreg,laneno), Ity_V256 );
}

IRExpr* getZMMRegLane128 ( UInt zmmreg, Int laneno ) {
   return IRExpr_Get( zmmGuestRegLane128offset(zmmreg,laneno), Ity_V128 );
}

IRExpr* getZMMRegLane64 ( UInt zmmreg, Int laneno ) {
   return IRExpr_Get( zmmGuestRegLane64offset(zmmreg,laneno), Ity_I64 );
}

IRExpr* getZMMRegLane64F ( UInt zmmreg, Int laneno ) {
   return IRExpr_Get( zmmGuestRegLane64offset(zmmreg,laneno), Ity_F64 );
}

IRExpr* getZMMRegLane32 ( UInt zmmreg, Int laneno ) {
   return IRExpr_Get( zmmGuestRegLane32offset(zmmreg,laneno), Ity_I32 );
}

IRExpr* getZMMRegLane32F ( UInt zmmreg, Int laneno ) {
   return IRExpr_Get( zmmGuestRegLane32offset(zmmreg,laneno), Ity_F32 );
}

void putZMMReg ( UInt zmmreg, IRExpr* e ) {
   vassert(typeOfIRExpr(irsb->tyenv,e) == Ity_V512);
   stmt( IRStmt_Put( zmmGuestRegOffset(zmmreg), e ) );
}

void putZMMRegLane256 ( UInt zmmreg, Int laneno, IRExpr* e ) {
   vassert(typeOfIRExpr(irsb->tyenv,e) == Ity_V256);
   stmt( IRStmt_Put( zmmGuestRegLane256offset(zmmreg,laneno), e ) );
}

void putZMMRegLane128 ( UInt zmmreg, Int laneno, IRExpr* e ) {
   vassert(typeOfIRExpr(irsb->tyenv,e) == Ity_V128);
   stmt( IRStmt_Put( zmmGuestRegLane128offset(zmmreg,laneno), e ) );
}

void putZMMRegLane64 ( UInt zmmreg, Int laneno, IRExpr* e ) {
   vassert(typeOfIRExpr(irsb->tyenv,e) == Ity_I64);
   stmt( IRStmt_Put( zmmGuestRegLane64offset(zmmreg,laneno), e ) );
}

void putZMMRegLane64F ( UInt zmmreg, Int laneno, IRExpr* e ) {
   vassert(typeOfIRExpr(irsb->tyenv,e) == Ity_F64);
   stmt( IRStmt_Put( zmmGuestRegLane64offset(zmmreg,laneno), e ) );
}

void putZMMRegLane32 ( UInt zmmreg, Int laneno, IRExpr* e ) {
   vassert(typeOfIRExpr(irsb->tyenv,e) == Ity_I32);
   stmt( IRStmt_Put( zmmGuestRegLane32offset(zmmreg,laneno), e ) );
}

void putZMMRegLane32F ( UInt zmmreg, Int laneno, IRExpr* e ) {
   vassert(typeOfIRExpr(irsb->tyenv,e) == Ity_F32);
   stmt( IRStmt_Put( zmmGuestRegLane32offset(zmmreg,laneno), e ) );
}

IRExpr* getKReg ( UInt kreg ) {
   return IRExpr_Get( kGuestRegOffset(kreg), Ity_I64 );
}

IRExpr* getKRegLane32 ( UInt kreg, Int laneno ) {
   return IRExpr_Get( kGuestRegLane32Offset(kreg, laneno), Ity_I32 );
}

IRExpr* getKRegLane16 ( UInt kreg, Int laneno ) {
   return IRExpr_Get( kGuestRegLane16Offset(kreg, laneno), Ity_I16 );
}

IRExpr* getKRegLane8 ( UInt kreg, Int laneno ) {
   return IRExpr_Get( kGuestRegLane8Offset(kreg, laneno), Ity_I8 );
}

void putKReg ( UInt kreg, IRExpr* e ) {
   vassert(typeOfIRExpr(irsb->tyenv,e) == Ity_I64);
   stmt( IRStmt_Put( kGuestRegOffset(kreg), e ) );
}

void putKRegLane32 ( UInt kreg, Int laneno, IRExpr* e ) {
   vassert(typeOfIRExpr(irsb->tyenv,e) == Ity_I32);
   stmt( IRStmt_Put( kGuestRegLane32Offset(kreg, laneno), e ) );
}

void putKRegLane16 ( UInt kreg, Int laneno, IRExpr* e ) {
   vassert(typeOfIRExpr(irsb->tyenv,e) == Ity_I16);
   stmt( IRStmt_Put( kGuestRegLane16Offset(kreg, laneno), e ) );
}

void putKRegLane8 ( UInt kreg, Int laneno, IRExpr* e ) {
   vassert(typeOfIRExpr(irsb->tyenv,e) == Ity_I8);
   stmt( IRStmt_Put( kGuestRegLane8Offset(kreg, laneno), e ) );
}

IRExpr* mkV512 ( UInt val ) {
   return IRExpr_Const(IRConst_V512(val));
}


void breakupV512toV256s ( IRTemp t512,
      /*OUTs*/ IRTemp* t1, IRTemp* t0 )
{
   vassert(t0 && *t0 == IRTemp_INVALID);
   vassert(t1 && *t1 == IRTemp_INVALID);
   *t0 = newTemp(Ity_V256);
   *t1 = newTemp(Ity_V256);
   assign(*t1, unop(Iop_V512toV256_1, mkexpr(t512)));
   assign(*t0, unop(Iop_V512toV256_0, mkexpr(t512)));
}
static void breakupV512toV128s ( IRTemp t512,
      /*OUTs*/ IRTemp* t3, IRTemp* t2, IRTemp* t1, IRTemp* t0 )
{
   IRTemp t256_1 = IRTemp_INVALID;
   IRTemp t256_0 = IRTemp_INVALID;
   breakupV512toV256s( t512, &t256_1, &t256_0 );
   breakupV256toV128s( t256_1, t3, t2 );
   breakupV256toV128s( t256_0, t1, t0 );
}

void breakupV512to64s ( IRTemp t512,
      /*OUTs*/
      IRTemp* t7, IRTemp* t6,
      IRTemp* t5, IRTemp* t4,
      IRTemp* t3, IRTemp* t2,
      IRTemp* t1, IRTemp* t0 )
{
   IRTemp t256_1 = IRTemp_INVALID;
   IRTemp t256_0 = IRTemp_INVALID;
   breakupV512toV256s( t512, &t256_1, &t256_0 );
   breakupV256to64s( t256_1, t7, t6, t5, t4 );
   breakupV256to64s( t256_0, t3, t2, t1, t0 );
}


// helpers start
static void Merge_dst( IRTemp dst, IRTemp* s, Int multiplier);
static void Split_arg (IRTemp src, IRTemp* s, Int multiplier);


static IRTemp m_Copy ( IRTemp src1, IRTemp unused, UInt ntimes ) {
   return src1;
}

// Right rotation: res = (src >> rotate) | (src << (width-rotate))
// Left rotation: res = (src << rotate) | (src >> (width-rotate))
static IRTemp m_RotateRight32(IRTemp src, IRTemp rotate, UInt unused) {
   IRExpr* rotate_m32;
   if (typeOfIRTemp(irsb->tyenv, rotate) == Ity_I8) { // imm version
      rotate_m32 = binop(Iop_And8, mkexpr(rotate), mkU8(0x1F));
   } else { // vector version
      rotate_m32 = binop(Iop_And8, unop(Iop_32to8, mkexpr(rotate)), mkU8(0x1F));
   }
   IRTemp res = newTemp(Ity_I32);
   assign( res, binop(Iop_Or32,
            binop(Iop_Shr32, mkexpr(src), rotate_m32),
            binop(Iop_Shl32, mkexpr(src), binop(Iop_Sub8, mkU8(0x20), rotate_m32))) );
   return res;
}
static IRTemp m_RotateLeft32(IRTemp src, IRTemp rotate, UInt unused) {
   IRExpr* rotate_m32;
   if (typeOfIRTemp(irsb->tyenv, rotate) == Ity_I8) {
      rotate_m32 = binop(Iop_And8, mkexpr(rotate), mkU8(0x1F));
   } else {
      rotate_m32 = binop(Iop_And8, unop(Iop_32to8, mkexpr(rotate)), mkU8(0x1F));
   }
   IRTemp res = newTemp(Ity_I32);
   assign( res, binop(Iop_Or32,
            binop(Iop_Shl32, mkexpr(src), rotate_m32),
            binop(Iop_Shr32, mkexpr(src), binop(Iop_Sub8, mkU8(0x20), rotate_m32))) );
   return res;
}
static IRTemp m_RotateRight64(IRTemp src, IRTemp rotate, UInt unused) {
   IRExpr* rotate_m64;
   if (typeOfIRTemp(irsb->tyenv, rotate) == Ity_I8) {
      rotate_m64 = binop(Iop_And8, mkexpr(rotate), mkU8(0x3F));
   } else {
      rotate_m64 = binop(Iop_And8, unop(Iop_64to8, mkexpr(rotate)), mkU8(0x3F));
   }
   IRTemp res = newTemp(Ity_I64);
   assign( res, binop(Iop_Or64,
            binop(Iop_Shr64, mkexpr(src), rotate_m64),
            binop(Iop_Shl64, mkexpr(src), binop(Iop_Sub8, mkU8(0x40), rotate_m64))) );
   return res;
}
static IRTemp m_RotateLeft64(IRTemp src, IRTemp rotate, UInt unused) {
   IRExpr* rotate_m64;
   if (typeOfIRTemp(irsb->tyenv, rotate) == Ity_I8) {
      rotate_m64 = binop(Iop_And8, mkexpr(rotate), mkU8(0x3F));
   } else {
      rotate_m64 = binop(Iop_And8, unop(Iop_64to8, mkexpr(rotate)), mkU8(0x3F));
   }
   IRTemp res = newTemp(Ity_I64);
   assign( res, binop(Iop_Or64,
            binop(Iop_Shl64, mkexpr(src), rotate_m64),
            binop(Iop_Shr64, mkexpr(src), binop(Iop_Sub8, mkU8(0x40), rotate_m64))) );
   return res;
}

static IRTemp m_PMULDQ (IRTemp src1, IRTemp src2, UInt unused_p) {
   return math_PMULDQ_128(src1, src2);
}
static IRTemp m_PMULUDQ (IRTemp src1, IRTemp src2, UInt unused_p) {
   return math_PMULUDQ_128(src1, src2);
}
static IRTemp m_VPERMILPS ( IRTemp sV, IRTemp unused, UInt imm8 ) {
   return math_VPERMILPS_128(sV, imm8);
}
static IRTemp m_PERMILPS_VAR ( IRTemp dataV, IRTemp ctrlV, UInt unused_p) {
   return math_PERMILPS_VAR_128(dataV, ctrlV);
}
static IRTemp m_PERMILPD_VAR ( IRTemp dataV, IRTemp ctrlV, UInt unused_p) {
   return math_PERMILPD_VAR_128(dataV, ctrlV);
}
static IRTemp m_VPERMILPD_128 ( IRTemp sV, IRTemp unused, UInt imm8 ) {
   IRTemp res = newTemp(Ity_V128);
   IRTemp s0 = IRTemp_INVALID, s1 = IRTemp_INVALID;
   breakupV128to64s( sV, &s1, &s0 );
   assign(res, binop(Iop_64HLtoV128,
            mkexpr((imm8 & (1<<1)) ? s1 : s0),
            mkexpr((imm8 & (1<<0)) ? s1 : s0)));
   return res;
}
static IRTemp m_VPERMILPD_256 ( IRTemp sV, IRTemp unused, UInt imm8 ) {
   IRTemp res = newTemp(Ity_V256);
   IRTemp s_val[4], s_res[4];
   for (Int j = 0; j < 4; j++) {
      s_val[j] = IRTemp_INVALID;
      s_res[j] = newTemp(Ity_I64);
   }
   Split_arg(sV, &s_val, 4);
   for (Int j = 0; j < 4; j++) {
      assign(s_res[j], (imm8 & (1<<j)) ?
            mkexpr(s_val[1+2*(Int)(j/2)]) :
            mkexpr(s_val[2*(Int)(j/2)]));
   }
   Merge_dst(res, s_res, 4);
   return res;
}
static IRTemp m_VPERMILPD_512 ( IRTemp sV, IRTemp unused, UInt imm8 ) {
   IRTemp res = newTemp(Ity_V512);
   IRTemp s_val[8], s_res[8];
   for (Int j = 0; j < 8; j++) {
      s_val[j] = IRTemp_INVALID;
      s_res[j] = newTemp(Ity_I64);
   }
   Split_arg(sV, &s_val, 8);
   for (Int j = 0; j < 8; j++) {
      assign(s_res[j], (imm8 & (1<<j)) ?
            mkexpr(s_val[1+2*(Int)(j/2)]) :
            mkexpr(s_val[2*(Int)(j/2)]));
   }
   Merge_dst(res, s_res, 8);
   return res;
}

static IRTemp m_NAND (IRTemp src1, IRTemp src2, UInt unused_p) {
   IRTemp res = newTemp(Ity_V128);
   assign( res, binop(Iop_AndV128, unop(Iop_NotV128, mkexpr(src1)), mkexpr(src2)) );
   return res;
}

static IRTemp m_EXTRACT_256(IRTemp src, IRTemp unused, UInt imm_8) {
   IRTemp res = newTemp(Ity_V256);
   assign(res, unop( (imm_8&1) ? Iop_V512toV256_1 : Iop_V512toV256_0, mkexpr(src)));
   return res;
}
static IRTemp m_EXTRACT_128(IRTemp src, IRTemp unused, UInt imm_8) {
   IRTemp res = newTemp(Ity_V128);
   if (typeOfIRTemp(irsb->tyenv, src) == Ity_V256) // 256-bit src
      assign(res, unop( (imm_8&1) ? Iop_V256toV128_1 : Iop_V256toV128_0, mkexpr(src)));
   else { // 512-bit src
      IRTemp s[4] = {IRTemp_INVALID, IRTemp_INVALID, IRTemp_INVALID, IRTemp_INVALID};
      Split_arg(src, s, 4);
      assign(res, mkexpr(s[imm_8&3]));
   }
   return res;
}

static IRTemp m_EXTRACT_64(IRTemp src, IRTemp unused, UInt imm_8) {
   IRTemp res = newTemp(Ity_I64);
   assign(res, unop( (imm_8&1) ? Iop_V128HIto64 : Iop_V128to64, mkexpr(src)));
   return res;
}
static IRTemp m_EXTRACT_32(IRTemp src, IRTemp unused, UInt imm_8) {
   IRTemp res = newTemp(Ity_I32);
   IRTemp s[4] = {IRTemp_INVALID, IRTemp_INVALID, IRTemp_INVALID, IRTemp_INVALID};
   Split_arg(src, s, 4);
   assign(res, mkexpr(s[imm_8&3]));
   return res;
}

static IRTemp m_INSERT_256(IRTemp src1, IRTemp src2, UInt imm_8) {
   IRTemp s[2] = {IRTemp_INVALID, IRTemp_INVALID};
   breakupV512toV256s(src1, &s[1], &s[0]);
   s[imm_8&1] = src2;
   IRTemp res = newTemp(Ity_V512);
   assign( res, binop(Iop_V256HLtoV512, mkexpr(s[1]), mkexpr(s[0])) );
   return res;
}
static IRTemp m_INSERT_128(IRTemp src1, IRTemp src2, UInt imm_8) {
   IRTemp res;
   IRTemp s[4] = {IRTemp_INVALID, IRTemp_INVALID, IRTemp_INVALID, IRTemp_INVALID};
   if (typeOfIRTemp(irsb->tyenv, src1) == Ity_V256) { // 256-bit dst
      breakupV256toV128s(src1, &s[1], &s[0]);
      s[imm_8&1] = src2;
      res = newTemp(Ity_V256);
      assign( res, binop(Iop_V128HLtoV256, mkexpr(s[1]), mkexpr(s[0])) );
   } else { // 512-bit dst
      breakupV512toV128s( src1, &s[3], &s[2], &s[1], &s[0] );
      s[imm_8&3] = src2;
      res = newTemp(Ity_V512);
      assign( res, binop(Iop_V256HLtoV512,
               binop(Iop_V128HLtoV256, mkexpr(s[3]), mkexpr(s[2])),
               binop(Iop_V128HLtoV256, mkexpr(s[1]), mkexpr(s[0]))) );
   }
   return res;
}

static IRTemp m_Broadcast ( IRTemp src, IRTemp unused, UInt ntimes ) {
   setTupleType( FullVectorMem );

   IRTemp s[ntimes];
   for (Int i = 0; i < ntimes; i++) {
      s[i] = src;
   }
   IRTemp res;
   switch ( sizeofIRType(typeOfIRTemp(irsb->tyenv, src)) * ntimes ) { // dst size in bytes
      case 64: res = newTemp(Ity_V512); break;
      case 32: res = newTemp(Ity_V256); break;
      case 16: res = newTemp(Ity_V128); break;
      case 8:  res = newTemp(Ity_I64);  break;
      default: vpanic("unsupported broadcast dst size");
   }
   Merge_dst(res, s, ntimes);
   return res;
}
static IRTemp m_BROADCASTMB2Q (IRTemp src, IRTemp unused, UInt ntimes ) {
   IRTemp mask_low_8_to_64 = newTemp(Ity_I64);
   assign(mask_low_8_to_64, unop(Iop_8Uto64, unop(Iop_64to8, mkexpr(src))) );
   return m_Broadcast(mask_low_8_to_64, unused, ntimes);
}
static IRTemp m_BROADCASTMW2D (IRTemp src, IRTemp unused, UInt ntimes ) {
   IRTemp mask_low_16_to_32 = newTemp(Ity_I32);
   assign(mask_low_16_to_32, unop(Iop_16Uto32, unop(Iop_64to16, mkexpr(src))) );
   return m_Broadcast(mask_low_16_to_32, unused, ntimes);
}
static IRTemp m_BroadcastQuart ( IRTemp src, IRTemp unused, UInt ntimes ) {
   // For cases when ntimes > 16, so merge from serial elements is too unwieldy
   // Broadcast into a quarter of dst, then merge 4 parts together
   IRTemp res_4, res;
   switch ( sizeofIRType(typeOfIRTemp(irsb->tyenv, src)) * ntimes ) { // dst size in bytes
      case 64:
         res = newTemp(Ity_V512);
         res_4 = newTemp(Ity_V128);
         res_4 = m_Broadcast(src, unused, ntimes/4);
         assign(res, binop(Iop_V256HLtoV512,
                  binop(Iop_V128HLtoV256, mkexpr(res_4), mkexpr(res_4)),
                  binop(Iop_V128HLtoV256, mkexpr(res_4), mkexpr(res_4))));
         break;
      case 32:
         res = newTemp(Ity_V256);
         res_4 = newTemp(Ity_I64);
         res_4 = m_Broadcast(src, unused, ntimes/4);
         assign(res, binop(Iop_V128HLtoV256,
                  binop(Iop_64HLtoV128, mkexpr(res_4), mkexpr(res_4)),
                  binop(Iop_64HLtoV128, mkexpr(res_4), mkexpr(res_4))));
         break;
      default: vpanic("unsupported broadcast_bw dst\n");
   }
   return res;
}

static IRTemp m_SQRT64 (IRTemp src, IRTemp unused_ir, UInt unused) {
   IRTemp res = newTemp(Ity_I64);
   assign( res, unop(Iop_ReinterpF64asI64, binop(Iop_SqrtF64,
               get_FAKE_roundingmode(),
               unop(Iop_ReinterpI64asF64, mkexpr(src)))) );
   return res;
}
static IRTemp m_SQRT32(IRTemp src, IRTemp unused_ir, UInt unused) {
   IRTemp res = newTemp(Ity_I32);
   assign( res, unop(Iop_ReinterpF32asI32, binop(Iop_F64toF32,
               get_FAKE_roundingmode(),
               binop(Iop_SqrtF64,
                  get_FAKE_roundingmode(),
                  unop(Iop_F32toF64, unop(Iop_ReinterpI32asF32, mkexpr(src)))))) );
   return res;
}

static IRTemp m_I32S_to_F32(IRTemp src, IRTemp unused_ir, UInt unused) {
   IRTemp res = newTemp(Ity_I32);
   assign( res, unop(Iop_ReinterpF32asI32, binop(Iop_F64toF32,
               get_FAKE_roundingmode(),
               unop(Iop_I32StoF64, mkexpr(src)))) );
   return res;
}
static IRTemp m_I32S_to_F64( IRTemp src, IRTemp unused_ir, UInt unused) {
   IRTemp res = newTemp(Ity_I64);
   assign(res, unop(Iop_ReinterpF64asI64, unop(Iop_I32StoF64, mkexpr(src))));
   return res;
}
static IRTemp m_I32U_to_F32(IRTemp src, IRTemp unused_ir, UInt unused) {
   IRTemp res = newTemp(Ity_I32);
   assign( res, unop(Iop_ReinterpF32asI32, binop(Iop_I32UtoF32,
               get_FAKE_roundingmode(),
               mkexpr(src))) );
   return res;
}
static IRTemp m_I32U_to_F64 (IRTemp src, IRTemp unused_ir, UInt unused) {
   IRTemp res = newTemp(Ity_I64);
   assign( res, unop(Iop_ReinterpF64asI64, binop(Iop_I64UtoF64,
               get_FAKE_roundingmode(),
               unop(Iop_32Uto64, mkexpr(src)))) );
   return res;
}
static IRTemp m_I64S_to_F32 (IRTemp src, IRTemp unused_ir, UInt unused) {
   IRTemp res = newTemp(Ity_I32);
   assign( res, unop(Iop_ReinterpF32asI32, binop(Iop_F64toF32,
               get_FAKE_roundingmode(),
               binop(Iop_I64StoF64,
                  get_FAKE_roundingmode(),
                  mkexpr(src)))) );
   return res;
}
static IRTemp m_I64S_to_F64 (IRTemp src, IRTemp unused_ir, UInt unused) {
   IRTemp res = newTemp(Ity_I64);
   assign( res, unop(Iop_ReinterpF64asI64, binop(Iop_I64StoF64,
               get_FAKE_roundingmode(),
               mkexpr(src))) );
   return res;
}
static IRTemp m_I64U_to_F32 (IRTemp src, IRTemp unused_ir, UInt unused) {
   IRTemp res = newTemp(Ity_I32);
   assign( res, unop(Iop_ReinterpF32asI32, binop( Iop_F64toF32,
               get_FAKE_roundingmode(),
               binop(Iop_I64UtoF64,
                     get_FAKE_roundingmode(),
                     mkexpr(src)))) );
   return res;
}
static IRTemp m_I64U_to_F64 (IRTemp src, IRTemp unused_ir, UInt unused) {
   IRTemp res = newTemp(Ity_I64);
   assign( res, unop(Iop_ReinterpF64asI64, binop( Iop_I64UtoF64,
               get_FAKE_roundingmode(),
               mkexpr(src))) );
   return res;
}
static IRTemp m_F32_to_I32S( IRTemp rmode, IRTemp src, UInt unused) {
   IRTemp res = newTemp(Ity_I32);
   assign( res, binop( Iop_F64toI32S,
            mkexpr(rmode),
            unop(Iop_F32toF64, unop(Iop_ReinterpI32asF32, mkexpr(src)))) );
   return res;
}
static IRTemp m_F32_to_I32U( IRTemp rmode, IRTemp src, UInt unused) {
   IRTemp res = newTemp(Ity_I32);
   assign( res, binop( Iop_F32toI32U,
            mkexpr(rmode),
            unop(Iop_ReinterpI32asF32, mkexpr(src))) );
   return res;
}
static IRTemp m_F32_to_I64S (IRTemp rmode, IRTemp src, UInt unused ) {
   IRTemp res = newTemp(Ity_I64);
   assign( res, binop( Iop_F64toI64S,
            mkexpr(rmode),
            unop(Iop_F32toF64, unop(Iop_ReinterpI32asF32, mkexpr(src)))) );
   return res;
}
static IRTemp m_F32_to_I64U (IRTemp rmode, IRTemp src, UInt unused ) {
   IRTemp res = newTemp(Ity_I64);
   assign( res, binop(Iop_F64toI64U,
            mkexpr(rmode),
            unop(Iop_F32toF64, unop(Iop_ReinterpI32asF32, mkexpr(src)))));
   return res;
}
static IRTemp m_F32_to_F64 (IRTemp rmode, IRTemp src, UInt unused) {
   IRTemp res = newTemp(Ity_I64);
   assign( res, unop(Iop_ReinterpF64asI64,
            unop(Iop_F32toF64, unop(Iop_ReinterpI32asF32, mkexpr(src)))) );
   return res;
}
static IRTemp m_F64_to_I32S (IRTemp rmode, IRTemp src, UInt unused) {
   IRTemp res = newTemp(Ity_I32);
   assign( res, binop( Iop_F64toI32S,
            mkexpr(rmode),
            unop(Iop_ReinterpI64asF64, mkexpr(src))) );
   return res;
}
static IRTemp m_F64_to_I32U (IRTemp rmode, IRTemp src, UInt unused) {
   IRTemp res = newTemp(Ity_I32);
   assign( res, binop(Iop_F32toI32U,
            mkexpr(rmode),
            binop(Iop_F64toF32,
                  mkexpr(rmode),
                  unop(Iop_ReinterpI64asF64, mkexpr(src)))));
   return res;
}
static IRTemp m_F64_to_I64S( IRTemp rmode, IRTemp src, UInt unused) {
   IRTemp res = newTemp(Ity_I64);
   assign( res, binop(Iop_F64toI64S,
            mkexpr(rmode),
            unop(Iop_ReinterpI64asF64, mkexpr(src))) );
   return res;
}
static IRTemp m_F64_to_I64U (IRTemp rmode, IRTemp src, UInt unused) {
   IRTemp res = newTemp(Ity_I64);
   assign( res, binop(Iop_F64toI64U,
            mkexpr(rmode),
            unop(Iop_ReinterpI64asF64, mkexpr(src))) );
   return res;
}
static IRTemp m_F64_to_F32 (IRTemp rmode, IRTemp src, UInt unused) {
   IRTemp res = newTemp(Ity_I32);
   assign( res, unop(Iop_ReinterpF32asI32, binop(Iop_F64toF32,
               mkexpr(rmode),
               unop(Iop_ReinterpI64asF64, mkexpr(src)))) );
   return res;
}

/* truncation conversions */
static IRTemp m_F32T_to_I64S (IRTemp rmode, IRTemp src, UInt unused ) {
   IRTemp res = newTemp(Ity_I64);
   assign( res, binop(Iop_F64TtoI64S,
            mkexpr(rmode),
            unop(Iop_ReinterpF64asI64, unop(Iop_F32toF64, unop(Iop_ReinterpI32asF32, mkexpr(src))))) );
   return res;
}
static IRTemp m_F32T_to_I64U (IRTemp rmode, IRTemp src, UInt unused ) {
   IRTemp res = newTemp(Ity_I64);
   assign( res, unop(Iop_F64TtoI64U, unop(Iop_ReinterpF64asI64,
               unop(Iop_F32toF64, unop(Iop_ReinterpI32asF32, mkexpr(src))))) );
   return res;
}
static IRTemp m_F64T_to_I32S (IRTemp rmode, IRTemp src, UInt unused) {
   IRTemp res = newTemp(Ity_I32);
   assign( res, unop(Iop_F32TtoI32S, unop(Iop_ReinterpF32asI32,
               binop( Iop_F64toF32,
                  mkexpr(rmode),
                  unop(Iop_ReinterpI64asF64, mkexpr(src))))) );
   return res;
}
static IRTemp m_F64T_to_I32U (IRTemp rmode, IRTemp src, UInt unused) {
   IRTemp res = newTemp(Ity_I32);
   assign( res, unop(Iop_F32TtoI32U, unop(Iop_ReinterpF32asI32,
            binop( Iop_F64toF32,
               mkexpr(rmode),
               unop(Iop_ReinterpI64asF64, mkexpr(src))))) );
   return res;
}

/* stauration conversions */
static IRTemp m_I16U_to_I8U_sat( IRTemp src, IRTemp unused, UInt unused_i) {
   IRTemp res = newTemp(Ity_I8);
   IRExpr* upper_bound = binop(Iop_CmpLE32U, mkU32(0xFF), unop(Iop_16Uto32, mkexpr(src)) );
   assign( res, IRExpr_ITE( upper_bound, mkU8(0xFF), // if 255 <= src return 255
            unop(Iop_16to8, mkexpr(src))) );
   return res;
}

static IRTemp m_I32S_to_I8S_sat( IRTemp src, IRTemp unused, UInt unused_i) {
   IRTemp res = newTemp(Ity_I8);
   IRExpr* lower_bound = binop(Iop_CmpLE32S, mkexpr(src), unop(Iop_8Sto32, mkU8(0x80)) );
   IRExpr* upper_bound = binop(Iop_CmpLE32S, unop(Iop_8Sto32, mkU8(0x7F)), mkexpr(src) );
   assign( res, IRExpr_ITE( lower_bound, mkU8(0x80), // if src <= -128 return -128
                IRExpr_ITE( upper_bound, mkU8(0x7F), // if 127 <= src  return 127
                   unop(Iop_32to8, mkexpr(src)))) );
   return res;
}
static IRTemp m_I32U_to_I8U_sat( IRTemp src, IRTemp unused, UInt unused_i) {
   IRTemp res = newTemp(Ity_I8);
   IRExpr* upper_bound = binop(Iop_CmpLE32U, unop(Iop_8Uto32, mkU8(0xFF)), mkexpr(src) );
   assign( res, IRExpr_ITE( upper_bound, mkU8(0xFF), // if 255 <= src return 255
            unop(Iop_32to8, mkexpr(src))) );
   return res;
}
static IRTemp m_I64S_to_I8S_sat( IRTemp src, IRTemp unused, UInt unused_i) {
   IRTemp res = newTemp(Ity_I8);
   IRExpr* lower_bound = binop(Iop_CmpLE64S, mkexpr(src), unop(Iop_8Sto64, mkU8(0x80)) );
   IRExpr* upper_bound = binop(Iop_CmpLE64S, unop(Iop_8Sto64, mkU8(0x7F)), mkexpr(src) );
   assign( res, IRExpr_ITE( lower_bound, mkU8(0x80),
                IRExpr_ITE( upper_bound, mkU8(0x7F),
                   unop(Iop_64to8, mkexpr(src)))) );
   return res;
}
static IRTemp m_I64U_to_I8U_sat( IRTemp src, IRTemp unused, UInt unused_i) {
   IRTemp res = newTemp(Ity_I8);
   IRExpr* upper_bound = binop(Iop_CmpLE64U, unop(Iop_8Uto64, mkU8(0xFF)), mkexpr(src) );
   assign( res, IRExpr_ITE( upper_bound, mkU8(0xFF),
            unop(Iop_64to8, mkexpr(src))) );
   return res;
}
static IRTemp m_I32S_to_I16S_sat( IRTemp src, IRTemp unused, UInt unused_i) {
   IRTemp res = newTemp(Ity_I16);
   IRExpr* lower_bound = binop(Iop_CmpLE32S, mkexpr(src), unop(Iop_16Sto32, mkU16(0x8000)) );
   IRExpr* upper_bound = binop(Iop_CmpLE32S, unop(Iop_16Sto32, mkU16(0x7FFF)), mkexpr(src) );
   assign( res, IRExpr_ITE( lower_bound, mkU16(0x8000),
                IRExpr_ITE( upper_bound, mkU16(0x7FFF),
                   unop(Iop_32to16, mkexpr(src)))) );
   return res;
}
static IRTemp m_I32U_to_I16U_sat( IRTemp src, IRTemp unused, UInt unused_i) {
   IRTemp res = newTemp(Ity_I16);
   IRExpr* upper_bound = binop(Iop_CmpLE32U, unop(Iop_16Uto32, mkU16(0xFFFF)), mkexpr(src) );
   assign( res, IRExpr_ITE( upper_bound, mkU16(0xFFFF), unop(Iop_32to16, mkexpr(src))) );
   return res;
}
static IRTemp m_I64S_to_I16S_sat( IRTemp src, IRTemp unused, UInt unused_i) {
   IRTemp res = newTemp(Ity_I16);
   IRExpr* lower_bound = binop(Iop_CmpLE64S, mkexpr(src), unop(Iop_16Sto64, mkU16(0x8000)) );
   IRExpr* upper_bound = binop(Iop_CmpLE64S, unop(Iop_16Sto64, mkU16(0x7FFF)), mkexpr(src) );
   assign( res, IRExpr_ITE( lower_bound, mkU16(0x8000),
                IRExpr_ITE( upper_bound, mkU16(0x7FFF),
                   unop(Iop_64to16, mkexpr(src)))) );
   return res;
}
static IRTemp m_I64U_to_I16U_sat( IRTemp src, IRTemp unused, UInt unused_i) {
   IRTemp res = newTemp(Ity_I16);
   IRExpr* upper_bound = binop(Iop_CmpLE64U, unop(Iop_16Uto64, mkU16(0xFFFF)), mkexpr(src) );
   assign( res, IRExpr_ITE( upper_bound, mkU16(0xFFFF), unop(Iop_64to16, mkexpr(src))) );
   return res;
}
static IRTemp m_I64S_to_I32S_sat( IRTemp src, IRTemp unused, UInt unused_i) {
   IRTemp res = newTemp(Ity_I32);
   IRExpr* lower_bound = binop(Iop_CmpLE64S, mkexpr(src), unop(Iop_32Sto64, mkU32(0x80000000)) );
   IRExpr* upper_bound = binop(Iop_CmpLE64S, unop(Iop_32Sto64, mkU32(0x7FFFFFFF)), mkexpr(src) );
   assign( res, IRExpr_ITE( lower_bound, mkU32(0x80000000),
                IRExpr_ITE( upper_bound, mkU32(0x7FFFFFFF),
                   unop(Iop_64to32, mkexpr(src)))) );
   return res;
}
static IRTemp m_I64U_to_I32U_sat( IRTemp src, IRTemp unused, UInt unused_i) {
   IRTemp res = newTemp(Ity_I32);
   IRExpr* upper_bound = binop(Iop_CmpLE64U, unop(Iop_32Uto64, mkU32(0xFFFFFFFF)), mkexpr(src) );
   assign( res, IRExpr_ITE( upper_bound, mkU32(0xFFFFFFFF), unop(Iop_64to32, mkexpr(src))) );
   return res;
}

static IRTemp m_ANDN (IRTemp src1, IRTemp src2, UInt unused ) {
   IROp and_op, not_op;
   switch (typeOfIRTemp(irsb->tyenv, src1)) {
      case Ity_V512: and_op = Iop_AndV512; not_op = Iop_NotV512; break;
      case Ity_V256: and_op = Iop_AndV256; not_op = Iop_NotV256; break;
      case Ity_V128: and_op = Iop_AndV128; not_op = Iop_NotV128; break;
      case Ity_I64:  and_op = Iop_And64;   not_op = Iop_Not64;   break;
      case Ity_I32:  and_op = Iop_And32;   not_op = Iop_Not32;   break;
      case Ity_I16:  and_op = Iop_And16;   not_op = Iop_Not16;   break;
      case Ity_I8:   and_op = Iop_And8;    not_op = Iop_Not8;    break;
      default: vpanic("invalid ANDN length");
   }
   IRTemp res = newTemp(typeOfIRTemp(irsb->tyenv, src1));
   assign( res, binop(and_op, mkexpr(src2), unop(not_op, mkexpr(src1))) );
   return res;
}
static IRTemp m_XNOR (IRTemp src1, IRTemp src2, UInt unused ) {
   IROp xor_op, not_op;
   switch (typeOfIRTemp(irsb->tyenv, src1)) {
      case Ity_I64: xor_op = Iop_Xor64; not_op = Iop_Not64; break;
      case Ity_I32: xor_op = Iop_Xor32; not_op = Iop_Not32; break;
      case Ity_I16: xor_op = Iop_Xor16; not_op = Iop_Not16; break;
      case Ity_I8:  xor_op = Iop_Xor8;  not_op = Iop_Not8;  break;
      default: vpanic("invalid XNOR length");
   }
   IRTemp res = newTemp(typeOfIRTemp(irsb->tyenv, src1));
   assign( res, unop(not_op, binop(xor_op, mkexpr(src2), mkexpr(src1))) );
   return res;
}
static IRTemp m_ORTEST (IRTemp src1, IRTemp src2, UInt unused ) {
   IRExpr* or_temp;
   ULong FULL = 0;
   switch (typeOfIRTemp(irsb->tyenv, src2)) {
      case Ity_I64:
         or_temp = binop(Iop_Or64, mkexpr(src1), mkexpr(src2));
         FULL = -1;
         break;
      case Ity_I32:
         or_temp = unop(Iop_32Uto64, binop(Iop_Or32, unop(Iop_64to32, mkexpr(src1)), mkexpr(src2)));
         FULL = 0xFFFFFFFF;
         break;
      case Ity_I16:
         or_temp = unop(Iop_16Uto64, binop(Iop_Or16, unop(Iop_64to16, mkexpr(src1)), mkexpr(src2)));
         FULL = 0xFFFF;
         break;
      case Ity_I8:
         or_temp = unop(Iop_8Uto64, binop(Iop_Or8, unop(Iop_64to8, mkexpr(src1)), mkexpr(src2)));
         FULL = 0xFF;
         break;
      default: vpanic("invalid ORTEST length");
   }
   IRExpr* carry_flag = IRExpr_ITE(
         binop(Iop_CmpEQ64, or_temp, mkU64(FULL)),
         mkU64(AMD64G_CC_MASK_C),
         mkU64(0));
   IRExpr* zero_flag = IRExpr_ITE(
         binop(Iop_CmpEQ64, or_temp, mkU64(0)),
         mkU64(AMD64G_CC_MASK_Z),
         mkU64(0));

   IRTemp old_rflags = newTemp(Ity_I64);
   assign(old_rflags, mk_amd64g_calculate_rflags_all());
   const ULong maskOSAP = AMD64G_CC_MASK_O | AMD64G_CC_MASK_S | AMD64G_CC_MASK_A | AMD64G_CC_MASK_P;
   const ULong maskCZ = AMD64G_CC_MASK_C | AMD64G_CC_MASK_Z;

   IRTemp new_rflags = newTemp(Ity_I64);
   assign(new_rflags, binop(Iop_Or64,
            binop(Iop_And64, mkexpr(old_rflags), mkU64(maskOSAP)),
            binop(Iop_And64,
               binop(Iop_Or64, carry_flag, zero_flag),
               mkU64(maskCZ))));

   /* Set all fields even though only OFFB_CC_DEP1 is used.
    * For some reason it does not work otherwise */
   stmt( IRStmt_Put( OFFB_CC_OP,   mkU64(AMD64G_CC_OP_COPY) ));
   stmt( IRStmt_Put( OFFB_CC_DEP1, mkexpr(new_rflags) ));
   stmt( IRStmt_Put( OFFB_CC_DEP2, mkU64(0) ));
   stmt( IRStmt_Put( OFFB_CC_NDEP, mkU64(0) ));

   return src1;
}
static IRTemp m_TEST (IRTemp src1, IRTemp src2, UInt unused ) {
   IRExpr *and_temp, *andN_temp;
   switch (typeOfIRTemp(irsb->tyenv, src2)) {
      case Ity_I64:
         and_temp  = binop(Iop_And64, mkexpr(src1), mkexpr(src2));
         andN_temp = binop(Iop_And64, unop(Iop_Not64, mkexpr(src1)), mkexpr(src2));
         break;
      case Ity_I32:
         and_temp  = unop(Iop_32Uto64, binop(Iop_And32, unop(Iop_64to32, mkexpr(src1)), mkexpr(src2)));
         andN_temp = unop(Iop_32Uto64, binop(Iop_And32, unop(Iop_Not32, unop(Iop_64to32, mkexpr(src1))), mkexpr(src2)));
         break;
      case Ity_I16:
         and_temp  = unop(Iop_16Uto64, binop(Iop_And16, unop(Iop_64to16, mkexpr(src1)), mkexpr(src2)));
         andN_temp = unop(Iop_16Uto64, binop(Iop_And16, unop(Iop_Not16, unop(Iop_64to16, mkexpr(src1))), mkexpr(src2)));
         break;
      case Ity_I8:
         and_temp  = unop(Iop_8Uto64, binop(Iop_And8, unop(Iop_64to8, mkexpr(src1)), mkexpr(src2)));
         andN_temp = unop(Iop_8Uto64, binop(Iop_And8, unop(Iop_Not8, unop(Iop_64to8, mkexpr(src1))), mkexpr(src2)));
         break;
      default: vpanic("invalid ORTEST length");
   }
   IRExpr* carry_flag = IRExpr_ITE(
         binop(Iop_CmpEQ64, andN_temp, mkU64(0)),
         mkU64(AMD64G_CC_MASK_C),
         mkU64(0));
   IRExpr* zero_flag = IRExpr_ITE(
         binop(Iop_CmpEQ64, and_temp, mkU64(0)),
         mkU64(AMD64G_CC_MASK_Z),
         mkU64(0));

   IRTemp old_rflags = newTemp(Ity_I64);
   assign(old_rflags, mk_amd64g_calculate_rflags_all());
   const ULong maskOSAP = AMD64G_CC_MASK_O | AMD64G_CC_MASK_S | AMD64G_CC_MASK_A | AMD64G_CC_MASK_P;
   const ULong maskCZ = AMD64G_CC_MASK_C | AMD64G_CC_MASK_Z;

   IRTemp new_rflags = newTemp(Ity_I64);
   assign(new_rflags, binop(Iop_Or64,
            binop(Iop_And64, mkexpr(old_rflags), mkU64(maskOSAP)),
            binop(Iop_And64,
               binop(Iop_Or64, carry_flag, zero_flag),
               mkU64(maskCZ))));

   /* Set all fields even though only OFFB_CC_DEP1 is used.
    * For some reason it does not work otherwise */
   stmt( IRStmt_Put( OFFB_CC_OP,   mkU64(AMD64G_CC_OP_COPY) ));
   stmt( IRStmt_Put( OFFB_CC_DEP1, mkexpr(new_rflags) ));
   stmt( IRStmt_Put( OFFB_CC_DEP2, mkU64(0) ));
   stmt( IRStmt_Put( OFFB_CC_NDEP, mkU64(0) ));

   return src1;
}

static IRTemp m_Shuffe128 (IRTemp src_to_low, IRTemp src_to_high, UInt imm8 ) {
   IRTemp sl[4], sh[4], res;
   for (Int i=0; i<4; i++) {
      sh[i] = IRTemp_INVALID;
      sl[i] = IRTemp_INVALID;
   }
   if (typeOfIRTemp(irsb->tyenv, src_to_low) == Ity_V512) {
      res = newTemp(Ity_V512);
      breakupV512toV128s( src_to_high, &sh[3], &sh[2], &sh[1], &sh[0] );
      breakupV512toV128s( src_to_low,  &sl[3], &sl[2], &sl[1], &sl[0] );
      assign( res, binop(Iop_V256HLtoV512,
               binop(Iop_V128HLtoV256, mkexpr(sh[(imm8>>6)&3]),  mkexpr(sh[(imm8>>4)&3])),
               binop(Iop_V128HLtoV256, mkexpr(sl[(imm8>>2)&3]),  mkexpr(sl[(imm8>>0)&3]))) );
   } else {
      res = newTemp(Ity_V256);
      breakupV256toV128s( src_to_high, &sh[1], &sh[0] );
      breakupV256toV128s( src_to_low,  &sl[1], &sl[0] );
      assign( res, binop(Iop_V128HLtoV256, mkexpr(sh[(imm8>>1)&1]), mkexpr(sl[(imm8>>0)&1])));
   }
   return res;
}

static IRTemp m_perm_imm_64x4 (IRTemp values, IRTemp unused, UInt imm8) {
   IRTemp res = newTemp(Ity_V256);
   IRTemp s_val[4], s_res[4];
   for (Int j = 0; j < 4; j++) {
      s_val[j] = IRTemp_INVALID;
      s_res[j] = newTemp(Ity_I64);
   }
   Split_arg(values, &s_val, 4);
   for (Int j = 0; j < 4; j++)
      assign(s_res[j], mkexpr( s_val[ (imm8>>(j*2)) & 3 ] ));
   Merge_dst(res, s_res, 4);
   return res;
}

static IRTemp m_PABS_64 (IRTemp src, IRTemp unused, UInt laneszB) {
   if (laneszB < 8)
      return math_PABS_MMX(src, laneszB);

   IRTemp res     = newTemp(Ity_I64);
   IRTemp srcNeg  = newTemp(Ity_I64);
   IRTemp negMask = newTemp(Ity_I64);
   IRTemp posMask = newTemp(Ity_I64);

   assign( negMask, binop(Iop_Sar64, mkexpr(src), mkU8(8*laneszB-1)) );
   assign( posMask, unop(Iop_Not64, mkexpr(negMask)) );
   assign( srcNeg,  binop(Iop_Sub64, mkU64(0), mkexpr(src)) );
   assign( res, binop(Iop_Or64,
            binop(Iop_And64, mkexpr(src),    mkexpr(posMask)),
            binop(Iop_And64, mkexpr(srcNeg), mkexpr(negMask)) ));
   return res;
}

static IRTemp m_SHUFPD_512 (IRTemp src1, IRTemp src2, UInt imm8) {
   IRTemp sVhi = IRTemp_INVALID, sVlo = IRTemp_INVALID;
   IRTemp dVhi = IRTemp_INVALID, dVlo = IRTemp_INVALID;
   breakupV512toV256s( src1, &sVhi, &sVlo );
   breakupV512toV256s( src2, &dVhi, &dVlo );
   IRTemp rVhi = math_SHUFPD_256(sVhi, dVhi, (imm8 >> 4) & 0xF);
   IRTemp rVlo = math_SHUFPD_256(sVlo, dVlo, imm8 & 0xF);
   IRTemp res  = newTemp(Ity_V512);
   assign(res, binop(Iop_V256HLtoV512, mkexpr(rVhi), mkexpr(rVlo)));
   return res;
}

static IRTemp m_PSHUFB (IRTemp src1, IRTemp src2, UInt unused) {
   return math_PSHUFB_XMM(src1, src2);
}
static IRTemp m_PMADDWD (IRTemp src1, IRTemp src2, UInt unused) {
   return math_PMADDWD_128(src1, src2);
}
static IRTemp m_PSADBW (IRTemp src1, IRTemp src2, UInt unused) {
   return math_PSADBW_128(src1, src2);
}

static IRTemp m_PSHUFD (IRTemp src, IRTemp unused, UInt imm8) {
   IRTemp res = newTemp(Ity_V128);
   IRTemp s[4] = { IRTemp_INVALID, IRTemp_INVALID, IRTemp_INVALID, IRTemp_INVALID};
   breakupV128to32s( src, &s[3], &s[2], &s[1], &s[0] );
   assign( res, mkV128from32s( s[(imm8>>6)&3], s[(imm8>>4)&3],
                              s[(imm8>>2)&3], s[(imm8>>0)&3] ) );
   return res;
}
static IRTemp m_PSHUFHW (IRTemp src, IRTemp unused, UInt imm8) {
   IRTemp res = newTemp(Ity_V128);
   IRTemp srcMut = newTemp(Ity_I64);
   assign(srcMut, unop(Iop_V128HIto64, mkexpr(src)));
   IRTemp s[4] = { IRTemp_INVALID, IRTemp_INVALID, IRTemp_INVALID, IRTemp_INVALID};
   breakup64to16s( srcMut, &s[3], &s[2], &s[1], &s[0] );
   IRTemp resMut = newTemp(Ity_I64);
   assign( resMut, mk64from16s( s[(imm8>>6)&3], s[(imm8>>4)&3],
                                s[(imm8>>2)&3], s[(imm8>>0)&3] ) );
   assign(res, binop(Iop_64HLtoV128, mkexpr(resMut), unop(Iop_V128to64, mkexpr(src))));
   return res;
}
static IRTemp m_PSHUFLW (IRTemp src, IRTemp unused, UInt imm8) {
   IRTemp res = newTemp(Ity_V128);
   IRTemp srcMut = newTemp(Ity_I64);
   assign(srcMut, unop(Iop_V128to64, mkexpr(src)));
   IRTemp s[4] = { IRTemp_INVALID, IRTemp_INVALID, IRTemp_INVALID, IRTemp_INVALID};
   breakup64to16s( srcMut, &s[3], &s[2], &s[1], &s[0] );
   IRTemp resMut = newTemp(Ity_I64);
   assign( resMut, mk64from16s( s[(imm8>>6)&3], s[(imm8>>4)&3],
            s[(imm8>>2)&3], s[(imm8>>0)&3] ) );
   assign(res, binop(Iop_64HLtoV128, unop(Iop_V128HIto64, mkexpr(src)), mkexpr(resMut)));
   return res;
}

static IRTemp m_PMULHRSW(IRTemp src1, IRTemp src2, UInt unused) {
   IRTemp res = newTemp(Ity_I64);
   assign(res, dis_PMULHRSW_helper(mkexpr(src1), mkexpr(src2)));
   return res;
}

static IRTemp m_MOVXBW (IRTemp src, IRTemp unused, UInt signextend) {
   // Implementation through Iop_8Uto16 requires split into 32 elements
   // As few instructions (so far) require it, going for a rip-off of dis_PMOVxXBW_128 instead
   IRTemp res = newTemp(Ity_V128);
   if (signextend)
      assign(res, binop(Iop_SarN16x8, binop(Iop_ShlN16x8,
                  binop( Iop_InterleaveLO8x16, mkV128(0), unop(Iop_64UtoV128, mkexpr(src)) ),
                  mkU8(8)), mkU8(8)));
   else
      assign(res, binop(Iop_InterleaveLO8x16, mkV128(0), unop(Iop_64UtoV128, mkexpr(src))));
   return res;
}

static IRTemp m_VPMOVB2M(IRTemp src, IRTemp unused, UInt vl) {
   IRTemp res, cmp;
   IRTemp elem = newTemp(Ity_I8);
   assign(elem, mkU8(1<<7));
   IROp cmp_op;
   switch (vl) {
      case 0:
         res = newTemp(Ity_V128);
         cmp_op = Iop_Cmp8Ux16;
         cmp = m_Broadcast(elem, unused, 16);
         break;
      case 1:
         res = newTemp(Ity_V256);
         cmp_op = Iop_Cmp8Ux32;
         cmp = m_BroadcastQuart(elem, unused, 32);
         break;
      case 2:
         res = newTemp(Ity_V512);
         cmp_op = Iop_Cmp8Ux64;
         cmp = m_BroadcastQuart(elem, unused, 64);
         break;
   }
   IRTemp dummy = newTemp(Ity_I64);
   assign(dummy, mkU64(0));
   assign(res, qop(cmp_op, mkexpr(dummy), mkexpr(cmp), mkexpr(src), mkU8(0x1)));
   return res;
}
static IRTemp m_VPMOVW2M(IRTemp src, IRTemp unused, UInt vl) {
   IRTemp res, cmp;
   IRTemp elem = newTemp(Ity_I16);
   assign(elem, mkU16(1<<15));
   IROp cmp_op;
   switch (vl) {
      case 0:
         res = newTemp(Ity_V128);
         cmp_op = Iop_Cmp16Ux8;
         cmp = m_Broadcast(elem, unused, 8);
         break;
      case 1:
         res = newTemp(Ity_V256);
         cmp_op = Iop_Cmp16Ux16;
         cmp = m_Broadcast(elem, unused, 16);
         break;
      case 2:
         res = newTemp(Ity_V512);
         cmp_op = Iop_Cmp16Ux32;
         cmp = m_BroadcastQuart(elem, unused, 32);
         break;
   }
   IRTemp dummy = newTemp(Ity_I64);
   assign(dummy, mkU64(0));
   assign(res, qop(cmp_op, mkexpr(dummy), mkexpr(cmp), mkexpr(src), mkU8(0x1)));
   return res;
}
static IRTemp m_VPMOVD2M(IRTemp src, IRTemp unused, UInt vl) {
   IRTemp res, cmp;
   IRTemp elem = newTemp(Ity_I32);
   assign(elem, mkU32(1ULL<<31));
   IROp cmp_op;
   switch (vl) {
      case 0:
         res = newTemp(Ity_V128);
         cmp_op = Iop_Cmp32Ux4;
         cmp = m_Broadcast(elem, unused, 4);
         break;
      case 1:
         res = newTemp(Ity_V256);
         cmp_op = Iop_Cmp32Ux8;
         cmp = m_Broadcast(elem, unused, 8);
         break;
      case 2:
         res = newTemp(Ity_V512);
         cmp_op = Iop_Cmp32Ux16;
         cmp = m_Broadcast(elem, unused, 16);
         break;
   }
   IRTemp dummy = newTemp(Ity_I64);
   assign(dummy, mkU64(0));
   assign(res, qop(cmp_op, mkexpr(dummy), mkexpr(cmp), mkexpr(src), mkU8(0x1)));
   return res;
}
static IRTemp m_VPMOVQ2M(IRTemp src, IRTemp unused, UInt vl) {
   IRTemp res, cmp;
   IRTemp elem = newTemp(Ity_I64);
   assign(elem, mkU64(1ULL<<63));
   IROp cmp_op;
   switch (vl) {
      case 0:
         res = newTemp(Ity_V128);
         cmp_op = Iop_Cmp64Ux2;
         cmp = m_Broadcast(elem, unused, 2);
         break;
      case 1:
         res = newTemp(Ity_V256);
         cmp_op = Iop_Cmp64Ux4;
         cmp = m_Broadcast(elem, unused, 4);
         break;
      case 2:
         res = newTemp(Ity_V512);
         cmp_op = Iop_Cmp64Ux8;
         cmp = m_Broadcast(elem, unused, 8);
         break;
   }
   IRTemp dummy = newTemp(Ity_I64);
   assign(dummy, mkU64(0));
   assign(res, qop(cmp_op, mkexpr(dummy), mkexpr(cmp), mkexpr(src), mkU8(0x1)));
   return res;
}

static IRTemp m_Scale32(IRTemp src1, IRTemp src2, UInt unused) {
   IRExpr* pow_floor = binop( Iop_I32UtoF32,
         get_FAKE_roundingmode(),
         binop(Iop_F32toI32U,
            mkU32(Irrm_NegINF),
            unop(Iop_ReinterpI32asF32, mkexpr(src2))));

   IRTemp res = newTemp(Ity_I32);
   assign(res, unop(Iop_ReinterpF32asI32, binop(Iop_F64toF32,
               get_evex_roundingmode(),
               triop(Iop_ScaleF64, get_evex_roundingmode(),
                  unop(Iop_F32toF64, unop(Iop_ReinterpI32asF32, mkexpr(src1))),
                  unop(Iop_F32toF64, pow_floor)))) );
   return res;
}
static IRTemp m_Scale64(IRTemp src1, IRTemp src2, UInt unused) {
   IRExpr* pow_floor = binop( Iop_I64UtoF64,
         get_FAKE_roundingmode(),
         binop(Iop_F64toI64U,
            mkU32(Irrm_NegINF),
            unop(Iop_ReinterpI64asF64, mkexpr(src2))));

   IRTemp res = newTemp(Ity_I64);
   assign(res, unop(Iop_ReinterpF64asI64,
            triop(Iop_ScaleF64, mkU32(Irrm_NegINF),
               unop(Iop_ReinterpI64asF64, mkexpr(src1)),
               pow_floor)));
   return res;
}
static IRTemp m_kShiftR(IRTemp src, IRTemp unused, UInt imm8) {
   IRTemp res = newTemp(typeOfIRTemp(irsb->tyenv, src));
   IROp shiftR;
   IRExpr *zero;
   UInt max;
   switch (typeOfIRTemp(irsb->tyenv, src)) {
      case Ity_I8:  shiftR = Iop_Shr8;  max = 0x7;  zero = mkU8(0);  break;
      case Ity_I16: shiftR = Iop_Shr16; max = 0xF;  zero = mkU16(0); break;
      case Ity_I32: shiftR = Iop_Shr32; max = 0x1F; zero = mkU32(0); break;
      case Ity_I64: shiftR = Iop_Shr64; max = 0x3F; zero = mkU64(0); break;
      default: vpanic("wrong mask shift");
   }
   assign( res, (imm8 > max) ? zero : binop(shiftR, mkexpr(src), mkU8(imm8)) );
   return res;
}
static IRTemp m_kShiftL(IRTemp src, IRTemp unused, UInt imm8) {
   IRTemp res = newTemp(typeOfIRTemp(irsb->tyenv, src));
   IROp shiftL;
   IRExpr *zero;
   UInt max;
   switch (typeOfIRTemp(irsb->tyenv, src)) {
      case Ity_I8:  shiftL = Iop_Shl8;  max = 0x7;  zero = mkU8(0);  break;
      case Ity_I16: shiftL = Iop_Shl16; max = 0xF;  zero = mkU16(0); break;
      case Ity_I32: shiftL = Iop_Shl32; max = 0x1F; zero = mkU32(0); break;
      case Ity_I64: shiftL = Iop_Shl64; max = 0x3F; zero = mkU64(0); break;
      default: vpanic("wrong mask shift");
   }
   assign( res, (imm8 > max) ? zero : binop(shiftL, mkexpr(src), mkU8(imm8)) );
   return res;
}

// helpers end


static void breakup32to8s ( IRTemp t32,
      IRTemp* t3, IRTemp* t2, IRTemp* t1, IRTemp* t0 )
{
   IRTemp hi16 = newTemp(Ity_I16);
   IRTemp lo16 = newTemp(Ity_I16);
   assign( hi16, unop(Iop_32HIto16, mkexpr(t32)) );
   assign( lo16, unop(Iop_32to16,   mkexpr(t32)) );

   vassert(t0 && *t0 == IRTemp_INVALID);
   vassert(t1 && *t1 == IRTemp_INVALID);
   vassert(t2 && *t2 == IRTemp_INVALID);
   vassert(t3 && *t3 == IRTemp_INVALID);

   *t0 = newTemp(Ity_I8);
   *t1 = newTemp(Ity_I8);
   *t2 = newTemp(Ity_I8);
   *t3 = newTemp(Ity_I8);
   assign( *t0, unop(Iop_16to8,   mkexpr(lo16)) );
   assign( *t1, unop(Iop_16HIto8, mkexpr(lo16)) );
   assign( *t2, unop(Iop_16to8,   mkexpr(hi16)) );
   assign( *t3, unop(Iop_16HIto8, mkexpr(hi16)) );
}
static void breakup64to32s ( IRTemp t64, IRTemp* t1, IRTemp* t0 )
{
   vassert(t0 && *t0 == IRTemp_INVALID);
   vassert(t1 && *t1 == IRTemp_INVALID);
   *t0 = newTemp(Ity_I32);
   *t1 = newTemp(Ity_I32);
   assign( *t0, unop(Iop_64to32,   mkexpr(t64)) );
   assign( *t1, unop(Iop_64HIto32, mkexpr(t64)) );
}

static void breakup32to16s ( IRTemp t32, IRTemp* t1, IRTemp* t0 )
{
   vassert(t0 && *t0 == IRTemp_INVALID);
   vassert(t1 && *t1 == IRTemp_INVALID);
   *t0 = newTemp(Ity_I16);
   *t1 = newTemp(Ity_I16);
   assign( *t0, unop(Iop_32to16,   mkexpr(t32)) );
   assign( *t1, unop(Iop_32HIto16, mkexpr(t32)) );
}
static void breakup16to8s ( IRTemp t16, IRTemp* t1, IRTemp* t0 )
{
   vassert(t0 && *t0 == IRTemp_INVALID);
   vassert(t1 && *t1 == IRTemp_INVALID);
   *t0 = newTemp(Ity_I8);
   *t1 = newTemp(Ity_I8);
   assign( *t0, unop(Iop_16to8,   mkexpr(t16)) );
   assign( *t1, unop(Iop_16HIto8, mkexpr(t16)) );
}



#include <guest_AVX512.h>

IRExpr* mask_expr(Prefix pfx, IRExpr* unmasked, IRExpr* original, Int ins_id)
{

   UInt mask = getEvexMask();
   if (!mask) {
      DIP("\n");
      return unmasked;
   }
   DIP("{k%x} \n", mask);

   IRType dst_type   = typeOfIRExpr(irsb->tyenv, unmasked);
   IRTemp res        = newTemp(dst_type);
   UInt   tuple_type = getTupleType();
   // Workaround for EXPAND and COMPRESS instructions: they use different tuple type
   // for memory displacemnet, but mask entire reister
   if (INS_ARR[ins_id].opcode >= 0x88 && INS_ARR[ins_id].opcode <= 0x8B)
      tuple_type = FullVector;

   if ( INS_ARR[ins_id].args[0] == _kG ) {
      // Destination is a full mask register: each opmask bit corresponds
      // to a result bit; only zeroing mode available
      return (binop(Iop_And64, unmasked, getKReg(mask)));
   }

   /* Serial case. Result is a vector, but only the last element is masked. */
   if (tuple_type == Tuple1Scalar) {
      // Only the lowest mask bit and vector elements are used
      IRTemp cond = newTemp(Ity_I1);
      assign( cond, unop(Iop_64to1, getKReg(mask)) );

      IRExpr* masked = NULL;
      switch (dst_type) {
         case Ity_V128: {
            if (getDstW() == W_32)
               masked = binop(Iop_SetV128lo32, unmasked, getZeroMode() ? mkU32(0): unop(Iop_V128to32, original));
            else
               masked = binop(Iop_SetV128lo64, unmasked, getZeroMode() ? mkU64(0): unop(Iop_V128to64, original));
            break;
         }
         case Ity_I64: masked = getZeroMode() ? mkU64(0): original; break;
         case Ity_I32: masked = getZeroMode() ? mkU32(0): original; break;
         default: vpanic("unknown IR type");
      }
      assign(res, IRExpr_ITE(mkexpr(cond), unmasked, masked));
      return (mkexpr(res));
   }

   /* Vector cases */
   /* Set auxiliary IRs depending on the destination width */
   IROp bitAnd = Iop_INVALID;
   IROp bitOr  = Iop_INVALID;
   IROp bitNot = Iop_INVALID;
   switch (dst_type) {
      case Ity_V512: bitAnd=Iop_AndV512; bitOr=Iop_OrV512; bitNot=Iop_NotV512; break;
      case Ity_V256: bitAnd=Iop_AndV256; bitOr=Iop_OrV256; bitNot=Iop_NotV256; break;
      case Ity_V128: bitAnd=Iop_AndV128; bitOr=Iop_OrV128; bitNot=Iop_NotV128; break;
      case Ity_I64:  bitAnd=Iop_And64;   bitOr=Iop_Or64;   bitNot=Iop_Not64;   break;
      case Ity_I32:  bitAnd=Iop_And32;   bitOr=Iop_Or32;   bitNot=Iop_Not32;   break;
      case Ity_I16:  bitAnd=Iop_And16;   bitOr=Iop_Or16;   bitNot=Iop_Not16;   break;
      default: vpanic("unsupported mask length");
   }

   /* Do the masking */
   /* Determine granularity. mask_vec expands each mask bit to the element size (0->0, 1->0xFF..FF) */
   IRTemp mask_vec = newTemp(dst_type);
   switch (dst_type) {
      case Ity_V512:
         assign(mask_vec, binop(Iop_ExpandBitsToV512, getKReg(mask), mkU8(getDstW())));
         break;
      case Ity_V256:
         assign(mask_vec, binop(Iop_ExpandBitsToV256, getKReg(mask), mkU8(getDstW())));
         break;
      case Ity_V128:
         assign(mask_vec, binop(Iop_ExpandBitsToV128, getKReg(mask), mkU8(getDstW())));
         break;
      case Ity_I64:
         assign(mask_vec, binop(Iop_ExpandBitsToInt, getKReg(mask), mkU8(getDstW())));
         break;
      case Ity_I32:
         assign(mask_vec, unop (Iop_64to32,
                  binop(Iop_ExpandBitsToInt, getKReg(mask), mkU8(getDstW()))));
         break;
      case Ity_I16:
         assign(mask_vec, unop (Iop_64to16,
                  binop(Iop_ExpandBitsToInt, getKReg(mask), mkU8(getDstW()))));
         break;
      default: vpanic("weird mask");
   }

   if (getZeroMode()) {
      //zero out unmasked elements
      assign(res, binop(bitAnd, unmasked, mkexpr(mask_vec)));
   }
   else {
      // zero out unmasked elements, fill the gaps with the original values
      assign(res, binop(bitOr,
               binop(bitAnd, unmasked, mkexpr(mask_vec)),
               binop(bitAnd, original, unop(bitNot, mkexpr(mask_vec)))));
   }
   return (mkexpr(res));
}

static Long Get_Instr_Args(
      const VexAbiInfo* vbi, Prefix pfx, Long delta,
      const Int ins_id, IRTemp* arg, Int* argN, Int* int_val,
      Int rG, Int rV, Int *rE_back, IRTemp* addr, UChar modrm)
{
   HChar dis_buf[50];
   Int alen = 0;
   Int rE = 0;

   if (epartIsReg(modrm)) {
      rE = eregOfRexRM32(pfx, modrm);
      *rE_back = rE;
      delta += 1;
   } else {
      // We need to read memory address and update delta before reading imm8 at the new delta
      // To parse memory address, disAMode needs to know if imm8 is present
      // Hence this ugly workaround.
      Int extra_bytes = 0;
      for (int i=0; i<5; i++)
         if (INS_ARR[ins_id].args[i] == _imm8)
            extra_bytes = 1;
      *addr = disAMode( &alen, vbi, pfx, delta, dis_buf, extra_bytes );
      delta += alen;
   }

#define REG_ARG(get, name, reg)  {DIP("%s ",name(reg)); assign(arg[i],get(reg));}
   for (int i=0; i<5; i++) {
      enum op_encoding op_code = INS_ARR[ins_id].args[i];
      if (op_code == _none) {
         (*argN)--; // not an argument
         continue;
      }

      IRType op_type = INS_ARR[ins_id].arg_type[i];
      arg[i] = newTemp(op_type);

      // imm8 is a special case
      if (op_code == _imm8) {
         *int_val = getUChar(delta++);
         assign(arg[i], mkU8(*int_val));
         DIP("imm8 %x", *int_val);
         continue;
      }

      if (op_code == _rmode) {
         assign(arg[i], mkU32(get_evex_roundingmode()));
         DIP("rmode %x", get_evex_roundingmode());
         continue;
      }

      // memory reference
      switch (op_code) {
         case _rmE: case _imE: case _kmE:
            if (epartIsReg(modrm)) break; // else fallthrough
         case _mG: case _mE: {
            if (getEvexb()) {
               IRType width = getDstW() ? Ity_I64 : Ity_I32;
               IRTemp elem = newTemp(width);
               assign (elem, loadLE(width, mkexpr(*addr)));
               arg[i] = m_Broadcast(elem, elem, sizeofIRType(op_type)/sizeofIRType(width));
               DIP("%s 1toN", dis_buf);
               continue;
            } else {
               assign( arg[i], loadLE( op_type, mkexpr(*addr)) );
               DIP("%s ", dis_buf);
               continue;
            }
         }
         default: break;
      }

      int reg = 0;
      switch (op_code) {
         case _rG: case _iG: case _kG: reg = rG; break;
         case _rV: case _kV:           reg = rV; break;
         case _rmE: case _imE: case _kmE:
         case _rE: case _iE: case _kE: reg = rE; break;
         case _kM:                     reg = getEvexMask(); break;
         default: vpanic("!");
      }
      // scalar register
      if ((op_code == _iG) || (op_code == _iE) || (op_code == _imE)) {
         switch (op_type) {
            case Ity_I8:
               assign(arg[i], unop(Iop_16to8, getIReg16(reg)));
               DIP("%s ", nameIReg16(reg));
               break;
            case Ity_I16: REG_ARG (getIReg16, nameIReg16, reg); break;
            case Ity_I32: REG_ARG (getIReg32, nameIReg32, reg); break;
            case Ity_I64: REG_ARG (getIReg64, nameIReg64, reg); break;
            default: vpanic("\nunimplemented scalar register size\n");
         }
         continue;
      }
      // mask register
      if ((op_code == _kE) || (op_code == _kM) || (op_code == _kG) ||
            (op_code == _kV) || (op_code == _kmE)) {
         DIP("%s ", nameKReg(reg));
         switch (op_type) {
            case Ity_I64: assign(arg[i], getKReg(reg)); break;
            case Ity_I32: assign(arg[i], getKRegLane32(reg, 0)); break;
            case Ity_I16: assign(arg[i], getKRegLane16(reg, 0)); break;
            case Ity_I8:  assign(arg[i], getKRegLane8(reg, 0)); break;
            default:      vpanic ("unknown mask length\n");
         }
         continue;
      }

      // vector register as the only remaining option
      switch (op_type) {
         case Ity_V512: REG_ARG (getZMMReg, nameZMMReg, reg); break;
         case Ity_V256: REG_ARG (getYMMReg, nameYMMReg, reg); break;
         case Ity_V128: REG_ARG (getXMMReg, nameXMMReg, reg); break;
         case Ity_I64:  assign(arg[i], getXMMRegLane64(reg, 0));
                        DIP("%s ", nameXMMReg(reg));
                        break;
         case Ity_I32:  assign(arg[i], getXMMRegLane32(reg, 0));
                        DIP("%s ", nameXMMReg(reg));
                        break;
         case Ity_I16:  assign(arg[i], getXMMRegLane16(reg, 0));
                        DIP("%s ", nameXMMReg(reg));
                        break;
         default:       vpanic ("unknown vector length\n");
      }
   }
#undef REG_ARG
   DIP("   arity %x,", *argN);
   return delta;
}

static void Split_arg (IRTemp src, IRTemp* s, Int multiplier) {
   switch (multiplier) {
      case 1: s[0] = src; return;
      case 2:
              switch( typeOfIRTemp(irsb->tyenv, src) ) {
                 case Ity_V512: breakupV512toV256s( src, &s[1], &s[0] ); return;
                 case Ity_V256: breakupV256toV128s( src, &s[1], &s[0] ); return;
                 case Ity_V128: breakupV128to64s  ( src, &s[1], &s[0] ); return;
                 case Ity_I64:  breakup64to32s    ( src, &s[1], &s[0] ); return;
                 case Ity_I32:  breakup32to16s    ( src, &s[1], &s[0] ); return;
                 case Ity_I16:  breakup16to8s     ( src, &s[1], &s[0] ); return;
                 default: vpanic(" cannot split in 2\n");
              }
              break;
      case 4:
              switch( typeOfIRTemp(irsb->tyenv, src) ) {
                 case Ity_V512: breakupV512toV128s( src, &s[3], &s[2], &s[1], &s[0] ); return;
                 case Ity_V256: breakupV256to64s  ( src, &s[3], &s[2], &s[1], &s[0] ); return;
                 case Ity_V128: breakupV128to32s  ( src, &s[3], &s[2], &s[1], &s[0] ); return;
                 case Ity_I64:  breakup64to16s    ( src, &s[3], &s[2], &s[1], &s[0] ); return;
                 case Ity_I32:  breakup32to8s     ( src, &s[3], &s[2], &s[1], &s[0] ); return;
                 default: vpanic(" cannot split in 4\n");
              }
              break;
      case 8:
              switch( typeOfIRTemp(irsb->tyenv, src) ) {
                 case Ity_V512: breakupV512to64s ( src, &s[7], &s[6], &s[5], &s[4], &s[3], &s[2], &s[1], &s[0] ); return;
                 case Ity_V256: breakupV256to32s ( src, &s[7], &s[6], &s[5], &s[4], &s[3], &s[2], &s[1], &s[0] ); return;
                 case Ity_V128: {
                    IRTemp tmpHi = IRTemp_INVALID;
                    IRTemp tmpLo = IRTemp_INVALID;
                    breakupV128to64s(src, &tmpHi, &tmpLo);
                    breakup64to16s( tmpHi, &s[7], &s[6], &s[5], &s[4] );
                    breakup64to16s( tmpLo, &s[3], &s[2], &s[1], &s[0] );
                    return;
                 }
                 case Ity_I64: {
                    IRTemp tmpHi = IRTemp_INVALID;
                    IRTemp tmpLo = IRTemp_INVALID;
                    breakup64to32s(src, &tmpHi, &tmpLo);
                    breakup32to8s( tmpHi, &s[7], &s[6], &s[5], &s[4] );
                    breakup32to8s( tmpLo, &s[3], &s[2], &s[1], &s[0] );
                    return;
                 }
                 default: vpanic(" cannot split in 8\n");
              }
              break;
      case 16:
              switch( typeOfIRTemp(irsb->tyenv, src) ) {
                 case Ity_V512: {
                    IRTemp tmpHi = IRTemp_INVALID;
                    IRTemp tmpLo = IRTemp_INVALID;
                    breakupV512toV256s(src, &tmpHi, &tmpLo);
                    breakupV256to32s (tmpHi, &s[15], &s[14], &s[13], &s[12], &s[11], &s[10], &s[9], &s[8] );
                    breakupV256to32s (tmpLo, &s[7], &s[6], &s[5], &s[4], &s[3], &s[2], &s[1], &s[0] );
                    return;
                 }
                 case Ity_V256: {
                    IRTemp tmp1, tmp2, tmp3, tmp4;
                    tmp1 = tmp2 = tmp3 = tmp4 = IRTemp_INVALID;
                    breakupV256to64s( src, &tmp4, &tmp3, &tmp2, &tmp1);
                    breakup64to16s( tmp4, &s[15], &s[14], &s[13], &s[12] );
                    breakup64to16s( tmp3, &s[11], &s[10], &s[9],  &s[8]  );
                    breakup64to16s( tmp2, &s[7],  &s[6],  &s[5],  &s[4]  );
                    breakup64to16s( tmp1, &s[3],  &s[2],  &s[1],  &s[0]  );
                    return;
                 }
                 case Ity_V128: {
                    IRTemp tmp1, tmp2, tmp3, tmp4;
                    tmp1 = tmp2 = tmp3 = tmp4 = IRTemp_INVALID;
                    breakupV128to32s( src, &tmp4, &tmp3, &tmp2, &tmp1);
                    breakup32to8s( tmp4, &s[15], &s[14], &s[13], &s[12] );
                    breakup32to8s( tmp3, &s[11], &s[10], &s[9],  &s[8]  );
                    breakup32to8s( tmp2, &s[7],  &s[6],  &s[5],  &s[4]  );
                    breakup32to8s( tmp1, &s[3],  &s[2],  &s[1],  &s[0]  );
                    return;
                 }
                 default: vpanic(" cannot split in 16\n");
              }
              break;
      case 32: /* yeesh. for vector shifts only. */
              switch( typeOfIRTemp(irsb->tyenv, src) ) {
                 case Ity_V512: {
                    IRTemp t[8];
                    for (Int i=0; i<8; i++)
                       t[i] = IRTemp_INVALID;
                    breakupV512to64s ( src, &t[7], &t[6], &t[5], &t[4], &t[3], &t[2], &t[1], &t[0] );
                    breakup64to16s( t[7], &s[31], &s[30], &s[29], &s[28] );
                    breakup64to16s( t[6], &s[27], &s[26], &s[25], &s[24] );
                    breakup64to16s( t[5], &s[23], &s[22], &s[21], &s[20] );
                    breakup64to16s( t[4], &s[19], &s[18], &s[17], &s[16] );
                    breakup64to16s( t[3], &s[15], &s[14], &s[13], &s[12] );
                    breakup64to16s( t[2], &s[11], &s[10], &s[9],  &s[8]  );
                    breakup64to16s( t[1], &s[7],  &s[6],  &s[5],  &s[4]  );
                    breakup64to16s( t[0], &s[3],  &s[2],  &s[1],  &s[0]  );
                    return;
                 }
                 default: vpanic("You don'ti want to split in 32.\n");
              }
              break;
      default: vex_printf("%x ", multiplier);
               vpanic("unsupported multiplier\n");
   }
}

static void Merge_dst( IRTemp dst, IRTemp* s, Int multiplier) {
   switch (multiplier) {
      case 1:
         assign(dst, mkexpr(s[0])); break;
      case 2:
         switch( typeOfIRTemp(irsb->tyenv, dst) ) {
            case Ity_V512: assign(dst, binop(Iop_V256HLtoV512, mkexpr(s[1]), mkexpr(s[0]))); return;
            case Ity_V256: assign(dst, binop(Iop_V128HLtoV256, mkexpr(s[1]), mkexpr(s[0]))); return;
            case Ity_V128: assign(dst, binop(Iop_64HLtoV128,   mkexpr(s[1]), mkexpr(s[0]))); return;
            case Ity_I64:  assign(dst, binop(Iop_32HLto64,     mkexpr(s[1]), mkexpr(s[0]))); return;
            case Ity_I32:  assign(dst, binop(Iop_16HLto32,     mkexpr(s[1]), mkexpr(s[0]))); return;
            case Ity_I16:  assign(dst, binop(Iop_8HLto16,      mkexpr(s[1]), mkexpr(s[0]))); return;
            default: vpanic("unsupported 2x multiplier\n");
         }
      case 4:
#define merge_4( iop_half, iop_quarter ) \
         assign( dst, binop( iop_half, \
                  binop(iop_quarter, mkexpr(s[3]), mkexpr(s[2])), \
                  binop(iop_quarter, mkexpr(s[1]), mkexpr(s[0]))) );
         switch( typeOfIRTemp(irsb->tyenv, dst) ) {
            case Ity_V512: merge_4( Iop_V256HLtoV512, Iop_V128HLtoV256); return;
            case Ity_V256: merge_4( Iop_V128HLtoV256, Iop_64HLtoV128);   return;
            case Ity_V128: merge_4( Iop_64HLtoV128,   Iop_32HLto64);     return;
            case Ity_I64:  merge_4( Iop_32HLto64,     Iop_16HLto32);     return;
            case Ity_I32:  merge_4( Iop_16HLto32,     Iop_8HLto16);      return;
            default: vpanic("unsupported 4x multiplier\n");
#undef merge_4
         }
      case 8:
#define merge_8( iop_half, iop_quarter, iop_oct ) \
         assign( dst, binop(iop_half,\
                  binop(iop_quarter, \
                     binop(iop_oct, mkexpr(s[7]), mkexpr(s[6])), \
                     binop(iop_oct, mkexpr(s[5]), mkexpr(s[4]))), \
                  binop(iop_quarter, \
                     binop(iop_oct, mkexpr(s[3]), mkexpr(s[2])), \
                     binop(iop_oct, mkexpr(s[1]), mkexpr(s[0])))) );
         switch( typeOfIRTemp(irsb->tyenv, dst) ) {
            case Ity_V512: merge_8(Iop_V256HLtoV512, Iop_V128HLtoV256, Iop_64HLtoV128); return;
            case Ity_V256: merge_8(Iop_V128HLtoV256, Iop_64HLtoV128,   Iop_32HLto64);   return;
            case Ity_V128: merge_8(Iop_64HLtoV128,   Iop_32HLto64,     Iop_16HLto32);   return;
            case Ity_I64:  merge_8(Iop_32HLto64,     Iop_16HLto32,     Iop_8HLto16);    return;
            default: vpanic("unsupported 8x multiplier\n");
#undef merge_8
         }
      case 16:
#define merge_16( iop_half, iop_quarter, iop_oct, iop_sed) \
         assign( dst, binop(iop_half,\
                  binop(iop_quarter, \
                     binop(iop_oct, \
                        binop(iop_sed, mkexpr(s[15]), mkexpr(s[14])), \
                        binop(iop_sed, mkexpr(s[13]), mkexpr(s[12]))), \
                     binop(iop_oct, \
                        binop(iop_sed, mkexpr(s[11]), mkexpr(s[10])), \
                        binop(iop_sed, mkexpr(s[ 9]), mkexpr(s[8])))), \
                  binop(iop_quarter, \
                     binop(iop_oct, \
                        binop(iop_sed, mkexpr(s[7]), mkexpr(s[6])), \
                        binop(iop_sed, mkexpr(s[5]), mkexpr(s[4]))), \
                     binop(iop_oct, \
                        binop(iop_sed, mkexpr(s[3]), mkexpr(s[2])), \
                        binop(iop_sed, mkexpr(s[1]), mkexpr(s[0]))))));
         switch( typeOfIRTemp(irsb->tyenv, dst) ) {
            case Ity_V512: merge_16(Iop_V256HLtoV512, Iop_V128HLtoV256, Iop_64HLtoV128, Iop_32HLto64); return;
            case Ity_V256: merge_16(Iop_V128HLtoV256, Iop_64HLtoV128,   Iop_32HLto64,   Iop_16HLto32); return;
            case Ity_V128: merge_16(Iop_64HLtoV128,   Iop_32HLto64,     Iop_16HLto32,   Iop_8HLto16);  return;
            default: vpanic("unsupported 16x multiplier\n");
         }
      case 32: // dst = 512
         switch( typeOfIRTemp(irsb->tyenv, dst) ) {
            case Ity_V512:
               assign( dst, binop(Iop_V256HLtoV512,
                        binop(Iop_V128HLtoV256, binop(Iop_64HLtoV128, binop(Iop_32HLto64,
                                 binop(Iop_16HLto32, mkexpr(s[31]), mkexpr(s[30])),
                                 binop(Iop_16HLto32, mkexpr(s[29]), mkexpr(s[28]))), binop(Iop_32HLto64,
                                 binop(Iop_16HLto32, mkexpr(s[27]), mkexpr(s[26])),
                                 binop(Iop_16HLto32, mkexpr(s[25]), mkexpr(s[24])))), binop(Iop_64HLtoV128, binop(Iop_32HLto64,
                                 binop(Iop_16HLto32, mkexpr(s[23]), mkexpr(s[22])),
                                 binop(Iop_16HLto32, mkexpr(s[21]), mkexpr(s[20]))), binop(Iop_32HLto64,
                                 binop(Iop_16HLto32, mkexpr(s[19]), mkexpr(s[18])),
                                 binop(Iop_16HLto32, mkexpr(s[17]), mkexpr(s[16]))))),
                        binop(Iop_V128HLtoV256, binop(Iop_64HLtoV128, binop(Iop_32HLto64,
                                 binop(Iop_16HLto32, mkexpr(s[15]), mkexpr(s[14])),
                                 binop(Iop_16HLto32, mkexpr(s[13]), mkexpr(s[12]))), binop(Iop_32HLto64,
                                 binop(Iop_16HLto32, mkexpr(s[11]), mkexpr(s[10])),
                                 binop(Iop_16HLto32, mkexpr(s[9]), mkexpr(s[8])))), binop(Iop_64HLtoV128, binop(Iop_32HLto64,
                                 binop(Iop_16HLto32, mkexpr(s[7]), mkexpr(s[6])),
                                 binop(Iop_16HLto32, mkexpr(s[5]), mkexpr(s[4]))), binop(Iop_32HLto64,
                                 binop(Iop_16HLto32, mkexpr(s[3]), mkexpr(s[2])),
                                 binop(Iop_16HLto32, mkexpr(s[1]), mkexpr(s[0])))))) );
               return;
            case Ity_V256:
               assign( dst, binop(Iop_V128HLtoV256,
                        binop(Iop_64HLtoV128, binop(Iop_32HLto64, binop(Iop_16HLto32,
                                 binop(Iop_8HLto16, mkexpr(s[31]), mkexpr(s[30])),
                                 binop(Iop_8HLto16, mkexpr(s[29]), mkexpr(s[28]))), binop(Iop_16HLto32,
                                 binop(Iop_8HLto16, mkexpr(s[27]), mkexpr(s[26])),
                                 binop(Iop_8HLto16, mkexpr(s[25]), mkexpr(s[24])))), binop(Iop_32HLto64, binop(Iop_16HLto32,
                                 binop(Iop_8HLto16, mkexpr(s[23]), mkexpr(s[22])),
                                 binop(Iop_8HLto16, mkexpr(s[21]), mkexpr(s[20]))), binop(Iop_16HLto32,
                                 binop(Iop_8HLto16, mkexpr(s[19]), mkexpr(s[18])),
                                 binop(Iop_8HLto16, mkexpr(s[17]), mkexpr(s[16]))))),
                        binop(Iop_64HLtoV128, binop(Iop_32HLto64, binop(Iop_16HLto32,
                                 binop(Iop_8HLto16, mkexpr(s[15]), mkexpr(s[14])),
                                 binop(Iop_8HLto16, mkexpr(s[13]), mkexpr(s[12]))), binop(Iop_16HLto32,
                                 binop(Iop_8HLto16, mkexpr(s[11]), mkexpr(s[10])),
                                 binop(Iop_8HLto16, mkexpr(s[9]), mkexpr(s[8])))), binop(Iop_32HLto64, binop(Iop_16HLto32,
                                 binop(Iop_8HLto16, mkexpr(s[7]), mkexpr(s[6])),
                                 binop(Iop_8HLto16, mkexpr(s[5]), mkexpr(s[4]))), binop(Iop_16HLto32,
                                 binop(Iop_8HLto16, mkexpr(s[3]), mkexpr(s[2])),
                                 binop(Iop_8HLto16, mkexpr(s[1]), mkexpr(s[0])))))) );
               return;
            default: vpanic("panic");
         }
      default: vpanic("unsupported multiplier\n");
   }
}

static void Put_Instr_Result(Int ins_id, IRExpr* masked,
      Int rG, Int rV, Int rE, IRTemp addr, UChar modrm)
{
   enum op_encoding op_code = INS_ARR[ins_id].args[0];

   // memory store
   switch (INS_ARR[ins_id].args[0]) {
      case _rmE: case _imE: case _kmE:
         if (epartIsReg(modrm)) break; // else fallthrough
      case _mG: case _mE:
         storeLE( mkexpr(addr), masked );
         return;
   }

   // register store
   int reg = 0;
   switch (INS_ARR[ins_id].args[0]) {
      case _rG: case _iG: case _kG:  reg = rG; break;
      case _rmE: case _imE: case _kmE:
      case _rE: case _iE: case _kE:  reg = rE; break;
      case _rV:                      reg = rV; break;
      default: vpanic("unknown destination register\n");
   }

   IRType op_type = INS_ARR[ins_id].arg_type[0];

   // scalar register
   if ((op_code == _iG) || (op_code == _iE) || (op_code == _imE)) {
      switch (op_type) { // put the data
         case Ity_I8:  putIReg16( reg, unop(Iop_8Uto16, masked) ); break;
         case Ity_I16: putIReg16( reg, masked ); break;
         case Ity_I32: putIReg32( reg, masked ); break;
         case Ity_I64: putIReg64( reg, masked ); break;
         default: vpanic("!");
      }
      return;
   }
   // mask register
   if ((op_code == _kE) || (op_code == _kG) || (op_code == _kmE)) {
      switch (op_type) { // put the data
         case Ity_I8:  putKRegLane8(reg, 0, masked);  break;
         case Ity_I16: putKRegLane16(reg, 0, masked); break;
         case Ity_I32: putKRegLane32(reg, 0, masked); break;
         case Ity_I64: putKReg(reg, masked);          break;
         default: vpanic("!");
      }
      return;
   }
   // vector register
   putZMMReg(reg, mkV512(0)); // zero out the entire register beforehand
   switch (op_type) { // put the data
      case Ity_I16:  stmt(IRStmt_Put( xmmGuestRegLane16offset(reg, 0), masked)); break;
      case Ity_I32:  putXMMRegLane32(reg, 0, masked); break;
      case Ity_I64:  putXMMRegLane64(reg, 0, masked); break;
      case Ity_V128: putXMMReg( reg, masked ); break;
      case Ity_V256: putYMMReg( reg, masked ); break;
      case Ity_V512: putZMMReg( reg, masked ); break;
      default: vpanic("!");
   }
   // It is a tempting idea to put rV in the remainder of xmm in serial cases
   // But it does not work for FMA instructions, so handle it at IR construction instead
   return;
}



static Long dis_FMA_512 ( const VexAbiInfo* vbi, Prefix pfx, Long delta, Int ins_id )
{
   /* determine instruction type and parameters */
   UChar opc = INS_ARR[ins_id].opcode;
   Bool scalar = ((opc & 0xF) > 7 && (opc & 1));

   Bool negateRes   = False;
   Bool negateZeven = False;
   Bool negateZodd  = False;
   switch (opc & 0xF) {
      case 0x6:           negateZeven = True; break;
      case 0x7:           negateZodd = True;  break;
      case 0x8: case 0x9: break;
      case 0xA: case 0xB: negateZeven = True; negateZodd = True; break;
      case 0xC: case 0xD: negateRes = True; negateZeven = True; negateZodd = True; break;
      case 0xE: case 0xF: negateRes = True; break;
      default:  vpanic("dis_FMA_512(amd64)"); break;
   }
   Bool is_64 = (getDstW() == W_64);
   IRType partial_type = is_64 ? Ity_I64     : Ity_I32;
   IROp   negate       = is_64 ? Iop_NegF64  : Iop_NegF32;
   IROp   mult_add     = is_64 ? Iop_MAddF64 : Iop_MAddF32;
   IROp   reinterprI2F = is_64 ? Iop_ReinterpI64asF64 : Iop_ReinterpI32asF32;
   IROp   reinterprF2I = is_64 ? Iop_ReinterpF64asI64 : Iop_ReinterpF32asI32;
   IROp   get_lowest   = is_64 ? Iop_V128to64 : Iop_V128to32;
   IROp   merge_lowest = is_64 ? Iop_SetV128lo64 : Iop_SetV128lo32;
   UInt element_count = sizeofIRType(INS_ARR[ins_id].arg_type[0]) / sizeofIRType(partial_type);
   DIP(" %s ", INS_ARR[ins_id].name);

   /* retrieve arguments. Assuming the master file has them in dst, 1, 2, 3 order */
   Int argN = 4; // MUST be so!
   IRTemp arg[argN];
   IRTemp s[argN][element_count];
   IRExpr* expr_s[argN][element_count];
   IRTemp res[element_count];

   Long delta_out = delta;
   UChar modrm = getUChar(delta_out);
   Int rG = gregOfRexRM32(pfx, modrm);
   Int rV = getEVexNvvvv(pfx);
   Int rE = 0;
   IRTemp addr = IRTemp_INVALID;
   Int int_val = -1;
   delta_out = Get_Instr_Args(vbi, pfx, delta_out, ins_id, arg, &argN, &int_val, rG, rV, &rE, &addr, modrm);

   IRTemp unmasked = newTemp(INS_ARR[ins_id].arg_type[0]);
   if (scalar)
   {
      for (Int i = 1; i <= argN; i++)
         expr_s[i][0] = unop(reinterprI2F, unop(get_lowest, mkexpr(arg[i])));
      if (negateZodd)
         expr_s[3][0] = unop(negate, expr_s[3][0]);
      expr_s[0][0] = IRExpr_Qop(mult_add, get_evex_roundingmode(), expr_s[1][0], expr_s[2][0], expr_s[3][0]);
      if (negateRes)
         expr_s[0][0] = unop(negate, expr_s[0][0]);
      assign(unmasked, binop(merge_lowest, mkexpr(arg[0]), unop(reinterprF2I, expr_s[0][0])));
   }
   else
   {
      // NOTE argN is 3 now
      /* Split the sources */
      for (Int i = 1; i <= argN; i++) {
         for (Int j = 0; j < element_count; j++)
            s[i][j] = IRTemp_INVALID;
         Split_arg(arg[i], s[i], element_count);
         for (Int j = 0; j < element_count; j++)
            expr_s[i][j] = unop(reinterprI2F, mkexpr(s[i][j]));
      }

      /* Multiply-add and negate values */
      for (Int j = 0; j < element_count; j++) {
         if ((j & 1) ? negateZodd : negateZeven)
            expr_s[3][j] = unop(negate, expr_s[3][j]);
         expr_s[0][j] = IRExpr_Qop(mult_add, get_evex_roundingmode(), expr_s[1][j], expr_s[2][j], expr_s[3][j]);
         if (negateRes)
            expr_s[0][j] = unop(negate, expr_s[0][j]);
         res[j] = newTemp(partial_type);
         assign(res[j], unop(reinterprF2I, expr_s[0][j]));
      }
      /* Merge back */
      Merge_dst(unmasked, res, element_count);
   }
   IRExpr* masked = mask_expr(pfx, mkexpr(unmasked), mkexpr(arg[0]), ins_id);
   Put_Instr_Result(ins_id, masked, rG, rV, rE, addr, modrm);
   return delta_out;
}


static ULong dis_GATHER_512 ( const VexAbiInfo* vbi, Prefix pfx, Long delta,
      UInt rG, UInt ins_id)
{
   HChar  dis_buf[50];
   Int    alen, i, vscale;
   IRTemp addr, cond;
   UInt   rI;
   UInt   mask = getEvexMask();
   IRExpr* address;

   addr = disAVSIBMode ( &alen, vbi, pfx, delta, dis_buf,
         &rI, INS_ARR[ins_id].arg_type[1], &vscale );
   delta += alen;
   DIP (" %s %s,", INS_ARR[ins_id].name, dis_buf);
   switch (INS_ARR[ins_id].arg_type[0]) {
      case Ity_V512: DIP("%s", nameZMMReg(rG)); break;
      case Ity_V256: DIP("%s", nameYMMReg(rG)); break;
      case Ity_V128: DIP("%s", nameXMMReg(rG)); break;
   }
   DIP (",%s\n", nameKReg(mask));

   IRType reg_w = (getDstW() == W_64) ? Ity_I64 : Ity_I32;
   IRType ind_w = (INS_ARR[ins_id].args[1] == _vm64) ? Ity_I64 : Ity_I32;
   Int count = sizeofIRType(INS_ARR[ins_id].arg_type[0]) / sizeofIRType(reg_w);

   for (i = 0; i < count; i++) {
      cond = newTemp(Ity_I1);
      assign( cond, unop(Iop_64to1, binop(Iop_Shr64, getKReg(mask), mkU8(i))));
      address = (ind_w == Ity_I64) ?
         getZMMRegLane64(rI, i) :
         unop(Iop_32Sto64, getZMMRegLane32(rI, i));

      switch (vscale) {
         case 2: address = binop(Iop_Shl64, address, mkU8(1)); break;
         case 4: address = binop(Iop_Shl64, address, mkU8(2)); break;
         case 8: address = binop(Iop_Shl64, address, mkU8(3)); break;
         default: break;
      }
      address = binop(Iop_Add64, mkexpr(addr), address);
      address = handleAddrOverrides(vbi, pfx, address);
      address = IRExpr_ITE(mkexpr(cond), address, getIReg64(R_RSP));

      if (reg_w == Ity_I64) {
         putZMMRegLane64( rG, i, IRExpr_ITE(mkexpr(cond),
                  loadLE(Ity_I64, address),
                  getZMMRegLane64(rG, i)) );
      } else {
         putZMMRegLane32( rG, i, IRExpr_ITE(mkexpr(cond),
                  loadLE(Ity_I32, address),
                  getZMMRegLane32(rG, i)) );
      }
   }

   // Fill the upper part of result vector register.
   switch (INS_ARR[ins_id].arg_type[0]) {
      case Ity_V128: putZMMRegLane128( rG, 1, mkV128(0) );
      case Ity_V256: putZMMRegLane256( rG, 1, mkV256(0) );
      default: break;
   }

   //assume success
   putKReg(mask, mkU64(0));
   return delta;
}

static ULong dis_SCATTER_512 ( const VexAbiInfo* vbi, Prefix pfx, Long delta,
      UInt rG, UInt ins_id)
{
   HChar  dis_buf[50];
   Int    alen, i, vscale;
   IRTemp addr, cond;
   UInt   rI;
   UInt   mask = getEvexMask();
   IRExpr* address;

   addr = disAVSIBMode ( &alen, vbi, pfx, delta, dis_buf,
         &rI, INS_ARR[ins_id].arg_type[0], &vscale );
   delta += alen;

   DIP (" %s \n", INS_ARR[ins_id].name);
   switch (INS_ARR[ins_id].arg_type[1]) {
      case Ity_V512: DIP("%s", nameZMMReg(rG)); break;
      case Ity_V256: DIP("%s", nameYMMReg(rG)); break;
      case Ity_V128: DIP("%s", nameXMMReg(rG)); break;
   }
   DIP (",%s,%s\n", dis_buf, nameKReg(mask));

   IRType ind_w = (INS_ARR[ins_id].args[0] == _vm64) ? Ity_I64 : Ity_I32;
   IRType reg_w = (getDstW() == W_64) ? Ity_I64 : Ity_I32;
   Int count = sizeofIRType(INS_ARR[ins_id].arg_type[1]) / sizeofIRType(reg_w);

   for (i = 0; i < count; i++) {
      cond = newTemp(Ity_I1);
      assign( cond, unop(Iop_64to1, binop(Iop_Shr64, getKReg(mask), mkU8(i))));
      // ind_w instead of reg_w is counterintuitive to me here, but it's the only working option
      address = (ind_w == Ity_I64) ?
         getZMMRegLane64(rI, i) :
         unop(Iop_32Sto64, getZMMRegLane32(rI, i));

      switch (vscale) {
         case 2: address = binop(Iop_Shl64, address, mkU8(1)); break;
         case 4: address = binop(Iop_Shl64, address, mkU8(2)); break;
         case 8: address = binop(Iop_Shl64, address, mkU8(3)); break;
         default: break;
      }
      address = binop(Iop_Add64, mkexpr(addr), address);
      address = handleAddrOverrides(vbi, pfx, address);
      address = IRExpr_ITE(mkexpr(cond), address, getIReg64(R_RSP));

      if (reg_w == Ity_I64) {
         storeLE(address, IRExpr_ITE(mkexpr(cond),
                  getZMMRegLane64(rG, i),
                  loadLE(Ity_I64, address)));
      } else {
         storeLE(address, IRExpr_ITE(mkexpr(cond),
                  getZMMRegLane32(rG, i),
                  loadLE(Ity_I32, address)));
      }
   }

   //assume success
   putKReg(mask, mkU64(0));
   return delta;
}

static Long dis_shift_512 ( const VexAbiInfo* vbi, Prefix pfx, Long delta, Int ins_id )
{
   /* determine instruction type and parameters */
   IRType part_ty;
   IRExpr* shift_max;
   UInt width = getDstW();
   switch (width) {
      case W_8:  part_ty = Ity_I8;  shift_max = mkU64(0x8);  break;
      case W_16: part_ty = Ity_I16; shift_max = mkU64(0x10); break;
      case W_32: part_ty = Ity_I32; shift_max = mkU64(0x20); break;
      case W_64: part_ty = Ity_I64; shift_max = mkU64(0x40); break;
      default: vpanic("unsupported width");
   }
   IROp shift_op = Iop_INVALID;
   switch (INS_ARR[ins_id].parameter) {
      case 0: /* SHL */
         switch (width) {
            case W_8:  shift_op = Iop_Shl8;  break;
            case W_16: shift_op = Iop_Shl16; break;
            case W_32: shift_op = Iop_Shl32; break;
            case W_64: shift_op = Iop_Shl64; break;
         }
         break;
      case 1: /* SHR */
         switch (width) {
            case W_8:  shift_op = Iop_Shr8;  break;
            case W_16: shift_op = Iop_Shr16; break;
            case W_32: shift_op = Iop_Shr32; break;
            case W_64: shift_op = Iop_Shr64; break;
         }
         break;
      case 2: /* SAR */
         switch (width) {
            case W_8:  shift_op = Iop_Sar8;  break;
            case W_16: shift_op = Iop_Sar16; break;
            case W_32: shift_op = Iop_Sar32; break;
            case W_64: shift_op = Iop_Sar64; break;
         }
         break;
      default: vpanic("unsupported shift\n");
   }

   UInt element_count = sizeofIRType(INS_ARR[ins_id].arg_type[0]) / sizeofIRType(part_ty);

   /* retrieve arguments */
   Int argN = 3;
   IRTemp arg[argN];
   IRTemp s[argN][element_count];
   IRTemp res[element_count];

   Long delta_out = delta;
   UChar modrm = getUChar(delta_out);
   Int rG = gregOfRexRM32(pfx, modrm);
   Int rV = getEVexNvvvv(pfx);
   Int rE = 0;
   IRTemp addr = IRTemp_INVALID;
   Int int_val = -1;
   DIP(" %s ", INS_ARR[ins_id].name);
   delta_out = Get_Instr_Args(vbi, pfx, delta_out, ins_id, arg, &argN, &int_val, rG, rV, &rE, &addr, modrm);

   /* Split the 1st source; the 2nd vector source can be split or replicated,
    * depending on the instruction */
   IRType full_ty = typeOfIRTemp(irsb->tyenv, arg[2]);
   Bool replicate = (full_ty == Ity_I8) || ((full_ty == Ity_V128) && (getTupleType() == Mem128));
   IRType partial_type = replicate ? full_ty : part_ty;

   for (Int j=0; j<element_count; j++) {
      s[1][j] = IRTemp_INVALID;
      s[2][j] = IRTemp_INVALID;
      res[j]  = newTemp(part_ty);
   }
   Split_arg(arg[1], s[1], element_count);
   if (!replicate)
      Split_arg(arg[2], s[2], element_count);

   // Generating lot of redundant values here; fortunately, IR optimization handles it
   /* Shift each element */
   for (Int j=0; j<element_count; j++) {
      /* We need the second src as a 64-bit int for the comparison,
       * and as a 8-bit int for shift */
      IRExpr *amt_cmp, *amt_shift;
      switch (partial_type) {
         case Ity_V128: // serial xmm case
            // if upper xmm != 0 shift by the max value
            amt_cmp = IRExpr_ITE(
                  binop( Iop_CmpLE64U, unop(Iop_V128HIto64, mkexpr(arg[2])), mkU64(0) ),
                  unop(Iop_V128to64, mkexpr(arg[2])),
                  shift_max);
            amt_shift = unop(Iop_64to8, unop(Iop_V128to64, mkexpr(arg[2])));
            break;
         case Ity_I64:
            amt_cmp = mkexpr(s[2][j]);
            amt_shift = unop(Iop_64to8, mkexpr(s[2][j]));
            break;
         case Ity_I32:
            amt_cmp = unop(Iop_32Uto64, mkexpr(s[2][j]));
            amt_shift = unop(Iop_32to8, mkexpr(s[2][j]));
            break;
         case Ity_I16:
            amt_cmp = unop(Iop_16Uto64, mkexpr(s[2][j]));
            amt_shift = unop(Iop_16to8, mkexpr(s[2][j]));
            break;
         case Ity_I8:
            // packed byte vector or imm8?
            if (INS_ARR[ins_id].args[2] == _imm8) {
               amt_cmp = unop(Iop_8Uto64, mkexpr(arg[2]));
               amt_shift = mkexpr(arg[2]);
            } else {
               amt_cmp = unop(Iop_8Uto64, mkexpr(s[2][j]));
               amt_shift = mkexpr(s[2][j]);
            }
            break;
         default: vpanic("invalid type of shift amount\n");
      }

      IRExpr *default_val;
      switch (INS_ARR[ins_id].parameter) {
         case 0:
         case 1:
            switch (width) {
               case W_8:  default_val = mkU8(0);  break;
               case W_16: default_val = mkU16(0); break;
               case W_32: default_val = mkU32(0); break;
               case W_64: default_val = mkU64(0); break;
            }
            break;
         case 2:
            switch (width) {
               case W_8:  default_val = binop(shift_op, mkexpr(s[1][j]), mkU8(0x8));  break;
               case W_16: default_val = binop(shift_op, mkexpr(s[1][j]), mkU8(0xF));  break;
               case W_32: default_val = binop(shift_op, mkexpr(s[1][j]), mkU8(0x1F)); break;
               case W_64: default_val = binop(shift_op, mkexpr(s[1][j]), mkU8(0x3F)); break;
            }
            break;
      }
      assign( res[j], IRExpr_ITE(
               binop(Iop_CmpLT64U, amt_cmp, shift_max),
               binop(shift_op, mkexpr(s[1][j]), amt_shift),
               default_val));
   }

   /* Merge back and mask the result */
   IRTemp unmasked = newTemp(INS_ARR[ins_id].arg_type[0]);
   Merge_dst(unmasked, res, element_count);
   IRExpr* masked = mask_expr(pfx, mkexpr(unmasked), mkexpr(arg[0]), ins_id);

   Put_Instr_Result(ins_id, masked, rG, rV, rE, addr, modrm);
   return delta_out;
}

static Long dis_EVEX_exceptions (const VexAbiInfo* vbi, Prefix pfx, Escape esc, Long delta, Int ins_id) {
   UChar opcode = getUChar(delta);
   delta += 1;
   UChar modrm = getUChar(delta);
   UInt rG = gregOfRexRM32(pfx, modrm);
   UInt rV = getEVexNvvvv(pfx);
   UInt mask = getEvexMask();
   HChar dis_buf[50];
   Int alen = 0;
   UInt imm8;

   switch (opcode) {
      case 0x10:
         /* VMOVSD */
         if (haveF2no66noF3(pfx) && (esc == ESC_0F) && getRexW(pfx)) {
            if (epartIsReg(modrm)) {
               UInt rE = eregOfRexRM(pfx, modrm);
               delta += 1;
               DIP(" VMOVSD %s,%s,%s{%s}\n", nameXMMReg(rE), nameXMMReg(rV), nameXMMReg(rG), nameKReg(mask));
               putXMMReg(rG, mask_expr(pfx,
                        binop(Iop_SetV128lo64, getXMMReg(rV), getXMMRegLane64(rE, 0)),
                        getXMMReg(rG), ins_id) );
            }
            else {
               IRTemp addr = disAMode(&alen, vbi, pfx, delta, dis_buf, 0);
               delta += alen;
               DIP(" VMOVSD %s,%s{%s}\n", dis_buf, nameXMMReg(rG), nameKReg(mask));
               putXMMReg(rG, mask_expr(pfx,
                        binop(Iop_SetV128lo64, mkV128(0), loadLE(Ity_I64, mkexpr(addr))),
                        getXMMReg(rG), ins_id) );
            }
            putZMMRegLane128(rG, 1, mkV128(0));
            putZMMRegLane256(rG, 1, mkV256(0));
         }
         /* VMOVSS */
         if (haveF3no66noF2(pfx) && (esc == ESC_0F) && !getRexW(pfx)) {
            if (epartIsReg(modrm)) {
               UInt rE = eregOfRexRM(pfx, modrm);
               delta += 1;
               DIP(" VMOVSS %s,%s,%s{%s}\n", nameXMMReg(rE), nameXMMReg(rV), nameXMMReg(rG), nameKReg(mask));
               putXMMReg(rG, mask_expr(pfx,
                        binop(Iop_SetV128lo32, getXMMReg(rV), getXMMRegLane32(rE, 0)),
                        getXMMReg(rG), ins_id) );
            }
            else {
               IRTemp addr = disAMode ( &alen, vbi, pfx, delta, dis_buf, 0 );
               delta += alen;
               DIP(" VMOVSS %s,%s{%s}\n", dis_buf, nameXMMReg(rG), nameKReg(mask));
               putXMMReg(rG, mask_expr(pfx,
                        binop(Iop_SetV128lo32, mkV128(0), loadLE(Ity_I32, mkexpr(addr))),
                        getXMMReg(rG), ins_id) );
            }
            putZMMRegLane128(rG, 1, mkV128(0));
            putZMMRegLane256(rG, 1, mkV256(0));
         }
         /* VPSRLVW */
         if (have66noF2noF3(pfx) && (esc == ESC_0F38) && getRexW(pfx))
            delta = dis_shift_512( vbi, pfx, delta, ins_id );
         break;
      case 0x11:
          /* VMOVSD */
         if (haveF2no66noF3(pfx) && (esc == ESC_0F) && getRexW(pfx)) {
            if (epartIsReg(modrm)) {
               UInt rE = eregOfRexRM(pfx, modrm);
               delta += 1;
               DIP(" VMOVSD %s,%s,%s{%s}\n", nameXMMReg(rE), nameXMMReg(rV), nameXMMReg(rG), nameKReg(mask));
               putXMMReg(rE, mask_expr(pfx,
                        binop(Iop_SetV128lo64, getXMMReg(rV), getXMMRegLane64(rG, 0)),
                        getXMMReg(rE), ins_id) );
            }
            else {
               IRTemp addr = disAMode (&alen, vbi, pfx, delta, dis_buf, 0);
               delta += alen;
               DIP(" VMOVSD %s,%s{%s}\n", nameXMMReg(rG), dis_buf, nameKReg(mask));
               storeLE( mkexpr(addr), IRExpr_ITE( unop(Iop_64to1, getKReg(mask)),
                        getXMMRegLane64(rG, 0), // masked
                        loadLE( Ity_I64, mkexpr(addr)))); //unmasked
            }
         }
         /* VMOVSS */
         if (haveF3no66noF2(pfx) && (esc == ESC_0F) && !getRexW(pfx)) {
            if (epartIsReg(modrm)) {
               UInt rE = eregOfRexRM(pfx, modrm);
               delta += 1;
               DIP(" VMOVSS %s,%s,%s{%s}\n", nameXMMReg(rE), nameXMMReg(rV), nameXMMReg(rG), nameKReg(mask));
               putXMMReg(rE, mask_expr(pfx,
                        binop(Iop_SetV128lo32, getXMMReg(rV), getXMMRegLane32(rG, 0)),
                        getXMMReg(rE), ins_id) );
            }
            else {
               IRTemp addr = disAMode ( &alen, vbi, pfx, delta, dis_buf, 0 );
               delta += alen;
               DIP(" VMOVSS %s,%s{%s}\n", nameXMMReg(rG), dis_buf, nameKReg(mask));
               storeLE( mkexpr(addr), IRExpr_ITE( unop(Iop_64to1, getKReg(mask)),
                        getXMMRegLane32(rG, 0), // masked
                        loadLE( Ity_I32, mkexpr(addr)))); //unmasked
            }
         }
         /* VPSRAVW */
         if (have66noF2noF3(pfx) && (esc == ESC_0F38) && getRexW(pfx))
            delta = dis_shift_512( vbi, pfx, delta, ins_id );
         break;
      case 0x12:
         /* VMOVHLPS */
         if (haveNo66noF2noF3(pfx) && (esc == ESC_0F) && !getRexW(pfx) && epartIsReg(modrm)) {
            setTupleType( FullVectorMem );
            UInt rE = eregOfRexRM(pfx, modrm);
            delta += 1;
            DIP(" VMOVHLPS %s,%s,%s\n", nameXMMReg(rE), nameXMMReg(rV), nameXMMReg(rG));
            putYMMRegLoAndZU(rG, binop(Iop_64HLtoV128,
                     getXMMRegLane64(rV, 1),
                     getXMMRegLane64(rE, 1)));
            putZMMRegLane256(rG, 1, mkV256(0));
         }
         /* VMOVLPS */
         if ( haveNo66noF2noF3(pfx) && (esc == ESC_0F) && !getRexW(pfx) && !epartIsReg(modrm) ) {
            setTupleType( Tuple2 );
            IRTemp addr = disAMode ( &alen, vbi, pfx, delta, dis_buf, 0 );
            delta += alen;
            DIP(" VMOVLPS %s,%s,%s\n", dis_buf, nameXMMReg(rV), nameXMMReg(rG));
            putYMMRegLoAndZU(rG, binop(Iop_64HLtoV128,
                     getXMMRegLane64(rV, 1),
                     loadLE(Ity_I64, mkexpr(addr))));
            putZMMRegLane256(rG, 1, mkV256(0));
         }
         /* VPSLLVW */
         if (have66noF2noF3(pfx) && (esc == ESC_0F38) && getRexW(pfx))
            delta = dis_shift_512( vbi, pfx, delta, ins_id );
         break;
      case 0x14: {
         /* VPEXTRB */
         IRTemp xmm_vec = newTemp(Ity_V128);
         assign(xmm_vec, getXMMReg(rG));
         IRTemp s[16];
         for (Int i=0; i<16; i++)
            s[i] = IRTemp_INVALID;
         Split_arg(xmm_vec, s, 16);
         if (epartIsReg(modrm)) {
            UInt rE = eregOfRexRM(pfx, modrm);
            imm8 = (Int)getUChar(delta+1);
            DIP(" VPEXTRB $%d,%s,%s\n", imm8, nameXMMReg(rG), nameIReg64(rE));
            delta += 1+1;
            putIReg64(rE, unop(Iop_8Uto64, mkexpr(s[ imm8&0xF ])));
         } else {
            IRTemp addr = IRTemp_INVALID;
            addr = disAMode( &alen, vbi, pfx, delta, dis_buf, 1 );
            imm8 = (Int)getUChar(delta+alen);
            DIP(" VPEXTRB $%d,%s,%s\n", imm8, nameXMMReg(rG), dis_buf );
            delta += alen+1;
            storeLE(mkexpr(addr), mkexpr(s[ imm8&0xF ]));
         }
         break;
      }
      case 0x15: {
         /* VPEXTRW */
         IRTemp xmm_vec = newTemp(Ity_V128);
         assign(xmm_vec, getXMMReg(rG));
         IRTemp s[8];
         for (Int i=0; i<8; i++)
            s[i] = IRTemp_INVALID;
         Split_arg(xmm_vec, s, 8);
         if (epartIsReg(modrm)) {
            UInt rE = eregOfRexRM(pfx, modrm);
            imm8 = (Int)getUChar(delta+1);
            DIP(" VPEXTRW $%d,%s,%s\n", imm8, nameXMMReg(rG), nameIReg64(rE));
            delta += 1+1;
            putIReg64(rE, unop(Iop_16Uto64, mkexpr(s[ imm8&0x7 ])));
         } else {
            IRTemp addr = IRTemp_INVALID;
            addr = disAMode( &alen, vbi, pfx, delta, dis_buf, 1 );
            imm8 = (Int)getUChar(delta+alen);
            DIP(" VPEXTRW $%d,%s,%s\n", imm8, nameXMMReg(rG), dis_buf );
            delta += alen+1;
            storeLE(mkexpr(addr), mkexpr(s[ imm8&0x7 ]));
         }
         break;
      }
      case 0x16:
         /* VMOVHPS */
         if ( haveNo66noF2noF3(pfx) && (esc == ESC_0F) && !getRexW(pfx) && !epartIsReg(modrm) ) {
            setTupleType( Tuple2 );
            IRTemp addr = disAMode ( &alen, vbi, pfx, delta, dis_buf, 0 );
            delta += alen;
            DIP(" VMOVHPS %s, %s, %s\n", dis_buf, nameXMMReg(rV), nameXMMReg(rG));
            IRTemp res = newTemp(Ity_V128);
            assign(res, binop(Iop_64HLtoV128,
                     loadLE(Ity_I64, mkexpr(addr)),
                     getXMMRegLane64(rV, 0)));
            putYMMRegLoAndZU(rG, mkexpr(res));
            putZMMRegLane256(rG, 1, mkV256(0));
         }
         /* VMOVLHPS */
         if (haveNo66noF2noF3(pfx) && (esc == ESC_0F) && !getRexW(pfx) && epartIsReg(modrm)) {
            setTupleType( FullVectorMem );
            UInt rE = eregOfRexRM32(pfx, modrm);
            delta += 1;
            DIP(" VMOVLHPS %s,%s,%s\n", nameXMMReg(rE), nameXMMReg(rV), nameXMMReg(rG));
            IRTemp res = newTemp(Ity_V128);
            assign(res, binop(Iop_64HLtoV128,
                     getXMMRegLane64(rE, 0),
                     getXMMRegLane64(rV, 0)));
            putYMMRegLoAndZU(rG, mkexpr(res));
            putZMMRegLane256(rG, 1, mkV256(0));
         }
         break;
      case 0x21: {
         /*VINSERTPS*/
         IRTemp d2ins = newTemp(Ity_I32); /* comes from the E part */
         if ( epartIsReg( modrm ) ) {
            UInt   rE = eregOfRexRM(pfx, modrm);
            IRTemp vE = newTemp(Ity_V128);
            assign( vE, getXMMReg(rE) );
            IRTemp dsE[4] = { IRTemp_INVALID, IRTemp_INVALID, IRTemp_INVALID, IRTemp_INVALID};
            breakupV128to32s( vE, &dsE[3], &dsE[2], &dsE[1], &dsE[0] );
            imm8 = getUChar(delta+1);
            d2ins = dsE[(imm8 >> 6) & 3]; /* "imm8_count_s" */
            delta += 1+1;
            DIP( " INSERTPS $%u, %s,%s\n", imm8, nameXMMReg(rE), nameXMMReg(rG) );
         } else {
            IRTemp addr = disAMode( &alen, vbi, pfx, delta, dis_buf, 1 );
            assign( d2ins, loadLE( Ity_I32, mkexpr(addr) ) );
            imm8 = getUChar(delta+alen);
            delta += alen+1;
            DIP( "INSERTPS $%u, %s,%s\n", imm8, dis_buf, nameXMMReg(rG) );
         }
         IRTemp vV = newTemp(Ity_V128);
         assign( vV, getXMMReg(rV) );
         putYMMRegLoAndZU( rG, mkexpr(math_INSERTPS( vV, d2ins, imm8 )) );
         break;
      }
      case 0x2E:
      case 0x2F: /* VUCOMISS, VUCOMISD, VCOMISS, VCOMISD*/
         DIP(" VCOMIS ");
         stmt( IRStmt_Put( OFFB_CC_OP,   mkU64(AMD64G_CC_OP_COPY) ));
         stmt( IRStmt_Put( OFFB_CC_DEP2, mkU64(0) ));

         IRTemp lhs = newTemp(Ity_F64);
         assign(lhs, getRexW(pfx) ?
               getXMMRegLane64F(rG, 0) :
               unop(Iop_F32toF64, getXMMRegLane32F(rG, 0)));
         IRTemp rhs = newTemp(Ity_F64);
         if (epartIsReg(modrm)) {
            UInt rE = eregOfRexRM(pfx, modrm);
            delta += 1;
            assign(rhs, getRexW(pfx) ?
                  getXMMRegLane64F(rE, 0) :
                  unop(Iop_F32toF64, getXMMRegLane32F(rE, 0)));
         } else {
            IRTemp addr = disAMode(&alen, vbi, pfx, delta, dis_buf, 0);
            delta += alen;
            assign(rhs, getRexW(pfx) ?
                  loadLE(Ity_F64, mkexpr(addr)) :
                  unop(Iop_F32toF64, loadLE(Ity_F32, mkexpr(addr))));
         }
         stmt( IRStmt_Put( OFFB_CC_DEP1, binop(Iop_And64,
                     unop( Iop_32Uto64, binop(Iop_CmpF64, mkexpr(lhs), mkexpr(rhs))),
                     mkU64(0x45))) );
         break;
      case 0x55: {
         Int argN = 5, rE = 0, int_val = -1;
         IRTemp arg[argN];
         IRTemp addr = IRTemp_INVALID;
         Int width = getRexW(pfx);
         DIP(" VFIXUPIMMS ");
         delta = Get_Instr_Args(vbi, pfx, delta, ins_id, arg, &argN, &int_val, rG, rV, &rE, &addr, modrm);
         IRTemp res = newTemp(Ity_V128);
         if (width)
            assign( res, binop(Iop_SetV128lo64, getXMMReg(rV),
                     qop(Iop_FixupImm64,
                        unop(Iop_V128to64, mkexpr(arg[0])),
                        unop(Iop_V128to64, mkexpr(arg[1])),
                        unop(Iop_V128to64, mkexpr(arg[2])),
                        mkU8(int_val))) );
         else
            assign( res, binop(Iop_SetV128lo32, getXMMReg(rV),
                     qop(Iop_FixupImm32,
                        unop(Iop_V128to32, mkexpr(arg[0])),
                        unop(Iop_V128to32, mkexpr(arg[1])),
                        unop(Iop_V128to32, mkexpr(arg[2])),
                        mkU8(int_val))) );
         Put_Instr_Result(ins_id, mkexpr(res), rG, rV, rE, addr, modrm);
         break;
      }
      case 0x64 ... 0x66: {
         /* VPBLENDMB, VPBLENDMW, VPBLENDMD, VPBLENDMQ, VBLENDMPS, VBLENDMPD */
         Int argN = 3, rE = 0, int_val = -1;
         IRTemp arg[argN];
         IRTemp addr = IRTemp_INVALID;
         DIP(" VBLEND ");
         delta = Get_Instr_Args(vbi, pfx, delta, ins_id, arg, &argN, &int_val, rG, rV, &rE, &addr, modrm);
         IRExpr* masked = mask_expr( pfx, mkexpr(arg[2]), mkexpr(arg[1]), ins_id);
         Put_Instr_Result(ins_id, masked, rG, rV, rE, addr, modrm);
         break;
      }
      case 0x8A:  /* VCOMPRESSPS, VCOMPRESSPD */
      case 0x8B:{ /* VPCOMPRESSD, VPCOMPRESSQ */
         Int argN = 5, rE = 0, int_val = -1;
         IRTemp arg[argN];
         IRTemp addr = IRTemp_INVALID;
         IROp compress_op;
         switch (getEVexL()) {
            case 0: compress_op = getRexW(pfx) ? Iop_Compress64x2 : Iop_Compress32x4;  break;
            case 1: compress_op = getRexW(pfx) ? Iop_Compress64x4 : Iop_Compress32x8;  break;
            case 2: compress_op = getRexW(pfx) ? Iop_Compress64x8 : Iop_Compress32x16; break;
         }
         DIP(" VCOMPRESS ");
         delta = Get_Instr_Args(vbi, pfx, delta, ins_id, arg, &argN, &int_val, rG, rV, &rE, &addr, modrm);
         IRExpr* result = qop(compress_op, mkexpr(arg[0]), mkexpr(arg[1]), getKReg(mask), mkU8(getZeroMode()));
         Put_Instr_Result(ins_id, result, rG, rV, rE, addr, modrm);
         break;
      }
      case 0x90 ... 0x93:
         delta = dis_GATHER_512(vbi, pfx, delta, rG, ins_id);
         break;
      case 0xA0 ... 0xA3:
         delta = dis_SCATTER_512(vbi, pfx, delta, rG, ins_id);
         break;
      case 0x96 ... 0x9F:
      case 0xA6 ... 0xAF:
      case 0xB6 ... 0xBF:  /* VFMAS */
         if (have66noF2noF3(pfx))
            delta = dis_FMA_512( vbi, pfx, delta, ins_id );
         break;
      case 0xC5: {
         /* VPEXTRW */
         UInt rE = eregOfRexRM(pfx, modrm);
         IRTemp xmm_vec = newTemp(Ity_V128);
         assign(xmm_vec, getXMMReg(rE));
         IRTemp s[8];
         for (Int i=0; i<8; i++)
            s[i] = IRTemp_INVALID;
         Split_arg(xmm_vec, s, 8);
         imm8 = (Int)getUChar(delta+1);
         DIP(" VPEXTRW $%d,%s,%s\n", imm8, nameXMMReg(rE), nameIReg64(rG));
         delta += 1+1;
         putIReg64(rG, unop(Iop_16Uto64, mkexpr(s[ imm8&0x7 ])));
         break;
      }
      case 0x45 ... 0x47:
      case 0x71 ... 0x73:
      case 0xD1: case 0xD3:
      case 0xE1: case 0xE2:
      case 0xF1: case 0xF3:
         delta = dis_shift_512( vbi, pfx, delta, ins_id );
         break;
      default:
         vex_printf("%x\n", opcode);
         vpanic("AVX-512 exception not implemented\n");
   }
   return delta;
}


static Long ParseEVEX (Prefix* pfx, Escape* esc, const Long deltaIn)
{
   evex=0;
   Long delta = deltaIn;
   UChar evex1 = getUChar(delta);
   UChar evex2 = getUChar(delta+1);
   UChar evex3 = getUChar(delta+2);
   delta += 3;
   DIP("62 %02x %02x %02x\t", evex1, evex2, evex3);

   /* Snarf contents of byte 1 */

   /* ! These four are stored in reverse despite the documentation claiming otherwise ! */
   /* R */  *pfx |= (evex1 & (1<<7)) ? 0 : PFX_REXR;
   /* X */  *pfx |= (evex1 & (1<<6)) ? 0 : PFX_REXX;
   /* B */  *pfx |= (evex1 & (1<<5)) ? 0 : PFX_REXB;
   /* R' */ evex |= (evex1 & (1<<4)) ? 0 : EVEX_R1;

   /* mm */
   switch (evex1 & 0x3) {
      case 1: *esc = ESC_0F;   break;
      case 2: *esc = ESC_0F38; break;
      case 3: *esc = ESC_0F3A; break;
      default: vassert(0);
   }

   /* Snarf contents of byte 2. vvvv is stored in reverse despite the documentation claiming otherwise */
   /* W */  *pfx |= (evex2 & (1<<7)) ? PFX_REXW : 0;
   /* v3 */ *pfx |= (evex2 & (1<<6)) ? 0 : PFX_VEXnV3;
   /* v2 */ *pfx |= (evex2 & (1<<5)) ? 0 : PFX_VEXnV2;
   /* v1 */ *pfx |= (evex2 & (1<<4)) ? 0 : PFX_VEXnV1;
   /* v0 */ *pfx |= (evex2 & (1<<3)) ? 0 : PFX_VEXnV0;
   /* pp */
   switch (evex2 & 3) {
      case 0: break;
      case 1: *pfx |= PFX_66; break;
      case 2: *pfx |= PFX_F3; break;
      case 3: *pfx |= PFX_F2; break;
      default: vassert(0);
   }

   /* Snarf contents of byte 3 */
   /* z */   setZeroMode((evex3 & (1<<7)) >> 7);
   /* L' */  setEVexL((evex3 & (3<<5)) >> 5);
   /* L  */  *pfx  |= (evex3 & (1<<5)) ? PFX_VEXL  : 0;
   /* b  */  evex |= (evex3 & (1<<4)) ? EVEX_EVEXb : 0;
   /* ~v4*/  evex |= (evex3 & (1<<3)) ? 0: EVEX_VEXnV4;
   /* ~m */  setEvexMask( evex3 & 0x7 );

   return delta;
}



static char* get_pfx_name(Prefix pfx) {
   if (haveF2(pfx)) return "F2";
   if (haveF3(pfx)) return "F3";
   if (have66(pfx)) return "66";
   return "NA";
}
static char* get_esc_name(Escape esc) {
   switch (esc) {
      case ESC_NONE: return "ESC_NONE";
      case ESC_0F:   return "ESC_0F";
      case ESC_0F38: return "ESC_0F38";
      case ESC_0F3A: return "ESC_0F3A";
      default: return "no esc";
   }
}
static Int IRTypeVL(IRType ty) {
   switch (ty) {
      case Ity_I8:
      case Ity_I16: case Ity_F16:
      case Ity_I32: case Ity_F32: case Ity_D32:
      case Ity_I64: case Ity_F64: case Ity_D64:
      case Ity_F128: case Ity_I128: case Ity_V128:
         return 0;
      case Ity_V256: return 1;
      case Ity_V512: return 2;
      default: vpanic("invalid VL");
   }
}

Long dis__EVEX( const VexAbiInfo* vbi, Prefix pfx, Long delta ) {

   /* Parse instruction prefix */
   int esc = ESC_NONE;
   delta++;
   delta = ParseEVEX(&pfx, &esc, delta);

   /* Find array entry with a matching opcode, prefix and vector length */
   UChar opcode = getUChar(delta);
   UChar modrm = getUChar(delta+1);
   Long prefix = (pfx & PFX_66) | (pfx & PFX_F2) | (pfx & PFX_F3);
   int width = getRexW(pfx); // Get dst W?
   int ins_id=0;
   while (INS_ARR[ins_id].opcode != opcode) { // match opcode
      ins_id++;
   }
   while (INS_ARR[ins_id].opcode == opcode && // match esc and prefix
         (INS_ARR[ins_id].pfx != prefix || INS_ARR[ins_id].esc != esc)) {
      ins_id++;
   }

   if (INS_ARR[ins_id].opcode != opcode) {
      vex_printf("pfx %s esc %s w %x, opcode 0x%x\n",
            get_pfx_name(prefix), get_esc_name(esc), width, opcode);
      vpanic("esc or pfx not implemented");
   }

   if (INS_ARR[ins_id].src_w != WIG) { // match width
      while ((INS_ARR[ins_id].opcode == opcode) && (INS_ARR[ins_id].src_w != WIG) &&
            (INS_ARR[ins_id].src_w != width+W0 )) {
         ins_id++;
      }
   } else {
      setWIG();
   }


   // Find matching "/2" or "/4" specifier, if it exists. I do not have any justification to its existence.
   if (INS_ARR[ins_id].misc != NULL) {
      if ((char)INS_ARR[ins_id].misc[0] == '/') {
         while ((char)INS_ARR[ins_id].misc[1] != gregLO3ofRM(modrm) + '0') {
            ins_id++;
         }
      }
   }

   if (INS_ARR[ins_id].misc != "noVL") {
      int first_arg = 1;
      if ((INS_ARR[ins_id].args[1] == INS_ARR[ins_id].args[0]) //  the 1st src is dst
         || (INS_ARR[ins_id].args[1] == _rmode)) // the 1st source is rounding mode
         first_arg = 2;
      while ( (getEVexL() != IRTypeVL(INS_ARR[ins_id].arg_type[0])) &&
            (getEVexL() != IRTypeVL(INS_ARR[ins_id].arg_type[first_arg])) ) { // match dst or src1 length
         ins_id++;
      }
   }

   if (INS_ARR[ins_id].opcode != opcode ||
       INS_ARR[ins_id].pfx    != prefix ||
       INS_ARR[ins_id].esc    != esc    ) {
      vex_printf("id %x - pfx %s esc %s w %x, opcode 0x%x, g %x\n",
            ins_id, get_pfx_name(prefix), get_esc_name(esc), width, opcode, gregLO3ofRM(modrm));
      vpanic("width, length or /N not implemented");
   }

   IROp irop = INS_ARR[ins_id].irop;
   setTupleType( INS_ARR[ins_id].type );
   setDstW( INS_ARR[ins_id].dst_w );
   if (INS_ARR[ins_id].mask == MASK_MERGE)
      setZeroMode( False );
   if (INS_ARR[ins_id].mask == MASK_ZERO)
      setZeroMode( True );


   Int argN = 5;
   IRTemp arg[argN];
#define MAX_MULTIPLIER 32
   IRTemp s[argN][MAX_MULTIPLIER];
   IRTemp res[MAX_MULTIPLIER];

   /* Handle exceptions */
   if ((irop == Iop_INVALID) && INS_ARR[ins_id].opFn == NULL ) {
      delta = dis_EVEX_exceptions ( vbi, pfx, esc, delta, ins_id );
      evex = 0;
      return delta;
   }

   /* retrieve arguments */
   delta++;
   Int rG = gregOfRexRM32(pfx, modrm);
   Int rV = getEVexNvvvv(pfx);
   Int rE = 0;
   IRTemp addr = IRTemp_INVALID;
   Int int_val = -1;
   DIP(" %s ", INS_ARR[ins_id].name);
   delta = Get_Instr_Args(vbi, pfx, delta, ins_id, arg, &argN, &int_val, rG, rV, &rE, &addr, modrm);
   if (INS_ARR[ins_id].parameter != -1)
      int_val = INS_ARR[ins_id].parameter;

   IRTemp unmasked = newTemp(INS_ARR[ins_id].arg_type[0]);

   /* Now. We can have EVEX version of something serial, or we can have a truly vector instruction.
    * Vector instructions can reuse AVX-2 IRs, since we split the sources and merge back the results.
    * When they cannot reuse, it still sometimes makes sense to implement one 128-bit IR
    * and use it for 128-, 256- and 512-bit VL instruction versions.
    * In some cases (full vector permutation, for example), we have to use full vector IRs.
    * Serial instructions can reuse AVX-2 IRs - then they use 128-bit arguments.
    * When they cannot reuse, we can call a serial IR for the lower bits, and copy the higher bits */

   if (INS_ARR[ins_id].misc == "xmm_u" || INS_ARR[ins_id].misc == "xmm_b") // Serial case
   {
      IROp get_low = width ? Iop_V128to64 : Iop_V128to32;
      IROp merge_xmm = (getDstW() == W_64) ? Iop_SetV128lo64 : Iop_SetV128lo32;

      // Run IR on the lowest element
      if (irop == Iop_INVALID) { // Function
         if (INS_ARR[ins_id].misc == "xmm_u") { // Unary, use the same arg
            if (INS_ARR[ins_id].args[1] != _rmode) {
               assign (unmasked, binop(merge_xmm,
                        mkexpr(arg[1]),
                        mkexpr(INS_ARR[ins_id].opFn( arg[2], arg[2], int_val))));
            } else {
               assign (unmasked, binop(merge_xmm,
                        mkexpr(arg[2]),
                        mkexpr(INS_ARR[ins_id].opFn( arg[1], arg[3], int_val))));
            }
         } else { // Binary, use both args
            IRTemp arg1_l = newTemp( width ? Ity_I64 : Ity_I32 );
            assign (arg1_l, unop(get_low, mkexpr(arg[1])));
            assign (unmasked, binop(merge_xmm,
                     mkexpr(arg[1]),
                     mkexpr(INS_ARR[ins_id].opFn( arg1_l, arg[2], int_val))));
         }
      }
      else { // IRop
         if (INS_ARR[ins_id].misc == "xmm_u") { // Unary
            if (int_val != -1 ) { // imm8
               assign(unmasked, binop(merge_xmm, mkexpr(arg[1]),
                        binop(irop, mkexpr(arg[2]), mkU8(int_val))));
            } else {
               assign(unmasked, binop(merge_xmm, mkexpr(arg[1]),
                        unop(irop, mkexpr(arg[2]))));
            }
         } else { //binary
            if (int_val != -1 ) { // imm8
               assign(unmasked, binop(merge_xmm,  mkexpr(arg[1]),
                        triop(irop, unop(get_low, mkexpr(arg[1])), mkexpr(arg[2]), mkU8(int_val))));
            } else {
               assign(unmasked, binop(merge_xmm, mkexpr(arg[1]),
                        binop (irop, unop(get_low, mkexpr(arg[1])), mkexpr(arg[2]))));
            }
         }
      }
   }
   else /* NOT a serial xmm */
   {
      /* Split each reg/mem source into <multiplier> parts */
      for (Int i=1; i<argN; i++) {
         if ((INS_ARR[ins_id].args[i] != _rmode) &&
               (INS_ARR[ins_id].args[i] != _imm8)) {
            for (Int j=0; j<INS_ARR[ins_id].multiplier; j++) {
               s[i][j] = IRTemp_INVALID;
            }
            Split_arg(arg[i], s[i], INS_ARR[ins_id].multiplier);
         } else { // Copy rmode and imm8
            for (Int j=0; j<INS_ARR[ins_id].multiplier; j++)
               s[i][j] = arg[i];
         }
      }
      IRType partial_type;
      switch ((UInt) (sizeofIRType(INS_ARR[ins_id].arg_type[0]) / INS_ARR[ins_id].multiplier)) {
         case 64: partial_type = Ity_V512; break;
         case 32: partial_type = Ity_V256; break;
         case 16: partial_type = Ity_V128; break;
         case 8:  partial_type = Ity_I64;  break;
         case 4:  partial_type = Ity_I32;  break;
         case 2:  partial_type = Ity_I16;  break;
         case 1:  partial_type = Ity_I8;   break;
         default: vpanic("partial width not implemented\n");
      }

      for (Int i=0; i < INS_ARR[ins_id].multiplier; i++) {
         res[i] = newTemp(partial_type);
         if (irop == Iop_INVALID) { // Function
            res[i] = INS_ARR[ins_id].opFn(s[1][i], s[2][i], int_val);
         } else { // IRop
            switch (argN) {
               case 2:
                  if (INS_ARR[ins_id].parameter != -1 ) {
                     assign(res[i], binop(irop, mkexpr(s[1][i]), mkU8(int_val)));
                  } else {
                     assign(res[i], unop(irop, mkexpr(s[1][i])));
                  }
                  break;
               case 3:
                  if (INS_ARR[ins_id].parameter != -1 ) {
                     assign(res[i], triop(irop, mkexpr(s[1][i]), mkexpr(s[2][i]), mkU8(int_val)));
                  } else {
                     assign(res[i], binop (irop, mkexpr(s[1][i]), mkexpr(s[2][i])));
                  }
                  break;
               case 4:
                  if (INS_ARR[ins_id].parameter != -1 ) {
                     assign(res[i], qop(irop, mkexpr(s[1][i]), mkexpr(s[2][i]), mkexpr(s[3][i]), mkU8(int_val)));
                  } else {
                     assign(res[i], triop(irop, mkexpr(s[1][i]), mkexpr(s[2][i]), mkexpr(s[3][i])));
                  }
                  break;
               case 5: assign(res[i], qop(irop, mkexpr(s[1][i]), mkexpr(s[2][i]), mkexpr(s[3][i]), mkexpr(s[4][i]))); break;
               default: vpanic("invalid arity\n");
            }
         }
      }
      Merge_dst(unmasked, res, INS_ARR[ins_id].multiplier);
   }

   IRExpr* masked;
   if (INS_ARR[ins_id].mask != MASK_NONE) {
      masked = mask_expr(pfx, mkexpr(unmasked), mkexpr(arg[0]), ins_id);
   } else {
      masked = mkexpr(unmasked);
   }

   Put_Instr_Result(ins_id, masked, rG, rV, rE, addr, modrm);

   evex = 0;
   return delta;
}

Long opmask_operation_decode ( const VexAbiInfo* vbi, UInt esc, Prefix pfx, Long deltaIN )
{
   // NOTE: NOT EVEX prefixed. DO not use the "evex" variable!
   Long delta = deltaIN;

   /* Find array entry with a matching opcode, prefix and vector length */
   UChar  opcode = getUChar(delta-1);
   UChar  modrm = getUChar(delta);

   Long prefix = (pfx & PFX_66) | (pfx & PFX_F2) | (pfx & PFX_F3);
   int width = getRexW(pfx);

   int ins_id=0;
   while (INS_ARR[ins_id].opcode != opcode) // match opcode
      ins_id++;
   while (INS_ARR[ins_id].opcode == opcode && // match esc and prefix
         (INS_ARR[ins_id].pfx != prefix || INS_ARR[ins_id].esc != esc))
      ins_id++;
   if (INS_ARR[ins_id].src_w != WIG) // match width
      while ((INS_ARR[ins_id].opcode == opcode) && (INS_ARR[ins_id].src_w != width+W0 ))
         ins_id++;
   else
      setWIG();

   if (INS_ARR[ins_id].opcode != opcode ||
         INS_ARR[ins_id].pfx    != prefix ||
         INS_ARR[ins_id].esc    != esc    ) {
      vex_printf("id %x - pfx %s esc %s w %x, opcode 0x%x\n",
            ins_id, get_pfx_name(prefix), get_esc_name(esc), width, opcode);
      vpanic("opmask instruction is not implemented\n");
   }

   IROp irop = INS_ARR[ins_id].irop;
   setDstW( INS_ARR[ins_id].dst_w );

   /* retrieve arguments */
   Int argN = 5;
   IRTemp arg[argN];
   Int rG = gregOfRexRM(pfx, modrm);
   Int rV = getVexNvvvv(pfx);
   Int rE = 0;
   IRTemp addr = IRTemp_INVALID;
   Int int_val = -1;
   DIP(" %s ", INS_ARR[ins_id].name);
   delta = Get_Instr_Args(vbi, pfx, delta, ins_id, arg, &argN, &int_val, rG, rV, &rE, &addr, modrm);
   if (INS_ARR[ins_id].parameter != -1)
      int_val = INS_ARR[ins_id].parameter;

   IRTemp res = newTemp(INS_ARR[ins_id].arg_type[0]);

   if (irop == Iop_INVALID) { // Function
      res = INS_ARR[ins_id].opFn(arg[1], arg[2], int_val);
   } else { // IRop
      switch (argN) {
         case 2:
            if (INS_ARR[ins_id].parameter != -1 )
               assign(res, binop(irop, mkexpr(arg[1]), mkU8(int_val)));
            else
               assign(res, unop(irop, mkexpr(arg[1])));
            break;
         case 3:
            if (INS_ARR[ins_id].parameter != -1 )
               assign(res, triop(irop, mkexpr(arg[1]), mkexpr(arg[2]), mkU8(int_val)));
            else
               assign(res, binop (irop, mkexpr(arg[1]), mkexpr(arg[2])));
            break;
         default: vpanic("invalid arity\n");
      }
   }

   // if result is mask register, zero it beforehand
   enum op_encoding op_code = INS_ARR[ins_id].args[0];
   if (op_code == _kG)
      putKReg(rG, mkU64(0));
   if ( (op_code == _kE) || ((op_code == _kmE) && epartIsReg(modrm)) )
      putKReg(rE, mkU64(0));

   Put_Instr_Result(ins_id, mkexpr(res), rG, rV, rE, addr, modrm);
   return delta;
}

#endif /* ndef AVX_512 */
/*--------------------------------------------------------------------*/
/*--- end                                       guest_amd64_toIR.c ---*/
/*--------------------------------------------------------------------*/
