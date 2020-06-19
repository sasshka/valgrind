/*--------------------------------------------------------------------*/
/*--- Instrument IR to perform memory checking operations.         ---*/
/*---                                        mc_translate_AVX512.c ---*/
/*--------------------------------------------------------------------*/

/*
   This file is part of MemCheck, a heavyweight Valgrind tool for
   detecting memory errors.

   Copyright (C) 2000-2017 Julian Seward
      jseward@acm.org

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
*/

#ifdef AVX_512
#include <memcheck_AVX512.h>

IRAtom* mkPCastTo( MCEnv* mce, IRType dst_ty, IRAtom* vbits );
void complainIfUndefined ( MCEnv* mce, IRAtom* atom, IRExpr *guard );
IRAtom* mkLazy2 ( MCEnv* mce, IRType finalVty, IRAtom* va1, IRAtom* va2 );
static void setHelperAnns ( MCEnv* mce, IRDirty* di );

IRType shadowTypeV_512 ( IRType ty ) {
   if (ty == Ity_V512) {
      return Ity_V512;
   }
   ppIRType(ty);
   VG_(tool_panic)("memcheck:shadowTypeV");
}

IRExpr* definedOfType_512 ( IRType ty ) {
   if (ty == Ity_V512) {
      return IRExpr_Const(IRConst_V512(0x0000000000000000));
   }
   VG_(tool_panic)("memcheck:definedOfType");
}

/*------------------------------------------------------------*/
/*--- Constructing definedness primitive ops               ---*/
/*------------------------------------------------------------*/

IRAtom* mkDifDV512 ( MCEnv* mce, IRAtom* a1, IRAtom* a2 ) {
   tl_assert(isShadowAtom(mce,a1));
   tl_assert(isShadowAtom(mce,a2));
   return assignNew('V', mce, Ity_V512, binop(Iop_AndV512, a1, a2));
}

IRAtom* mkUifUV512 ( MCEnv* mce, IRAtom* a1, IRAtom* a2 ) {
   tl_assert(isShadowAtom(mce,a1));
   tl_assert(isShadowAtom(mce,a2));
   return assignNew('V', mce, Ity_V512, binop(Iop_OrV512, a1, a2));
}

IRAtom* mkUifU_512 ( MCEnv* mce, IRType ty, IRAtom* a1, IRAtom* a2 ) {
   if (ty == Ity_V512) {
      return mkUifUV512(mce, a1, a2);
   }
   VG_(printf)("\n"); ppIRType(ty); VG_(printf)("\n");
   VG_(tool_panic)("memcheck:mkUifU");
}

/* --------- 'Improvement' functions for AND/OR. --------- */

IRAtom* mkImproveANDV512 ( MCEnv* mce, IRAtom* data, IRAtom* vbits ) {
   tl_assert(isOriginalAtom(mce, data));
   tl_assert(isShadowAtom(mce, vbits));
   tl_assert(sameKindedAtoms(data, vbits));
   return assignNew('V', mce, Ity_V512, binop(Iop_OrV512, data, vbits));
}

IRAtom* mkImproveORV512 ( MCEnv* mce, IRAtom* data, IRAtom* vbits ) {
   tl_assert(isOriginalAtom(mce, data));
   tl_assert(isShadowAtom(mce, vbits));
   tl_assert(sameKindedAtoms(data, vbits));
   return assignNew(
         'V', mce, Ity_V512,
         binop(Iop_OrV512,
            assignNew('V', mce, Ity_V512, unop(Iop_NotV512, data)),
            vbits) );
}

/* --------- Pessimising casts. --------- */

/* The function returns an expression of type DST_TY. If any of the VBITS
   is undefined (value == 1) the resulting expression has all bits set to
   1. Otherwise, all bits are 0. */

IRAtom* CollapseTo1( MCEnv* mce, IRAtom* vbits )
{
   IRType src_ty = typeOfIRExpr(mce->sb->tyenv, vbits);
   switch (src_ty) {
      case Ity_V256: {
         IRAtom* tmp1 = NULL;
         IRAtom* tmp_128_hi = assignNew('V', mce, Ity_V128, unop(Iop_V256toV128_1, vbits));
         IRAtom* tmp_128_lo = assignNew('V', mce, Ity_V128, unop(Iop_V256toV128_0, vbits));
         IRAtom* tmp_128    = assignNew('V', mce, Ity_V128, binop(Iop_OrV128, tmp_128_hi, tmp_128_lo));
         IRAtom* tmp_64_hi  = assignNew('V', mce,  Ity_I64, unop(Iop_V128HIto64, tmp_128));
         IRAtom* tmp_64_lo  = assignNew('V', mce,  Ity_I64, unop(Iop_V128to64, tmp_128));
         IRAtom* tmp_64 = assignNew('V', mce, Ity_I64, binop(Iop_Or64, tmp_64_hi, tmp_64_lo));
         tmp1 = assignNew('V', mce, Ity_I1, unop(Iop_CmpNEZ64, tmp_64));
         return tmp1;
      }
      case Ity_V512: {
         IRAtom* tmp1 = NULL;
         IRAtom* tmp_256_hi = assignNew('V', mce, Ity_V256, unop(Iop_V512toV256_1, vbits));
         IRAtom* tmp_256_lo = assignNew('V', mce, Ity_V256, unop(Iop_V512toV256_0, vbits));
         IRAtom* tmp_256    = assignNew('V', mce, Ity_V256, binop(Iop_OrV256, tmp_256_hi, tmp_256_lo));
         IRAtom* tmp_128_hi = assignNew('V', mce, Ity_V128, unop(Iop_V256toV128_1, tmp_256));
         IRAtom* tmp_128_lo = assignNew('V', mce, Ity_V128, unop(Iop_V256toV128_0, tmp_256));
         IRAtom* tmp_128    = assignNew('V', mce, Ity_V128, binop(Iop_OrV128, tmp_128_hi, tmp_128_lo));
         IRAtom* tmp_64_hi  = assignNew('V', mce,  Ity_I64, unop(Iop_V128HIto64, tmp_128));
         IRAtom* tmp_64_lo  = assignNew('V', mce,  Ity_I64, unop(Iop_V128to64, tmp_128));
         IRAtom* tmp_64 = assignNew('V', mce, Ity_I64, binop(Iop_Or64, tmp_64_hi, tmp_64_lo));
         tmp1 = assignNew('V', mce, Ity_I1, unop(Iop_CmpNEZ64, tmp_64));
         return tmp1;
      }
      default: break;
   }
   ppIRType(src_ty);
   VG_(tool_panic)("mkPCastTo(1)");
}

IRAtom* Widen1to512(MCEnv* mce, IRType dst_ty, IRAtom* tmp1) {
   if( dst_ty == Ity_V512 ) {
      tmp1 = assignNew('V', mce, Ity_I64,  unop(Iop_1Sto64, tmp1));
      tmp1 = assignNew('V', mce, Ity_V128, binop(Iop_64HLtoV128, tmp1, tmp1));
      tmp1 = assignNew('V', mce, Ity_V256, binop(Iop_V128HLtoV256, tmp1, tmp1));
      tmp1 = assignNew('V', mce, Ity_V512, binop(Iop_V256HLtoV512, tmp1, tmp1));
      return tmp1;
   }
   ppIRType(dst_ty);
   VG_(tool_panic)("mkPCastTo(2)");
}

IRAtom* mkPCast64x8 ( MCEnv* mce, IRAtom* at ) {
   return assignNew('V', mce, Ity_V512, unop(Iop_CmpNEZ64x8, at));
}

IRAtom* mkPCast32x16 ( MCEnv* mce, IRAtom* at ) {
   return assignNew('V', mce, Ity_V512, unop(Iop_CmpNEZ32x16, at));
}

IRAtom* binary64x8 ( MCEnv* mce, IRAtom* vatomX, IRAtom* vatomY ) {
   IRAtom* at;
   tl_assert(isShadowAtom(mce, vatomX));
   tl_assert(isShadowAtom(mce, vatomY));
   at = mkUifUV512(mce, vatomX, vatomY);
   at = assignNew('V', mce, Ity_V512, mkPCast64x8(mce, at));
   return at;
}

IRAtom* binary32x16 ( MCEnv* mce, IRAtom* vatomX, IRAtom* vatomY )
{
   IRAtom* at;
   tl_assert(isShadowAtom(mce, vatomX));
   tl_assert(isShadowAtom(mce, vatomY));
   at = mkUifUV512(mce, vatomX, vatomY);
   at = assignNew('V', mce, Ity_V512, mkPCast32x16(mce, at));
   return at;
}

/*------------------------------------------------------------*/
/*--- Generate shadow values from all kinds of IRExprs.    ---*/
/*------------------------------------------------------------*/
#define qop(_op, _arg1, _arg2, _arg3, _arg4) \
   IRExpr_Qop((_op),(_arg1),(_arg2),(_arg3),(_arg4))
IRAtom* expr2vbits_Qop_EVEX ( MCEnv* mce, IROp op,
      IRAtom* atom3, IRAtom* atom4,
      IRAtom* vatom1, IRAtom* vatom2, IRAtom* vatom3, IRAtom* vatom4 )
{
   switch (MC_ARR[INDEX(op)].type) {
      case MC_SPECIAL:
         break;
      case MC_SAME:
         return assignNew('V', mce, MC_ARR[INDEX(op)].args[0],
               qop(op, vatom1, vatom2, vatom3, vatom4));
      case MC_ANY: {
         IRAtom* at = mkPCastTo(mce, Ity_I1, vatom1);
         at = assignNew('V', mce, Ity_I1, binop(Iop_Or1, at, mkPCastTo(mce, Ity_I1, vatom2)));
         at = assignNew('V', mce, Ity_I1, binop(Iop_Or1, at, mkPCastTo(mce, Ity_I1, vatom3)));
         at = assignNew('V', mce, Ity_I1, binop(Iop_Or1, at, mkPCastTo(mce, Ity_I1, vatom4)));
         return mkPCastTo(mce, MC_ARR[INDEX(op)].args[0], at);
      }
      default:
         ppIROp(op);
         VG_(tool_panic)("memcheck:expr2vbits_Triop_EVEX");
   }
   switch (op) {
      case Iop_Compress32x4: case Iop_Compress32x8: case Iop_Compress32x16:
      case Iop_Compress64x2: case Iop_Compress64x4: case Iop_Compress64x8:
         return assignNew('V', mce, MC_ARR[INDEX(op)].args[0],
               qop(op, vatom1, vatom2, atom3, atom4));
   }
   ppIROp(op);
   VG_(tool_panic)("memcheck:expr2vbits_Qop");
}

IRAtom* expr2vbits_Triop_EVEX ( MCEnv* mce, IROp op,
      IRAtom* atom3,
      IRAtom* vatom1, IRAtom* vatom2, IRAtom* vatom3 )
{
   switch (MC_ARR[INDEX(op)].type) {
      case MC_SPECIAL:
         break;
      case MC_SAME:
         return assignNew('V', mce, MC_ARR[INDEX(op)].args[0], triop(op, vatom1, vatom2, vatom3));
      case MC_ANY: {
         IRAtom* at = mkPCastTo(mce, Ity_I1, vatom1);
         at = assignNew('V', mce, Ity_I1, binop(Iop_Or1, at, mkPCastTo(mce, Ity_I1, vatom2)));
         at = assignNew('V', mce, Ity_I1, binop(Iop_Or1, at, mkPCastTo(mce, Ity_I1, vatom3)));
         return mkPCastTo(mce, MC_ARR[INDEX(op)].args[0], at);
      }
      default:
         ppIROp(op);
         VG_(tool_panic)("memcheck:expr2vbits_Triop_EVEX");
   }
   switch (op) {
      case Iop_Align32x4: case Iop_Align32x8:  case Iop_Align32x16:
      case Iop_Align64x2: case Iop_Align64x4:  case Iop_Align64x8:
         return assignNew('V', mce, MC_ARR[INDEX(op)].args[0], triop(op, vatom1, vatom2, atom3));
      case Iop_PermI8x16: case Iop_PermI8x32:  case Iop_PermI8x64:
      case Iop_PermI16x8: case Iop_PermI16x16: case Iop_PermI16x32:
      case Iop_PermI32x4: case Iop_PermI32x8:  case Iop_PermI32x16:
      case Iop_PermI64x2: case Iop_PermI64x4:  case Iop_PermI64x8:
         return assignNew('V', mce, MC_ARR[INDEX(op)].args[0], triop(op, vatom1, vatom2, atom3));
      default:
         ppIROp(op);
         VG_(tool_panic)("memcheck:expr2vbits_Triop_EVEX");
   }
}

#define INDEX(ir) ((ir)-Iop_FIRST_EVEX)
IRAtom* expr2vbits_Binop_EVEX ( MCEnv* mce, IROp op,
                           IRAtom* atom1, IRAtom* atom2,
                           IRAtom* vatom1, IRAtom* vatom2)
{
   switch (MC_ARR[INDEX(op)].type) {
      case MC_SPECIAL:
         break;
      case MC_SAME:
         return assignNew('V', mce, MC_ARR[INDEX(op)].args[0], binop(op, vatom1, vatom2));
      case MC_ANY: {
         IRAtom* at = mkPCastTo(mce, Ity_I1, vatom1);
         at = assignNew('V', mce, Ity_I1, binop(Iop_Or1, at, mkPCastTo(mce, Ity_I1, vatom2)));
         return mkPCastTo(mce, MC_ARR[INDEX(op)].args[0], at);
      }
      default:
         VG_(tool_panic)("memcheck:expr2vbits_Binop_EVEX");
   }

   /* MC_SPECIAL cases */
   IRAtom* (*uifu)    (MCEnv*, IRAtom*, IRAtom*);
   IRAtom* (*difd)    (MCEnv*, IRAtom*, IRAtom*);
   IRAtom* (*improve) (MCEnv*, IRAtom*, IRAtom*);
   uifu = mkUifUV512;
   difd = mkDifDV512;

   switch (op) {
      case Iop_AndV512:
         improve = mkImproveANDV512;
         return
            assignNew( 'V', mce, Ity_V512,
                  difd(mce, uifu(mce, vatom1, vatom2),
                     difd(mce, improve(mce, atom1, vatom1),
                        improve(mce, atom2, vatom2) ) ) );
      case Iop_OrV512:
         improve = mkImproveORV512;
         return
            assignNew( 'V', mce, Ity_V512,
                  difd(mce, uifu(mce, vatom1, vatom2),
                     difd(mce, improve(mce, atom1, vatom1),
                        improve(mce, atom2, vatom2) ) ) );
      case Iop_Perm32x16:
      case Iop_Perm64x4:  case Iop_Perm64x8:
         return assignNew('V', mce, MC_ARR[INDEX(op)].args[0], binop(op, atom1, vatom2));
      default:
         ppIROp(op);
         VG_(tool_panic)("memcheck:expr2vbits_Binop");
   }
}

IRExpr* expr2vbits_Unop_EVEX ( MCEnv* mce, IROp op, IRAtom* vatom )
{
   switch (MC_ARR[INDEX(op)].type) {
      case MC_SPECIAL:
         break;
      case MC_SAME:
         return assignNew('V', mce, MC_ARR[INDEX(op)].args[0], unop(op, vatom));
      case MC_ANY:
         return mkPCastTo(mce, MC_ARR[INDEX(op)].args[0], mkPCastTo(mce, Ity_I1, vatom));
      default:
         VG_(tool_panic)("memcheck:expr2vbits_Unop");
   }

   /* MC_SPECIAL cases */
   switch (op) {
      case Iop_NotV512:
         return vatom;
      default:
         ppIROp(op);
         VG_(tool_panic)("memcheck:expr2vbits_Unop_EVEX");
   }
}

void do_shadow_Store_512 ( MCEnv* mce,
                       IREndness end,
                       IRAtom* addr, UInt bias,
                       IRAtom* data, IRAtom* vdata,
                       IRAtom* guard )
{
   tl_assert(end == Iend_LE);
   if (MC_(clo_mc_level) == 1) {
      vdata = IRExpr_Const( IRConst_V512(V_BITS32_DEFINED) );
   }
   complainIfUndefined( mce, addr, guard );

   IRType tyAddr = mce->hWordTy;
   IROp mkAdd = (tyAddr==Ity_I32) ? Iop_Add32 : Iop_Add64;
   void* helper = &MC_(helperc_STOREV64le);
   const HChar* hname = "MC_(helperc_STOREV64le)";

   IRDirty *diQ[8];
   IRAtom  *addrQ[8], *vdataQ[8], *eBiasQ[8];
   for (Int i = 0; i < 8; i++) {
      eBiasQ[i] = tyAddr==Ity_I32 ? mkU32(bias + i*8 ) : mkU64(bias + i*8);
      addrQ[i]  = assignNew('V', mce, tyAddr, binop(mkAdd, addr, eBiasQ[i]) );
      vdataQ[i] = assignNew('V', mce, Ity_I64, unop(Iop_V512to64_0+i, vdata));
      diQ[i]    = unsafeIRDirty_0_N(
            1/*regparms*/,
            hname, VG_(fnptr_to_fnentry)( helper ),
            mkIRExprVec_2( addrQ[i], vdataQ[i] )
            );
      if (guard)
         diQ[i]->guard = guard;
      setHelperAnns( mce, diQ[i] );
      stmt( 'V', mce, IRStmt_Dirty(diQ[i]) );
   }
   return;
}

#endif /* ndef AVX_512 */
/*--------------------------------------------------------------------*/
/*--- end                                    mc_translate_AVX512.c ---*/
/*--------------------------------------------------------------------*/
