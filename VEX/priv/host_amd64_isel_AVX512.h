/*---------------------------------------------------------------*/
/*--- begin                          host_amd64_isel_AVX512.h ---*/
/*---------------------------------------------------------------*/

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
#ifdef AVX_512
#ifndef __HOST_AMD64_ISEL512_H
#define __HOST_AMD64_ISEL512_H

#include "libvex_basictypes.h"
#include "libvex_ir.h"
#include "libvex.h"
#include "ir_match.h"
#include "host_generic_regs.h"
#include "host_amd64_defs.h"
#include "host_AVX512.h"

#define MULT512 4

typedef
struct {
    /* Constant -- are set at the start and do not change. */
    IRTypeEnv*   type_env;
    HReg*        vregmaps[MULT512];
    Int          n_vregmap;
    UInt         hwcaps;
    Bool         chainingAllowed;
    Addr64       max_ga;
    /* These are modified as we go along. */
    HInstrArray* code;
    Int          vreg_ctr;
}
ISelEnv;

#define vregmap vregmaps[0]
#define vregmapHI vregmaps[1]

AMD64RMI*     iselIntExpr_RMI     ( ISelEnv* env, const IRExpr* e );
HReg          iselIntExpr_R       ( ISelEnv* env, const IRExpr* e );
AMD64RI*      iselIntExpr_RI      ( ISelEnv* env, const IRExpr* e );
AMD64CondCode iselCondCode_C      ( ISelEnv* env, const IRExpr* e );
HReg          iselDblExpr         ( ISelEnv* env, const IRExpr* e );
HReg          iselFltExpr         ( ISelEnv* env, const IRExpr* e );
HReg          iselVecExpr         ( ISelEnv* env, const IRExpr* e );
void          iselDVecExpr        ( /*OUT*/HReg* rHi, HReg* rLo,  ISelEnv* env, const IRExpr* e );
void          iselExpr512         ( /*OUT*/HReg *dst, ISelEnv* env, const IRExpr* e );

void iselStmt ( ISelEnv* env, IRStmt* stmt );
void iselStmt_512 (ISelEnv* env, IRStmt* stmt);

void doHelperCall ( /*OUT*/UInt*   stackAdjustAfterCall, /*OUT*/RetLoc* retloc,
                    ISelEnv* env, IRExpr* guard, IRCallee* cee, IRType retTy, IRExpr** args );
void doHelperCall_512( /*OUT*/UInt* stackAdjustAfterCall, /*OUT*/RetLoc* retloc,
        ISelEnv* env, IRExpr* guard, IRCallee* cee, IRType retTy, IRExpr** args);

HReg newVRegI ( ISelEnv* env );
HReg newVRegV ( ISelEnv* env );
void addInstr ( ISelEnv* env, AMD64Instr* instr );
AMD64Instr* mk_iMOVsd_RR ( HReg src, HReg dst );
AMD64Instr* mk_vMOVsd_RR ( HReg src, HReg dst );
void add_to_rsp ( ISelEnv* env, Int n );
void sub_from_rsp ( ISelEnv* env, Int n );
HReg generate_zeroes_V128 ( ISelEnv* env );
HReg do_sse_NotV128 ( ISelEnv* env, HReg src );

#endif /*__HOST_AMD64_ISEL512_H*/
#endif /* ndef AVX_512 */
/*---------------------------------------------------------------*/
/*--- end                            host_amd64_isel_AVX512.h ---*/
/*---------------------------------------------------------------*/
