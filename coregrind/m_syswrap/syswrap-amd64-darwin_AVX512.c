
/*--------------------------------------------------------------------*/
/*--- Darwin-specific syscalls       syswrap-amd64-darwin_AVX512.c ---*/
/*--------------------------------------------------------------------*/

/*
   This file is part of Valgrind, a dynamic binary instrumentation
   framework.

   Copyright (C) 2005-2017 Apple Inc.
      Greg Parker  gparker@apple.com

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
#if defined(VGP_amd64_darwin)

#include "config.h"                // DARWIN_VERS

void x86_float_state64_from_vex_AVX512(x86_float_state64_t *mach,
                                       VexGuestAMD64State *vex)
{
   VG_(memcpy)(&mach->__fpu_xmm0,  &vex->guest_ZMM0,   sizeof(mach->__fpu_xmm0));
   VG_(memcpy)(&mach->__fpu_xmm1,  &vex->guest_ZMM1,   sizeof(mach->__fpu_xmm1));
   VG_(memcpy)(&mach->__fpu_xmm2,  &vex->guest_ZMM2,   sizeof(mach->__fpu_xmm2));
   VG_(memcpy)(&mach->__fpu_xmm3,  &vex->guest_ZMM3,   sizeof(mach->__fpu_xmm3));
   VG_(memcpy)(&mach->__fpu_xmm4,  &vex->guest_ZMM4,   sizeof(mach->__fpu_xmm4));
   VG_(memcpy)(&mach->__fpu_xmm5,  &vex->guest_ZMM5,   sizeof(mach->__fpu_xmm5));
   VG_(memcpy)(&mach->__fpu_xmm6,  &vex->guest_ZMM6,   sizeof(mach->__fpu_xmm6));
   VG_(memcpy)(&mach->__fpu_xmm7,  &vex->guest_ZMM7,   sizeof(mach->__fpu_xmm7));
   VG_(memcpy)(&mach->__fpu_xmm8,  &vex->guest_ZMM8,   sizeof(mach->__fpu_xmm8));
   VG_(memcpy)(&mach->__fpu_xmm9,  &vex->guest_ZMM9,   sizeof(mach->__fpu_xmm9));
   VG_(memcpy)(&mach->__fpu_xmm10, &vex->guest_ZMM10,  sizeof(mach->__fpu_xmm10));
   VG_(memcpy)(&mach->__fpu_xmm11, &vex->guest_ZMM11,  sizeof(mach->__fpu_xmm11));
   VG_(memcpy)(&mach->__fpu_xmm12, &vex->guest_ZMM12,  sizeof(mach->__fpu_xmm12));
   VG_(memcpy)(&mach->__fpu_xmm13, &vex->guest_ZMM13,  sizeof(mach->__fpu_xmm13));
   VG_(memcpy)(&mach->__fpu_xmm14, &vex->guest_ZMM14,  sizeof(mach->__fpu_xmm14));
   VG_(memcpy)(&mach->__fpu_xmm15, &vex->guest_ZMM15,  sizeof(mach->__fpu_xmm15));
}

void x86_float_state64_to_vex_AVX512(const x86_float_state64_t *mach,
                                     VexGuestAMD64State *vex)
{
   VG_(memcpy)(&vex->guest_ZMM0,  &mach->__fpu_xmm0,  sizeof(mach->__fpu_xmm0));
   VG_(memcpy)(&vex->guest_ZMM1,  &mach->__fpu_xmm1,  sizeof(mach->__fpu_xmm1));
   VG_(memcpy)(&vex->guest_ZMM2,  &mach->__fpu_xmm2,  sizeof(mach->__fpu_xmm2));
   VG_(memcpy)(&vex->guest_ZMM3,  &mach->__fpu_xmm3,  sizeof(mach->__fpu_xmm3));
   VG_(memcpy)(&vex->guest_ZMM4,  &mach->__fpu_xmm4,  sizeof(mach->__fpu_xmm4));
   VG_(memcpy)(&vex->guest_ZMM5,  &mach->__fpu_xmm5,  sizeof(mach->__fpu_xmm5));
   VG_(memcpy)(&vex->guest_ZMM6,  &mach->__fpu_xmm6,  sizeof(mach->__fpu_xmm6));
   VG_(memcpy)(&vex->guest_ZMM7,  &mach->__fpu_xmm7,  sizeof(mach->__fpu_xmm7));
   VG_(memcpy)(&vex->guest_ZMM8,  &mach->__fpu_xmm8,  sizeof(mach->__fpu_xmm8));
   VG_(memcpy)(&vex->guest_ZMM9,  &mach->__fpu_xmm9,  sizeof(mach->__fpu_xmm9));
   VG_(memcpy)(&vex->guest_ZMM10, &mach->__fpu_xmm10, sizeof(mach->__fpu_xmm10));
   VG_(memcpy)(&vex->guest_ZMM11, &mach->__fpu_xmm11, sizeof(mach->__fpu_xmm11));
   VG_(memcpy)(&vex->guest_ZMM12, &mach->__fpu_xmm12, sizeof(mach->__fpu_xmm12));
   VG_(memcpy)(&vex->guest_ZMM13, &mach->__fpu_xmm13, sizeof(mach->__fpu_xmm13));
   VG_(memcpy)(&vex->guest_ZMM14, &mach->__fpu_xmm14, sizeof(mach->__fpu_xmm14));
   VG_(memcpy)(&vex->guest_ZMM15, &mach->__fpu_xmm15, sizeof(mach->__fpu_xmm15));
}
#endif // defined(VGP_amd64_darwin)
#endif /* ndef AVX_512 */

/*--------------------------------------------------------------------*/
/*--- end                                                          ---*/
/*--------------------------------------------------------------------*/
