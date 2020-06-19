/*--------------------------------------------------------------------*/
/*--- Header for AVX-512 for DARWIN  syswrap-amd64-darwin_AVX512.h ---*/
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
#ifndef __SYSWRAP_AVX512_H
#define __SYSWRAP_AVX512_H

#if defined(VGP_amd64_darwin)
#include "config.h"                // DARWIN_VERS
void x86_float_state64_from_vex_AVX512(x86_float_state64_t *mach, VexGuestAMD64State *vex);
void x86_float_state64_to_vex_AVX512(const x86_float_state64_t *mach, VexGuestAMD64State *vex);
#endif

#endif /* __SYSWRAP_AVX512_H */
#endif /* ndef AVX_512 */
/*--------------------------------------------------------------------*/
/*--- end                                                          ---*/
/*--------------------------------------------------------------------*/
