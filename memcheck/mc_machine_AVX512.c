/*--------------------------------------------------------------------*/
/*--- Contains machine-specific (guest-state-layout-specific)      ---*/
/*--- support for origin tracking for AVX-512 machines             ---*/
/*---                                             mc_machine_512.c ---*/
/*--------------------------------------------------------------------*/

/*
   This file is part of MemCheck, a heavyweight Valgrind tool for
   detecting memory errors.

   Copyright (C) 2008-2017 OpenWorks Ltd
      info@open-works.co.uk

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

Int start_zmm_vector_register( Int offset, Int sz ) {
# if defined(VGA_amd64)
# define START_VECTOR_REGS (offsetof(VexGuestAMD64State, guest_ZMM0))
# define SIZE_VECTOR_REG   (sizeof(((VexGuestAMD64State*)0)->guest_ZMM0))
   if ( offset >= START_VECTOR_REGS && sz <= SIZE_VECTOR_REG) {
      // all vector registers have the same length
      Int reg_number = ((offset - START_VECTOR_REGS) / SIZE_VECTOR_REG);
      if (reg_number <= 32) // 32 is a valid pseudo register
         return START_VECTOR_REGS + reg_number*SIZE_VECTOR_REG;
   }
   else
# undef START_VECTOR_REGS
# undef SIZE_VECTOR_REG
# endif /* VGA_amd64 */
   return 0;
}

Int start_mask_register( Int offset, Int sz ) {
# if defined(VGA_amd64)
# define START_MASK_REGS (offsetof(VexGuestAMD64State, guest_K0))
# define SIZE_MASK_REG   (sizeof(((VexGuestAMD64State*)0)->guest_K0))
   if ( offset >= START_MASK_REGS && sz <= SIZE_MASK_REG ) {
      // all mask registers have the same length
      Int reg_number = ((offset - START_MASK_REGS) / SIZE_MASK_REG);
      if (reg_number <= 7)
         return START_MASK_REGS + reg_number*SIZE_MASK_REG;
   }
# undef START_MASK_REGS
# undef SIZE_MASK_REG
# endif /* VGA_amd64 */
   return 0;
}

#endif /* ndef AVX_512 */
/*--------------------------------------------------------------------*/
/*--- end                                        mc_machine_512.c  ---*/
/*--------------------------------------------------------------------*/
