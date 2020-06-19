/*---------------------------------------------------------------*/
/*--- begin                      guest_amd64_helpers_AVX512.c ---*/
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
#include "libvex_guest_amd64.h"
#include "guest_amd64_defs.h"
#include "libvex_ir.h"

/* Claim to be the following AVX-512 capable CPU.
   With the following change: claim that XSaveOpt is not available, by
   cpuid(eax=0xD,ecx=1).eax[0] returns 0, compared to 1 on the real
   CPU.
*/
void amd64g_dirtyhelper_CPUID_avx512 ( VexGuestAMD64State* st ) {
#  define SET_ABCD(_a,_b,_c,_d)                \
      do { st->guest_RAX = (ULong)(_a);        \
           st->guest_RBX = (ULong)(_b);        \
           st->guest_RCX = (ULong)(_c);        \
           st->guest_RDX = (ULong)(_d);        \
      } while (0);

   UInt old_eax = (UInt)st->guest_RAX;
   UInt old_ecx = (UInt)st->guest_RCX;

   switch (old_eax) {
      case 0x00000000:
         SET_ABCD(0x00000016, 0X756e6547, 0x6c65746e, 0x49656e69);
         break;
      case 0x00000001:
         /* Don't advertise RDRAND support, bit 30 in ECX.  */
         SET_ABCD(0x00050654, 0x43400800, 0x3ffafbff, 0xbfebfbff);
         break;
      case 0x00000002:
         SET_ABCD(0x76036301, 0x00f0b5ff, 0x00000000, 0x00c30000);
         break;
      case 0x00000004:
         switch (old_ecx) {
            case 0x00000000:
               SET_ABCD(0xfc00c121, 0x01c0003f, 0x0000003f, 0x00000000);
               break;
            case 0x00000001:
               SET_ABCD(0xfc00c122, 0x01c0003f, 0x0000003f, 0x00000000);
               break;
            case 0x00000002:
               SET_ABCD(0xfc01c143, 0x03c0003f, 0x000003ff, 0x00000000);
               break;
            case 0x00000003:
            default:
               SET_ABCD(0x7c0fc163, 0x0280003f, 0x0000bfff, 0x00000004);
               break;
         }
         break;
      case 0x00000005:
         SET_ABCD(0x00000040, 0x00000040, 0x00000003, 0x00002020);
         break;
      case 0x00000006:
         SET_ABCD(0x00000ef7, 0x00000002, 0x00000009, 0x00000000);
         break;
      case 0x00000007:
         switch (old_ecx) {
             case 0x00000000:
                 SET_ABCD(0x00000000, 0xd39ffffb, 0x00000008, 0xc000000);
                 break;
             default:
                 SET_ABCD(0x00000000, 0x00000000, 0x00000000, 0x00000000);
                 break;
         }
         break;
      case 0x0000000a:
         SET_ABCD(0x07300404, 0x00000000, 0x00000000, 0x00000603);
         break;
      case 0x0000000b:
         switch (old_ecx) {
            case 0x00000000:
               SET_ABCD(0x00000001, 0x00000002, 0x00000100, 0x00000043);
               break;
            case 0x00000001:
               SET_ABCD(0x00000006, 0x00000030, 0x00000201, 0x00000043);
               break;
            default:
               SET_ABCD(0x00000000, 0x00000000, old_ecx, 0x00000043);
               break;
         }
         break;
      case 0x0000000d:
         switch (old_ecx) {
            case 0x00000000:
               // important
               SET_ABCD(0x000002ff, 0x00000340, 0x00000340, 0x00000000);
               break;
            case 0x00000001:
               // important
               SET_ABCD(0x00000000, 0x00000000, 0x00000000, 0x00000000);
               break;
            case 0x00000002:
               SET_ABCD(0x00000100, 0x00000240, 0x00000000, 0x00000000);
               break;
            default:
               SET_ABCD(0x00000040, 0x0003c000, 0x00000000, 0x00000000);
               break;
         }
         break;
      case 0x80000000:
         SET_ABCD(0x80000008, 0x00000000, 0x00000000, 0x00000000);
         break;
      case 0x80000001:
         SET_ABCD(0x00000000, 0x00000000, 0x00000121, 0x2c100800);
         break;
      case 0x80000002:
         SET_ABCD(0x65746e49, 0x2952286c, 0x6f655820, 0x2952286e);
         break;
      case 0x80000003:
         SET_ABCD(0x616c5020, 0x756e6974, 0x3138206d, 0x43203836);
         break;
      case 0x80000004:
         SET_ABCD(0x40205550, 0x372e3220, 0x7a484730, 0x00000000)
         break;
      case 0x80000006:
         SET_ABCD(0x00000000, 0x00000000, 0x01006040, 0x00000000);
         break;
      case 0x80000007:
         SET_ABCD(0x00000000, 0x00000000, 0x00000000, 0x00000100);
         break;
      case 0x80000008:
         SET_ABCD(0x0000302e, 0x00000000, 0x00000000, 0x00000000);
         break;
      case 0x00000003:
      case 0x00000008:
      case 0x00000009:
      case 0x0000000c:
      case 0x80000005:
         SET_ABCD(0x00000000, 0x00000000, 0x00000000, 0x00000000);
         break;
      default:
         SET_ABCD(0x000000e7, 0x00000a80, 0x00000a80, 0x00000000);
         break;
   }
#  undef SET_ABCD
}

/* Initialise ZMM registers */
void LibVEX_GuestAMD64_initialise_ZMM ( /*OUT*/VexGuestAMD64State* vex_state )
{
#  define AVXZERO(_zmm) \
      do { \
         for (Int i=0; i<16; i++) { \
            _zmm[i]=0; \
         } \
      } while (0)
   vex_state->guest_SSEROUND = (ULong)Irrm_NEAREST;
   AVXZERO(vex_state->guest_ZMM0);
   AVXZERO(vex_state->guest_ZMM1);
   AVXZERO(vex_state->guest_ZMM2);
   AVXZERO(vex_state->guest_ZMM3);
   AVXZERO(vex_state->guest_ZMM4);
   AVXZERO(vex_state->guest_ZMM5);
   AVXZERO(vex_state->guest_ZMM6);
   AVXZERO(vex_state->guest_ZMM7);
   AVXZERO(vex_state->guest_ZMM8);
   AVXZERO(vex_state->guest_ZMM9);
   AVXZERO(vex_state->guest_ZMM10);
   AVXZERO(vex_state->guest_ZMM11);
   AVXZERO(vex_state->guest_ZMM12);
   AVXZERO(vex_state->guest_ZMM13);
   AVXZERO(vex_state->guest_ZMM14);
   AVXZERO(vex_state->guest_ZMM15);
   AVXZERO(vex_state->guest_ZMM16);
   AVXZERO(vex_state->guest_ZMM17);
   AVXZERO(vex_state->guest_ZMM18);
   AVXZERO(vex_state->guest_ZMM19);
   AVXZERO(vex_state->guest_ZMM20);
   AVXZERO(vex_state->guest_ZMM21);
   AVXZERO(vex_state->guest_ZMM22);
   AVXZERO(vex_state->guest_ZMM23);
   AVXZERO(vex_state->guest_ZMM24);
   AVXZERO(vex_state->guest_ZMM25);
   AVXZERO(vex_state->guest_ZMM26);
   AVXZERO(vex_state->guest_ZMM27);
   AVXZERO(vex_state->guest_ZMM28);
   AVXZERO(vex_state->guest_ZMM29);
   AVXZERO(vex_state->guest_ZMM30);
   AVXZERO(vex_state->guest_ZMM31);
   AVXZERO(vex_state->guest_ZMM32);
#  undef AVXZERO
}

#endif /* ndef AVX_512 */
/*---------------------------------------------------------------*/
/*--- end                        guest_amd64_helpers_AVX512.c ---*/
/*---------------------------------------------------------------*/
