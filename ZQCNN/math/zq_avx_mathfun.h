/* 
   AVX implementation of sin, cos, sincos, exp and log

   Based on "sse_mathfun.h", by Julien Pommier
   http://gruntthepeon.free.fr/ssemath/

   Copyright (C) 2012 Giovanni Garberoglio
   Interdisciplinary Laboratory for Computational Science (LISC)
   Fondazione Bruno Kessler and University of Trento
   via Sommarive, 18
   I-38123 Trento (Italy)

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  (this is the zlib license)
*/

#ifndef _ZQ_AVX_MATHFUN_H_
#define _ZQ_AVX_MATHFUN_H_
#include "../ZQ_CNN_CompileConfig.h"
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
#include <immintrin.h>

#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

	/* natural logarithm computed for 8 simultaneous float
	   return NaN for x <= 0
	*/
	__m256 zq_mm256_log_ps(__m256 x);

	__m256 zq_mm256_exp_ps(__m256 x);


	/* evaluation of 8 sines at onces using AVX intrisics

	   The code is the exact rewriting of the cephes sinf function.
	   Precision is excellent as long as x < 8192 (I did not bother to
	   take into account the special handling they have for greater values
	   -- it does not return garbage for arguments over 8192, though, but
	   the extra precision is missing).

	   Note that it is such that sinf((float)M_PI) = 8.74e-8, which is the
	   surprising but correct result.

	*/
	__m256 zq_mm256_sin_ps(__m256 x);

	/* almost the same as sin_ps */
	__m256 zq_mm256_cos_ps(__m256 x);

	/* since sin256_ps and cos256_ps are almost identical, sincos256_ps could replace both of them..
	   it is almost as fast, and gives you a free cosine with your sine */
	void zq_mm256_sincos_ps(__m256 x, __m256 *s, __m256 *c);

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif
#endif
#endif