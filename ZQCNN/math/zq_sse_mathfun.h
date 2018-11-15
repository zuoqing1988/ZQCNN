/* SIMD (SSE1+MMX or SSE2) implementation of sin, cos, exp and log
Inspired by Intel Approximate Math library, and based on the
corresponding algorithms of the cephes math library
The default is to use the SSE1 version. If you define USE_SSE2 the
the SSE2 intrinsics will be used in place of the MMX intrinsics. Do
not expect any significant performance improvement with SSE2.
*/

/* Copyright (C) 2007  Julien Pommier
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


#ifndef _ZQ_SSE_MATHFUN_H_
#define _ZQ_SSE_MATHFUN_H_
#include "../ZQ_CNN_CompileConfig.h"
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
#include <xmmintrin.h>

#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

	/* natural logarithm computed for 4 simultaneous float
	return NaN for x <= 0
	*/
	__m128 zq_mm128_log_ps(__m128 x);

	__m128 zq_mm128_exp_ps(__m128 x);



	/* evaluation of 4 sines at onces, using only SSE1+MMX intrinsics so
	it runs also on old athlons XPs and the pentium III of your grand
	mother.
	The code is the exact rewriting of the cephes sinf function.
	Precision is excellent as long as x < 8192 (I did not bother to
	take into account the special handling they have for greater values
	-- it does not return garbage for arguments over 8192, though, but
	the extra precision is missing).
	Note that it is such that sinf((float)M_PI) = 8.74e-8, which is the
	surprising but correct result.
	Performance is also surprisingly good, 1.33 times faster than the
	macos vsinf SSE2 function, and 1.5 times faster than the
	__vrs4_sinf of amd's ACML (which is only available in 64 bits). Not
	too bad for an SSE1 function (with no special tuning) !
	However the latter libraries probably have a much better handling of NaN,
	Inf, denormalized and other special arguments..
	On my core 1 duo, the execution of this function takes approximately 95 cycles.
	From what I have observed on the experiments with Intel AMath lib, switching to an
	SSE2 version would improve the perf by only 10%.
	Since it is based on SSE intrinsics, it has to be compiled at -O2 to
	deliver full speed.
	*/
	__m128 zq_mm128_sin_ps(__m128 x);

	/* almost the same as sin_ps */
	__m128 zq_mm128_cos_ps(__m128 x);

	/* since sin_ps and cos_ps are almost identical, sincos_ps could replace both of them..
	it is almost as fast, and gives you a free cosine with your sine */
	void zq_mm128_sincos_ps(__m128 x, __m128 *s, __m128 *c);

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif
#endif
#endif
