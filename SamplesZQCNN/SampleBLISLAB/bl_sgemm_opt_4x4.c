/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas at Austin nor the names
      of its contributors may be used to endorse or promote products
      derived derived derived from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/
#if defined(_WIN32)
#else
#include "arm_neon.h" //NEON

#include <stdio.h>

#include "bl_sgemm_kernel.h"

#define inc_t unsigned long long 



void bl_sgemm_opt_4x4(
                        int              k,
                        float*     a,
                        float*     b,
                        float*     c,
                        unsigned long long ldc,
                        aux_t*         data
                      )
{

    const inc_t cs_c = ldc;
    const inc_t rs_c = 1;
    float alpha_val = 1.0, beta_val = 1.0;
    float *alpha, *beta;

    alpha = &alpha_val;
    beta  = &beta_val;


	//void* a_next = bli_auxinfo_next_a( data );
	//void* b_next = bli_auxinfo_next_b( data );

    //void* a_next = bli_auxinfo_next_a( data );
	//void* b_next = bli_auxinfo_next_b( data );

    //float* a_next = data->a_next;
    float* b_next = data->b_next;



	float32x4_t alphav;
	alphav = vmovq_n_f32( *alpha );

	float32x4_t av1;
	float32x4_t av2;
	float32x4_t av3;
	float32x4_t av4;

	float32x4_t bv1;
	float32x4_t bv2;
	float32x4_t bv3;
	float32x4_t bv4;

	dim_t  k_iter = k/4;
	dim_t  k_left = k%4;
	dim_t  i; 

	// Vector for column 0
	float32x4_t cv0;
	// Vector for column 1
	float32x4_t cv1;
	// Vector for column 2
	float32x4_t cv2;
	// Vector for column 3
	float32x4_t cv3;

	if( rs_c == 1 )
	{
		// Load column 0
 		cv0 = vld1q_f32( c + 0*rs_c + 0*cs_c ); 
	
		// Load column 1
 		cv1 = vld1q_f32( c + 0*rs_c + 1*cs_c ); 
	
		// Load column 2
 		cv2 = vld1q_f32( c + 0*rs_c + 2*cs_c ); 
	
		// Load column 3
 		cv3 = vld1q_f32( c + 0*rs_c + 3*cs_c ); 
	}	
	else
	{
		// Load column 0
		cv0 = vld1q_lane_f32( c + 0*rs_c + 0*cs_c, cv0, 0);
		cv0 = vld1q_lane_f32( c + 1*rs_c + 0*cs_c, cv0, 1);
		cv0 = vld1q_lane_f32( c + 2*rs_c + 0*cs_c, cv0, 2);
		cv0 = vld1q_lane_f32( c + 3*rs_c + 0*cs_c, cv0, 3);
	
		// Load column 1
		cv1 = vld1q_lane_f32( c + 0*rs_c + 1*cs_c, cv1, 0);
		cv1 = vld1q_lane_f32( c + 1*rs_c + 1*cs_c, cv1, 1);
		cv1 = vld1q_lane_f32( c + 2*rs_c + 1*cs_c, cv1, 2);
		cv1 = vld1q_lane_f32( c + 3*rs_c + 1*cs_c, cv1, 3);
	
		// Load column 2
		cv2 = vld1q_lane_f32( c + 0*rs_c + 2*cs_c, cv2, 0);
		cv2 = vld1q_lane_f32( c + 1*rs_c + 2*cs_c, cv2, 1);
		cv2 = vld1q_lane_f32( c + 2*rs_c + 2*cs_c, cv2, 2);
		cv2 = vld1q_lane_f32( c + 3*rs_c + 2*cs_c, cv2, 3);
	
		// Load column 3
		cv3 = vld1q_lane_f32( c + 0*rs_c + 3*cs_c, cv3, 0);
		cv3 = vld1q_lane_f32( c + 1*rs_c + 3*cs_c, cv3, 1);
		cv3 = vld1q_lane_f32( c + 2*rs_c + 3*cs_c, cv3, 2);
		cv3 = vld1q_lane_f32( c + 3*rs_c + 3*cs_c, cv3, 3);

	}

	// Vector for accummulating column 0
	float32x4_t abv0;
	// Initialize vector to 0.0
	abv0 = vmovq_n_f32( 0.0 );

	// Vector for accummulating column 1
	float32x4_t abv1;
	// Initialize vector to 0.0
	abv1 = vmovq_n_f32( 0.0 );

	// Vector for accummulating column 2
	float32x4_t abv2;
	// Initialize vector to 0.0
	abv2 = vmovq_n_f32( 0.0 );

	// Vector for accummulating column 3
	float32x4_t abv3;
	// Initialize vector to 0.0
	abv3 = vmovq_n_f32( 0.0 );

	for ( i = 0; i < k_iter; ++i ) 
	{ 
		// Begin iter 0
 		av1 = vld1q_f32( a ); 

		__builtin_prefetch( a + 224 );
		__builtin_prefetch( b + 224 );
	
 		bv1 = vld1q_f32( b ); 

		abv0 = vmlaq_lane_f32( abv0, av1, vget_low_f32(bv1), 0 );
		abv1 = vmlaq_lane_f32( abv1, av1, vget_low_f32(bv1), 1 );
		abv2 = vmlaq_lane_f32( abv2, av1, vget_high_f32(bv1), 0 );
		abv3 = vmlaq_lane_f32( abv3, av1, vget_high_f32(bv1), 1 );


		av2 = vld1q_f32( a+4 ); 

		//__builtin_prefetch( a + 116 );
		//__builtin_prefetch( b + 116 );
	
 		bv2 = vld1q_f32( b+4 ); 

		abv0 = vmlaq_lane_f32( abv0, av2, vget_low_f32(bv2), 0 );
		abv1 = vmlaq_lane_f32( abv1, av2, vget_low_f32(bv2), 1 );
		abv2 = vmlaq_lane_f32( abv2, av2, vget_high_f32(bv2), 0 );
		abv3 = vmlaq_lane_f32( abv3, av2, vget_high_f32(bv2), 1 );

		av3 = vld1q_f32( a+8 ); 

		//__builtin_prefetch( a + 120 );
		//__builtin_prefetch( b + 120 );
	
 		bv3 = vld1q_f32( b+8 ); 

		abv0 = vmlaq_lane_f32( abv0, av3, vget_low_f32(bv3), 0 );
		abv1 = vmlaq_lane_f32( abv1, av3, vget_low_f32(bv3), 1 );
		abv2 = vmlaq_lane_f32( abv2, av3, vget_high_f32(bv3), 0 );
		abv3 = vmlaq_lane_f32( abv3, av3, vget_high_f32(bv3), 1 );


		av4 = vld1q_f32( a+12); 

		//__builtin_prefetch( a + 124 );
		//__builtin_prefetch( b + 124 );
	
 		bv4 = vld1q_f32( b+12); 

		abv0 = vmlaq_lane_f32( abv0, av4, vget_low_f32(bv4), 0 );
		abv1 = vmlaq_lane_f32( abv1, av4, vget_low_f32(bv4), 1 );
		abv2 = vmlaq_lane_f32( abv2, av4, vget_high_f32(bv4), 0 );
		abv3 = vmlaq_lane_f32( abv3, av4, vget_high_f32(bv4), 1 );



		a += 16; 
		b += 16; 
	} 

	for ( i = 0; i < k_left; ++i ) 
	{ 
 		av1 = vld1q_f32( a ); 

		__builtin_prefetch( a + 112 );
		__builtin_prefetch( b + 112 );
	
 		bv1 = vld1q_f32( b ); 

		abv0 = vmlaq_lane_f32( abv0, av1, vget_low_f32(bv1), 0 );
		abv1 = vmlaq_lane_f32( abv1, av1, vget_low_f32(bv1), 1 );
		abv2 = vmlaq_lane_f32( abv2, av1, vget_high_f32(bv1), 0 );
		abv3 = vmlaq_lane_f32( abv3, av1, vget_high_f32(bv1), 1 );

		a += 4; 
		b += 4; 
	}

	//__builtin_prefetch( a_next );
	__builtin_prefetch( b_next );

	cv0 = vmulq_n_f32( cv0, *beta );
	cv1 = vmulq_n_f32( cv1, *beta );
	cv2 = vmulq_n_f32( cv2, *beta );
	cv3 = vmulq_n_f32( cv3, *beta );

	cv0 = vmlaq_f32( cv0, abv0, alphav );
	cv1 = vmlaq_f32( cv1, abv1, alphav );
	cv2 = vmlaq_f32( cv2, abv2, alphav );
	cv3 = vmlaq_f32( cv3, abv3, alphav );

	if( rs_c == 1 )
	{
		// Store column 0
  		vst1q_f32( c + 0*rs_c + 0*cs_c, cv0 ); 
		// Store column 1
  		vst1q_f32( c + 0*rs_c + 1*cs_c, cv1 ); 
		// Store column 2
  		vst1q_f32( c + 0*rs_c + 2*cs_c, cv2 ); 
		// Store column 3
  		vst1q_f32( c + 0*rs_c + 3*cs_c, cv3 ); 
	}
	else{
		// Store column 0
		vst1q_lane_f32( c + 0*rs_c + 0*cs_c, cv0, 0);
		vst1q_lane_f32( c + 1*rs_c + 0*cs_c, cv0, 1);
		vst1q_lane_f32( c + 2*rs_c + 0*cs_c, cv0, 2);
		vst1q_lane_f32( c + 3*rs_c + 0*cs_c, cv0, 3);
	
		// Store column 1
		vst1q_lane_f32( c + 0*rs_c + 1*cs_c, cv1, 0);
		vst1q_lane_f32( c + 1*rs_c + 1*cs_c, cv1, 1);
		vst1q_lane_f32( c + 2*rs_c + 1*cs_c, cv1, 2);
		vst1q_lane_f32( c + 3*rs_c + 1*cs_c, cv1, 3);
	
		// Store column 2
		vst1q_lane_f32( c + 0*rs_c + 2*cs_c, cv2, 0);
		vst1q_lane_f32( c + 1*rs_c + 2*cs_c, cv2, 1);
		vst1q_lane_f32( c + 2*rs_c + 2*cs_c, cv2, 2);
		vst1q_lane_f32( c + 3*rs_c + 2*cs_c, cv2, 3);
	
		// Store column 3
		vst1q_lane_f32( c + 0*rs_c + 3*cs_c, cv3, 0);
		vst1q_lane_f32( c + 1*rs_c + 3*cs_c, cv3, 1);
		vst1q_lane_f32( c + 2*rs_c + 3*cs_c, cv3, 2);
		vst1q_lane_f32( c + 3*rs_c + 3*cs_c, cv3, 3);
	}
}

#endif


