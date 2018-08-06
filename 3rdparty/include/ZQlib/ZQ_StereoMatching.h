#ifndef _ZQ_STEREO_MATCHING_H_
#define _ZQ_STEREO_MATCHING_H_
#pragma once

#include "ZQ_DoubleImage.h"
#include <stdio.h>
#include <stdint.h>
#include <smmintrin.h>
#include <tmmintrin.h>


namespace ZQ
{
	class ZQ_StereoMatching
	{
	private:

		/* row1    : [width1 x nChannels] uchar array
		*  row2    : [width2 x nChannels] uchar array
		*  dotValue: [W*D]	uchar array
		* W = __max(width1, width2+maxD);
		* D = maxD + 1;
		*/
		static void _computeMatchingCostNCC_Dot_OneRow(const unsigned char* row1, const unsigned char* row2, int width1, int width2, int nChannels, int maxD, float* dotValue)
		{
			int W = __max(width1, width2 + maxD);
			int D = maxD + 1;
			memset(dotValue, 0, sizeof(float)*W*D);
			for (int c = 0; c < nChannels; c++)
			{
				for (int w = 0; w < W; w++)
				{
					int coordX1 = w;
					float val1;
					if (coordX1 >= width1)
					{
						val1 = row1[(width1 - 1)*nChannels + c];
					}
					else
					{
						val1 = row1[coordX1*nChannels + c];
					}
					for (int d = 0; d < D; d++)
					{
						int coordX2 = w - d;
						float val2;
						if (coordX2 < 0)
						{
							val2 = row2[c];
						}
						else if (coordX2 >= width2)
						{
							val2 = row2[(width2 - 1)*nChannels + c];
						}
						else
						{
							val2 = row2[coordX2*nChannels + c];
						}
						dotValue[w*D + d] += val1*val2;
					}
				}
			}
		}


		/* row1    : [width1 x nChannels] uchar array
		*  row2    : [width2 x nChannels] uchar array
		*  pixDiff : [W*D]	uchar array
		* W = __max(width1, width2+maxD);
		* D = maxD + 1;
		*/
		static void _computeMatchingCostBT_OneRow(const unsigned char* row1, const unsigned char* row2, int width1, int width2, int nChannels, int maxD, short* pixDiff, int truncate_value)
		{
			int W = __max(width1, width2 + maxD);
			int D = maxD + 1;
			unsigned char* buffer = new unsigned char[(width1 + width2) * 2];
			unsigned char* left_half1 = buffer;
			unsigned char* right_half1 = left_half1 + width1;
			unsigned char* left_half2 = right_half1 + width1;
			unsigned char* right_half2 = left_half2 + width2;
			memset(pixDiff, 0, sizeof(short)*W*D);
			for (int c = 0; c < nChannels; c++)
			{
				// left img
				left_half1[0] = row1[c];
				left_half1[width1-1] = ((int)row1[(width1 - 2)*nChannels + c] + (int)row1[(width1 - 1)*nChannels + c]) / 2;
				right_half1[0] = ((int)row1[c] + (int)row1[nChannels + c]) / 2;
				right_half1[width1 - 1] = row1[(width1 - 1)*nChannels + c];
				for (int w = 1; w < width1 - 1; w++)
				{
					int off = w*nChannels + c;
					left_half1[w] = ((int)row1[off - nChannels] + (int)row1[off]) / 2;
					right_half1[w] = ((int)row1[off + nChannels] + (int)row1[off]) / 2;
				}

				// right img
				left_half2[0] = row2[c];
				left_half2[width2 - 1] = ((int)row2[(width2 - 2)*nChannels + c] + (int)row2[(width2 - 1)*nChannels + c]) / 2;
				right_half2[0] = ((int)row2[c] + (int)row2[nChannels + c]) / 2;
				right_half2[width2 - 1] = row2[(width2 - 1)*nChannels + c];
				for (int w = 1; w < width2 - 1; w++)
				{
					int off = w*nChannels + c;
					left_half2[w] = ((int)row2[off - nChannels] + (int)row2[off]) / 2;
					right_half2[w] = ((int)row2[off + nChannels] + (int)row2[off]) / 2;
				}

				for (int w = 0; w < W; w++)
				{
					int coordX1 = w;
					int val1, left_val1, right_val1;
					if (coordX1 >= width1)
					{
						val1 = row1[(width1 - 1)*nChannels + c];
						left_val1 = val1;
						right_val1 = val1;
					}
					else
					{
						val1 = row1[coordX1*nChannels + c];
						left_val1 = left_half1[coordX1];
						right_val1 = right_half1[coordX1];
					}
					for (int d = 0; d < D; d++)
					{
						int coordX2 = w - d;
						int val2, left_val2, right_val2;
						if (coordX2 < 0)
						{
							val2 = row2[c];
							left_val2 = val2;
							right_val2 = val2;

						}
						else if (coordX2 >= width2)
						{
							val2 = row2[(width2 - 1)*nChannels + c];
							left_val2 = val2;
							right_val2 = val2;
						}
						else
						{
							val2 = row2[coordX2*nChannels + c];
							left_val2 = left_half2[coordX2];
							right_val2 = right_half2[coordX2];
						}

						int diff1 = abs(val1 - val2);
						int diff2 = abs(val1 - left_val2);
						int diff3 = abs(val1 - right_val2);
						int diff4 = abs(left_val1 - val2);
						int diff5 = abs(right_val1 - val2);
						int min_diff = __min(diff1, diff2);
						min_diff = __min(min_diff, diff3);
						min_diff = __min(min_diff, diff4);
						min_diff = __min(min_diff, diff5);
						min_diff = __min(min_diff, truncate_value);
						pixDiff[w*D + d] += min_diff;
					}
					
				}
				
			}
			delete []buffer;
		}

		static void _computeMatchingCostBT_OneRow_SSE(const unsigned char* row1, const unsigned char* row2, int width1, int width2, int nChannels, int maxD, short* pixDiff, int truncate_value)
		{
			int W = __max(width1, width2 + maxD);
			int D = maxD + 1;
			short* buffer = new short[(width1 + D - 1 + width2 + 2*(D - 1)) * 3];
			short* pad_row1 = buffer;
			short* left_half1 = pad_row1 + width1 + D - 1;
			short* right_half1 = left_half1 + width1 + D - 1;
			short* pad_row2 = right_half1 + width1 + D - 1;
			short* left_half2 = pad_row2 + width2 + 2*(D - 1);
			short* right_half2 = left_half2 + width2 + 2*(D - 1);
			memset(pixDiff, 0, sizeof(short)*W*D);

			/*if (D % 16 != 0)
			{
				printf("error: D must be 16* for SSE\n");
				return;
			}
			*/
			int pad_W2 = width2 + 2 * (D - 1);
			int pad_W2_1 = pad_W2 - 1;

			__m128i _tr_val = _mm_set1_epi16((short)truncate_value);
			__m128i z = _mm_setzero_si128();
			
			for (int c = 0; c < nChannels; c++)
			{
				// left img 
				left_half1[0] = row1[c];
				left_half1[width1 - 1] = ((int)row1[(width1 - 2)*nChannels + c] + (int)row1[(width1 - 1)*nChannels + c]) / 2;
				right_half1[0] = ((int)row1[c] + (int)row1[nChannels + c]) / 2;
				right_half1[width1 - 1] = row1[(width1 - 1)*nChannels + c];
				for (int w = 1; w < width1 - 1; w++)
				{
					int off = w*nChannels + c;

					left_half1[w] = ((int)row1[off - nChannels] + (int)row1[off]) / 2;
					right_half1[w] = ((int)row1[off + nChannels] + (int)row1[off]) / 2;
				}
				for (int w = 0; w < width1; w++)
					pad_row1[w] = row1[w*nChannels + c];
				for (int w = width1; w < width1 + D - 1; w++)
				{
					pad_row1[w] = left_half1[w] = right_half1[w] = row1[(width1 - 1)*nChannels + c];
				}

				// right img
				left_half2[pad_W2_1 -(D - 1 + 0)] = row2[c];
				left_half2[pad_W2_1 - (D - 1 + width2 - 1)] = ((int)row2[(width2 - 2)*nChannels + c] + (int)row2[(width2 - 1)*nChannels + c]) / 2;
				right_half2[pad_W2_1 - (D - 1 + 0)] = ((int)row2[c] + (int)row2[nChannels + c]) / 2;
				right_half2[pad_W2_1 - (D - 1 + width2 - 1)] = row2[(width2 - 1)*nChannels + c];
				for (int w = 1; w < width2 - 1; w++)
				{
					int off = w*nChannels + c;
					left_half2[pad_W2_1 - (D - 1 + w)] = ((int)row2[off - nChannels] + (int)row2[off]) / 2;
					right_half2[pad_W2_1 - (D - 1 + w)] = ((int)row2[off + nChannels] + (int)row2[off]) / 2;
				}
				for (int w = 0; w < width2; w++)
					pad_row2[pad_W2_1 - (D - 1 + w)] = row2[w*nChannels + c];
				for (int w = 0; w < D - 1; w++)
				{
					pad_row2[pad_W2_1 - w] = left_half2[pad_W2_1 - w] = right_half2[pad_W2_1 - w] = row2[c];
					pad_row2[pad_W2_1 - (D - 1 + width2 + w)] = row2[(width2 - 1)*nChannels + c];
					left_half2[pad_W2_1 - (D - 1 + width2 + w)] = row2[(width2 - 1)*nChannels + c];
					right_half2[pad_W2_1 - (D - 1 + width2 + w)] = row2[(width2 - 1)*nChannels + c];
				}

				/******/

				for (int w = 0; w < W; w++)
				{
					short u_p = pad_row1[w];
					short u_l = left_half1[w];
					short u_r = right_half1[w];

					__m128i _u_p = _mm_set1_epi16((short)u_p);
					__m128i _u_l = _mm_set1_epi16((short)u_l);
					__m128i _u_r = _mm_set1_epi16((short)u_r);
					

					int d = 0;
					for (; d < D - 7; d += 8)
					{
						int st_X2 = pad_W2_1 - (w + D - 1 - d);
						__m128i _v_p = _mm_loadu_si128((const __m128i*)(pad_row2 + st_X2));
						__m128i _v_l = _mm_loadu_si128((const __m128i*)(left_half2 + st_X2));
						__m128i _v_r = _mm_loadu_si128((const __m128i*)(right_half2 + st_X2));
						__m128i _diff1 = _mm_abs_epi16(_mm_sub_epi16(_u_p, _v_p));
						__m128i _diff2 = _mm_abs_epi16(_mm_sub_epi16(_u_p, _v_l));
						__m128i _diff3 = _mm_abs_epi16(_mm_sub_epi16(_u_p, _v_r));
						__m128i _diff4 = _mm_abs_epi16(_mm_sub_epi16(_u_l, _v_p));
						__m128i _diff5 = _mm_abs_epi16(_mm_sub_epi16(_u_r, _v_p));
						__m128i diff = _mm_min_epi16(_diff1, _diff2);
						diff = _mm_min_epi16(diff, _diff3);
						diff = _mm_min_epi16(diff, _diff4);
						diff = _mm_min_epi16(diff, _diff5);
						diff = _mm_min_epi16(diff, _tr_val);
						short* pos = pixDiff + w*D + d;
						__m128i c = _mm_loadu_si128((__m128i*)pos);
						_mm_storeu_si128((__m128i*)pos, _mm_adds_epi16(c, diff));
					}


					for (; d < D; d++)
					{
						int st_X2 = pad_W2_1 - (w + D - 1 - d);
						short v_p = pad_row2[st_X2];
						short v_l = left_half2[st_X2];
						short v_r = right_half2[st_X2];
						int diff1 = abs((int)u_p - (int)v_p);
						int diff2 = abs((int)u_p - (int)v_l);
						int diff3 = abs((int)u_p - (int)v_r);
						int diff4 = abs((int)u_l - (int)v_p);
						int diff5 = abs((int)u_r - (int)v_p);
						int min_diff = __min(diff1, diff2);
						min_diff = __min(min_diff, diff3);
						min_diff = __min(min_diff, diff4);
						min_diff = __min(min_diff, diff5);
						min_diff = __min(min_diff, truncate_value);
						pixDiff[w*D + d] += min_diff;
					}
				}
			}
			delete[]buffer;
		}

		/*
		*	val: W*D
		*	row_sum: W*D
		*/
		template<class T1, class T2>
		static void _cumulate_row(const T1* val, int W, int D, int half_win_size_W, T2* row_sum)
		{
		
			for (int d = 0; d < D; d++)
			{
				//compute the first
				row_sum[0*D+d] = val[0*D+d]*(half_win_size_W+1);
				for (int w = 1; w <= half_win_size_W; w++)
					row_sum[0 * D + d] += val[w*D + d];
				
				//compute x = 1 to half_win_W
				for (int x = 1; x <= half_win_size_W; x++)
				{
					row_sum[x*D + d] = row_sum[(x - 1)*D + d] - val[0 * D + d] + val[(x + half_win_size_W)*D + d];
				}
				//compute x = half_win_W +1 to W - 1 - half_win_W
				for (int x = half_win_size_W + 1; x < W - half_win_size_W; x++)
				{
					row_sum[x*D + d] = row_sum[(x - 1)*D + d] - val[(x - half_win_size_W - 1)*D + d] + val[(x + half_win_size_W)*D + d];
				}
				//compute x = W - half_win_W to W-1
				for (int x = W - half_win_size_W; x < W; x++)
				{
					row_sum[x*D + d] = row_sum[(x - 1)*D + d] - val[(x - half_win_size_W - 1)*D + d] + val[(W - 1)*D + d];
				}
			}
		}

		/******************************************************** 
		*cost: [W x height x D] int array;
		*		W = __max(width1, width2+maxD);
		*		D = maxD + 1;
		*********************************************************/
		static bool _computeMatchingCost_SAD_BT(const ZQ_DImage<unsigned char>& img1, const ZQ_DImage<unsigned char>& img2, ZQ_DImage<int>& cost,
			int maxD, int half_sad_win_size_W, int half_sad_win_size_H, int truncate_value)
		{
			int width1 = img1.width();
			int height = img1.height();
			int nChannels = img1.nchannels();
			int width2 = img2.width();
			if (!img2.matchDimension(width2, height, nChannels))
				return false;
			if (maxD <= 0)
				return false;

			int winH = 2 * half_sad_win_size_H + 1;
			int winW = 2 * half_sad_win_size_W + 1;
			int D = maxD + 1;
			int W = __max(width1, width2 + maxD);
			if (!cost.matchDimension(W, height, D))
				cost.allocate(W, height, D);
			else
				cost.reset();

			const unsigned char*& img1_data = img1.data();
			const unsigned char*& img2_data = img2.data();
			int*& cost_data = cost.data();

			int NROWS = winH + 1;
			ZQ_DImage<short> pixDiff(W, 1, D);
			ZQ_DImage<int> row_sum(W, NROWS, D);
			short*& pixDiff_data = pixDiff.data();
			int*& row_sum_data = row_sum.data();
			
			for (int h = 0; h < height; h++)
			{
				if (h == 0)
				{
					for (int hh = 0; hh <= half_sad_win_size_H; hh++)
					{
						_computeMatchingCostBT_OneRow(img1_data+hh*width1*nChannels, img2_data+hh*width2*nChannels, width1, width2, nChannels, maxD, pixDiff_data, truncate_value);
						_cumulate_row(pixDiff_data, W, D, half_sad_win_size_W, row_sum_data + (hh + half_sad_win_size_H)*W*D);
						if (hh == 0)
						{
							for (int i = 0; i < half_sad_win_size_H; i++)
								memcpy(row_sum_data + i*W*D, row_sum_data + (hh + half_sad_win_size_H)*W*D, sizeof(int)*W*D);
						}
					}
					
					for (int d = 0; d < D; d++)
					{
						for (int w = 0; w < W; w++)
						{
							cost_data[(0*W+w)*D + d] = 0;
							for (int hh = 0; hh < winH; hh++)
								cost_data[(0 * W + w)*D + d] += row_sum_data[hh*W*D + w*D + d];
						}
					}
				}
				else
				{
					int row_idx = h + half_sad_win_size_H;
					int last_row_idx = row_idx - 1;
					
					int add_store_kk = (row_idx + half_sad_win_size_H) % NROWS;
					int last_store_kk = (last_row_idx + half_sad_win_size_H) % NROWS;
					int sub_store_kk = (row_idx - half_sad_win_size_H - 1) % NROWS;
					int* add_row_data = row_sum_data + add_store_kk*W*D;
					int* last_row_data = row_sum_data + last_store_kk*W*D;
					int* sub_row_data = row_sum_data + sub_store_kk*W*D;

					if (row_idx >= height)
						memcpy(add_row_data, last_row_data, sizeof(int)*W*D);
					else
					{
						_computeMatchingCostBT_OneRow(img1_data + row_idx*width1*nChannels, img2_data + row_idx*width2*nChannels, width1, width2, nChannels, maxD, pixDiff_data, truncate_value);
						_cumulate_row(pixDiff_data, W, D, half_sad_win_size_W, add_row_data);
					}

					for (int d = 0; d < D; d++)
					{
						for (int w = 0; w < W; w++)
						{
							cost_data[(h * W + w)*D + d] = cost_data[((h - 1)*W + w)*D + d] - sub_row_data[w*D + d] + add_row_data[w*D + d];
						}
					}	
				}
			}

			
			return true;
		}

		/*padding D-1 cols at the beginning for img1,
		padding D-1 cols at the endding for img2,
		and the two images are squared and summed in window
		*/
		static void _compute_denom_for_NCC(const ZQ_DImage<unsigned char>& img1, const ZQ_DImage<unsigned char>& img2, ZQ_DImage<float>& denom1, ZQ_DImage<float>& denom2, int half_win_size_W, int half_win_size_H, int D)
		{
			int win_W = half_win_size_W * 2 + 1;
			int win_H = half_win_size_H * 2 + 1;
			int width1 = img1.width();
			int height1 = img1.height();
			int nChannels1 = img1.nchannels();
			int width2 = img2.width();
			int height2 = img2.height();
			int nChannels2 = img2.nchannels();
			

			for (int pass = 0; pass < 2; pass++)
			{
				int pad_W, pad_H, width, height, nChannels;
				int denom_W;
				int w_st1, w_st2, w_st3;
				const unsigned char* img_data;
				float* denom_data;
				if (pass == 0)
				{
					pad_W = width1 + win_W + D - 1;
					pad_H = height1 + win_H;
					denom_W = pad_W - win_W;
					width = width1;
					height = height1;
					nChannels = nChannels1;
					img_data = img1.data();
					w_st1 = 1;
					w_st2 = half_win_size_W + 2;
					w_st3 = half_win_size_W + width;
					if (!denom1.matchDimension(denom_W, height, 1))
						denom1.allocate(denom_W, height, 1);
					else
						denom1.reset();
					denom_data = denom1.data();
				}
				else
				{
					pad_W = width2 + win_W + 2*(D - 1);
					pad_H = height2 + win_H;
					denom_W = pad_W - win_W;
					width = width2;
					height = height2;
					nChannels = nChannels2;
					img_data = img2.data();
					w_st1 = 1;
					w_st2 = half_win_size_W + D + 1;
					w_st3 = half_win_size_W + width + D - 1;
					if (!denom2.matchDimension(denom_W, height, 1))
						denom2.allocate(denom_W, height, 1);
					else
						denom2.reset();
					denom_data = denom2.data();
				}
				
				ZQ_DImage<float> tmp(pad_W, height);
				float*& tmp_data = tmp.data();
				ZQ_DImage<float> sum(pad_W, pad_H);
				float*& sum_data = sum.data();
				// row sum
				for (int h = 0; h < height; h++)
				{
					float tmp_val = 0;
					for (int c = 0; c < nChannels; c++)
					{
						float vv = (float)img_data[h*width*nChannels + c];
						tmp_val += vv*vv;
					}
					for (int w = w_st1; w < w_st2; w++)
					{
						tmp_data[h*pad_W + w] = tmp_val*w;
					}
					for (int w = w_st2; w < w_st3; w++)
					{
						float tmp_val = 0;
						for (int c = 0; c < nChannels; c++)
						{
							float vv = (float)img_data[(h*width + w - w_st2 + 1)*nChannels + c];
							tmp_val += vv*vv;
						}
						tmp_data[h*pad_W + w] = tmp_data[h*pad_W + w - 1] + tmp_val;
					}
					tmp_val = 0;
					for (int c = 0; c < nChannels; c++)
					{
						float vv = (float)img_data[(h*width + width - 1)*nChannels + c];
						tmp_val += vv*vv;
					}
					for (int w = w_st3; w < pad_W; w++)
					{
						tmp_data[h*pad_W + w] = tmp_data[h*pad_W + w - 1] + tmp_val;
					}
				}

				// col sum
				
				for (int w = 0; w < pad_W; w++)
				{
					for (int h = 1; h <= half_win_size_H + 1; h++)
					{
						sum_data[h*pad_W + w] = h*tmp_data[w];
					}
					for (int h = half_win_size_H + 2; h <= half_win_size_H + height1 - 1; h++)
					{
						sum_data[h*pad_W + w] = sum_data[(h - 1)*pad_W + w] + tmp_data[(h - half_win_size_H - 1)*pad_W + w];
					}
					for (int h = half_win_size_H + height1; h < height1 + win_H; h++)
						sum_data[h*pad_W + w] = sum_data[(h - 1)*pad_W + w] + tmp_data[(height1 - 1)*pad_W + w];
				}

				
				for (int h = 0; h < height; h++)
				{
					for (int w = 0; w < denom_W; w++)
					{
						denom_data[h*denom_W + w] = sum_data[(h + win_H)*pad_W + (w + win_W)] + sum_data[h*pad_W + w]
							- sum_data[(h + win_H)*pad_W + w] - sum_data[h*pad_W + (w + win_W)];
					}
				}
			}
		}

		/********************************************************
		*cost: [W x height x D] int array;
		*		W = __max(width1, width2+maxD);
		*		D = maxD + 1;
		*********************************************************/
		static bool _computeMatchingCost_NCC(const ZQ_DImage<unsigned char>& img1, const ZQ_DImage<unsigned char>& img2, ZQ_DImage<float>& cost,
			int maxD, int half_win_size_W, int half_win_size_H)
		{
			int width1 = img1.width();
			int height = img1.height();
			int nChannels = img1.nchannels();
			int width2 = img2.width();
			if (!img2.matchDimension(width2, height, nChannels))
				return false;
			if (maxD <= 0)
				return false;

			int winH = 2 * half_win_size_H + 1;
			int winW = 2 * half_win_size_W + 1;
			int D = maxD + 1;
			int W = __max(width1, width2 + maxD);
			if (!cost.matchDimension(W, height, D))
				cost.allocate(W, height, D);
			else
				cost.reset();

			
			ZQ_DImage<float> denom1, denom2;
			_compute_denom_for_NCC(img1,img2,denom1, denom2, half_win_size_W, half_win_size_H, D);
			int denom_W1 = denom1.width();
			int denom_W2 = denom2.width();
			
			const unsigned char*& img1_data = img1.data();
			const unsigned char*& img2_data = img2.data();
			float*& cost_data = cost.data();
			float*& denom1_data = denom1.data();
			float*& denom2_data = denom2.data();

			int NROWS = winH + 1;
			ZQ_DImage<float> num(W,1,D);
			ZQ_DImage<float> dotVal(W, 1, D);
			ZQ_DImage<float> row_sum(W, NROWS, D);
			float*& num_data = num.data();
			float*& dotVal_data = dotVal.data();
			float*& row_sum_data = row_sum.data();

			float eps = 1e-32;
			for (int h = 0; h < height; h++)
			{
				if (h == 0)
				{
					for (int hh = 0; hh <= half_win_size_H; hh++)
					{
						_computeMatchingCostNCC_Dot_OneRow(img1_data + hh*width1*nChannels, img2_data + hh*width2*nChannels, width1, width2, nChannels, maxD, dotVal_data);
						_cumulate_row(dotVal_data, W, D, half_win_size_W, row_sum_data + (hh + half_win_size_H)*W*D);
						if (hh == 0)
						{
							for (int i = 0; i < half_win_size_H; i++)
								memcpy(row_sum_data + i*W*D, row_sum_data + (hh + half_win_size_H)*W*D, sizeof(float)*W*D);
						}
					}
					for (int w = 0; w < W; w++)
					{
						for (int d = 0; d < D; d++)
						{
							num_data[w*D + d] = 0;
							for (int hh = 0; hh < winH; hh++)
								num_data[w*D + d] += row_sum_data[hh*W*D + w*D + d];
							float denom_val1 = denom1_data[h*denom_W1 + w];
							float denom_val2 = denom2_data[h*denom_W2 + (w + D - 1 - d)];
							if (denom_val1 <= 0 || denom_val2 <= 0)
								cost_data[(0 * W + w)*D + d] = 0;
							else
								cost_data[(0 * W + w)*D + d] = num_data[w*D + d] / sqrt((double)denom_val1*denom_val2);
						}
					}
				}
				else
				{
					int row_idx = h + half_win_size_H;
					int last_row_idx = row_idx - 1;

					int add_store_kk = (row_idx + half_win_size_H) % NROWS;
					int last_store_kk = (last_row_idx + half_win_size_H) % NROWS;
					int sub_store_kk = (row_idx - half_win_size_H - 1) % NROWS;
					float* add_row_data = row_sum_data + add_store_kk*W*D;
					float* last_row_data = row_sum_data + last_store_kk*W*D;
					float* sub_row_data = row_sum_data + sub_store_kk*W*D;

					if (row_idx >= height)
						memcpy(add_row_data, last_row_data, sizeof(float)*W*D);
					else
					{
						_computeMatchingCostNCC_Dot_OneRow(img1_data + row_idx*width1*nChannels, img2_data + row_idx*width2*nChannels, width1, width2, nChannels, maxD, dotVal_data);
						_cumulate_row(dotVal_data, W, D, half_win_size_W, add_row_data);
					}

					for (int w = 0; w < W; w++)
					{
						for (int d = 0; d < D; d++)
						{
							num_data[w*D + d] += -sub_row_data[w*D + d] + add_row_data[w*D + d];
							float denom_val1 = denom1_data[h*denom_W1 + w];
							float denom_val2 = denom2_data[h*denom_W2 + (w + D - 1 - d)];
							if (denom_val1 <= 0 || denom_val2 <= 0)
								cost_data[(h * W + w)*D + d] = 0;
							else
								cost_data[(h * W + w)*D + d] = num_data[w*D + d] / sqrt((double)denom_val1*denom_val2);
						}
					}
				}
			}
			return true;
		}
	public:
		static bool _computeDisparityBM_SAD_BT(const ZQ_DImage<unsigned char>& img1, const ZQ_DImage<unsigned char>& img2,
			ZQ_DImage<float>& left_disparity, ZQ_DImage<float>& right_disparity, int maxD,
			int half_sad_win_size_W = 3, int half_sad_win_size_H = 3, float uniqueness = 1.15f, int truncate_value = 64)
		{
			int width1 = img1.width();
			int height = img1.height();
			int nChannels = img1.nchannels();
			int width2 = img2.width();
			if (!img2.matchDimension(width2, height, nChannels))
				return false;
			if (maxD <= 0)
				return false;

			if(!left_disparity.matchDimension(width1, height, 1))
				left_disparity.allocate(width1, height, 1);
			if (!right_disparity.matchDimension(width2, height, 1))
				right_disparity.allocate(width2, height, 1);

			const int MAX_COST = INT_MAX;

			float*& left_disparity_data = left_disparity.data();
			float*& right_disparity_data = right_disparity.data();

			int winH = 2 * half_sad_win_size_H + 1;
			int winW = 2 * half_sad_win_size_W + 1;
			int D = maxD + 1;
			int W = __max(width1, width2 + maxD);

			const unsigned char*& img1_data = img1.data();
			const unsigned char*& img2_data = img2.data();
			
			int NROWS = winH + 1;
			ZQ_DImage<short> pixDiff(W, 1, D);
			ZQ_DImage<int> row_sum(W, NROWS, D);
			ZQ_DImage<int> cost(W, 1, D);
			short*& pixDiff_data = pixDiff.data();
			int*& row_sum_data = row_sum.data();
			int*& cost_data = cost.data();
		
			for (int h = 0; h < height; h++)
			{
				/*compute cost of current row Begin*/
				if (h == 0)
				{
					for (int hh = 0; hh <= half_sad_win_size_H; hh++)
					{
						_computeMatchingCostBT_OneRow(img1_data + hh*width1*nChannels, img2_data + hh*width2*nChannels, width1, width2, nChannels, maxD, pixDiff_data, truncate_value);
						_cumulate_row(pixDiff_data, W, D, half_sad_win_size_W, row_sum_data + (hh + half_sad_win_size_H)*W*D);
						if (hh == 0)
						{
							for (int i = 0; i < half_sad_win_size_H; i++)
								memcpy(row_sum_data + i*W*D, row_sum_data + (hh + half_sad_win_size_H)*W*D, sizeof(int)*W*D);
						}
					}

					for (int d = 0; d < D; d++)
					{
						for (int w = 0; w < W; w++)
						{
							cost_data[w*D + d] = 0;
							for (int hh = 0; hh < winH; hh++)
								cost_data[w*D + d] += row_sum_data[hh*W*D + w*D + d];
						}
					}
				}
				else
				{
					int row_idx = h + half_sad_win_size_H;
					int last_row_idx = row_idx - 1;

					int add_store_kk = (row_idx + half_sad_win_size_H) % NROWS;
					int last_store_kk = (last_row_idx + half_sad_win_size_H) % NROWS;
					int sub_store_kk = (row_idx - half_sad_win_size_H - 1) % NROWS;
					int* add_row_data = row_sum_data + add_store_kk*W*D;
					int* last_row_data = row_sum_data + last_store_kk*W*D;
					int* sub_row_data = row_sum_data + sub_store_kk*W*D;

					if (row_idx >= height)
						memcpy(add_row_data, last_row_data, sizeof(int)*W*D);
					else
					{
						_computeMatchingCostBT_OneRow(img1_data + row_idx*width1*nChannels, img2_data + row_idx*width2*nChannels, width1, width2, nChannels, maxD, pixDiff_data, truncate_value);
						_cumulate_row(pixDiff_data, W, D, half_sad_win_size_W, add_row_data);
					}

					for (int d = 0; d < D; d++)
					{
						for (int w = 0; w < W; w++)
						{
							cost_data[w*D + d] += - sub_row_data[w*D + d] + add_row_data[w*D + d];
						}
					}
				}
				/*compute cost of current row End*/

				//for left
				for (int w = 0; w < width1; w++)
				{
					int min_d = -1;
					int min_sum = MAX_COST;
					for (int d = 0; d < D; d++)
					{
						int tmp_sum = cost_data[w*D + d];
						if (tmp_sum < min_sum)
						{
							min_sum = tmp_sum;
							min_d = d;
						}
					}
					/*check uniqueness begin*/
					int min_d2;
					for (min_d2 = 0; min_d2 < D; min_d2++)
					{
						if (cost_data[w*D+min_d2]  < min_sum*uniqueness && std::abs(min_d - min_d2) > 1)
							break;
					}
					if (min_d2 < D)
					{
						left_disparity_data[h*width1 + w] = -1;
						continue;
					}
					/*check uniqueness end*/

					/*interpolation for sub-pixel precision begin*/
					int d = min_d;
					float final_d;
					if (0 < d && d < D - 1)
					{
						// do subpixel quadratic interpolation:
						//   fit parabola into (x1=d-1, y1=Sp[d-1]), (x2=d, y2=Sp[d]), (x3=d+1, y3=Sp[d+1])
						//   then find minimum of the parabola.
						int* sum_p = cost_data + w*D;
						float denom2 = sum_p[d - 1] + sum_p[d + 1] - 2 * sum_p[d];
						if (denom2 == 0)
							final_d = d;
						else
							final_d = (float)d + (sum_p[d - 1] - sum_p[d + 1]) / (denom2 * 2);
					}
					else
						final_d = d;
					left_disparity_data[h*width1 + w] = final_d;
					/*interpolation for sub-pixel precision end*/
				}


				//for right
				for (int w = 0; w < width2; w++)
				{
					int min_d = -1;
					int min_sum = MAX_COST;
					for (int d = 0; d < D; d++)
					{
						int tmp_sum = cost_data[(w + d)*D + d];
						if (tmp_sum < min_sum)
						{
							min_sum = tmp_sum;
							min_d = d;
						}
					}
					/*check uniqueness begin*/
					int min_d2;
					for (min_d2 = 0; min_d2 < D; min_d2++)
					{
						if (cost_data[(w+min_d2)*D+min_d2]  < min_sum*uniqueness && std::abs(min_d - min_d2) > 1)
							break;
					}
					if (min_d2 < D)
					{
						right_disparity_data[h*width2 + w] = -1;
						continue;
					}
					/*check uniqueness end*/

					/*interpolation for sub-pixel precision begin*/
					int d = min_d;
					float final_d;
					if (0 < d && d < D - 1)
					{
						// do subpixel quadratic interpolation:
						//   fit parabola into (x1=d-1, y1=Sp[d-1]), (x2=d, y2=Sp[d]), (x3=d+1, y3=Sp[d+1])
						//   then find minimum of the parabola.
						int sum_p2 = cost_data[(w + d)*D + d];
						int sum_p1 = cost_data[(w + d - 1)*D + d - 1];
						int sum_p3 = cost_data[(w + d + 1)*D + d + 1];
						float denom2 = sum_p1 + sum_p3 - 2 * sum_p2;
						if (denom2 == 0)
							final_d = d;
						else
							final_d = (float)d + (sum_p1 - sum_p3) / (denom2 * 2);
					}
					else
						final_d = d;
					right_disparity_data[h*width2 + w] = final_d;
					/*interpolation for sub-pixel precision end*/
				}
			}
			return true;
		}

		static bool _computeDisparityBM_NCC(const ZQ_DImage<unsigned char>& img1, const ZQ_DImage<unsigned char>& img2,
			ZQ_DImage<float>& left_disparity, ZQ_DImage<float>& right_disparity, int maxD,
			int half_win_size_W = 7, int half_win_size_H = 7, float uniqueness = 1.0f)
		{
			int width1 = img1.width();
			int height = img1.height();
			int nChannels = img1.nchannels();
			int width2 = img2.width();
			if (!img2.matchDimension(width2, height, nChannels))
				return false;
			if (maxD <= 0)
				return false;

			if (!left_disparity.matchDimension(width1, height, 1))
				left_disparity.allocate(width1, height, 1);
			if (!right_disparity.matchDimension(width2, height, 1))
				right_disparity.allocate(width2, height, 1);

			
			float*& left_disparity_data = left_disparity.data();
			float*& right_disparity_data = right_disparity.data();

			int winH = 2 * half_win_size_H + 1;
			int winW = 2 * half_win_size_W + 1;
			int D = maxD + 1;
			int W = __max(width1, width2 + maxD);
			
			ZQ_DImage<float> denom1, denom2;
			_compute_denom_for_NCC(img1, img2, denom1, denom2, half_win_size_W, half_win_size_H, D);
			int denom_W1 = denom1.width();
			int denom_W2 = denom2.width();

			const unsigned char*& img1_data = img1.data();
			const unsigned char*& img2_data = img2.data();
			
			float*& denom1_data = denom1.data();
			float*& denom2_data = denom2.data();

			int NROWS = winH + 1;
			ZQ_DImage<float> cost(W, 1, D);
			ZQ_DImage<float> num(W, 1, D);
			ZQ_DImage<float> dotVal(W, 1, D);
			ZQ_DImage<float> row_sum(W, NROWS, D);
			float*& cost_data = cost.data();
			float*& num_data = num.data();
			float*& dotVal_data = dotVal.data();
			float*& row_sum_data = row_sum.data();

			float eps = 1e-32;
			for (int h = 0; h < height; h++)
			{
				/*compute cost of current row Begin*/
				if (h == 0)
				{
					for (int hh = 0; hh <= half_win_size_H; hh++)
					{
						_computeMatchingCostNCC_Dot_OneRow(img1_data + hh*width1*nChannels, img2_data + hh*width2*nChannels, width1, width2, nChannels, maxD, dotVal_data);
						_cumulate_row(dotVal_data, W, D, half_win_size_W, row_sum_data + (hh + half_win_size_H)*W*D);
						if (hh == 0)
						{
							for (int i = 0; i < half_win_size_H; i++)
								memcpy(row_sum_data + i*W*D, row_sum_data + (hh + half_win_size_H)*W*D, sizeof(float)*W*D);
						}
					}

					for (int w = 0; w < W; w++)
					{
						for (int d = 0; d < D; d++)
						{
							num_data[w*D + d] = 0;
							for (int hh = 0; hh < winH; hh++)
								num_data[w*D + d] += row_sum_data[hh*W*D + w*D + d];
							float denom_val1 = denom1_data[h*denom_W1 + w];
							float denom_val2 = denom2_data[h*denom_W2 + (w + D - 1 - d)];
							if (denom_val1 <= 0 || denom_val2 <= 0)
								cost_data[w*D + d] = 0;
							else
								cost_data[w*D + d] = num_data[w*D + d] / sqrt((double)denom_val1*denom_val2);
						}
					}
				}
				else
				{
					int row_idx = h + half_win_size_H;
					int last_row_idx = row_idx - 1;

					int add_store_kk = (row_idx + half_win_size_H) % NROWS;
					int last_store_kk = (last_row_idx + half_win_size_H) % NROWS;
					int sub_store_kk = (row_idx - half_win_size_H - 1) % NROWS;
					float* add_row_data = row_sum_data + add_store_kk*W*D;
					float* last_row_data = row_sum_data + last_store_kk*W*D;
					float* sub_row_data = row_sum_data + sub_store_kk*W*D;

					if (row_idx >= height)
						memcpy(add_row_data, last_row_data, sizeof(float)*W*D);
					else
					{
						_computeMatchingCostNCC_Dot_OneRow(img1_data + row_idx*width1*nChannels, img2_data + row_idx*width2*nChannels, width1, width2, nChannels, maxD, dotVal_data);
						_cumulate_row(dotVal_data, W, D, half_win_size_W, add_row_data);
					}

					for (int w = 0; w < W; w++)
					{
						for (int d = 0; d < D; d++)
						{
							num_data[w*D + d] += -sub_row_data[w*D + d] + add_row_data[w*D + d];
							float denom_val1 = denom1_data[h*denom_W1 + w];
							float denom_val2 = denom2_data[h*denom_W2 + (w + D - 1 - d)];
							if (denom_val1 <= 0 || denom_val2 <= 0)
								cost_data[w*D + d] = 0;
							else
								cost_data[w*D + d] = num_data[w*D + d] / sqrt((double)denom_val1*denom_val2);
						}
					}
				}
				/*compute cost of current row End*/

				//for left
				for (int w = 0; w < width1; w++)
				{
					int max_d = 0;
					float max_c = cost_data[w*D + 0];
					for (int d = 1; d < D; d++)
					{
						float tmp_c = cost_data[w*D + d];
						if (tmp_c > max_c)
						{
							max_c = tmp_c;
							max_d = d;
						}
					}
					/*check uniqueness begin*/
					int max_d2;
					for (max_d2 = 0; max_d2 < D; max_d2++)
					{
						if (cost_data[w*D + max_d2]*uniqueness  > max_c && std::abs(max_d - max_d2) > 1)
							break;
					}
					if (max_d2 < D)
					{
						left_disparity_data[h*width1 + w] = -1;
						continue;
					}
					/*check uniqueness end*/

					/*interpolation for sub-pixel precision begin*/
					int d = max_d;
					float final_d;
					if (0 < d && d < D - 1)
					{
						// do subpixel quadratic interpolation:
						//   fit parabola into (x1=d-1, y1=Sp[d-1]), (x2=d, y2=Sp[d]), (x3=d+1, y3=Sp[d+1])
						//   then find minimum of the parabola.
						float* sum_p = cost_data + w*D;
						float denom2 = sum_p[d - 1] + sum_p[d + 1] - 2 * sum_p[d];
						if (denom2 == 0)
							final_d = d;
						else
							final_d = (float)d + (sum_p[d - 1] - sum_p[d + 1]) / (denom2 * 2);
					}
					else
						final_d = d;
					left_disparity_data[h*width1 + w] = final_d;
					/*interpolation for sub-pixel precision end*/
				}


				//for right
				for (int w = 0; w < width2; w++)
				{
					int max_d = -1;
					float max_c = cost_data[(w + 0)*D + 0];
					for (int d = 1; d < D; d++)
					{
						float tmp_c = cost_data[(w + d)*D + d];
						if (tmp_c > max_c)
						{
							max_c = tmp_c;
							max_d = d;
						}
					}
					/*check uniqueness begin*/
					int max_d2;
					for (max_d2 = 0; max_d2 < D; max_d2++)
					{
						if (cost_data[(w + max_d2)*D + max_d2]*uniqueness  > max_c && std::abs(max_d - max_d2) > 1)
							break;
					}
					if (max_d2 < D)
					{
						right_disparity_data[h*width2 + w] = -1;
						continue;
					}
					/*check uniqueness end*/

					/*interpolation for sub-pixel precision begin*/
					int d = max_d;
					float final_d;
					if (0 < d && d < D - 1)
					{
						// do subpixel quadratic interpolation:
						//   fit parabola into (x1=d-1, y1=Sp[d-1]), (x2=d, y2=Sp[d]), (x3=d+1, y3=Sp[d+1])
						//   then find minimum of the parabola.
						float sum_p2 = cost_data[(w + d)*D + d];
						float sum_p1 = cost_data[(w + d - 1)*D + d - 1];
						float sum_p3 = cost_data[(w + d + 1)*D + d + 1];
						float denom2 = sum_p1 + sum_p3 - 2 * sum_p2;
						if (denom2 == 0)
							final_d = d;
						else
							final_d = (float)d + (sum_p1 - sum_p3) / (denom2 * 2);
					}
					else
						final_d = d;
					right_disparity_data[h*width2 + w] = final_d;
					/*interpolation for sub-pixel precision end*/
				}
			}
			return true;
		}

		static bool _computeDisparityBM_SSE(const ZQ_DImage<unsigned char>& img1, const ZQ_DImage<unsigned char>& img2,
			ZQ_DImage<float>& left_disparity, ZQ_DImage<float>& right_disparity, int maxD,
			int half_sad_win_size_W = 3, int half_sad_win_size_H = 3, float uniqueness = 1.15f, int truncate_value = 64)
		{
			int width1 = img1.width();
			int height = img1.height();
			int nChannels = img1.nchannels();
			int width2 = img2.width();
			if (!img2.matchDimension(width2, height, nChannels))
				return false;
			if (maxD <= 0)
				return false;

			if (!left_disparity.matchDimension(width1, height, 1))
				left_disparity.allocate(width1, height, 1);
			if (!right_disparity.matchDimension(width2, height, 1))
				right_disparity.allocate(width2, height, 1);

			const int MAX_COST = INT_MAX;

			float*& left_disparity_data = left_disparity.data();
			float*& right_disparity_data = right_disparity.data();

			int winH = 2 * half_sad_win_size_H + 1;
			int winW = 2 * half_sad_win_size_W + 1;
			int D = maxD + 1;
			int W = __max(width1, width2 + maxD);

			const unsigned char*& img1_data = img1.data();
			const unsigned char*& img2_data = img2.data();

			int NROWS = winH + 1;
			ZQ_DImage<short> pixDiff(W, 1, D);
			ZQ_DImage<short> pixDiff2(W, 1, D);
			ZQ_DImage<int> row_sum(W, NROWS, D);
			ZQ_DImage<int> cost(W, 1, D);
			short*& pixDiff_data = pixDiff.data();
			short*& pixDiff2_data = pixDiff2.data();
			int*& row_sum_data = row_sum.data();
			int*& cost_data = cost.data();

			for (int h = 0; h < height; h++)
			{
				/*compute cost of current row Begin*/
				if (h == 0)
				{
					for (int hh = 0; hh <= half_sad_win_size_H; hh++)
					{
						_computeMatchingCostBT_OneRow_SSE(img1_data + hh*width1*nChannels, img2_data + hh*width2*nChannels, width1, width2, nChannels, maxD, pixDiff_data, truncate_value);
						//_computeMatchingCostBT_OneRow(img1_data + hh*width1*nChannels, img2_data + hh*width2*nChannels, width1, width2, nChannels, maxD, pixDiff_data, truncate_value);
						_cumulate_row(pixDiff_data, W, D, half_sad_win_size_W, row_sum_data + (hh + half_sad_win_size_H)*W*D);
						if (hh == 0)
						{
							for (int i = 0; i < half_sad_win_size_H; i++)
								memcpy(row_sum_data + i*W*D, row_sum_data + (hh + half_sad_win_size_H)*W*D, sizeof(int)*W*D);
						}
					}

					for (int d = 0; d < D; d++)
					{
						for (int w = 0; w < W; w++)
						{
							cost_data[w*D + d] = 0;
							for (int hh = 0; hh < winH; hh++)
								cost_data[w*D + d] += row_sum_data[hh*W*D + w*D + d];
						}
					}
				}
				else
				{
					int row_idx = h + half_sad_win_size_H;
					int last_row_idx = row_idx - 1;

					int add_store_kk = (row_idx + half_sad_win_size_H) % NROWS;
					int last_store_kk = (last_row_idx + half_sad_win_size_H) % NROWS;
					int sub_store_kk = (row_idx - half_sad_win_size_H - 1) % NROWS;
					int* add_row_data = row_sum_data + add_store_kk*W*D;
					int* last_row_data = row_sum_data + last_store_kk*W*D;
					int* sub_row_data = row_sum_data + sub_store_kk*W*D;

					if (row_idx >= height)
						memcpy(add_row_data, last_row_data, sizeof(int)*W*D);
					else
					{
						_computeMatchingCostBT_OneRow_SSE(img1_data + row_idx*width1*nChannels, img2_data + row_idx*width2*nChannels, width1, width2, nChannels, maxD, pixDiff_data, truncate_value);
						//_computeMatchingCostBT_OneRow(img1_data + row_idx*width1*nChannels, img2_data + row_idx*width2*nChannels, width1, width2, nChannels, maxD, pixDiff_data, truncate_value);
						_cumulate_row(pixDiff_data, W, D, half_sad_win_size_W, add_row_data);
					}

					for (int d = 0; d < D; d++)
					{
						for (int w = 0; w < W; w++)
						{
							cost_data[w*D + d] += -sub_row_data[w*D + d] + add_row_data[w*D + d];
						}
					}
				}
				/*compute cost of current row End*/

				//for left
				for (int w = 0; w < width1; w++)
				{
					int min_d = -1;
					int min_sum = MAX_COST;
					for (int d = 0; d < D; d++)
					{
						int tmp_sum = cost_data[w*D + d];
						if (tmp_sum < min_sum)
						{
							min_sum = tmp_sum;
							min_d = d;
						}
					}
					/*check uniqueness begin*/
					int min_d2;
					for (min_d2 = 0; min_d2 < D; min_d2++)
					{
						if (cost_data[w*D + min_d2]  < min_sum*uniqueness && std::abs(min_d - min_d2) > 1)
							break;
					}
					if (min_d2 < D)
					{
						left_disparity_data[h*width1 + w] = -1;
						continue;
					}
					/*check uniqueness end*/

					/*interpolation for sub-pixel precision begin*/
					int d = min_d;
					float final_d;
					if (0 < d && d < D - 1)
					{
						// do subpixel quadratic interpolation:
						//   fit parabola into (x1=d-1, y1=Sp[d-1]), (x2=d, y2=Sp[d]), (x3=d+1, y3=Sp[d+1])
						//   then find minimum of the parabola.
						int* sum_p = cost_data + w*D;
						float denom2 = sum_p[d - 1] + sum_p[d + 1] - 2 * sum_p[d];
						if (denom2 == 0)
							final_d = d;
						else
							final_d = (float)d + (sum_p[d - 1] - sum_p[d + 1]) / (denom2 * 2);
					}
					else
						final_d = d;
					left_disparity_data[h*width1 + w] = final_d;
					/*interpolation for sub-pixel precision end*/
				}


				//for right
				for (int w = 0; w < width2; w++)
				{
					int min_d = -1;
					int min_sum = MAX_COST;
					for (int d = 0; d < D; d++)
					{
						int tmp_sum = cost_data[(w + d)*D + d];
						if (tmp_sum < min_sum)
						{
							min_sum = tmp_sum;
							min_d = d;
						}
					}
					/*check uniqueness begin*/
					int min_d2;
					for (min_d2 = 0; min_d2 < D; min_d2++)
					{
						if (cost_data[(w + min_d2)*D + min_d2]  < min_sum*uniqueness && std::abs(min_d - min_d2) > 1)
							break;
					}
					if (min_d2 < D)
					{
						right_disparity_data[h*width2 + w] = -1;
						continue;
					}
					/*check uniqueness end*/

					/*interpolation for sub-pixel precision begin*/
					int d = min_d;
					float final_d;
					if (0 < d && d < D - 1)
					{
						// do subpixel quadratic interpolation:
						//   fit parabola into (x1=d-1, y1=Sp[d-1]), (x2=d, y2=Sp[d]), (x3=d+1, y3=Sp[d+1])
						//   then find minimum of the parabola.
						int sum_p2 = cost_data[(w + d)*D + d];
						int sum_p1 = cost_data[(w + d - 1)*D + d - 1];
						int sum_p3 = cost_data[(w + d + 1)*D + d + 1];
						float denom2 = sum_p1 + sum_p3 - 2 * sum_p2;
						if (denom2 == 0)
							final_d = d;
						else
							final_d = (float)d + (sum_p1 - sum_p3) / (denom2 * 2);
					}
					else
						final_d = d;
					right_disparity_data[h*width2 + w] = final_d;
					/*interpolation for sub-pixel precision end*/
				}
			}
			return true;
		}

		/* a naive implementation which will allocate large amounts of memory*/
		static bool _computeDisparityBM_naivie(const ZQ_DImage<unsigned char>& img1, const ZQ_DImage<unsigned char>& img2, 
			ZQ_DImage<float>& left_disparity, ZQ_DImage<float>& right_disparity, int maxD,
			int half_sad_win_size_W = 3, int half_sad_win_size_H = 3, float uniqueness = 1.15f, int truncate_value = 64)
		{
			ZQ_DImage<int> cost;
			if (!_computeMatchingCost_SAD_BT(img1, img2, cost, maxD, half_sad_win_size_W, half_sad_win_size_H, truncate_value))
				return false;
			int*& cost_data = cost.data();

			int W = cost.width();
			int height = cost.height();
			int D = cost.nchannels();
			int width1 = img1.width();
			int width2 = img2.width();
			if (!left_disparity.matchDimension(width1, height, 1))
				left_disparity.allocate(width1, height, 1);
			if (!right_disparity.matchDimension(width2, height, 1))
				right_disparity.allocate(width2, height, 1);

			const int MAX_COST = INT_MAX;
			
			float*& left_disparity_data = left_disparity.data();
			float*& right_disparity_data = right_disparity.data();

			//for left
			for (int h = 0; h < height; h++)
			{
				for (int w = 0; w < width1; w++)
				{
					int min_d = -1;
					int min_sum = MAX_COST;
					for (int d = 0; d < D; d++)
					{
						int tmp_sum = cost_data[(h*W + w)*D + d];
						if (tmp_sum < min_sum)
						{
							min_sum = tmp_sum;
							min_d = d;
						}
					}
					/*check uniqueness begin*/
					int min_d2;
					for (min_d2 = 0; min_d2 < D; min_d2++)
					{
						if (cost_data[(h*W + w)*D + min_d2]  < min_sum*uniqueness && std::abs(min_d - min_d2) > 1)
							break;
					}
					if (min_d2 < D)
					{
						left_disparity_data[h*width1 + w] = -1;
						continue;
					}
					/*check uniqueness end*/

					/*interpolation for sub-pixel precision begin*/
					int d = min_d;
					float final_d;
					if (0 < d && d < D - 1)
					{
						// do subpixel quadratic interpolation:
						//   fit parabola into (x1=d-1, y1=Sp[d-1]), (x2=d, y2=Sp[d]), (x3=d+1, y3=Sp[d+1])
						//   then find minimum of the parabola.
						int* sum_p = cost_data + (h*W + w)*D;
						float denom2 = sum_p[d - 1] + sum_p[d + 1] - 2 * sum_p[d];
						if (denom2 == 0)
							final_d = d;
						else
							final_d = (float)d + (sum_p[d - 1] - sum_p[d + 1]) / (denom2 * 2);
					}
					else
						final_d = d;
					left_disparity_data[h*width1 + w] = final_d;
					/*interpolation for sub-pixel precision end*/
				}
			}
			
			//for right
			for (int h = 0; h < height; h++)
			{
				for (int w = 0; w < width2; w++)
				{
					int min_d = -1;
					int min_sum = MAX_COST;
					for (int d = 0; d < D; d++)
					{
						int tmp_sum = cost_data[(h*W + w+d)*D + d];
						if (tmp_sum < min_sum)
						{
							min_sum = tmp_sum;
							min_d = d;
						}
					}
					/*check uniqueness begin*/
					int min_d2;
					for (min_d2 = 0; min_d2 < D; min_d2++)
					{
						if (cost_data[(h*W+w+min_d2)*D+min_d2]  < min_sum*uniqueness && std::abs(min_d - min_d2) > 1)
							break;
					}
					if (min_d2 < D)
					{
						right_disparity_data[h*width2 + w] = -1;
						continue;
					}
					/*check uniqueness end*/

					/*interpolation for sub-pixel precision begin*/
					int d = min_d;
					float final_d;
					if (0 < d && d < D - 1)
					{
						// do subpixel quadratic interpolation:
						//   fit parabola into (x1=d-1, y1=Sp[d-1]), (x2=d, y2=Sp[d]), (x3=d+1, y3=Sp[d+1])
						//   then find minimum of the parabola.
						int sum_p2 = cost_data[(h*W + w + d)*D + d];
						int sum_p1 = cost_data[(h*W + w + d - 1)*D + d - 1];
						int sum_p3 = cost_data[(h*W + w + d + 1)*D + d + 1];
						float denom2 = sum_p1 + sum_p3 - 2 * sum_p2;
						if (denom2 == 0)
							final_d = d;
						else
							final_d = (float)d + (sum_p1 - sum_p3) / (denom2 * 2);
					}
					else
						final_d = d;
					right_disparity_data[h*width2 + w] = final_d;
					/*interpolation for sub-pixel precision end*/
				}
			}
			return true;
		}

		/* a naive implementation which will allocate large amounts of memory*/
		static bool _computeDisparityBM_NCC_naivie(const ZQ_DImage<unsigned char>& img1, const ZQ_DImage<unsigned char>& img2,
			ZQ_DImage<float>& left_disparity, ZQ_DImage<float>& right_disparity, int maxD,
			int half_sad_win_size_W = 3, int half_sad_win_size_H = 3, float uniqueness = 1.0f)
		{
			ZQ_DImage<float> cost;
			if (!_computeMatchingCost_NCC(img1, img2, cost, maxD, half_sad_win_size_W, half_sad_win_size_H))
				return false;
			cost.Multiplywith(-1);
			cost.Addwith(1);
			float*& cost_data = cost.data();

			int W = cost.width();
			int height = cost.height();
			int D = cost.nchannels();
			int width1 = img1.width();
			int width2 = img2.width();
			if (!left_disparity.matchDimension(width1, height, 1))
				left_disparity.allocate(width1, height, 1);
			if (!right_disparity.matchDimension(width2, height, 1))
				right_disparity.allocate(width2, height, 1);

			
			float*& left_disparity_data = left_disparity.data();
			float*& right_disparity_data = right_disparity.data();

			//for left
			for (int h = 0; h < height; h++)
			{
				for (int w = 0; w < width1; w++)
				{
					int min_d = 0;
					float min_sum = cost_data[(h*W+w)*D+0];
					for (int d = 1; d < D; d++)
					{
						float tmp_sum = cost_data[(h*W + w)*D + d];
						if (tmp_sum < min_sum)
						{
							min_sum = tmp_sum;
							min_d = d;
						}
					}
					/*check uniqueness begin*/
					int min_d2;
					for (min_d2 = 0; min_d2 < D; min_d2++)
					{
						if (cost_data[(h*W + w)*D + min_d2]  < min_sum*uniqueness && std::abs(min_d - min_d2) > 1)
							break;
					}
					if (min_d2 < D)
					{
						left_disparity_data[h*width1 + w] = -1;
						continue;
					}
					/*check uniqueness end*/

					/*interpolation for sub-pixel precision begin*/
					int d = min_d;
					float final_d;
					if (0 < d && d < D - 1)
					{
						// do subpixel quadratic interpolation:
						//   fit parabola into (x1=d-1, y1=Sp[d-1]), (x2=d, y2=Sp[d]), (x3=d+1, y3=Sp[d+1])
						//   then find minimum of the parabola.
						float* sum_p = cost_data + (h*W + w)*D;
						float denom2 = sum_p[d - 1] + sum_p[d + 1] - 2 * sum_p[d];
						if (denom2 == 0)
							final_d = d;
						else
							final_d = (float)d + (sum_p[d - 1] - sum_p[d + 1]) / (denom2 * 2);
					}
					else
						final_d = d;
					left_disparity_data[h*width1 + w] = final_d;
					/*interpolation for sub-pixel precision end*/
				}
			}

			//for right
			for (int h = 0; h < height; h++)
			{
				for (int w = 0; w < width2; w++)
				{
					int min_d = 0;
					float min_sum = cost_data[(h*W + w + 0)*D + 0];
					for (int d = 1; d < D; d++)
					{
						int tmp_sum = cost_data[(h*W + w + d)*D + d];
						if (tmp_sum < min_sum)
						{
							min_sum = tmp_sum;
							min_d = d;
						}
					}
					/*check uniqueness begin*/
					int min_d2;
					for (min_d2 = 0; min_d2 < D; min_d2++)
					{
						if (cost_data[(h*W + w + min_d2)*D + min_d2]  < min_sum*uniqueness && std::abs(min_d - min_d2) > 1)
							break;
					}
					if (min_d2 < D)
					{
						right_disparity_data[h*width2 + w] = -1;
						continue;
					}
					/*check uniqueness end*/

					/*interpolation for sub-pixel precision begin*/
					int d = min_d;
					float final_d;
					if (0 < d && d < D - 1)
					{
						// do subpixel quadratic interpolation:
						//   fit parabola into (x1=d-1, y1=Sp[d-1]), (x2=d, y2=Sp[d]), (x3=d+1, y3=Sp[d+1])
						//   then find minimum of the parabola.
						int sum_p2 = cost_data[(h*W + w + d)*D + d];
						int sum_p1 = cost_data[(h*W + w + d - 1)*D + d - 1];
						int sum_p3 = cost_data[(h*W + w + d + 1)*D + d + 1];
						float denom2 = sum_p1 + sum_p3 - 2 * sum_p2;
						if (denom2 == 0)
							final_d = d;
						else
							final_d = (float)d + (sum_p1 - sum_p3) / (denom2 * 2);
					}
					else
						final_d = d;
					right_disparity_data[h*width2 + w] = final_d;
					/*interpolation for sub-pixel precision end*/
				}
			}
			return true;
		}

		static bool _computeDisparitySGBM_fullDP(const ZQ_DImage<unsigned char>& img1, const ZQ_DImage<unsigned char>& img2, 
			ZQ_DImage<float>& left_disparity, ZQ_DImage<float>& right_disparity, int maxD, int P1, int P2, 
			int half_sad_win_size_W = 3, int half_sad_win_size_H = 3, float uniqueness = 1.15f, int truncate_value = 64, int Ndir = 8)
		{
			if (Ndir != 8 && Ndir != 16)
				return false;

			ZQ_DImage<int> cost;
			if (!_computeMatchingCost_SAD_BT(img1, img2, cost, maxD, half_sad_win_size_W, half_sad_win_size_H, truncate_value))
				return false;
			int*& cost_data = cost.data();

			int W = cost.width();
			int height = cost.height();
			int D = cost.nchannels();
			int width1 = img1.width();
			int width2 = img2.width();
			if (!left_disparity.matchDimension(width1, height, 1))
				left_disparity.allocate(width1, height, 1);
			if (!right_disparity.matchDimension(width2, height, 1))
				right_disparity.allocate(width2, height, 1);


			////////
			int NBoarder = Ndir == 8 ? 1 : 2;
			int WL2 = width1 + 2*NBoarder;
			int WR2 = width2 + 2*NBoarder;
			int D2 = D + 2;
			int H2 = height + 2*NBoarder;
			float*& left_disparity_data = left_disparity.data();
			float*& right_disparity_data = right_disparity.data();
			float* disparity_data = 0;
			const int MAX_COST = INT_MAX;
			
			int dir_x[8] = { 1, 1, 0, -1, 2, 1, -1, -2 };
			int dir_y[8] = { 0, 1, 1, 1, 1, 2, 2, 1 };


			
			ZQ_DImage<int> sum_Lr(width1,height,D);
			ZQ_DImage<int> Lr(WL2,H2,D2);
			ZQ_DImage<int> minLr(WL2,H2);
			int*& sum_Lr_data = sum_Lr.data();
			int*& Lr_data = Lr.data();
			int*& minLr_data = minLr.data();

			/*left disp : pass = 0,
			right disp: pass =1
			*/
			
			for (int pass = 0; pass < 2; pass++)
			{
				int width, W2;
				if (pass == 0)
				{
					width = width1;
					disparity_data = left_disparity_data;
				}
				else
				{
					if (width1 != width2)
					{
						sum_Lr.allocate(width2, height, D);
						Lr.allocate(WR2, H2, D2);
						minLr.allocate(WR2, H2);
					}
					else
					{
						sum_Lr.reset();
						Lr.reset();
						minLr.reset();
					}
					width = width2;
					disparity_data = right_disparity_data;
				}
				W2 = width + 2*NBoarder;

				int halfNdir = Ndir / 2;
				for (int dd = 0; dd < halfNdir; dd++)
				{
					for (int h = 0; h < height; h++)
					{
						for (int w = 0; w < width; w++)
						{
							int cur_px = w + NBoarder;
							int cur_py = h + NBoarder;

							int last_px = cur_px - dir_x[dd];
							int	last_py = cur_py - dir_y[dd];
							int minLr_p_r = minLr_data[last_py * W2 + last_px];
							int* sum_Lr_p = sum_Lr_data + (h*width + w)*D;
							int* Lr_p = Lr_data + (cur_py*W2 + cur_px)*D2 + 1;
							int* Lr_p_r = Lr_data + (last_py*W2 + last_px)*D2 + 1;
							
							Lr_p_r[-1] = MAX_COST - P1;
							Lr_p_r[D] = MAX_COST - P1;
						
							int cur_min = MAX_COST;
							for (int d = 0; d < D; d++)
							{
								//Lr(p, d) = C(p, d) + min(Lr(p - r, d), Lr(p - r, d - 1) + P1, Lr(p - r, d + 1) + P1, minLr(p - r) + P2) - minLr(p-r);

								int val0 = Lr_p_r[d];
								int val1 = Lr_p_r[d - 1] + P1;
								int val2 = Lr_p_r[d + 1] + P1;
								int val3 = minLr_p_r + P2;
								int min_val = __min(val0, val1);
								min_val = __min(min_val, val2);
								min_val = __min(min_val, val3);
								int Lr_p_d = (min_val - minLr_p_r);
								Lr_p_d += pass == 0 ? cost_data[(h*W + w)*D + d] : cost_data[(h*W + w + d)*D + d];
								cur_min = __min(cur_min, Lr_p_d);
								Lr_p[d] = Lr_p_d;
								sum_Lr_p[d] += Lr_p_d;
							}
							minLr_data[(h + NBoarder)*W2 + w + NBoarder] = cur_min;
						}
					}

					minLr.reset();
					for (int h = height - 1; h >= 0; h--)
					{
						for (int w = width - 1; w >= 0; w--)
						{
							int cur_px = w + NBoarder;
							int cur_py = h + NBoarder;

							int last_px = cur_px + dir_x[dd];
							int	last_py = cur_py + dir_y[dd];
							int minLr_p_r = minLr_data[last_py * W2 + last_px];
							int* sum_Lr_p = sum_Lr_data + (h*width + w)*D;
							int* Lr_p = Lr_data + (cur_py*W2 + cur_px)*D2 + 1;
							int* Lr_p_r = Lr_data + (last_py*W2 + last_px)*D2 + 1;
							
							Lr_p_r[-1] = MAX_COST - P1;
							Lr_p_r[D] = MAX_COST - P1;
							
							int cur_min = MAX_COST;
							for (int d = 0; d < D; d++)
							{
								//Lr(p, d) = C(p, d) + min(Lr(p - r, d), Lr(p - r, d - 1) + P1, Lr(p - r, d + 1) + P1, minLr(p - r) + P2) - minLr(p-r);

								int val0 = Lr_p_r[d];
								int val1 = Lr_p_r[d - 1] + P1;
								int val2 = Lr_p_r[d + 1] + P1;
								int val3 = minLr_p_r + P2;
								int min_val = __min(val0, val1);
								min_val = __min(min_val, val2);
								min_val = __min(min_val, val3);
								int Lr_p_d = min_val - minLr_p_r;
								Lr_p_d += pass == 0 ? cost_data[(h*W + w)*D + d] : cost_data[(h*W + w + d)*D + d];
								cur_min = __min(cur_min, Lr_p_d);
								Lr_p[d] = Lr_p_d;
								sum_Lr_p[d] += Lr_p_d;
							}
							minLr_data[(h + NBoarder)*W2 + w + NBoarder] = cur_min;
						}
					}
				}

				///

				for (int h = 0; h < height; h++)
				{
					for (int w = 0; w < width; w++)
					{
						int min_d = -1;
						int min_sum = MAX_COST;
						for (int d = 0; d < D; d++)
						{
							int tmp_sum = sum_Lr_data[(h*width + w)*D + d];
							if (tmp_sum < min_sum)
							{
								min_sum = tmp_sum;
								min_d = d;
							}
						}
						/*check uniqueness begin*/
						int min_d2;
						for (min_d2 = 0; min_d2 < D; min_d2++)
						{
							if (sum_Lr_data[(h*width+w)*D+min_d2]  < min_sum*uniqueness && std::abs(min_d - min_d2) > 1)
								break;
						}
						if (min_d2 < D)
						{
							disparity_data[h*width + w] = -1;
							continue;
						}
						/*check uniqueness end*/

						/*interpolation for sub-pixel precision begin*/
						int d = min_d;
						float final_d;
						if (0 < d && d < D - 1)
						{
							// do subpixel quadratic interpolation:
							//   fit parabola into (x1=d-1, y1=Sp[d-1]), (x2=d, y2=Sp[d]), (x3=d+1, y3=Sp[d+1])
							//   then find minimum of the parabola.
							int* sum_p = sum_Lr_data + (h*width + w)*D;
							float denom2 = sum_p[d - 1] + sum_p[d + 1] - 2 * sum_p[d];
							if (denom2 == 0)
								final_d = d;
							else
								final_d = (float)d + (sum_p[d - 1] - sum_p[d + 1]) / (denom2 * 2);
						}
						else
							final_d = d;

						disparity_data[h*width + w] = final_d;
						/*interpolation for sub-pixel precision end*/
					}
				}
			}
			return true;
		}

		static bool _computeDisparitySGBM_NCC_fullDP(const ZQ_DImage<unsigned char>& img1, const ZQ_DImage<unsigned char>& img2,
			ZQ_DImage<float>& left_disparity, ZQ_DImage<float>& right_disparity, int maxD, float P1 = 0.0006, float P2 = 0.0024,
			int half_sad_win_size_W = 7, int half_sad_win_size_H = 7, float uniqueness = 1.0f, int Ndir = 8)
		{
			if (Ndir != 8 && Ndir != 16)
				return false;

			ZQ_DImage<float> cost;
			if (!_computeMatchingCost_NCC(img1, img2, cost, maxD, half_sad_win_size_W, half_sad_win_size_H))
				return false;
			cost.Multiplywith(-1);
			cost.Addwith(1);
			float*& cost_data = cost.data();

			int W = cost.width();
			int height = cost.height();
			int D = cost.nchannels();
			int width1 = img1.width();
			int width2 = img2.width();
			if (!left_disparity.matchDimension(width1, height, 1))
				left_disparity.allocate(width1, height, 1);
			if (!right_disparity.matchDimension(width2, height, 1))
				right_disparity.allocate(width2, height, 1);


			////////
			int NBoarder = Ndir == 8 ? 1 : 2;
			int WL2 = width1 + 2 * NBoarder;
			int WR2 = width2 + 2 * NBoarder;
			int D2 = D + 2;
			int H2 = height + 2 * NBoarder;
			float*& left_disparity_data = left_disparity.data();
			float*& right_disparity_data = right_disparity.data();
			float* disparity_data = 0;
			const float MAX_COST = 1e16;

			int dir_x[8] = { 1, 1, 0, -1, 2, 1, -1, -2 };
			int dir_y[8] = { 0, 1, 1, 1, 1, 2, 2, 1 };



			ZQ_DImage<float> sum_Lr(width1, height, D);
			ZQ_DImage<float> Lr(WL2, H2, D2);
			ZQ_DImage<float> minLr(WL2, H2);
			float*& sum_Lr_data = sum_Lr.data();
			float*& Lr_data = Lr.data();
			float*& minLr_data = minLr.data();

			/*left disp : pass = 0,
			right disp: pass =1
			*/

			for (int pass = 0; pass < 2; pass++)
			{
				int width, W2;
				if (pass == 0)
				{
					width = width1;
					disparity_data = left_disparity_data;
				}
				else
				{
					if (width1 != width2)
					{
						sum_Lr.allocate(width2, height, D);
						Lr.allocate(WR2, H2, D2);
						minLr.allocate(WR2, H2);
					}
					else
					{
						sum_Lr.reset();
						Lr.reset();
						minLr.reset();
					}
					width = width2;
					disparity_data = right_disparity_data;
				}
				W2 = width + 2 * NBoarder;

				int halfNdir = Ndir / 2;
				for (int dd = 0; dd < halfNdir; dd++)
				{
					for (int h = 0; h < height; h++)
					{
						for (int w = 0; w < width; w++)
						{
							int cur_px = w + NBoarder;
							int cur_py = h + NBoarder;

							int last_px = cur_px - dir_x[dd];
							int	last_py = cur_py - dir_y[dd];
							float minLr_p_r = minLr_data[last_py * W2 + last_px];
							float* sum_Lr_p = sum_Lr_data + (h*width + w)*D;
							float* Lr_p = Lr_data + (cur_py*W2 + cur_px)*D2 + 1;
							float* Lr_p_r = Lr_data + (last_py*W2 + last_px)*D2 + 1;

							Lr_p_r[-1] = MAX_COST - P1;
							Lr_p_r[D] = MAX_COST - P1;

							float cur_min = MAX_COST;
							for (int d = 0; d < D; d++)
							{
								//Lr(p, d) = C(p, d) + min(Lr(p - r, d), Lr(p - r, d - 1) + P1, Lr(p - r, d + 1) + P1, minLr(p - r) + P2) - minLr(p-r);

								float val0 = Lr_p_r[d];
								float val1 = Lr_p_r[d - 1] + P1;
								float val2 = Lr_p_r[d + 1] + P1;
								float val3 = minLr_p_r + P2;
								float min_val = __min(val0, val1);
								min_val = __min(min_val, val2);
								min_val = __min(min_val, val3);
								float Lr_p_d = (min_val - minLr_p_r);
								Lr_p_d += pass == 0 ? cost_data[(h*W + w)*D + d] : cost_data[(h*W + w + d)*D + d];
								cur_min = __min(cur_min, Lr_p_d);
								Lr_p[d] = Lr_p_d;
								sum_Lr_p[d] += Lr_p_d;
							}
							minLr_data[(h + NBoarder)*W2 + w + NBoarder] = cur_min;
						}
					}

					minLr.reset();
					for (int h = height - 1; h >= 0; h--)
					{
						for (int w = width - 1; w >= 0; w--)
						{
							int cur_px = w + NBoarder;
							int cur_py = h + NBoarder;

							int last_px = cur_px + dir_x[dd];
							int	last_py = cur_py + dir_y[dd];
							float minLr_p_r = minLr_data[last_py * W2 + last_px];
							float* sum_Lr_p = sum_Lr_data + (h*width + w)*D;
							float* Lr_p = Lr_data + (cur_py*W2 + cur_px)*D2 + 1;
							float* Lr_p_r = Lr_data + (last_py*W2 + last_px)*D2 + 1;

							Lr_p_r[-1] = MAX_COST - P1;
							Lr_p_r[D] = MAX_COST - P1;

							float cur_min = MAX_COST;
							for (int d = 0; d < D; d++)
							{
								//Lr(p, d) = C(p, d) + min(Lr(p - r, d), Lr(p - r, d - 1) + P1, Lr(p - r, d + 1) + P1, minLr(p - r) + P2) - minLr(p-r);

								float val0 = Lr_p_r[d];
								float val1 = Lr_p_r[d - 1] + P1;
								float val2 = Lr_p_r[d + 1] + P1;
								float val3 = minLr_p_r + P2;
								float min_val = __min(val0, val1);
								min_val = __min(min_val, val2);
								min_val = __min(min_val, val3);
								float Lr_p_d = min_val - minLr_p_r;
								Lr_p_d += pass == 0 ? cost_data[(h*W + w)*D + d] : cost_data[(h*W + w + d)*D + d];
								cur_min = __min(cur_min, Lr_p_d);
								Lr_p[d] = Lr_p_d;
								sum_Lr_p[d] += Lr_p_d;
							}
							minLr_data[(h + NBoarder)*W2 + w + NBoarder] = cur_min;
						}
					}
				}

				///

				for (int h = 0; h < height; h++)
				{
					for (int w = 0; w < width; w++)
					{
						int min_d = -1;
						float min_sum = MAX_COST;
						for (int d = 0; d < D; d++)
						{
							float tmp_sum = sum_Lr_data[(h*width + w)*D + d];
							if (tmp_sum < min_sum)
							{
								min_sum = tmp_sum;
								min_d = d;
							}
						}
						/*check uniqueness begin*/
						int min_d2;
						for (min_d2 = 0; min_d2 < D; min_d2++)
						{
							if (sum_Lr_data[(h*width + w)*D + min_d2]  < min_sum*uniqueness && std::abs(min_d - min_d2) > 1)
								break;
						}
						if (min_d2 < D)
						{
							disparity_data[h*width + w] = -1;
							continue;
						}
						/*check uniqueness end*/

						/*interpolation for sub-pixel precision begin*/
						int d = min_d;
						float final_d;
						if (0 < d && d < D - 1)
						{
							// do subpixel quadratic interpolation:
							//   fit parabola into (x1=d-1, y1=Sp[d-1]), (x2=d, y2=Sp[d]), (x3=d+1, y3=Sp[d+1])
							//   then find minimum of the parabola.
							float* sum_p = sum_Lr_data + (h*width + w)*D;
							float denom2 = sum_p[d - 1] + sum_p[d + 1] - 2 * sum_p[d];
							if (denom2 == 0)
								final_d = d;
							else
								final_d = (float)d + (sum_p[d - 1] - sum_p[d + 1]) / (denom2 * 2);
						}
						else
							final_d = d;

						disparity_data[h*width + w] = final_d;
						/*interpolation for sub-pixel precision end*/
					}
				}
			}
			return true;
		}

		

		static bool _computeDisparitySGBM_halfDP(const ZQ_DImage<unsigned char>& img1, const ZQ_DImage<unsigned char>& img2,
			ZQ_DImage<float>& left_disparity, ZQ_DImage<float>& right_disparity, int maxD, int P1, int P2,
			int half_sad_win_size_W = 3, int half_sad_win_size_H = 3, float uniqueness = 1.15f, int truncate_value = 64, int Ndir = 8)
		{
			if (Ndir != 8 && Ndir != 16)
				return false;

			int width1 = img1.width();
			int height = img1.height();
			int nChannels = img1.nchannels();
			int width2 = img2.width();
			if (!img2.matchDimension(width2, height, nChannels))
				return false;
			if (maxD <= 0)
				return false;

			if (!left_disparity.matchDimension(width1, height, 1))
				left_disparity.allocate(width1, height, 1);
			if (!right_disparity.matchDimension(width2, height, 1))
				right_disparity.allocate(width2, height, 1);

			const int MAX_COST = INT_MAX - P2;

			float*& left_disparity_data = left_disparity.data();
			float*& right_disparity_data = right_disparity.data();

			int winH = 2 * half_sad_win_size_H + 1;
			int winW = 2 * half_sad_win_size_W + 1;
			int D = maxD + 1;
			int W = __max(width1, width2 + maxD);

			const unsigned char*& img1_data = img1.data();
			const unsigned char*& img2_data = img2.data();

			int NROWS = winH + 1;
			ZQ_DImage<short> pixDiff(W, 1, D);
			ZQ_DImage<int> row_sum(W, NROWS, D);
			ZQ_DImage<int> cost(W, 1, D);
			short*& pixDiff_data = pixDiff.data();
			int*& row_sum_data = row_sum.data();
			int*& cost_data = cost.data();


			////////
			int NBoarder = Ndir == 8 ? 1 : 2;
			int WL2 = width1 + 2 * NBoarder;
			int WR2 = width2 + 2 * NBoarder;
			int D2 = D + 2;
			int H2 = height + 2 * NBoarder;
		
			const int dir_x[8] = { 1, 1, 0, -1, 2, 1, -1, -2 };
			const int dir_y[8] = { 0, 1, 1, 1, 1, 2, 2, 1 };

			int halfNdir = Ndir / 2;
			int halfNdir1 = halfNdir + 1;
			int NBS = NBoarder + 1;
			
			ZQ_DImage<int> left_Lr(WL2, NBS, D2*halfNdir1), right_Lr(WR2, NBS, D2*halfNdir1);
			ZQ_DImage<int> left_minLr(WL2, NBS, Ndir), right_minLr(WR2, NBS, Ndir);
			
			int*& left_Lr_data = left_Lr.data();
			int*& left_minLr_data = left_minLr.data();
			int*& right_Lr_data = right_Lr.data();
			int*& right_minLr_data = right_minLr.data();


			for (int h = 0; h < height; h++)
			{
				/*compute cost of current row Begin*/
				if (h == 0)
				{
					for (int hh = 0; hh <= half_sad_win_size_H; hh++)
					{
						_computeMatchingCostBT_OneRow(img1_data + hh*width1*nChannels, img2_data + hh*width2*nChannels, width1, width2, nChannels, maxD, pixDiff_data, truncate_value);
						_cumulate_row(pixDiff_data, W, D, half_sad_win_size_W, row_sum_data + (hh + half_sad_win_size_H)*W*D);
						if (hh == 0)
						{
							for (int i = 0; i < half_sad_win_size_H; i++)
								memcpy(row_sum_data + i*W*D, row_sum_data + (hh + half_sad_win_size_H)*W*D, sizeof(int)*W*D);
						}
					}

					for (int d = 0; d < D; d++)
					{
						for (int w = 0; w < W; w++)
						{
							cost_data[w*D + d] = 0;
							for (int hh = 0; hh < winH; hh++)
								cost_data[w*D + d] += row_sum_data[hh*W*D + w*D + d];
						}
					}
				}
				else
				{
					int row_idx = h + half_sad_win_size_H;
					int last_row_idx = row_idx - 1;

					int add_store_kk = (row_idx + half_sad_win_size_H) % NROWS;
					int last_store_kk = (last_row_idx + half_sad_win_size_H) % NROWS;
					int sub_store_kk = (row_idx - half_sad_win_size_H - 1) % NROWS;
					int* add_row_data = row_sum_data + add_store_kk*W*D;
					int* last_row_data = row_sum_data + last_store_kk*W*D;
					int* sub_row_data = row_sum_data + sub_store_kk*W*D;

					if (row_idx >= height)
						memcpy(add_row_data, last_row_data, sizeof(int)*W*D);
					else
					{
						_computeMatchingCostBT_OneRow(img1_data + row_idx*width1*nChannels, img2_data + row_idx*width2*nChannels, width1, width2, nChannels, maxD, pixDiff_data, truncate_value);
						_cumulate_row(pixDiff_data, W, D, half_sad_win_size_W, add_row_data);
					}

					for (int d = 0; d < D; d++)
					{
						for (int w = 0; w < W; w++)
						{
							cost_data[w*D + d] += -sub_row_data[w*D + d] + add_row_data[w*D + d];
						}
					}
				}
				/*compute cost of current row End*/

				/*****************************       compute Lr and find best d         Begin      ************************/
				/*****************************       pass == 0 for left,  pass == 1 for right      ************************/
				for (int pass = 0; pass < 2; pass++)
				{
					float* disparity_data = pass == 0 ? left_disparity_data : right_disparity_data;
					int* minLr = pass == 0 ? left_minLr_data : right_minLr_data;
					int* Lr = pass == 0 ? left_Lr_data : right_Lr_data;
					int W2 = pass == 0 ? WL2 : WR2;
					int width = pass == 0 ? width1 : width2;

					/*compute sumLr for p =(h,w) Begin*/
					ZQ_DImage<int> sum_Lr(width, 1, D);
					int*& sum_Lr_data = sum_Lr.data();

					for (int w = 0; w < width; w++)
					{
						int* tmp_minLr[3] = {
							minLr + (h % NBS*W2 + NBoarder + w)*halfNdir1,
							minLr + ((h + 1) % NBS*W2 + NBoarder + w)*halfNdir1,
							minLr + ((h + 2) % NBS*W2 + NBoarder + w)*halfNdir1
						};
						int* tmp_Lr[3] = {
							Lr + (h % NBS*W2 + w + NBoarder)*halfNdir1*D2 + 1,
							Lr + ((h + 1) % NBS*W2 + w + NBoarder)*halfNdir1*D2 + 1,
							Lr + ((h + 2) % NBS*W2 + w + NBoarder)*halfNdir1*D2 + 1
						};
						int** cur_minLr = &(tmp_minLr[2]);
						int** cur_Lr = &(tmp_Lr[2]);
						
						for (int dd = 0; dd < halfNdir; dd++)
						{
							int last_px = -dir_x[dd];
							int	last_py = -dir_y[dd];
							int* minLr_p_r = cur_minLr[last_py] + last_px*halfNdir1;
							int* Lr_p_r = cur_Lr[last_py] + last_px*halfNdir1*D2 + dd*D2;
							Lr_p_r[-1] = Lr_p_r[D] = MAX_COST;
							int* Lr_p = cur_Lr[0] + dd*D2;
							int cur_min = MAX_COST;
							for (int d = 0; d < D; d++)
							{
								//Lr(p, d) = C(p, d) + min(Lr(p - r, d), Lr(p - r, d - 1) + P1, Lr(p - r, d + 1) + P1, minLr(p - r) + P2) - minLr(p-r);
								int Lr_p_d = __min(Lr_p_r[d], __min(Lr_p_r[d - 1] + P1, __min(Lr_p_r[d + 1] + P1, minLr_p_r[dd] + P2))) - minLr_p_r[dd];
								Lr_p_d += pass == 0 ? cost_data[w*D + d] : cost_data[(w + d)*D + d];
								cur_min = __min(cur_min, Lr_p_d);
								Lr_p[d] = Lr_p_d;
								sum_Lr_data[w*D+d] += Lr_p_d;
							}
							cur_minLr[0][dd] = cur_min;
						}
					}

					for (int w = width - 1; w >= 0; w--)
					{
						int* tmp_minLr[3] = {
							minLr + (h % NBS*W2 + NBoarder + w)*halfNdir1 + halfNdir,
							minLr + ((h + 1) % NBS*W2 + NBoarder + w)*halfNdir1 + halfNdir,
							minLr + ((h + 2) % NBS*W2 + NBoarder + w)*halfNdir1 + halfNdir
						};
						int* tmp_Lr[3] = {
							Lr + (h % NBS*W2 + w + NBoarder)*halfNdir1*D2 + 1,
							Lr + ((h + 1) % NBS*W2 + w + NBoarder)*halfNdir1*D2 + 1,
							Lr + ((h + 2) % NBS*W2 + w + NBoarder)*halfNdir1*D2 + 1
						};
						int** cur_minLr = &(tmp_minLr[2]);
						int** cur_Lr = &(tmp_Lr[2]);
						
						for (int dd = 0; dd < 1; dd++)
						{
							int last_px = +dir_x[dd];
							int	last_py = -dir_y[dd];
							int* minLr_p_r = cur_minLr[last_py] + last_px*halfNdir1 + halfNdir;
							int* Lr_p_r = cur_Lr[last_py] + last_px*halfNdir1*D2;
							Lr_p_r[-1] = Lr_p_r[D] = MAX_COST;
							int* Lr_p = cur_Lr[0] + (dd+halfNdir)*D2;
							int cur_min = MAX_COST;
							for (int d = 0; d < D; d++)
							{
								//Lr(p, d) = C(p, d) + min(Lr(p - r, d), Lr(p - r, d - 1) + P1, Lr(p - r, d + 1) + P1, minLr(p - r) + P2) - minLr(p-r);
								int Lr_p_d = __min(Lr_p_r[d], __min(Lr_p_r[d - 1] + P1, __min(Lr_p_r[d + 1] + P1, minLr_p_r[dd] + P2))) - minLr_p_r[dd];
								Lr_p_d += pass == 0 ? cost_data[w*D + d] : cost_data[(w + d)*D + d];
								cur_min = __min(cur_min, Lr_p_d);
								Lr_p[d] = Lr_p_d;
								sum_Lr_data[w*D+d] += Lr_p_d;
							}
							cur_minLr[0][dd] = cur_min;
						}
					}
					/*compute sumLr for p =(h,w) End*/

					for (int w = 0; w < width;w++)
					{
						int min_d = -1;
						int min_sum = MAX_COST;
						for (int d = 0; d < D; d++)
						{
							int tmp_sum = sum_Lr_data[w*D+d];
							if (tmp_sum < min_sum)
							{
								min_sum = tmp_sum;
								min_d = d;
							}
						}
						/*check uniqueness begin*/
						int min_d2;
						for (min_d2 = 0; min_d2 < D; min_d2++)
						{
							if (sum_Lr_data[w*D+min_d2]  < min_sum*uniqueness && std::abs(min_d - min_d2) > 1)
								break;
						}
						if (min_d2 < D)
						{
							disparity_data[h*width + w] = -1;
							continue;
						}
						/*check uniqueness end*/

						/*interpolation for sub-pixel precision begin*/
						int d = min_d;
						float final_d;
						if (0 < d && d < D - 1)
						{
							// do subpixel quadratic interpolation:
							//   fit parabola into (x1=d-1, y1=Sp[d-1]), (x2=d, y2=Sp[d]), (x3=d+1, y3=Sp[d+1])
							//   then find minimum of the parabola.
							int* sum_p = sum_Lr_data + w*D;
							float denom2 = sum_p[d - 1] + sum_p[d + 1] - 2 * sum_p[d];
							if (denom2 == 0)
								final_d = d;
							else
								final_d = (float)d + (sum_p[d - 1] - sum_p[d + 1]) / (denom2 * 2);
						}
						else
							final_d = d;
						disparity_data[h*width + w] = final_d;
						/*interpolation for sub-pixel precision end*/
					}// loop for w

				}//loop for pass
				/*****************************       compute Lr and find best d         Begin      ************************/
				/*****************************       pass == 0 for left,  pass == 1 for right      ************************/

			}// loop for h

			return true;
		}

		static bool _computeDisparitySGBM_NCC_halfDP(const ZQ_DImage<unsigned char>& img1, const ZQ_DImage<unsigned char>& img2,
			ZQ_DImage<float>& left_disparity, ZQ_DImage<float>& right_disparity, int maxD, float P1 = 0.0006f, float P2 = 0.0024f,
			int half_win_size_W = 3, int half_win_size_H = 3, float uniqueness = 1.00f, int Ndir = 8)
		{
			if (Ndir != 8 && Ndir != 16)
				return false;

			int width1 = img1.width();
			int height = img1.height();
			int nChannels = img1.nchannels();
			int width2 = img2.width();
			if (!img2.matchDimension(width2, height, nChannels))
				return false;
			if (maxD <= 0)
				return false;

			if (!left_disparity.matchDimension(width1, height, 1))
				left_disparity.allocate(width1, height, 1);
			if (!right_disparity.matchDimension(width2, height, 1))
				right_disparity.allocate(width2, height, 1);

			const float MAX_COST = 1e16 - P2;

			float*& left_disparity_data = left_disparity.data();
			float*& right_disparity_data = right_disparity.data();

			int winH = 2 * half_win_size_H + 1;
			int winW = 2 * half_win_size_W + 1;
			int D = maxD + 1;
			int W = __max(width1, width2 + maxD);

			ZQ_DImage<float> denom1, denom2;
			_compute_denom_for_NCC(img1, img2, denom1, denom2, half_win_size_W, half_win_size_H, D);
			int denom_W1 = denom1.width();
			int denom_W2 = denom2.width();

			const unsigned char*& img1_data = img1.data();
			const unsigned char*& img2_data = img2.data();

			float*& denom1_data = denom1.data();
			float*& denom2_data = denom2.data();

			int NROWS = winH + 1;
			ZQ_DImage<float> cost(W, 1, D);
			ZQ_DImage<float> num(W, 1, D);
			ZQ_DImage<float> dotVal(W, 1, D);
			ZQ_DImage<float> row_sum(W, NROWS, D);
			float*& cost_data = cost.data();
			float*& num_data = num.data();
			float*& dotVal_data = dotVal.data();
			float*& row_sum_data = row_sum.data();

			////////
			int NBoarder = Ndir == 8 ? 1 : 2;
			int WL2 = width1 + 2 * NBoarder;
			int WR2 = width2 + 2 * NBoarder;
			int D2 = D + 2;
			int H2 = height + 2 * NBoarder;

			const int dir_x[8] = { 1, 1, 0, -1, 2, 1, -1, -2 };
			const int dir_y[8] = { 0, 1, 1, 1, 1, 2, 2, 1 };

			int halfNdir = Ndir / 2;
			int halfNdir1 = halfNdir + 1;
			int NBS = NBoarder + 1;

			ZQ_DImage<float> left_Lr(WL2, NBS, D2*halfNdir1), right_Lr(WR2, NBS, D2*halfNdir1);
			ZQ_DImage<float> left_minLr(WL2, NBS, Ndir), right_minLr(WR2, NBS, Ndir);

			float*& left_Lr_data = left_Lr.data();
			float*& left_minLr_data = left_minLr.data();
			float*& right_Lr_data = right_Lr.data();
			float*& right_minLr_data = right_minLr.data();


			for (int h = 0; h < height; h++)
			{
				/*compute cost of current row Begin*/
				if (h == 0)
				{
					for (int hh = 0; hh <= half_win_size_H; hh++)
					{
						_computeMatchingCostNCC_Dot_OneRow(img1_data + hh*width1*nChannels, img2_data + hh*width2*nChannels, width1, width2, nChannels, maxD, dotVal_data);
						_cumulate_row(dotVal_data, W, D, half_win_size_W, row_sum_data + (hh + half_win_size_H)*W*D);
						if (hh == 0)
						{
							for (int i = 0; i < half_win_size_H; i++)
								memcpy(row_sum_data + i*W*D, row_sum_data + (hh + half_win_size_H)*W*D, sizeof(float)*W*D);
						}
					}

					for (int w = 0; w < W; w++)
					{
						for (int d = 0; d < D; d++)
						{
							num_data[w*D + d] = 0;
							for (int hh = 0; hh < winH; hh++)
								num_data[w*D + d] += row_sum_data[hh*W*D + w*D + d];
							float denom_val1 = denom1_data[h*denom_W1 + w];
							float denom_val2 = denom2_data[h*denom_W2 + (w + D - 1 - d)];
							if (denom_val1 <= 0 || denom_val2 <= 0)
								cost_data[w*D + d] = 0;
							else
								cost_data[w*D + d] = num_data[w*D + d] / sqrt((double)denom_val1*denom_val2);
						}
					}
				}
				else
				{
					int row_idx = h + half_win_size_H;
					int last_row_idx = row_idx - 1;

					int add_store_kk = (row_idx + half_win_size_H) % NROWS;
					int last_store_kk = (last_row_idx + half_win_size_H) % NROWS;
					int sub_store_kk = (row_idx - half_win_size_H - 1) % NROWS;
					float* add_row_data = row_sum_data + add_store_kk*W*D;
					float* last_row_data = row_sum_data + last_store_kk*W*D;
					float* sub_row_data = row_sum_data + sub_store_kk*W*D;

					if (row_idx >= height)
						memcpy(add_row_data, last_row_data, sizeof(float)*W*D);
					else
					{
						_computeMatchingCostNCC_Dot_OneRow(img1_data + row_idx*width1*nChannels, img2_data + row_idx*width2*nChannels, width1, width2, nChannels, maxD, dotVal_data);
						_cumulate_row(dotVal_data, W, D, half_win_size_W, add_row_data);
					}

					for (int w = 0; w < W; w++)
					{
						for (int d = 0; d < D; d++)
						{
							num_data[w*D + d] += -sub_row_data[w*D + d] + add_row_data[w*D + d];
							float denom_val1 = denom1_data[h*denom_W1 + w];
							float denom_val2 = denom2_data[h*denom_W2 + (w + D - 1 - d)];
							if (denom_val1 <= 0 || denom_val2 <= 0)
								cost_data[w*D + d] = 0;
							else
								cost_data[w*D + d] =  num_data[w*D + d] / sqrt((double)denom_val1*denom_val2);
						}
					}
				}
				cost.Multiplywith(-1);
				cost.Addwith(1);
				/*compute cost of current row End*/

				/*****************************       compute Lr and find best d         Begin      ************************/
				/*****************************       pass == 0 for left,  pass == 1 for right      ************************/
				for (int pass = 0; pass < 2; pass++)
				{
					float* disparity_data = pass == 0 ? left_disparity_data : right_disparity_data;
					float* minLr = pass == 0 ? left_minLr_data : right_minLr_data;
					float* Lr = pass == 0 ? left_Lr_data : right_Lr_data;
					int W2 = pass == 0 ? WL2 : WR2;
					int width = pass == 0 ? width1 : width2;

					/*compute sumLr for p =(h,w) Begin*/
					ZQ_DImage<float> sum_Lr(width, 1, D);
					float*& sum_Lr_data = sum_Lr.data();

					for (int w = 0; w < width; w++)
					{
						float* tmp_minLr[3] = {
							minLr + (h % NBS*W2 + NBoarder + w)*halfNdir1,
							minLr + ((h + 1) % NBS*W2 + NBoarder + w)*halfNdir1,
							minLr + ((h + 2) % NBS*W2 + NBoarder + w)*halfNdir1
						};
						float* tmp_Lr[3] = {
							Lr + (h % NBS*W2 + w + NBoarder)*halfNdir1*D2 + 1,
							Lr + ((h + 1) % NBS*W2 + w + NBoarder)*halfNdir1*D2 + 1,
							Lr + ((h + 2) % NBS*W2 + w + NBoarder)*halfNdir1*D2 + 1
						};
						float** cur_minLr = &(tmp_minLr[2]);
						float** cur_Lr = &(tmp_Lr[2]);

						for (int dd = 0; dd < halfNdir; dd++)
						{
							int last_px = -dir_x[dd];
							int	last_py = -dir_y[dd];
							float* minLr_p_r = cur_minLr[last_py] + last_px*halfNdir1;
							float* Lr_p_r = cur_Lr[last_py] + last_px*halfNdir1*D2 + dd*D2;
							Lr_p_r[-1] = Lr_p_r[D] = MAX_COST;
							float* Lr_p = cur_Lr[0] + dd*D2;
							int cur_min = MAX_COST;
							for (int d = 0; d < D; d++)
							{
								//Lr(p, d) = C(p, d) + min(Lr(p - r, d), Lr(p - r, d - 1) + P1, Lr(p - r, d + 1) + P1, minLr(p - r) + P2) - minLr(p-r);
								int Lr_p_d = __min(Lr_p_r[d], __min(Lr_p_r[d - 1] + P1, __min(Lr_p_r[d + 1] + P1, minLr_p_r[dd] + P2))) - minLr_p_r[dd];
								Lr_p_d += pass == 0 ? cost_data[w*D + d] : cost_data[(w + d)*D + d];
								cur_min = __min(cur_min, Lr_p_d);
								Lr_p[d] = Lr_p_d;
								sum_Lr_data[w*D + d] += Lr_p_d;
							}
							cur_minLr[0][dd] = cur_min;
						}
					}

					for (int w = width - 1; w >= 0; w--)
					{
						float* tmp_minLr[3] = {
							minLr + (h % NBS*W2 + NBoarder + w)*halfNdir1 + halfNdir,
							minLr + ((h + 1) % NBS*W2 + NBoarder + w)*halfNdir1 + halfNdir,
							minLr + ((h + 2) % NBS*W2 + NBoarder + w)*halfNdir1 + halfNdir
						};
						float* tmp_Lr[3] = {
							Lr + (h % NBS*W2 + w + NBoarder)*halfNdir1*D2 + 1,
							Lr + ((h + 1) % NBS*W2 + w + NBoarder)*halfNdir1*D2 + 1,
							Lr + ((h + 2) % NBS*W2 + w + NBoarder)*halfNdir1*D2 + 1
						};
						float** cur_minLr = &(tmp_minLr[2]);
						float** cur_Lr = &(tmp_Lr[2]);

						for (int dd = 0; dd < 1; dd++)
						{
							int last_px = +dir_x[dd];
							int	last_py = -dir_y[dd];
							float* minLr_p_r = cur_minLr[last_py] + last_px*halfNdir1 + halfNdir;
							float* Lr_p_r = cur_Lr[last_py] + last_px*halfNdir1*D2;
							Lr_p_r[-1] = Lr_p_r[D] = MAX_COST;
							float* Lr_p = cur_Lr[0] + (dd + halfNdir)*D2;
							float cur_min = MAX_COST;
							for (int d = 0; d < D; d++)
							{
								//Lr(p, d) = C(p, d) + min(Lr(p - r, d), Lr(p - r, d - 1) + P1, Lr(p - r, d + 1) + P1, minLr(p - r) + P2) - minLr(p-r);
								float Lr_p_d = __min(Lr_p_r[d], __min(Lr_p_r[d - 1] + P1, __min(Lr_p_r[d + 1] + P1, minLr_p_r[dd] + P2))) - minLr_p_r[dd];
								Lr_p_d += pass == 0 ? cost_data[w*D + d] : cost_data[(w + d)*D + d];
								cur_min = __min(cur_min, Lr_p_d);
								Lr_p[d] = Lr_p_d;
								sum_Lr_data[w*D + d] += Lr_p_d;
							}
							cur_minLr[0][dd] = cur_min;
						}
					}
					/*compute sumLr for p =(h,w) End*/

					for (int w = 0; w < width; w++)
					{
						int min_d = -1;
						float min_sum = MAX_COST;
						for (int d = 0; d < D; d++)
						{
							float tmp_sum = sum_Lr_data[w*D + d];
							if (tmp_sum < min_sum)
							{
								min_sum = tmp_sum;
								min_d = d;
							}
						}
						/*check uniqueness begin*/
						int min_d2;
						for (min_d2 = 0; min_d2 < D; min_d2++)
						{
							if (sum_Lr_data[w*D + min_d2]  < min_sum*uniqueness && std::abs(min_d - min_d2) > 1)
								break;
						}
						if (min_d2 < D)
						{
							disparity_data[h*width + w] = -1;
							continue;
						}
						/*check uniqueness end*/

						/*interpolation for sub-pixel precision begin*/
						int d = min_d;
						float final_d;
						if (0 < d && d < D - 1)
						{
							// do subpixel quadratic interpolation:
							//   fit parabola into (x1=d-1, y1=Sp[d-1]), (x2=d, y2=Sp[d]), (x3=d+1, y3=Sp[d+1])
							//   then find minimum of the parabola.
							float* sum_p = sum_Lr_data + w*D;
							float denom2 = sum_p[d - 1] + sum_p[d + 1] - 2 * sum_p[d];
							if (denom2 == 0)
								final_d = d;
							else
								final_d = (float)d + (sum_p[d - 1] - sum_p[d + 1]) / (denom2 * 2);
						}
						else
							final_d = d;
						disparity_data[h*width + w] = final_d;
						/*interpolation for sub-pixel precision end*/
					}// loop for w

				}//loop for pass
				/*****************************       compute Lr and find best d         Begin      ************************/
				/*****************************       pass == 0 for left,  pass == 1 for right      ************************/

			}// loop for h

			return true;
		}

		static bool _computeDisparitySGBM_halfDP_SSE(const ZQ_DImage<unsigned char>& img1, const ZQ_DImage<unsigned char>& img2,
			ZQ_DImage<float>& left_disparity, ZQ_DImage<float>& right_disparity, int maxD, int P1, int P2,
			int half_sad_win_size_W = 3, int half_sad_win_size_H = 3, float uniqueness = 1.15f, int truncate_value = 64, int Ndir = 8)
		{
			if (Ndir != 8 && Ndir != 16)
				return false;

			int width1 = img1.width();
			int height = img1.height();
			int nChannels = img1.nchannels();
			int width2 = img2.width();
			if (!img2.matchDimension(width2, height, nChannels))
				return false;
			if (maxD <= 0)
				return false;

			if (!left_disparity.matchDimension(width1, height, 1))
				left_disparity.allocate(width1, height, 1);
			if (!right_disparity.matchDimension(width2, height, 1))
				right_disparity.allocate(width2, height, 1);

			const int MAX_COST = INT_MAX - P2;

			float*& left_disparity_data = left_disparity.data();
			float*& right_disparity_data = right_disparity.data();

			int winH = 2 * half_sad_win_size_H + 1;
			int winW = 2 * half_sad_win_size_W + 1;
			int D = maxD + 1;
			int W = __max(width1, width2 + maxD);

			const unsigned char*& img1_data = img1.data();
			const unsigned char*& img2_data = img2.data();

			int NROWS = winH + 1;
			ZQ_DImage<short> pixDiff(W, 1, D);
			ZQ_DImage<int> row_sum(W, NROWS, D);
			ZQ_DImage<int> cost(W, 1, D);
			short*& pixDiff_data = pixDiff.data();
			int*& row_sum_data = row_sum.data();
			int*& cost_data = cost.data();


			////////
			int NBoarder = Ndir == 8 ? 1 : 2;
			int WL2 = width1 + 2 * NBoarder;
			int WR2 = width2 + 2 * NBoarder;
			int D2 = D + 2;
			int H2 = height + 2 * NBoarder;

			const int dir_x[8] = { 1, 1, 0, -1, 2, 1, -1, -2 };
			const int dir_y[8] = { 0, 1, 1, 1, 1, 2, 2, 1 };

			int halfNdir = Ndir / 2;
			int halfNdir1 = halfNdir + 1;
			int NBS = NBoarder + 1;

			ZQ_DImage<int> left_Lr(WL2, NBS, D2*halfNdir1), right_Lr(WR2, NBS, D2*halfNdir1);
			ZQ_DImage<int> left_minLr(WL2, NBS, Ndir), right_minLr(WR2, NBS, Ndir);

			int*& left_Lr_data = left_Lr.data();
			int*& left_minLr_data = left_minLr.data();
			int*& right_Lr_data = right_Lr.data();
			int*& right_minLr_data = right_minLr.data();


			for (int h = 0; h < height; h++)
			{
				/*compute cost of current row Begin*/
				if (h == 0)
				{
					for (int hh = 0; hh <= half_sad_win_size_H; hh++)
					{
						_computeMatchingCostBT_OneRow_SSE(img1_data + hh*width1*nChannels, img2_data + hh*width2*nChannels, width1, width2, nChannels, maxD, pixDiff_data, truncate_value);
						_cumulate_row(pixDiff_data, W, D, half_sad_win_size_W, row_sum_data + (hh + half_sad_win_size_H)*W*D);
						if (hh == 0)
						{
							for (int i = 0; i < half_sad_win_size_H; i++)
								memcpy(row_sum_data + i*W*D, row_sum_data + (hh + half_sad_win_size_H)*W*D, sizeof(int)*W*D);
						}
					}

					for (int d = 0; d < D; d++)
					{
						for (int w = 0; w < W; w++)
						{
							cost_data[w*D + d] = 0;
							for (int hh = 0; hh < winH; hh++)
								cost_data[w*D + d] += row_sum_data[hh*W*D + w*D + d];
						}
					}
				}
				else
				{
					int row_idx = h + half_sad_win_size_H;
					int last_row_idx = row_idx - 1;

					int add_store_kk = (row_idx + half_sad_win_size_H) % NROWS;
					int last_store_kk = (last_row_idx + half_sad_win_size_H) % NROWS;
					int sub_store_kk = (row_idx - half_sad_win_size_H - 1) % NROWS;
					int* add_row_data = row_sum_data + add_store_kk*W*D;
					int* last_row_data = row_sum_data + last_store_kk*W*D;
					int* sub_row_data = row_sum_data + sub_store_kk*W*D;

					if (row_idx >= height)
						memcpy(add_row_data, last_row_data, sizeof(int)*W*D);
					else
					{
						_computeMatchingCostBT_OneRow_SSE(img1_data + row_idx*width1*nChannels, img2_data + row_idx*width2*nChannels, width1, width2, nChannels, maxD, pixDiff_data, truncate_value);
						_cumulate_row(pixDiff_data, W, D, half_sad_win_size_W, add_row_data);
					}

					for (int d = 0; d < D; d++)
					{
						for (int w = 0; w < W; w++)
						{
							cost_data[w*D + d] += -sub_row_data[w*D + d] + add_row_data[w*D + d];
						}
					}
				}
				/*compute cost of current row End*/

				/*****************************       compute Lr and find best d         Begin      ************************/
				/*****************************       pass == 0 for left,  pass == 1 for right      ************************/
				for (int pass = 0; pass < 2; pass++)
				{
					float* disparity_data = pass == 0 ? left_disparity_data : right_disparity_data;
					int* minLr = pass == 0 ? left_minLr_data : right_minLr_data;
					int* Lr = pass == 0 ? left_Lr_data : right_Lr_data;
					int W2 = pass == 0 ? WL2 : WR2;
					int width = pass == 0 ? width1 : width2;

					/*compute sumLr for p =(h,w) Begin*/
					ZQ_DImage<int> sum_Lr(width, 1, D);
					int*& sum_Lr_data = sum_Lr.data();

					for (int w = 0; w < width; w++)
					{
						int* tmp_minLr[3] = {
							minLr + (h % NBS*W2 + NBoarder + w)*halfNdir1,
							minLr + ((h + 1) % NBS*W2 + NBoarder + w)*halfNdir1,
							minLr + ((h + 2) % NBS*W2 + NBoarder + w)*halfNdir1
						};
						int* tmp_Lr[3] = {
							Lr + (h % NBS*W2 + w + NBoarder)*halfNdir1*D2 + 1,
							Lr + ((h + 1) % NBS*W2 + w + NBoarder)*halfNdir1*D2 + 1,
							Lr + ((h + 2) % NBS*W2 + w + NBoarder)*halfNdir1*D2 + 1
						};
						int** cur_minLr = &(tmp_minLr[2]);
						int** cur_Lr = &(tmp_Lr[2]);

						for (int dd = 0; dd < halfNdir; dd++)
						{
							int last_px = -dir_x[dd];
							int	last_py = -dir_y[dd];
							int* minLr_p_r = cur_minLr[last_py] + last_px*halfNdir1;
							int* Lr_p_r = cur_Lr[last_py] + last_px*halfNdir1*D2 + dd*D2;
							Lr_p_r[-1] = Lr_p_r[D] = MAX_COST;
							int* Lr_p = cur_Lr[0] + dd*D2;
							int cur_min = MAX_COST;
							for (int d = 0; d < D; d++)
							{
								//Lr(p, d) = C(p, d) + min(Lr(p - r, d), Lr(p - r, d - 1) + P1, Lr(p - r, d + 1) + P1, minLr(p - r) + P2) - minLr(p-r);
								int Lr_p_d = __min(Lr_p_r[d], __min(Lr_p_r[d - 1] + P1, __min(Lr_p_r[d + 1] + P1, minLr_p_r[dd] + P2))) - minLr_p_r[dd];
								Lr_p_d += pass == 0 ? cost_data[w*D + d] : cost_data[(w + d)*D + d];
								cur_min = __min(cur_min, Lr_p_d);
								Lr_p[d] = Lr_p_d;
								sum_Lr_data[w*D + d] += Lr_p_d;
							}
							cur_minLr[0][dd] = cur_min;
						}
					}

					for (int w = width - 1; w >= 0; w--)
					{
						int* tmp_minLr[3] = {
							minLr + (h % NBS*W2 + NBoarder + w)*halfNdir1 + halfNdir,
							minLr + ((h + 1) % NBS*W2 + NBoarder + w)*halfNdir1 + halfNdir,
							minLr + ((h + 2) % NBS*W2 + NBoarder + w)*halfNdir1 + halfNdir
						};
						int* tmp_Lr[3] = {
							Lr + (h % NBS*W2 + w + NBoarder)*halfNdir1*D2 + 1,
							Lr + ((h + 1) % NBS*W2 + w + NBoarder)*halfNdir1*D2 + 1,
							Lr + ((h + 2) % NBS*W2 + w + NBoarder)*halfNdir1*D2 + 1
						};
						int** cur_minLr = &(tmp_minLr[2]);
						int** cur_Lr = &(tmp_Lr[2]);

						for (int dd = 0; dd < 1; dd++)
						{
							int last_px = +dir_x[dd];
							int	last_py = -dir_y[dd];
							int* minLr_p_r = cur_minLr[last_py] + last_px*halfNdir1 + halfNdir;
							int* Lr_p_r = cur_Lr[last_py] + last_px*halfNdir1*D2;
							Lr_p_r[-1] = Lr_p_r[D] = MAX_COST;
							int* Lr_p = cur_Lr[0] + (dd + halfNdir)*D2;
							int cur_min = MAX_COST;
							for (int d = 0; d < D; d++)
							{
								//Lr(p, d) = C(p, d) + min(Lr(p - r, d), Lr(p - r, d - 1) + P1, Lr(p - r, d + 1) + P1, minLr(p - r) + P2) - minLr(p-r);
								int Lr_p_d = __min(Lr_p_r[d], __min(Lr_p_r[d - 1] + P1, __min(Lr_p_r[d + 1] + P1, minLr_p_r[dd] + P2))) - minLr_p_r[dd];
								Lr_p_d += pass == 0 ? cost_data[w*D + d] : cost_data[(w + d)*D + d];
								cur_min = __min(cur_min, Lr_p_d);
								Lr_p[d] = Lr_p_d;
								sum_Lr_data[w*D + d] += Lr_p_d;
							}
							cur_minLr[0][dd] = cur_min;
						}
					}
					/*compute sumLr for p =(h,w) End*/

					for (int w = 0; w < width; w++)
					{
						int min_d = -1;
						int min_sum = MAX_COST;
						for (int d = 0; d < D; d++)
						{
							int tmp_sum = sum_Lr_data[w*D + d];
							if (tmp_sum < min_sum)
							{
								min_sum = tmp_sum;
								min_d = d;
							}
						}
						/*check uniqueness begin*/
						int min_d2;
						for (min_d2 = 0; min_d2 < D; min_d2++)
						{
							if (sum_Lr_data[w*D + min_d2]  < min_sum*uniqueness && std::abs(min_d - min_d2) > 1)
								break;
						}
						if (min_d2 < D)
						{
							disparity_data[h*width + w] = -1;
							continue;
						}
						/*check uniqueness end*/

						/*interpolation for sub-pixel precision begin*/
						int d = min_d;
						float final_d;
						if (0 < d && d < D - 1)
						{
							// do subpixel quadratic interpolation:
							//   fit parabola into (x1=d-1, y1=Sp[d-1]), (x2=d, y2=Sp[d]), (x3=d+1, y3=Sp[d+1])
							//   then find minimum of the parabola.
							int* sum_p = sum_Lr_data + w*D;
							float denom2 = sum_p[d - 1] + sum_p[d + 1] - 2 * sum_p[d];
							if (denom2 == 0)
								final_d = d;
							else
								final_d = (float)d + (sum_p[d - 1] - sum_p[d + 1]) / (denom2 * 2);
						}
						else
							final_d = d;
						disparity_data[h*width + w] = final_d;
						/*interpolation for sub-pixel precision end*/
					}// loop for w

				}//loop for pass
				/*****************************       compute Lr and find best d         Begin      ************************/
				/*****************************       pass == 0 for left,  pass == 1 for right      ************************/

			}// loop for h

			return true;
		}

		static bool _computeDisparitySGBM_fullDP_OOC(const ZQ_DImage<unsigned char>& img1, const ZQ_DImage<unsigned char>& img2,
			ZQ_DImage<float>& left_disparity, ZQ_DImage<float>& right_disparity, int maxD, int P1, int P2,
			int half_sad_win_size_W = 3, int half_sad_win_size_H = 3, float uniqueness = 1.15f, int truncate_value = 64, int Ndir = 8)
		{
			if (Ndir != 8 && Ndir != 16)
				return false;

			int width1 = img1.width();
			int height = img1.height();
			int nChannels = img1.nchannels();
			int width2 = img2.width();
			if (!img2.matchDimension(width2, height, nChannels))
				return false;
			if (maxD <= 0)
				return false;

			if (!left_disparity.matchDimension(width1, height, 1))
				left_disparity.allocate(width1, height, 1);
			if (!right_disparity.matchDimension(width2, height, 1))
				right_disparity.allocate(width2, height, 1);

			const int MAX_COST = INT_MAX - P2;

			float*& left_disparity_data = left_disparity.data();
			float*& right_disparity_data = right_disparity.data();

			int winH = 2 * half_sad_win_size_H + 1;
			int winW = 2 * half_sad_win_size_W + 1;
			int D = maxD + 1;
			int W = __max(width1, width2 + maxD);

			const unsigned char*& img1_data = img1.data();
			const unsigned char*& img2_data = img2.data();
			
			
			int NROWS = winH + 1;
			ZQ_DImage<short> pixDiff(W, 1, D);
			ZQ_DImage<int> row_sum(W, NROWS, D);
			ZQ_DImage<int> cost(W, 1, D);
			short*& pixDiff_data = pixDiff.data();
			int*& row_sum_data = row_sum.data();
			int*& cost_data = cost.data();

			////////
			int NBoarder = Ndir == 8 ? 1 : 2;
			int WL2 = width1 + 2 * NBoarder;
			int WR2 = width2 + 2 * NBoarder;
			int D2 = D + 2;
			int H2 = height + 2 * NBoarder;

			const int dir_x[8] = { 1, 1, 0, -1, 2, 1, -1, -2 };
			const int dir_y[8] = { 0, 1, 1, 1, 1, 2, 2, 1 };

			int halfNdir = Ndir / 2;
			int NBS = NBoarder + 1;

			ZQ_DImage<int> left_Lr(WL2, NBS, D2*halfNdir), right_Lr(WR2, NBS, D2*halfNdir);
			ZQ_DImage<int> left_minLr(WL2, NBS, halfNdir), right_minLr(WR2, NBS, halfNdir);

			int*& left_Lr_data = left_Lr.data();
			int*& left_minLr_data = left_minLr.data();
			int*& right_Lr_data = right_Lr.data();
			int*& right_minLr_data = right_minLr.data();
			/***********************/

			const char* file_cost = "fulldp_ooc_1234567890_cost.tmp";
			const char* file_sumL_left = "fulldp_ooc_1234567890_sumL_left.tmp";
			const char* file_sumL_right = "fulldp_ooc_1234567890_sumL_right.tmp";
			printf("warning: need %I64d bytes for file \"%s\"\n", (int64_t)height*W*D*sizeof(int), file_cost);
			printf("         need %I64d bytes for file \"%s\"\n", (int64_t)height*width1*D*sizeof(int), file_sumL_left);
			printf("         need %I64d bytes for file \"%s\"\n", (int64_t)height*width2*D*sizeof(int), file_sumL_right);
			FILE* fid_cost = fopen(file_cost, "wb+");
			if (fid_cost == 0)
			{
				return false;
			}
			
			FILE* fid_sumL_left = fopen(file_sumL_left, "wb+");
			if (fid_sumL_left == 0)
			{
				fclose(fid_cost);
				return false;
			}

			FILE* fid_sumL_right = fopen(file_sumL_right, "wb+");
			if (fid_sumL_right == 0)
			{
				fclose(fid_cost);
				return false;
			}

			/**************************************************************************/
			for (int h = 0; h < height; h++)
			{
				/*compute cost of current row Begin*/
				if (h == 0)
				{
					for (int hh = 0; hh <= half_sad_win_size_H; hh++)
					{
						_computeMatchingCostBT_OneRow(img1_data + hh*width1*nChannels, img2_data + hh*width2*nChannels, width1, width2, nChannels, maxD, pixDiff_data, truncate_value);
						_cumulate_row(pixDiff_data, W, D, half_sad_win_size_W, row_sum_data + (hh + half_sad_win_size_H)*W*D);
						if (hh == 0)
						{
							for (int i = 0; i < half_sad_win_size_H; i++)
								memcpy(row_sum_data + i*W*D, row_sum_data + (hh + half_sad_win_size_H)*W*D, sizeof(int)*W*D);
						}
					}

					for (int d = 0; d < D; d++)
					{
						for (int w = 0; w < W; w++)
						{
							cost_data[w*D + d] = 0;
							for (int hh = 0; hh < winH; hh++)
								cost_data[w*D + d] += row_sum_data[hh*W*D + w*D + d];
						}
					}
				}
				else
				{
					int row_idx = h + half_sad_win_size_H;
					int last_row_idx = row_idx - 1;

					int add_store_kk = (row_idx + half_sad_win_size_H) % NROWS;
					int last_store_kk = (last_row_idx + half_sad_win_size_H) % NROWS;
					int sub_store_kk = (row_idx - half_sad_win_size_H - 1) % NROWS;
					int* add_row_data = row_sum_data + add_store_kk*W*D;
					int* last_row_data = row_sum_data + last_store_kk*W*D;
					int* sub_row_data = row_sum_data + sub_store_kk*W*D;

					if (row_idx >= height)
						memcpy(add_row_data, last_row_data, sizeof(int)*W*D);
					else
					{
						_computeMatchingCostBT_OneRow(img1_data + row_idx*width1*nChannels, img2_data + row_idx*width2*nChannels, width1, width2, nChannels, maxD, pixDiff_data, truncate_value);
						_cumulate_row(pixDiff_data, W, D, half_sad_win_size_W, add_row_data);
					}

					for (int d = 0; d < D; d++)
					{
						for (int w = 0; w < W; w++)
						{
							cost_data[w*D + d] += -sub_row_data[w*D + d] + add_row_data[w*D + d];
						}
					}
				}
				/*compute cost of current row End*/

				if (W*D != fwrite(cost_data, sizeof(int), W*D, fid_cost))
				{
					fclose(fid_cost);
					fclose(fid_sumL_left);
					fclose(fid_sumL_right);
					return false;
				}
			
				/*****************************       compute Lr and find best d         Begin      ************************/
				/*****************************       pass == 0 for left,  pass == 1 for right      ************************/
				for (int pass = 0; pass < 2; pass++)
				{
					int* minLr = pass == 0 ? left_minLr_data : right_minLr_data;
					int* Lr = pass == 0 ? left_Lr_data : right_Lr_data;
					int W2 = pass == 0 ? WL2 : WR2;
					int width = pass == 0 ? width1 : width2;
					FILE* fid = pass == 0 ? fid_sumL_left : fid_sumL_right;

					/*compute sumLr for p =(h,w) Begin*/
					ZQ_DImage<int> sum_Lr(width, 1, D);
					int*& sum_Lr_data = sum_Lr.data();

					for (int w = 0; w < width; w++)
					{
						int* tmp_minLr[3] = {
							minLr + (h % NBS*W2 + NBoarder + w)*halfNdir,
							minLr + ((h + 1) % NBS*W2 + NBoarder + w)*halfNdir,
							minLr + ((h + 2) % NBS*W2 + NBoarder + w)*halfNdir
						};
						int* tmp_Lr[3] = {
							Lr + (h % NBS*W2 + w + NBoarder)*halfNdir*D2 + 1,
							Lr + ((h + 1) % NBS*W2 + w + NBoarder)*halfNdir*D2 + 1,
							Lr + ((h + 2) % NBS*W2 + w + NBoarder)*halfNdir*D2 + 1
						};
						int** cur_minLr = &(tmp_minLr[2]);
						int** cur_Lr = &(tmp_Lr[2]);

						for (int dd = 0; dd < halfNdir; dd++)
						{
							int last_px = -dir_x[dd];
							int	last_py = -dir_y[dd];
							int* minLr_p_r = cur_minLr[last_py] + last_px*halfNdir;
							int* Lr_p_r = cur_Lr[last_py] + last_px*halfNdir*D2 + dd*D2;
							Lr_p_r[-1] = Lr_p_r[D] = MAX_COST;
							int* Lr_p = cur_Lr[0] + dd*D2;
							int cur_min = MAX_COST;
							for (int d = 0; d < D; d++)
							{
								//Lr(p, d) = C(p, d) + min(Lr(p - r, d), Lr(p - r, d - 1) + P1, Lr(p - r, d + 1) + P1, minLr(p - r) + P2) - minLr(p-r);
								int Lr_p_d = __min(Lr_p_r[d], __min(Lr_p_r[d - 1] + P1, __min(Lr_p_r[d + 1] + P1, minLr_p_r[dd] + P2))) - minLr_p_r[dd];
								Lr_p_d += pass == 0 ? cost_data[w*D + d] : cost_data[(w + d)*D + d];
								cur_min = __min(cur_min, Lr_p_d);
								Lr_p[d] = Lr_p_d;
								sum_Lr_data[w*D + d] += Lr_p_d;
							}
							cur_minLr[0][dd] = cur_min;
						}
					}
					/*compute sumLr for p =(h,w) End*/
					
					if (width*D != fwrite(sum_Lr_data, sizeof(int), width*D, fid))
					{
						fclose(fid_cost);
						fclose(fid_sumL_left);
						fclose(fid_sumL_right);
						return false;
					}

				}//loop for pass
				/*****************************       compute Lr and find best d         Begin      ************************/
				/*****************************       pass == 0 for left,  pass == 1 for right      ************************/
			}// loop for h

			/***************************************************************************************/
			/***************************************************************************************/
			left_Lr.reset(); right_Lr.reset();
			left_minLr.reset(); right_minLr.reset();
			for (int h = height - 1; h >= 0; h--)
			{
				_fseeki64(fid_cost, (int64_t)h*W*D*sizeof(int), SEEK_SET);
				//fseek(fid_cost, h*W*D*sizeof(int), SEEK_SET);
				if (W*D != fread(cost_data, sizeof(int), W*D, fid_cost))
				{
					fclose(fid_cost);
					fclose(fid_sumL_left);
					fclose(fid_sumL_right);
					return false;
				}

				/*****************************       compute Lr and find best d         Begin      ************************/
				/*****************************       pass == 0 for left,  pass == 1 for right      ************************/
				for (int pass = 0; pass < 2; pass++)
				{
					float* disparity_data = pass == 0 ? left_disparity_data : right_disparity_data;
					int* minLr = pass == 0 ? left_minLr_data : right_minLr_data;
					int* Lr = pass == 0 ? left_Lr_data : right_Lr_data;
					int W2 = pass == 0 ? WL2 : WR2;
					int width = pass == 0 ? width1 : width2;
					FILE* fid = pass == 0 ? fid_sumL_left : fid_sumL_right;

					/*compute sumLr for p =(h,w) Begin*/
					ZQ_DImage<int> sum_Lr(width, 1, D);
					int*& sum_Lr_data = sum_Lr.data();
					_fseeki64(fid, (int64_t)h*width*D*sizeof(int), SEEK_SET);
					//fseek(fid, h*width*D*sizeof(int), SEEK_SET);
					if (width*D != fread(sum_Lr_data, sizeof(int), width*D, fid))
					{
						fclose(fid_cost);
						fclose(fid_sumL_left);
						fclose(fid_sumL_right);
						return false;
					}

					for (int w = width - 1; w >= 0; w--)
					{
						int* tmp_minLr[3] = {
							minLr + ((h + 2) % NBS*W2 + NBoarder + w)*halfNdir,
							minLr + ((h + 3) % NBS*W2 + NBoarder + w)*halfNdir,
							minLr + ((h + 4) % NBS*W2 + NBoarder + w)*halfNdir
						};
						int* tmp_Lr[3] = {
							Lr + ((h + 2) % NBS*W2 + w + NBoarder)*halfNdir*D2 + 1,
							Lr + ((h + 3) % NBS*W2 + w + NBoarder)*halfNdir*D2 + 1,
							Lr + ((h + 4) % NBS*W2 + w + NBoarder)*halfNdir*D2 + 1
						};
						int** cur_minLr = &(tmp_minLr[0]);
						int** cur_Lr = &(tmp_Lr[0]);

						for (int dd = 0; dd < halfNdir; dd++)
						{
							int last_px = +dir_x[dd];
							int	last_py = +dir_y[dd];
							int* minLr_p_r = cur_minLr[last_py] + last_px*halfNdir;
							int* Lr_p_r = cur_Lr[last_py] + last_px*halfNdir*D2 + dd*D2;
							Lr_p_r[-1] = Lr_p_r[D] = MAX_COST;
							int* Lr_p = cur_Lr[0] + dd*D2;
							int cur_min = MAX_COST;
							for (int d = 0; d < D; d++)
							{
								//Lr(p, d) = C(p, d) + min(Lr(p - r, d), Lr(p - r, d - 1) + P1, Lr(p - r, d + 1) + P1, minLr(p - r) + P2) - minLr(p-r);
								int Lr_p_d = __min(Lr_p_r[d], __min(Lr_p_r[d - 1] + P1, __min(Lr_p_r[d + 1] + P1, minLr_p_r[dd] + P2))) - minLr_p_r[dd];
								Lr_p_d += pass == 0 ? cost_data[w*D + d] : cost_data[(w + d)*D + d];
								cur_min = __min(cur_min, Lr_p_d);
								Lr_p[d] = Lr_p_d;
								sum_Lr_data[w*D + d] += Lr_p_d;
							}
							cur_minLr[0][dd] = cur_min;
						}
					}
					/*compute sumLr for p =(h,w) End*/

					for (int w = 0; w < width; w++)
					{
						int min_d = -1;
						int min_sum = MAX_COST;
						for (int d = 0; d < D; d++)
						{
							int tmp_sum = sum_Lr_data[w*D + d];
							if (tmp_sum < min_sum)
							{
								min_sum = tmp_sum;
								min_d = d;
							}
						}
						/*check uniqueness begin*/
						int min_d2;
						for (min_d2 = 0; min_d2 < D; min_d2++)
						{
							if (sum_Lr_data[w*D + min_d2]  < min_sum*uniqueness && std::abs(min_d - min_d2) > 1)
								break;
						}
						if (min_d2 < D)
						{
							disparity_data[h*width + w] = -1;
							continue;
						}
						/*check uniqueness end*/

						/*interpolation for sub-pixel precision begin*/
						int d = min_d;
						float final_d;
						if (0 < d && d < D - 1)
						{
							// do subpixel quadratic interpolation:
							//   fit parabola into (x1=d-1, y1=Sp[d-1]), (x2=d, y2=Sp[d]), (x3=d+1, y3=Sp[d+1])
							//   then find minimum of the parabola.
							int* sum_p = sum_Lr_data + w*D;
							float denom2 = sum_p[d - 1] + sum_p[d + 1] - 2 * sum_p[d];
							if (denom2 == 0)
								final_d = d;
							else
								final_d = (float)d + (sum_p[d - 1] - sum_p[d + 1]) / (denom2 * 2);
						}
						else
							final_d = d;
						disparity_data[h*width + w] = final_d;
						/*interpolation for sub-pixel precision end*/
					}// loop for w

				}// loop for pass

			}// loop for h

			fclose(fid_cost);
			fclose(fid_sumL_left);
			fclose(fid_sumL_right);

			remove(file_cost);
			remove(file_sumL_left);
			remove(file_sumL_right);
			return true;
		}

		static bool _computeDisparitySGBM_NCC_fullDP_OOC(const ZQ_DImage<unsigned char>& img1, const ZQ_DImage<unsigned char>& img2,
			ZQ_DImage<float>& left_disparity, ZQ_DImage<float>& right_disparity, int maxD, float P1 = 0.0006f, float P2 = 0.0024f,
			int half_win_size_W = 3, int half_win_size_H = 3, float uniqueness = 1.00f, int Ndir = 8)
		{
			if (Ndir != 8 && Ndir != 16)
				return false;

			int width1 = img1.width();
			int height = img1.height();
			int nChannels = img1.nchannels();
			int width2 = img2.width();
			if (!img2.matchDimension(width2, height, nChannels))
				return false;
			if (maxD <= 0)
				return false;

			if (!left_disparity.matchDimension(width1, height, 1))
				left_disparity.allocate(width1, height, 1);
			if (!right_disparity.matchDimension(width2, height, 1))
				right_disparity.allocate(width2, height, 1);

			const float MAX_COST = 1e16 - P2;

			float*& left_disparity_data = left_disparity.data();
			float*& right_disparity_data = right_disparity.data();

			int winH = 2 * half_win_size_H + 1;
			int winW = 2 * half_win_size_W + 1;
			int D = maxD + 1;
			int W = __max(width1, width2 + maxD);

			ZQ_DImage<float> denom1, denom2;
			_compute_denom_for_NCC(img1, img2, denom1, denom2, half_win_size_W, half_win_size_H, D);
			int denom_W1 = denom1.width();
			int denom_W2 = denom2.width();

			const unsigned char*& img1_data = img1.data();
			const unsigned char*& img2_data = img2.data();

			float*& denom1_data = denom1.data();
			float*& denom2_data = denom2.data();

			int NROWS = winH + 1;
			ZQ_DImage<float> cost(W, 1, D);
			ZQ_DImage<float> num(W, 1, D);
			ZQ_DImage<float> dotVal(W, 1, D);
			ZQ_DImage<float> row_sum(W, NROWS, D);
			float*& cost_data = cost.data();
			float*& num_data = num.data();
			float*& dotVal_data = dotVal.data();
			float*& row_sum_data = row_sum.data();

			////////
			int NBoarder = Ndir == 8 ? 1 : 2;
			int WL2 = width1 + 2 * NBoarder;
			int WR2 = width2 + 2 * NBoarder;
			int D2 = D + 2;
			int H2 = height + 2 * NBoarder;

			const int dir_x[8] = { 1, 1, 0, -1, 2, 1, -1, -2 };
			const int dir_y[8] = { 0, 1, 1, 1, 1, 2, 2, 1 };

			int halfNdir = Ndir / 2;
			int NBS = NBoarder + 1;

			ZQ_DImage<float> left_Lr(WL2, NBS, D2*halfNdir), right_Lr(WR2, NBS, D2*halfNdir);
			ZQ_DImage<float> left_minLr(WL2, NBS, halfNdir), right_minLr(WR2, NBS, halfNdir);

			float*& left_Lr_data = left_Lr.data();
			float*& left_minLr_data = left_minLr.data();
			float*& right_Lr_data = right_Lr.data();
			float*& right_minLr_data = right_minLr.data();
			/***********************/

			const char* file_cost = "fulldp_ooc_1234567890_cost.tmp";
			const char* file_sumL_left = "fulldp_ooc_1234567890_sumL_left.tmp";
			const char* file_sumL_right = "fulldp_ooc_1234567890_sumL_right.tmp";
			printf("warning: need %I64d bytes for file \"%s\"\n", (int64_t)height*W*D*sizeof(float), file_cost);
			printf("         need %I64d bytes for file \"%s\"\n", (int64_t)height*width1*D*sizeof(float), file_sumL_left);
			printf("         need %I64d bytes for file \"%s\"\n", (int64_t)height*width2*D*sizeof(float), file_sumL_right);
			FILE* fid_cost = fopen(file_cost, "wb+");
			if (fid_cost == 0)
			{
				return false;
			}

			FILE* fid_sumL_left = fopen(file_sumL_left, "wb+");
			if (fid_sumL_left == 0)
			{
				fclose(fid_cost);
				return false;
			}

			FILE* fid_sumL_right = fopen(file_sumL_right, "wb+");
			if (fid_sumL_right == 0)
			{
				fclose(fid_cost);
				return false;
			}

			/**************************************************************************/
			for (int h = 0; h < height; h++)
			{
				/*compute cost of current row Begin*/
				if (h == 0)
				{
					for (int hh = 0; hh <= half_win_size_H; hh++)
					{
						_computeMatchingCostNCC_Dot_OneRow(img1_data + hh*width1*nChannels, img2_data + hh*width2*nChannels, width1, width2, nChannels, maxD, dotVal_data);
						_cumulate_row(dotVal_data, W, D, half_win_size_W, row_sum_data + (hh + half_win_size_H)*W*D);
						if (hh == 0)
						{
							for (int i = 0; i < half_win_size_H; i++)
								memcpy(row_sum_data + i*W*D, row_sum_data + (hh + half_win_size_H)*W*D, sizeof(float)*W*D);
						}
					}

					for (int w = 0; w < W; w++)
					{
						for (int d = 0; d < D; d++)
						{
							num_data[w*D + d] = 0;
							for (int hh = 0; hh < winH; hh++)
								num_data[w*D + d] += row_sum_data[hh*W*D + w*D + d];
							float denom_val1 = denom1_data[h*denom_W1 + w];
							float denom_val2 = denom2_data[h*denom_W2 + (w + D - 1 - d)];
							if (denom_val1 <= 0 || denom_val2 <= 0)
								cost_data[w*D + d] = 0;
							else
								cost_data[w*D + d] = num_data[w*D + d] / sqrt((double)denom_val1*denom_val2);
						}
					}
				}
				else
				{
					int row_idx = h + half_win_size_H;
					int last_row_idx = row_idx - 1;

					int add_store_kk = (row_idx + half_win_size_H) % NROWS;
					int last_store_kk = (last_row_idx + half_win_size_H) % NROWS;
					int sub_store_kk = (row_idx - half_win_size_H - 1) % NROWS;
					float* add_row_data = row_sum_data + add_store_kk*W*D;
					float* last_row_data = row_sum_data + last_store_kk*W*D;
					float* sub_row_data = row_sum_data + sub_store_kk*W*D;

					if (row_idx >= height)
						memcpy(add_row_data, last_row_data, sizeof(float)*W*D);
					else
					{
						_computeMatchingCostNCC_Dot_OneRow(img1_data + row_idx*width1*nChannels, img2_data + row_idx*width2*nChannels, width1, width2, nChannels, maxD, dotVal_data);
						_cumulate_row(dotVal_data, W, D, half_win_size_W, add_row_data);
					}

					for (int w = 0; w < W; w++)
					{
						for (int d = 0; d < D; d++)
						{
							num_data[w*D + d] += -sub_row_data[w*D + d] + add_row_data[w*D + d];
							float denom_val1 = denom1_data[h*denom_W1 + w];
							float denom_val2 = denom2_data[h*denom_W2 + (w + D - 1 - d)];
							if (denom_val1 <= 0 || denom_val2 <= 0)
								cost_data[w*D + d] = 0;
							else
								cost_data[w*D + d] = num_data[w*D + d] / sqrt((double)denom_val1*denom_val2);
						}
					}
				}
				/*compute cost of current row End*/
				cost.Multiplywith(-1);
				cost.Addwith(1);
				if (W*D != fwrite(cost_data, sizeof(float), W*D, fid_cost))
				{
					fclose(fid_cost);
					fclose(fid_sumL_left);
					fclose(fid_sumL_right);
					return false;
				}

				/*****************************       compute Lr and find best d         Begin      ************************/
				/*****************************       pass == 0 for left,  pass == 1 for right      ************************/
				for (int pass = 0; pass < 2; pass++)
				{
					float* minLr = pass == 0 ? left_minLr_data : right_minLr_data;
					float* Lr = pass == 0 ? left_Lr_data : right_Lr_data;
					int W2 = pass == 0 ? WL2 : WR2;
					int width = pass == 0 ? width1 : width2;
					FILE* fid = pass == 0 ? fid_sumL_left : fid_sumL_right;

					/*compute sumLr for p =(h,w) Begin*/
					ZQ_DImage<float> sum_Lr(width, 1, D);
					float*& sum_Lr_data = sum_Lr.data();

					for (int w = 0; w < width; w++)
					{
						float* tmp_minLr[3] = {
							minLr + (h % NBS*W2 + NBoarder + w)*halfNdir,
							minLr + ((h + 1) % NBS*W2 + NBoarder + w)*halfNdir,
							minLr + ((h + 2) % NBS*W2 + NBoarder + w)*halfNdir
						};
						float* tmp_Lr[3] = {
							Lr + (h % NBS*W2 + w + NBoarder)*halfNdir*D2 + 1,
							Lr + ((h + 1) % NBS*W2 + w + NBoarder)*halfNdir*D2 + 1,
							Lr + ((h + 2) % NBS*W2 + w + NBoarder)*halfNdir*D2 + 1
						};
						float** cur_minLr = &(tmp_minLr[2]);
						float** cur_Lr = &(tmp_Lr[2]);

						for (int dd = 0; dd < halfNdir; dd++)
						{
							int last_px = -dir_x[dd];
							int	last_py = -dir_y[dd];
							float* minLr_p_r = cur_minLr[last_py] + last_px*halfNdir;
							float* Lr_p_r = cur_Lr[last_py] + last_px*halfNdir*D2 + dd*D2;
							Lr_p_r[-1] = Lr_p_r[D] = MAX_COST;
							float* Lr_p = cur_Lr[0] + dd*D2;
							float cur_min = MAX_COST;
							for (int d = 0; d < D; d++)
							{
								//Lr(p, d) = C(p, d) + min(Lr(p - r, d), Lr(p - r, d - 1) + P1, Lr(p - r, d + 1) + P1, minLr(p - r) + P2) - minLr(p-r);
								float Lr_p_d = __min(Lr_p_r[d], __min(Lr_p_r[d - 1] + P1, __min(Lr_p_r[d + 1] + P1, minLr_p_r[dd] + P2))) - minLr_p_r[dd];
								Lr_p_d += pass == 0 ? cost_data[w*D + d] : cost_data[(w + d)*D + d];
								cur_min = __min(cur_min, Lr_p_d);
								Lr_p[d] = Lr_p_d;
								sum_Lr_data[w*D + d] += Lr_p_d;
							}
							cur_minLr[0][dd] = cur_min;
						}
					}
					/*compute sumLr for p =(h,w) End*/

					if (width*D != fwrite(sum_Lr_data, sizeof(float), width*D, fid))
					{
						fclose(fid_cost);
						fclose(fid_sumL_left);
						fclose(fid_sumL_right);
						return false;
					}

				}//loop for pass
				/*****************************       compute Lr and find best d         Begin      ************************/
				/*****************************       pass == 0 for left,  pass == 1 for right      ************************/
			}// loop for h

			/***************************************************************************************/
			/***************************************************************************************/
			left_Lr.reset(); right_Lr.reset();
			left_minLr.reset(); right_minLr.reset();
			for (int h = height - 1; h >= 0; h--)
			{
				_fseeki64(fid_cost, (int64_t)h*W*D*sizeof(float), SEEK_SET);
				//fseek(fid_cost, h*W*D*sizeof(float), SEEK_SET);
				if (W*D != fread(cost_data, sizeof(float), W*D, fid_cost))
				{
					fclose(fid_cost);
					fclose(fid_sumL_left);
					fclose(fid_sumL_right);
					return false;
				}

				/*****************************       compute Lr and find best d         Begin      ************************/
				/*****************************       pass == 0 for left,  pass == 1 for right      ************************/
				for (int pass = 0; pass < 2; pass++)
				{
					float* disparity_data = pass == 0 ? left_disparity_data : right_disparity_data;
					float* minLr = pass == 0 ? left_minLr_data : right_minLr_data;
					float* Lr = pass == 0 ? left_Lr_data : right_Lr_data;
					int W2 = pass == 0 ? WL2 : WR2;
					int width = pass == 0 ? width1 : width2;
					FILE* fid = pass == 0 ? fid_sumL_left : fid_sumL_right;

					/*compute sumLr for p =(h,w) Begin*/
					ZQ_DImage<float> sum_Lr(width, 1, D);
					float*& sum_Lr_data = sum_Lr.data();
					_fseeki64(fid, (int64_t)h*width*D*sizeof(float), SEEK_SET);
					//fseek(fid, h*width*D*sizeof(float), SEEK_SET);
					if (width*D != fread(sum_Lr_data, sizeof(float), width*D, fid))
					{
						fclose(fid_cost);
						fclose(fid_sumL_left);
						fclose(fid_sumL_right);
						return false;
					}

					for (int w = width - 1; w >= 0; w--)
					{
						float* tmp_minLr[3] = {
							minLr + ((h + 2) % NBS*W2 + NBoarder + w)*halfNdir,
							minLr + ((h + 3) % NBS*W2 + NBoarder + w)*halfNdir,
							minLr + ((h + 4) % NBS*W2 + NBoarder + w)*halfNdir
						};
						float* tmp_Lr[3] = {
							Lr + ((h + 2) % NBS*W2 + w + NBoarder)*halfNdir*D2 + 1,
							Lr + ((h + 3) % NBS*W2 + w + NBoarder)*halfNdir*D2 + 1,
							Lr + ((h + 4) % NBS*W2 + w + NBoarder)*halfNdir*D2 + 1
						};
						float** cur_minLr = &(tmp_minLr[0]);
						float** cur_Lr = &(tmp_Lr[0]);

						for (int dd = 0; dd < halfNdir; dd++)
						{
							int last_px = +dir_x[dd];
							int	last_py = +dir_y[dd];
							float* minLr_p_r = cur_minLr[last_py] + last_px*halfNdir;
							float* Lr_p_r = cur_Lr[last_py] + last_px*halfNdir*D2 + dd*D2;
							Lr_p_r[-1] = Lr_p_r[D] = MAX_COST;
							float* Lr_p = cur_Lr[0] + dd*D2;
							float cur_min = MAX_COST;
							for (int d = 0; d < D; d++)
							{
								//Lr(p, d) = C(p, d) + min(Lr(p - r, d), Lr(p - r, d - 1) + P1, Lr(p - r, d + 1) + P1, minLr(p - r) + P2) - minLr(p-r);
								float Lr_p_d = __min(Lr_p_r[d], __min(Lr_p_r[d - 1] + P1, __min(Lr_p_r[d + 1] + P1, minLr_p_r[dd] + P2))) - minLr_p_r[dd];
								Lr_p_d += pass == 0 ? cost_data[w*D + d] : cost_data[(w + d)*D + d];
								cur_min = __min(cur_min, Lr_p_d);
								Lr_p[d] = Lr_p_d;
								sum_Lr_data[w*D + d] += Lr_p_d;
							}
							cur_minLr[0][dd] = cur_min;
						}
					}
					/*compute sumLr for p =(h,w) End*/

					for (int w = 0; w < width; w++)
					{
						int min_d = -1;
						float min_sum = MAX_COST;
						for (int d = 0; d < D; d++)
						{
							float tmp_sum = sum_Lr_data[w*D + d];
							if (tmp_sum < min_sum)
							{
								min_sum = tmp_sum;
								min_d = d;
							}
						}
						/*check uniqueness begin*/
						int min_d2;
						for (min_d2 = 0; min_d2 < D; min_d2++)
						{
							if (sum_Lr_data[w*D + min_d2] < min_sum*uniqueness && std::abs(min_d - min_d2) > 1)
								break;
						}
						if (min_d2 < D)
						{
							disparity_data[h*width + w] = -1;
							continue;
						}
						/*check uniqueness end*/

						/*interpolation for sub-pixel precision begin*/
						int d = min_d;
						float final_d;
						if (0 < d && d < D - 1)
						{
							// do subpixel quadratic interpolation:
							//   fit parabola into (x1=d-1, y1=Sp[d-1]), (x2=d, y2=Sp[d]), (x3=d+1, y3=Sp[d+1])
							//   then find minimum of the parabola.
							float* sum_p = cost_data/*sum_Lr_data*/ + w*D;
							float denom2 = sum_p[d - 1] + sum_p[d + 1] - 2 * sum_p[d];
							if (denom2 == 0)
								final_d = d;
							else
								final_d = (float)d + (sum_p[d - 1] - sum_p[d + 1]) / (denom2 * 2);
						}
						else
							final_d = d;
						disparity_data[h*width + w] = final_d;
						/*interpolation for sub-pixel precision end*/
					}// loop for w

				}// loop for pass

			}// loop for h

			fclose(fid_cost);
			fclose(fid_sumL_left);
			fclose(fid_sumL_right);

			remove(file_cost);
			remove(file_sumL_left);
			remove(file_sumL_right);
			return true;
		}
	};
}

#endif