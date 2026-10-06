#include "layers_nchwc/zq_cnn_convolution_gemm_nchwc.h"
#include "layers_nchwc/zq_cnn_depthwise_convolution_nchwc.h"
#include "layers_nchwc/zq_cnn_innerproduct_gemm_nchwc.h"
#include "layers_nchwc/zq_cnn_addbias_nchwc.h"
#include "layers_nchwc/zq_cnn_softmax_nchwc.h"
#include "layers_nchwc/zq_cnn_pooling_nchwc.h"
#include "layers_nchwc/zq_cnn_prelu_nchwc.h"
#include "layers_nchwc/zq_cnn_relu_nchwc.h"
#include "layers_nchwc/zq_cnn_batchnormscale_nchwc.h"
#include "layers_nchwc/zq_cnn_eltwise_nchwc.h"
#include "ZQ_CNN_Forward_SSEUtils_NCHWC.h"
#include "ZQ_CNN_BBoxUtils.h"
#include <algorithm>
#include <math.h>
#include "ZQ_CNN_CompileConfig.h"
using namespace ZQ;

// 审计修复 2026-10-06（附录 F2）：池化前的 padding 准备。
// 六份 MaxPooling/AVGPooling 共用这一段 —— 同一段索引算术抄六遍，
// 抄错一份就又是一次**静默算错**，而且只有三行 pad 时才会暴露。
//
// NCHWC 的 `Padding(padW, padH, mode)` **只支持对称补**，而
// SAME/VALID 算出来的 pad 经常不对称（in=7,s=2,k=3 -> pad=(0,1)）。
// 所以这里补 `P = max(两侧)`，再把窗口起点往上/左挪到 `-pad_top/-pad_left`：
//
//   in_data = 数据首行 - pad_H_top*widthStep - pad_W_left*align
//   in_H    = 补齐区的行数 = in_H0 + pad_H_top + pad_H_bottom
//   in_W    = 补齐区的列数 = in_W0 + pad_W_left + pad_W_right
//
// 注意 in_H/in_W 取的是**意图上的补齐区**，不是物理可用的行数
// （后者是 in_H0 + P_H + pad_H_top，因为对称补法会在另一侧也多补 P_H 行）。
// 两者只在 pad_bottom > P_H 时才会不同，而 P_H = max(两侧) >= pad_bottom，
// 所以 `in_H0 + pt + pb <= in_H0 + P_H + pt` **恒成立**，物理上一定读得到。
// 取意图值而不是物理值，是为了让 suredivided 的判据
// （最后一个窗口放得下 = (need_H-1)*s + k <= in_H）说的是**语义**，
// 而不是"因为多补了几行所以碰巧放得下"。
// —— 第一版取物理值，pad_top > pad_bottom 的一档（pad=1,0 与 pad=2,1）就判错了，
// 12 个 AVG 用例变红。
//
// 为什么"多补的那几行永远读不到"（这是对称补法安全的关键）：
// 最后一个窗口要碰到的最大行号是 (need_H-1)*stride_H + kernel_H，
// 而 need_H 由 ceil((in_H0 + pad_top + pad_bottom - k)/s) + 1 给出，
// 于是 (need_H-1)*s + k <= in_H0 + pad_top + pad_bottom
//                      <= in_H0 + pad_top + P_H = in_H。 列方向同理。
// 多补出来的行在**分配之外**不可达，越界不会发生。
template <class T>
static bool zq_nchwc_pool_prepare_pad(T& input,
                                      int pad_H_top, int pad_H_bottom,
                                      int pad_W_left, int pad_W_right,
                                      const float*& in_data, int& in_H, int& in_W)
{
    const int in_H0 = input.GetH();
    const int in_W0 = input.GetW();
    const int align = input.GetAlignSize();
    const int P_H = pad_H_top > pad_H_bottom ? pad_H_top : pad_H_bottom;
    const int P_W = pad_W_left > pad_W_right ? pad_W_left : pad_W_right;
    if (P_H > 0 || P_W > 0)
    {
        // 非零 pad 时**就地**把输入补成带边框的张量 —— NCHW 那一支
        // （`ZQ_CNN_Forward_SSEUtils::MaxPooling`）就是这么做的。
        if (!input.Padding(P_W, P_H, 0))
            return false;
    }
    in_H = in_H0 + pad_H_top + pad_H_bottom;
    in_W = in_W0 + pad_W_left + pad_W_right;
    // Padding 可能重新分配，widthStep 会变 —— 必须在**之后**再读。
    in_data = input.GetFirstPixelPtr()
              - (long long)pad_H_top * input.GetWidthStep()
              - (long long)pad_W_left * align;
    return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::InnerProductWithBias(ZQ_CNN_Tensor4D_NCHWC1& input, const ZQ_CNN_Tensor4D_NCHWC1& filters,
	const ZQ_CNN_Tensor4D_NCHWC1& bias, ZQ_CNN_Tensor4D_NCHWC1& output, 
	void** buffer, __int64* buffer_len)
{
	double t1 = omp_get_wtime();
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	int filter_N = filters.GetN();
	int filter_H = filters.GetH();
	int filter_W = filters.GetW();
	int filter_C = filters.GetC();
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	int bias_C = bias.GetC();
	int need_H = 1;
	int need_W = 1;
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| need_H < 0 || need_W < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}
	if (filter_C != in_C || filter_N != bias_C)
		return false;

	int need_N = in_N;

	int need_C = filter_N;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int filter_sliceStep = filters.GetSliceStep();
	int filter_widthStep = filters.GetWidthStep();
	int filter_imStep = filters.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr();
	const float* filter_firstPixelData = filters.GetFirstPixelPtr();
	float* out_firstPixelData = output.GetFirstPixelPtr();
	const float* bias_firstPixelData = bias.GetFirstPixelPtr();

#if (ZQ_CNN_USE_BLAS_GEMM || ZQ_CNN_USE_MKL_GEMM || ZQ_CNN_USE_ZQ_GEMM)
	if (out_N >= 16 && filter_N >= 16)
	{
		zq_cnn_innerproduct_gemm_nchwc1_general_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
			filter_widthStep, filter_sliceStep, filter_imStep,
			out_firstPixelData, need_N, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, buffer, buffer_len);

	}
	else
#endif
	{
		if (in_W == in_widthStep && in_widthStep*in_H == in_sliceStep
			&& in_W == filter_widthStep && filter_widthStep*in_H == filter_sliceStep
			&& out_W == out_widthStep && out_widthStep * out_H == out_sliceStep)
		{
			zq_cnn_innerproduct_nchwc1_noborder_with_bias(in_firstPixelData, in_N, in_H*in_W*in_C, filter_firstPixelData, filter_N,
				// 审计修复 2026-10-02（附录 BN.1）：传 out_sliceStep 时，N>1 的结果会
				// 互相覆盖（out 是 [N,1,1,K]，sliceStep 与 imStep 差 K 倍）。
				out_firstPixelData, out_imStep, bias_firstPixelData);
		}
		else
		{
			zq_cnn_innerproduct_gemm_nchwc1_general_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep,
				out_firstPixelData, need_N, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, buffer, buffer_len);
		}
	}

	double t2 = omp_get_wtime();
	//printf("utils:conv: %.3f ms\n", (t2 - t1) * 1000);
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::InnerProductWithBiasPReLU(ZQ_CNN_Tensor4D_NCHWC1& input, const ZQ_CNN_Tensor4D_NCHWC1& filters,
	const ZQ_CNN_Tensor4D_NCHWC1& bias, const ZQ_CNN_Tensor4D_NCHWC1& slope, ZQ_CNN_Tensor4D_NCHWC1& output,
	void** buffer, __int64* buffer_len)
{
	double t1 = omp_get_wtime();
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	int filter_N = filters.GetN();
	int filter_H = filters.GetH();
	int filter_W = filters.GetW();
	int filter_C = filters.GetC();
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	int bias_C = bias.GetC();
	int slope_C = slope.GetC();
	int need_H = 1;
	int need_W = 1;
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| need_H < 0 || need_W < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}
	if (filter_C != in_C || filter_N != bias_C || filter_N != slope_C)
		return false;

	int need_N = in_N;

	int need_C = filter_N;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int filter_sliceStep = filters.GetSliceStep();
	int filter_widthStep = filters.GetWidthStep();
	int filter_imStep = filters.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr();
	const float* filter_firstPixelData = filters.GetFirstPixelPtr();
	float* out_firstPixelData = output.GetFirstPixelPtr();
	const float* bias_firstPixelData = bias.GetFirstPixelPtr();
	const float* slope_firstPixelData = slope.GetFirstPixelPtr();


#if (ZQ_CNN_USE_BLAS_GEMM || ZQ_CNN_USE_MKL_GEMM || ZQ_CNN_USE_ZQ_GEMM)
	if (out_N >= 16 && filter_N >= 16)
	{
		zq_cnn_innerproduct_gemm_nchwc1_general_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
			filter_widthStep, filter_sliceStep, filter_imStep,
			out_firstPixelData, need_N, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData, buffer, buffer_len);

	}
	else
#endif
	{
		if (in_W == in_widthStep && in_widthStep*in_H == in_sliceStep
			&& in_W == filter_widthStep && filter_widthStep*in_H == filter_sliceStep
			&& out_W == out_widthStep && out_widthStep * out_H == out_sliceStep)
		{
			zq_cnn_innerproduct_nchwc1_noborder_with_bias_prelu(in_firstPixelData, in_N, in_H*in_W*in_C, filter_firstPixelData, filter_N,
				// 审计修复 2026-10-02（附录 BN.1）：传 out_sliceStep 时，N>1 的结果会
				// 互相覆盖（out 是 [N,1,1,K]，sliceStep 与 imStep 差 K 倍）。
				out_firstPixelData, out_imStep, bias_firstPixelData, slope_firstPixelData);
		}
		else
		{
			zq_cnn_innerproduct_gemm_nchwc1_general_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep,
				out_firstPixelData, need_N, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData, buffer, buffer_len);
		}
	}

	double t2 = omp_get_wtime();
	//printf("utils:conv: %.3f ms\n", (t2 - t1) * 1000);
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::InnerProductWithPReLU(ZQ_CNN_Tensor4D_NCHWC1& input, const ZQ_CNN_Tensor4D_NCHWC1& filters,
	const ZQ_CNN_Tensor4D_NCHWC1& slope, ZQ_CNN_Tensor4D_NCHWC1& output, 
	void** buffer, __int64* buffer_len)
{
	double t1 = omp_get_wtime();
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	int filter_N = filters.GetN();
	int filter_H = filters.GetH();
	int filter_W = filters.GetW();
	int filter_C = filters.GetC();
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	int slope_C = slope.GetC();
	int need_H = 1;
	int need_W = 1;
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| need_H < 0 || need_W < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}
	if (filter_C != in_C || filter_N != slope_C)
		return false;

	int need_N = in_N;

	int need_C = filter_N;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int filter_sliceStep = filters.GetSliceStep();
	int filter_widthStep = filters.GetWidthStep();
	int filter_imStep = filters.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr();
	const float* filter_firstPixelData = filters.GetFirstPixelPtr();
	float* out_firstPixelData = output.GetFirstPixelPtr();
	const float* slope_firstPixelData = slope.GetFirstPixelPtr();


#if (ZQ_CNN_USE_BLAS_GEMM || ZQ_CNN_USE_MKL_GEMM || ZQ_CNN_USE_ZQ_GEMM)
	if (out_N >= 16 && filter_N >= 16)
	{
		zq_cnn_innerproduct_gemm_nchwc1_general(in_firstPixelData, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
			filter_widthStep, filter_sliceStep, filter_imStep,
			out_firstPixelData, need_N, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);

	}
	else
#endif
	{
		if (in_W == in_widthStep && in_widthStep*in_H == in_sliceStep
			&& in_W == filter_widthStep && filter_widthStep*in_H == filter_sliceStep
			&& out_W == out_widthStep && out_widthStep * out_H == out_sliceStep)
		{
			zq_cnn_innerproduct_nchwc1_noborder(in_firstPixelData, in_N, in_H*in_W*in_C, filter_firstPixelData, filter_N,
				// 审计修复 2026-10-02（附录 BN.1）：传 out_sliceStep 时，N>1 的结果会
				// 互相覆盖（out 是 [N,1,1,K]，sliceStep 与 imStep 差 K 倍）。
				out_firstPixelData, out_imStep);
		}
		else
		{
			zq_cnn_innerproduct_gemm_nchwc1_general(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep,
				out_firstPixelData, need_N, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
		}
	}
	zq_cnn_prelu_nchwc1(out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, slope_firstPixelData);

	double t2 = omp_get_wtime();
	//printf("utils:conv: %.3f ms\n", (t2 - t1) * 1000);
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::InnerProduct(ZQ_CNN_Tensor4D_NCHWC1& input, const ZQ_CNN_Tensor4D_NCHWC1& filters, 
	ZQ_CNN_Tensor4D_NCHWC1& output, void** buffer, __int64* buffer_len)
{
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	int filter_N = filters.GetN();
	int filter_H = filters.GetH();
	int filter_W = filters.GetW();
	int filter_C = filters.GetC();
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	int need_H = 1;
	int need_W = 1;
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| need_H < 0 || need_W < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}
	if (filter_C != in_C)
		return false;

	int need_N = in_N;
	int need_C = filter_N;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int filter_sliceStep = filters.GetSliceStep();
	int filter_widthStep = filters.GetWidthStep();
	int filter_imStep = filters.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr();
	const float* filter_firstPixelData = filters.GetFirstPixelPtr();
	float* out_firstPixelData = output.GetFirstPixelPtr();


#if (ZQ_CNN_USE_BLAS_GEMM || ZQ_CNN_USE_MKL_GEMM || ZQ_CNN_USE_ZQ_GEMM)
	if (out_N >= 16 && filter_N >= 16)
	{
		zq_cnn_innerproduct_gemm_nchwc1_general(in_firstPixelData, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
			filter_widthStep, filter_sliceStep, filter_imStep,
			out_firstPixelData, need_N, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);

	}
	else
#endif
	{
		if (in_W == in_widthStep && in_widthStep*in_H == in_sliceStep
			&& in_W == filter_widthStep && filter_widthStep*in_H == filter_sliceStep
			&& out_W == out_widthStep && out_widthStep * out_H == out_sliceStep)
		{
			zq_cnn_innerproduct_nchwc1_noborder(in_firstPixelData, in_N, in_H*in_W*in_C, filter_firstPixelData, filter_N,
				// 审计修复 2026-10-02（附录 BN.1）：传 out_sliceStep 时，N>1 的结果会
				// 互相覆盖（out 是 [N,1,1,K]，sliceStep 与 imStep 差 K 倍）。
				out_firstPixelData, out_imStep);
		}
		else
		{
			zq_cnn_innerproduct_gemm_nchwc1_general(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep,
				out_firstPixelData, need_N, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
		}
	}

	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::ConvolutionWithBias(ZQ_CNN_Tensor4D_NCHWC1& input, const ZQ_CNN_Tensor4D_NCHWC1& filters,
	const ZQ_CNN_Tensor4D_NCHWC1& bias, int strideH, int strideW, int dilation_H, int dilation_W, int padH, int padW,
	ZQ_CNN_Tensor4D_NCHWC1& output, void** buffer, __int64* buffer_len)
{
	double t1 = omp_get_wtime();
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	int filter_N = filters.GetN();
	int filter_H = filters.GetH();
	int filter_W = filters.GetW();
	int filter_C = filters.GetC();
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	int bias_C = bias.GetC();
	// 审计修复 2026-10-02（附录 BE.2 / BH）：strideH/strideW 来自**模型文件**
	// （不可信输入），下面这行是**整数除法** —— 除以 0 在 x86 上是 idiv，
	// 直接 SIGFPE、进程死。附录 BE 只修了 NCHW 那 7 处 wrapper，
	// NCHWC 这边是同一份代码复制过去的，同样要修。
	if (strideH <= 0 || strideW <= 0)
		return false;
	int need_H = (in_H - (filter_H - 1)*dilation_H - 1 + (padH << 1)) / strideH + 1;
	int need_W = (in_W - (filter_W - 1)*dilation_W - 1 + (padW << 1)) / strideW + 1;
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| need_H < 0 || need_W < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}
	if (filter_C != in_C || filter_N != bias_C)
		return false;

	int need_N = in_N;

	int need_C = filter_N;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	if (padH != 0 || padW != 0)
	{
		if (!input.Padding(padW, padH, 0))
			return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int filter_sliceStep = filters.GetSliceStep();
	int filter_widthStep = filters.GetWidthStep();
	int filter_imStep = filters.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr() - padH*in_widthStep - padW;
	const float* filter_firstPixelData = filters.GetFirstPixelPtr();
	float* out_firstPixelData = output.GetFirstPixelPtr();
	const float* bias_firstPixelData = bias.GetFirstPixelPtr();

	if (filter_H == 3 && filter_W == 3)
	{
		if (filter_C == 3)
		{
			zq_cnn_conv_no_padding_gemm_nchwc1_kernel3x3_C3_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, buffer, buffer_len);
		}
		else
		{
			zq_cnn_conv_no_padding_gemm_nchwc1_kernel3x3_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, buffer, buffer_len);
		}
	}
	else if (filter_H == 2 && filter_W == 2)
	{
		if (filter_C == 3)
		{
			zq_cnn_conv_no_padding_gemm_nchwc1_kernel2x2_C3_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, buffer, buffer_len);
		}
		else
		{
			zq_cnn_conv_no_padding_gemm_nchwc1_kernel2x2_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, buffer, buffer_len);
		}
	}
	else if (filter_H == 1 && filter_W == 1)
	{
		zq_cnn_conv_no_padding_gemm_nchwc1_kernel1x1_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
			filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
			out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, buffer, buffer_len);
	}
	else
	{
		zq_cnn_conv_no_padding_gemm_nchwc1_general_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
			filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
			out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, buffer, buffer_len);
	}
	
	double t2 = omp_get_wtime();
	//printf("utils:conv: %.3f ms\n", (t2 - t1) * 1000);
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::ConvolutionWithBiasPReLU(ZQ_CNN_Tensor4D_NCHWC1& input, const ZQ_CNN_Tensor4D_NCHWC1& filters,
	const ZQ_CNN_Tensor4D_NCHWC1& bias, const ZQ_CNN_Tensor4D_NCHWC1& slope, int strideH, int strideW,
	int dilation_H, int dilation_W, int padH, int padW, ZQ_CNN_Tensor4D_NCHWC1& output,
	void** buffer, __int64* buffer_len)
{
	double t1 = omp_get_wtime();
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	int filter_N = filters.GetN();
	int filter_H = filters.GetH();
	int filter_W = filters.GetW();
	int filter_C = filters.GetC();
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	int bias_C = bias.GetC();
	int slope_C = slope.GetC();
	// 审计修复 2026-10-02（附录 BE.2 / BH）：strideH/strideW 来自**模型文件**
	// （不可信输入），下面这行是**整数除法** —— 除以 0 在 x86 上是 idiv，
	// 直接 SIGFPE、进程死。附录 BE 只修了 NCHW 那 7 处 wrapper，
	// NCHWC 这边是同一份代码复制过去的，同样要修。
	if (strideH <= 0 || strideW <= 0)
		return false;
	int need_H = (in_H - (filter_H - 1)*dilation_H - 1 + (padH << 1)) / strideH + 1;
	int need_W = (in_W - (filter_W - 1)*dilation_W - 1 + (padW << 1)) / strideW + 1;
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| need_H < 0 || need_W < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}
	if (filter_C != in_C || filter_N != bias_C || filter_N != slope_C)
		return false;

	int need_N = in_N;

	int need_C = filter_N;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	if (padH != 0 || padW != 0)
	{
		if (!input.Padding(padW, padH, 0))
			return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int filter_sliceStep = filters.GetSliceStep();
	int filter_widthStep = filters.GetWidthStep();
	int filter_imStep = filters.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr() - padH*in_widthStep - padW;
	const float* filter_firstPixelData = filters.GetFirstPixelPtr();
	float* out_firstPixelData = output.GetFirstPixelPtr();
	const float* bias_firstPixelData = bias.GetFirstPixelPtr();
	const float* slope_firstPixelData = slope.GetFirstPixelPtr();

	if (filter_H == 3 && filter_W == 3)
	{
		if (filter_C == 3)
		{
			zq_cnn_conv_no_padding_gemm_nchwc1_kernel3x3_C3_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData, buffer, buffer_len);
		}
		else
		{
			zq_cnn_conv_no_padding_gemm_nchwc1_kernel3x3_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData, buffer, buffer_len);
		}
	}
	else if (filter_H == 2 && filter_W == 2)
	{
		if (filter_C == 3)
		{
			zq_cnn_conv_no_padding_gemm_nchwc1_kernel2x2_C3_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData, buffer, buffer_len);
		}
		else
		{
			zq_cnn_conv_no_padding_gemm_nchwc1_kernel2x2_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData, buffer, buffer_len);
		}
	}
	else if (filter_H == 1 && filter_W == 1)
	{
		zq_cnn_conv_no_padding_gemm_nchwc1_kernel1x1_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
			filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
			out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData, buffer, buffer_len);
	}
	else
	{
		zq_cnn_conv_no_padding_gemm_nchwc1_general_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
			filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
			out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData, buffer, buffer_len);
	}

	double t2 = omp_get_wtime();
	//printf("utils:conv: %.3f ms\n", (t2 - t1) * 1000);
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::ConvolutionWithPReLU(ZQ_CNN_Tensor4D_NCHWC1& input, const ZQ_CNN_Tensor4D_NCHWC1& filters,
	const ZQ_CNN_Tensor4D_NCHWC1& slope, int strideH, int strideW, int dilation_H, int dilation_W, int padH, int padW,
	ZQ_CNN_Tensor4D_NCHWC1& output, void** buffer, __int64* buffer_len)
{
	double t1 = omp_get_wtime();
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	int filter_N = filters.GetN();
	int filter_H = filters.GetH();
	int filter_W = filters.GetW();
	int filter_C = filters.GetC();
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	int slope_C = slope.GetC();
	// 审计修复 2026-10-02（附录 BE.2 / BH）：strideH/strideW 来自**模型文件**
	// （不可信输入），下面这行是**整数除法** —— 除以 0 在 x86 上是 idiv，
	// 直接 SIGFPE、进程死。附录 BE 只修了 NCHW 那 7 处 wrapper，
	// NCHWC 这边是同一份代码复制过去的，同样要修。
	if (strideH <= 0 || strideW <= 0)
		return false;
	int need_H = (in_H - (filter_H - 1)*dilation_H - 1 + (padH << 1)) / strideH + 1;
	int need_W = (in_W - (filter_W - 1)*dilation_W - 1 + (padW << 1)) / strideW + 1;
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| need_H < 0 || need_W < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}
	if (filter_C != in_C || filter_N != slope_C)
		return false;

	int need_N = in_N;

	int need_C = filter_N;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	if (padH != 0 || padW != 0)
	{
		if (!input.Padding(padW, padH, 0))
			return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int filter_sliceStep = filters.GetSliceStep();
	int filter_widthStep = filters.GetWidthStep();
	int filter_imStep = filters.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr() - padH*in_widthStep - padW;
	const float* filter_firstPixelData = filters.GetFirstPixelPtr();
	float* out_firstPixelData = output.GetFirstPixelPtr();
	const float* slope_firstPixelData = slope.GetFirstPixelPtr();

	
	if (filter_H == 3 && filter_W == 3)
	{
		if (filter_C == 3)
		{
			zq_cnn_conv_no_padding_gemm_nchwc1_kernel3x3_C3(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
		}
		else
		{
			zq_cnn_conv_no_padding_gemm_nchwc1_kernel3x3(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
		}
	}
	else if (filter_H == 2 && filter_W == 2)
	{
		if (filter_C == 3)
		{
			zq_cnn_conv_no_padding_gemm_nchwc1_kernel2x2_C3(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
		}
		else
		{
			zq_cnn_conv_no_padding_gemm_nchwc1_kernel2x2(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
		}
	}
	else if (filter_H == 1 && filter_W == 1)
	{
		zq_cnn_conv_no_padding_gemm_nchwc1_kernel1x1(in_firstPixelData, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
			filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
			out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
	}
	else
	{
		zq_cnn_conv_no_padding_gemm_nchwc1_general(in_firstPixelData, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
			filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
			out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
	}

	zq_cnn_prelu_nchwc1(out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, slope_firstPixelData);

	double t2 = omp_get_wtime();
	//printf("utils:conv: %.3f ms\n", (t2 - t1) * 1000);
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::Convolution(ZQ_CNN_Tensor4D_NCHWC1& input, const ZQ_CNN_Tensor4D_NCHWC1& filters, int strideH, int strideW,
	int dilation_H, int dilation_W, int padH, int padW,
	ZQ_CNN_Tensor4D_NCHWC1& output, void** buffer, __int64* buffer_len)
{
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	int filter_N = filters.GetN();
	int filter_H = filters.GetH();
	int filter_W = filters.GetW();
	int filter_C = filters.GetC();
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	// 审计修复 2026-10-02（附录 BE.2 / BH）：strideH/strideW 来自**模型文件**
	// （不可信输入），下面这行是**整数除法** —— 除以 0 在 x86 上是 idiv，
	// 直接 SIGFPE、进程死。附录 BE 只修了 NCHW 那 7 处 wrapper，
	// NCHWC 这边是同一份代码复制过去的，同样要修。
	if (strideH <= 0 || strideW <= 0)
		return false;
	int need_H = (in_H - (filter_H - 1)*dilation_H - 1 + (padH << 1)) / strideH + 1;
	int need_W = (in_W - (filter_W - 1)*dilation_W - 1 + (padW << 1)) / strideW + 1;
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| need_H < 0 || need_W < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}
	if (filter_C != in_C)
		return false;

	int need_N = in_N;
	int need_C = filter_N;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	if (padH != 0 || padW != 0)
	{
		if (!input.Padding(padW, padH, 0))
			return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int filter_sliceStep = filters.GetSliceStep();
	int filter_widthStep = filters.GetWidthStep();
	int filter_imStep = filters.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr() - padH*in_widthStep - padW;
	const float* filter_firstPixelData = filters.GetFirstPixelPtr();
	float* out_firstPixelData = output.GetFirstPixelPtr();

	if (filter_H == 3 && filter_W == 3)
	{
		if (filter_C == 3)
		{
			zq_cnn_conv_no_padding_gemm_nchwc1_kernel3x3_C3(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
		}
		else
		{
			zq_cnn_conv_no_padding_gemm_nchwc1_kernel3x3(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
		}
	}
	else if (filter_H == 2 && filter_W == 2)
	{
		if (filter_C == 3)
		{
			zq_cnn_conv_no_padding_gemm_nchwc1_kernel2x2_C3(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
		}
		else
		{
			zq_cnn_conv_no_padding_gemm_nchwc1_kernel2x2(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
		}
	}
	else if (filter_H == 1 && filter_W == 1)
	{
		zq_cnn_conv_no_padding_gemm_nchwc1_kernel1x1(in_firstPixelData, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
			filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
			out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
	}
	else
	{
		zq_cnn_conv_no_padding_gemm_nchwc1_general(in_firstPixelData, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
			filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
			out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
	}

	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::DepthwiseConvolutionWithBias(ZQ_CNN_Tensor4D_NCHWC1& input, const ZQ_CNN_Tensor4D_NCHWC1& filters,
	const ZQ_CNN_Tensor4D_NCHWC1& bias, int strideH, int strideW, int dilation_H, int dilation_W, 
	int padH, int padW, ZQ_CNN_Tensor4D_NCHWC1& output)
{
	double t1 = omp_get_wtime();
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	int filter_N = filters.GetN();
	int filter_H = filters.GetH();
	int filter_W = filters.GetW();
	int filter_C = filters.GetC();
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	float bias_C = (float)bias.GetC();
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| (in_H - filter_H + (padH << 1)) < 0 || (in_W - filter_W + (padW << 1)) < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}
	if (filter_C != in_C || filter_N != 1)
		return false;

	int need_N = in_N;
	// 审计修复 2026-10-02（附录 BE.2 / BH）：strideH/strideW 来自**模型文件**
	// （不可信输入），下面这行是**整数除法** —— 除以 0 在 x86 上是 idiv，
	// 直接 SIGFPE、进程死。附录 BE 只修了 NCHW 那 7 处 wrapper，
	// NCHWC 这边是同一份代码复制过去的，同样要修。
	if (strideH <= 0 || strideW <= 0)
		return false;
	int need_H = (in_H - (filter_H - 1)*dilation_H - 1 + (padH << 1)) / strideH + 1;
	int need_W = (in_W - (filter_W - 1)*dilation_W - 1 + (padW << 1)) / strideW + 1;
	int need_C = in_C;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	if (padH != 0 || padW != 0)
	{
		if (!input.Padding(padW, padH, 0))
			return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int filter_sliceStep = filters.GetSliceStep();
	int filter_widthStep = filters.GetWidthStep();
	int filter_imStep = filters.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr() - padH*in_widthStep - padW;
	const float* filter_firstPixelData = filters.GetFirstPixelPtr();
	float* out_firstPixelData = output.GetFirstPixelPtr();
	const float* bias_firstPixelData = bias.GetFirstPixelPtr();

	if (filter_H == 3 && filter_W == 3)
	{
		if (strideH == 1 && strideW == 1 && dilation_H == 1 && dilation_W == 1)
		{
			zq_cnn_depthwise_conv_no_padding_nchwc1_kernel3x3_s1d1_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData);
		}
		else if (strideH == 2 && strideW == 2 && dilation_H == 1 && dilation_W == 1)
		{
			zq_cnn_depthwise_conv_no_padding_nchwc1_kernel3x3_s2d1_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData);
		}
		else
		{
			zq_cnn_depthwise_conv_no_padding_nchwc1_kernel3x3_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData);
		}
	}
	else if (filter_H == 2 && filter_W == 2)
	{
		if (strideH == 1 && strideW == 1 && dilation_H == 1 && dilation_W == 1)
		{
			zq_cnn_depthwise_conv_no_padding_nchwc1_kernel2x2_s1d1_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData);
		}
		else
		{
			zq_cnn_depthwise_conv_no_padding_nchwc1_kernel2x2_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData);
		}
	}
	else if (filter_H == 5 && filter_W == 5
		&& strideH == 1 && strideW == 1 && dilation_H == 1 && dilation_W == 1)
	{
		zq_cnn_depthwise_conv_no_padding_nchwc1_kernel5x5_s1d1_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
			filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
			dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData);
	}
	else
	{
		zq_cnn_depthwise_conv_no_padding_nchwc1_general_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
			filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
			dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData);
	}
	
	double t2 = omp_get_wtime();
	//printf("utils:conv: %.3f ms\n", (t2 - t1) * 1000);
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::DepthwiseConvolutionWithBiasPReLU(ZQ_CNN_Tensor4D_NCHWC1& input, const ZQ_CNN_Tensor4D_NCHWC1& filters,
	const ZQ_CNN_Tensor4D_NCHWC1& bias, const ZQ_CNN_Tensor4D_NCHWC1& prelu_slope, int strideH, int strideW, int dilation_H, int dilation_W,
	int padH, int padW, ZQ_CNN_Tensor4D_NCHWC1& output)
{
	double t1 = omp_get_wtime();
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	int filter_N = filters.GetN();
	int filter_H = filters.GetH();
	int filter_W = filters.GetW();
	int filter_C = filters.GetC();
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	float bias_C = (float)bias.GetC();
	float slope_C = (float)prelu_slope.GetC();
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| (in_H - filter_H + (padH << 1)) < 0 || (in_W - filter_W + (padW << 1)) < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}
	if (filter_C != in_C || filter_N != 1)
		return false;

	int need_N = in_N;
	// 审计修复 2026-10-02（附录 BE.2 / BH）：strideH/strideW 来自**模型文件**
	// （不可信输入），下面这行是**整数除法** —— 除以 0 在 x86 上是 idiv，
	// 直接 SIGFPE、进程死。附录 BE 只修了 NCHW 那 7 处 wrapper，
	// NCHWC 这边是同一份代码复制过去的，同样要修。
	if (strideH <= 0 || strideW <= 0)
		return false;
	int need_H = (in_H - (filter_H - 1)*dilation_H - 1 + (padH << 1)) / strideH + 1;
	int need_W = (in_W - (filter_W - 1)*dilation_W - 1 + (padW << 1)) / strideW + 1;
	int need_C = in_C;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	if (padH != 0 || padW != 0)
	{
		if (!input.Padding(padW, padH, 0))
			return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int filter_sliceStep = filters.GetSliceStep();
	int filter_widthStep = filters.GetWidthStep();
	int filter_imStep = filters.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr() - padH*in_widthStep - padW;
	const float* filter_firstPixelData = filters.GetFirstPixelPtr();
	float* out_firstPixelData = output.GetFirstPixelPtr();
	const float* bias_firstPixelData = bias.GetFirstPixelPtr();
	const float* slope_firstPixelData = prelu_slope.GetFirstPixelPtr();

	if (filter_H == 3 && filter_W == 3)
	{
		if (strideH == 1 && strideW == 1 && dilation_H == 1 && dilation_W == 1)
		{
			zq_cnn_depthwise_conv_no_padding_nchwc1_kernel3x3_s1d1_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData);
		}
		else if (strideH == 2 && strideW == 2 && dilation_H == 1 && dilation_W == 1)
		{
			zq_cnn_depthwise_conv_no_padding_nchwc1_kernel3x3_s2d1_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData);
		}
		else
		{
			zq_cnn_depthwise_conv_no_padding_nchwc1_kernel3x3_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData);
		}
	}
	else if (filter_H == 2 && filter_W == 2)
	{
		if (strideH == 1 && strideW == 1 && dilation_H == 1 && dilation_W == 1)
		{
			zq_cnn_depthwise_conv_no_padding_nchwc1_kernel2x2_s1d1_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData);
		}
		else
		{
			zq_cnn_depthwise_conv_no_padding_nchwc1_kernel2x2_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData);
		}
	}
	else if (filter_H == 5 && filter_W == 5
		&& strideH == 1 && strideW == 1 && dilation_H == 1 && dilation_W == 1)
	{
		zq_cnn_depthwise_conv_no_padding_nchwc1_kernel5x5_s1d1_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
			filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
			dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData);
	}
	else
	{
		zq_cnn_depthwise_conv_no_padding_nchwc1_general_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
			filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
			dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData);
	}

	double t2 = omp_get_wtime();
	//printf("utils:conv: %.3f ms\n", (t2 - t1) * 1000);
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::DepthwiseConvolution(ZQ_CNN_Tensor4D_NCHWC1& input, const ZQ_CNN_Tensor4D_NCHWC1& filters,
	int strideH, int strideW, int dilation_H, int dilation_W,
	int padH, int padW, ZQ_CNN_Tensor4D_NCHWC1& output)
{
	double t1 = omp_get_wtime();
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	int filter_N = filters.GetN();
	int filter_H = filters.GetH();
	int filter_W = filters.GetW();
	int filter_C = filters.GetC();
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| (in_H - filter_H + (padH << 1)) < 0 || (in_W - filter_W + (padW << 1)) < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}
	if (filter_C != in_C || filter_N != 1)
		return false;

	int need_N = in_N;
	// 审计修复 2026-10-02（附录 BE.2 / BH）：strideH/strideW 来自**模型文件**
	// （不可信输入），下面这行是**整数除法** —— 除以 0 在 x86 上是 idiv，
	// 直接 SIGFPE、进程死。附录 BE 只修了 NCHW 那 7 处 wrapper，
	// NCHWC 这边是同一份代码复制过去的，同样要修。
	if (strideH <= 0 || strideW <= 0)
		return false;
	int need_H = (in_H - (filter_H - 1)*dilation_H - 1 + (padH << 1)) / strideH + 1;
	int need_W = (in_W - (filter_W - 1)*dilation_W - 1 + (padW << 1)) / strideW + 1;
	int need_C = in_C;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	if (padH != 0 || padW != 0)
	{
		if (!input.Padding(padW, padH, 0))
			return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int filter_sliceStep = filters.GetSliceStep();
	int filter_widthStep = filters.GetWidthStep();
	int filter_imStep = filters.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr() - padH*in_widthStep - padW;
	const float* filter_firstPixelData = filters.GetFirstPixelPtr();
	float* out_firstPixelData = output.GetFirstPixelPtr();

	if (filter_H == 3 && filter_W == 3)
	{
		if (strideH == 1 && strideW == 1 && dilation_H == 1 && dilation_W == 1)
		{
			zq_cnn_depthwise_conv_no_padding_nchwc1_kernel3x3_s1d1(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
		}
		else if (strideH == 2 && strideW == 2 && dilation_H == 1 && dilation_W == 1)
		{
			zq_cnn_depthwise_conv_no_padding_nchwc1_kernel3x3_s2d1(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
		}
		else
		{
			zq_cnn_depthwise_conv_no_padding_nchwc1_kernel3x3(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
		}
	}
	else if (filter_H == 2 && filter_W == 2)
	{
		if (strideH == 1 && strideW == 1 && dilation_H == 1 && dilation_W == 1)
		{
			zq_cnn_depthwise_conv_no_padding_nchwc1_kernel2x2_s1d1(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
		}
		else
		{
			zq_cnn_depthwise_conv_no_padding_nchwc1_kernel2x2(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
		}
	}
	else if (filter_H == 5 && filter_W == 5
		&& strideH == 1 && strideW == 1 && dilation_H == 1 && dilation_W == 1)
	{
		zq_cnn_depthwise_conv_no_padding_nchwc1_kernel5x5_s1d1(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
			filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
			dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
	}
	else
	{
		zq_cnn_depthwise_conv_no_padding_nchwc1_general(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
			filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
			dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
	}

	double t2 = omp_get_wtime();
	//printf("utils:conv: %.3f ms\n", (t2 - t1) * 1000);
	return true;
}

void ZQ_CNN_Forward_SSEUtils_NCHWC::MaxPooling(ZQ_CNN_Tensor4D_NCHWC1 &input, ZQ_CNN_Tensor4D_NCHWC1 &output, int kernel_H, int kernel_W,
	int stride_H, int stride_W, int pad_H_top, int pad_H_bottom,
	int pad_W_left, int pad_W_right, bool global_pool)
{
	int in_N = input.GetN();
	int in_H0 = input.GetH();
	int in_W0 = input.GetW();
	int in_C = input.GetC();
	int need_W, need_H;
	int need_N = in_N;
	int need_C = in_C;
	if (global_pool)
	{
		need_H = 1;
		need_W = 1;
		kernel_H = in_H0;
		kernel_W = in_W0;
		stride_H = 1;
		stride_W = 1;
		// 审计 F2：global_pool 不需要 padding（NCHW 那一支也不加），
		// 显式清零，免得外层传进来的 pad 让 need/use 尺寸互相打架。
		pad_H_top = pad_H_bottom = pad_W_left = pad_W_right = 0;
	}
	else
	{
		need_W = (int)ceil((float)(in_W0 + pad_W_left + pad_W_right - kernel_W) / stride_W + 1);
		need_H = (int)ceil((float)(in_H0 + pad_H_top + pad_H_bottom - kernel_H) / stride_H + 1);
	}

	if (need_W <= 0 || need_H <= 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
			return;
		// 审计修复 2026-10-06（附录 IR.1）：这一行原来缺，NCHW 版（`ZQ_CNN_Forward_SSEUtils.h:1601`）有。
		// `ChangeSize(0,0,0,0,0,0)` 是**成功**的（把 output 置空并返回 true），
		// 所以少了它就会继续往下走到 `(in_H - kernel_H) % stride_H`：
		// `stride_H == 0` 时那是**整数 idiv 除零 -> SIGFPE**，进程直接死。
		// 触发：`stride_H <= 0 || stride_W <= 0`。
		// 经模型文件当前不可达（`ZQ_CNN_Layer_NCHWC.h:2184-2192` 的 ReadParam
		// 已有 `stride_H <= 0 -> return false`），但这 6 个是 public static，
		// 外部可直接调 —— 而 NCHW 版专门补的那行 `return ;` 就是这条路径的防线。
		// 顺带：即便 stride > 0，`need_H == 0`（in_H=3, kernel_H=5, stride_H=2）时
		// NCHWC 会把 output 置成 0 大小后**继续调内核**（内核的循环条件让它空转，
		// 实际不写内存），NCHW 则直接返回 —— 也是行为分叉。
		return;
	}

	// 审计 F2：`% stride == 0` 是**无 pad** 时的等价写法，有 pad 时不成立。
	// 直接说清它真正要表达的事：最后一个窗口不用裁边。
	const float* in_data = 0;
	int in_H = 0, in_W = 0;
	if (!zq_nchwc_pool_prepare_pad(input, pad_H_top, pad_H_bottom, pad_W_left, pad_W_right,
		in_data, in_H, in_W))
		return;
	bool suredivided = ((need_H - 1)*stride_H + kernel_H <= in_H)
		&& ((need_W - 1)*stride_W + kernel_W <= in_W);
	if (output.GetN() != need_N || output.GetH() != need_H || output.GetW() != need_W || output.GetC() != need_C)
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
		return;
	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* out_data = output.GetFirstPixelPtr();

	if (suredivided)
	{
		if (kernel_H == 2 && kernel_W == 2)
		{
			zq_cnn_maxpooling_nopadding_suredivided_nchwc1_kernel2x2(in_data, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, kernel_H, kernel_W, stride_H, stride_W,
				out_data, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
		}
		else if (kernel_H == 3 && kernel_W == 3)
		{
			zq_cnn_maxpooling_nopadding_suredivided_nchwc1_kernel3x3(in_data, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, kernel_H, kernel_W, stride_H, stride_W,
				out_data, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
		}
		else
		{
			zq_cnn_maxpooling_nopadding_suredivided_nchwc1_general(in_data, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, kernel_H, kernel_W, stride_H, stride_W,
				out_data, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
		}
	}
	else
	{
		zq_cnn_maxpooling_nopadding_nodivided_nchwc1_general(in_data, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, kernel_H, kernel_W, stride_H, stride_W,
			out_data, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
	}
}

void ZQ_CNN_Forward_SSEUtils_NCHWC::AVGPooling(ZQ_CNN_Tensor4D_NCHWC1 &input, ZQ_CNN_Tensor4D_NCHWC1 &output, int kernel_H, int kernel_W,
	int stride_H, int stride_W, int pad_H_top, int pad_H_bottom,
	int pad_W_left, int pad_W_right, bool global_pool)
{
	int in_N = input.GetN();
	int in_H0 = input.GetH();
	int in_W0 = input.GetW();
	int in_C = input.GetC();
	int need_W, need_H;
	int need_N = in_N;
	int need_C = in_C;
	if (global_pool)
	{
		need_H = 1;
		need_W = 1;
		kernel_H = in_H0;
		kernel_W = in_W0;
		stride_H = 1;
		stride_W = 1;
		// 审计 F2：global_pool 不需要 padding（NCHW 那一支也不加），
		// 显式清零，免得外层传进来的 pad 让 need/use 尺寸互相打架。
		pad_H_top = pad_H_bottom = pad_W_left = pad_W_right = 0;
	}
	else
	{
		need_W = (int)ceil((float)(in_W0 + pad_W_left + pad_W_right - kernel_W) / stride_W + 1);
		need_H = (int)ceil((float)(in_H0 + pad_H_top + pad_H_bottom - kernel_H) / stride_H + 1);
	}

	if (need_W <= 0 || need_H <= 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
			return;
		// 审计修复 2026-10-06（附录 IR.1）：这一行原来缺，NCHW 版（`ZQ_CNN_Forward_SSEUtils.h:1601`）有。
		// `ChangeSize(0,0,0,0,0,0)` 是**成功**的（把 output 置空并返回 true），
		// 所以少了它就会继续往下走到 `(in_H - kernel_H) % stride_H`：
		// `stride_H == 0` 时那是**整数 idiv 除零 -> SIGFPE**，进程直接死。
		// 触发：`stride_H <= 0 || stride_W <= 0`。
		// 经模型文件当前不可达（`ZQ_CNN_Layer_NCHWC.h:2184-2192` 的 ReadParam
		// 已有 `stride_H <= 0 -> return false`），但这 6 个是 public static，
		// 外部可直接调 —— 而 NCHW 版专门补的那行 `return ;` 就是这条路径的防线。
		// 顺带：即便 stride > 0，`need_H == 0`（in_H=3, kernel_H=5, stride_H=2）时
		// NCHWC 会把 output 置成 0 大小后**继续调内核**（内核的循环条件让它空转，
		// 实际不写内存），NCHW 则直接返回 —— 也是行为分叉。
		return;
	}

	// 审计 F2：`% stride == 0` 是**无 pad** 时的等价写法，有 pad 时不成立。
	// 直接说清它真正要表达的事：最后一个窗口不用裁边。
	const float* in_data = 0;
	int in_H = 0, in_W = 0;
	if (!zq_nchwc_pool_prepare_pad(input, pad_H_top, pad_H_bottom, pad_W_left, pad_W_right,
		in_data, in_H, in_W))
		return;
	bool suredivided = ((need_H - 1)*stride_H + kernel_H <= in_H)
		&& ((need_W - 1)*stride_W + kernel_W <= in_W);
	if (output.GetN() != need_N || output.GetH() != need_H || output.GetW() != need_W || output.GetC() != need_C)
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
		return;
	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* out_data = output.GetFirstPixelPtr();

	if (suredivided)
	{
		if (kernel_H == 2 && kernel_W == 2)
		{
			zq_cnn_avgpooling_nopadding_suredivided_nchwc1_kernel2x2(in_data, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, kernel_H, kernel_W, stride_H, stride_W,
				out_data, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
		}
		else if (kernel_H == 3 && kernel_W == 3)
		{
			zq_cnn_avgpooling_nopadding_suredivided_nchwc1_kernel3x3(in_data, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, kernel_H, kernel_W, stride_H, stride_W,
				out_data, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
		}
		else
		{
			zq_cnn_avgpooling_nopadding_suredivided_nchwc1_general(in_data, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, kernel_H, kernel_W, stride_H, stride_W,
				out_data, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
		}
	}
	else
	{
		zq_cnn_avgpooling_nopadding_nodivided_nchwc1_general(in_data, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, kernel_H, kernel_W, stride_H, stride_W,
			out_data, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
	}
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::AddBiasPReLU(ZQ_CNN_Tensor4D_NCHWC1 &input, const ZQ_CNN_Tensor4D_NCHWC1& bias, const ZQ_CNN_Tensor4D_NCHWC1& slope)
{
	int N = input.GetN();
	int H = input.GetH();
	int W = input.GetW();
	int C = input.GetC();
	if (N <= 0 || H <= 0 || W <= 0 || C <= 0)
		return true;
	if (bias.GetC() != C || slope.GetC() != C)
		return false;
	float* data = input.GetFirstPixelPtr();
	const float* bias_Data = bias.GetFirstPixelPtr();
	const float* slope_Data = slope.GetFirstPixelPtr();
	int imStep = input.GetImageStep();
	int widthStep = input.GetWidthStep();
	int sliceStep = input.GetSliceStep();
	bool sure_lessthan1 = true;
	for (int c = 0; c < C; c++)
	{
		if (slope_Data[c] > 1)
		{
			sure_lessthan1 = false;
			break;
		}
	}
	if (sure_lessthan1)
	{
		zq_cnn_addbias_prelu_nchwc1_sure_slope_lessthan1(data, N, H, W, C, widthStep, sliceStep, imStep, bias_Data, slope_Data);
	}
	else
	{
		zq_cnn_addbias_prelu_nchwc1(data, N, H, W, C, widthStep, sliceStep, imStep, bias_Data, slope_Data);
	}
	
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::PReLU(ZQ_CNN_Tensor4D_NCHWC1 &input, const ZQ_CNN_Tensor4D_NCHWC1& slope)
{
	int N = input.GetN();
	int H = input.GetH();
	int W = input.GetW();
	int C = input.GetC();
	if (N <= 0 || H <= 0 || W <= 0 || C <= 0)
		return true;
	if (slope.GetC() != C)
		return false;
	float* data = input.GetFirstPixelPtr();
	const float* slope_Data = slope.GetFirstPixelPtr();
	int imStep = input.GetImageStep();
	int widthStep = input.GetWidthStep();
	int sliceStep = input.GetSliceStep();
	bool sure_lessthan1 = true;
	for (int c = 0; c < C; c++)
	{
		if (slope_Data[c] > 1)
		{
			sure_lessthan1 = false;
			break;
		}
	}
	if (sure_lessthan1)
	{
		zq_cnn_prelu_nchwc1_sure_slope_lessthan1(data, N, H, W, C, widthStep, sliceStep, imStep, slope_Data);
	}
	else
	{
		zq_cnn_prelu_nchwc1(data, N, H, W, C, widthStep, sliceStep, imStep, slope_Data);
	}

	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::ReLU(ZQ_CNN_Tensor4D_NCHWC1 &input, float slope)
{
	int N = input.GetN();
	int H = input.GetH();
	int W = input.GetW();
	int C = input.GetC();
	if (N <= 0 || H <= 0 || W <= 0 || C <= 0)
		return true;
	float* data = input.GetFirstPixelPtr();
	int imStep = input.GetImageStep();
	int widthStep = input.GetWidthStep();
	int sliceStep = input.GetSliceStep();
	
	zq_cnn_relu_nchwc1(data, N, H, W, C, widthStep, sliceStep, imStep, slope);
	
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::Softmax(ZQ_CNN_Tensor4D_NCHWC1 &input, int axis)
{
	if (axis < 0 || axis >= 4)
		return false;
	int N = input.GetN();
	int H = input.GetH();
	int W = input.GetW();
	int C = input.GetC();
	if (N <= 0 || H <= 0 || W <= 0 || C <= 0)
		return true;
	float* data = input.GetFirstPixelPtr();
	int sliceStep = input.GetSliceStep();
	int widthStep = input.GetWidthStep();
	int imStep = input.GetImageStep();

	if (axis == 1)
		zq_cnn_softmax_nchwc1_C(data, N, H, W, C, widthStep, sliceStep, imStep);
	else
		return false;
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::BatchNorm_b_a(ZQ_CNN_Tensor4D_NCHWC1 &input, const ZQ_CNN_Tensor4D_NCHWC1& b, const ZQ_CNN_Tensor4D_NCHWC1& a)
{
	int N = input.GetN();
	int H = input.GetH();
	int W = input.GetW();
	int C = input.GetC();
	if (N <= 0 || H <= 0 || W <= 0 || C <= 0)
		return true;
	if (b.GetC() != C || a.GetC() != C)
		return false;
	float* data = input.GetFirstPixelPtr();
	const float* b_data = b.GetFirstPixelPtr();
	const float* a_data = a.GetFirstPixelPtr();
	int widthStep = input.GetWidthStep();
	int sliceStep = input.GetSliceStep();
	int imStep = input.GetImageStep();

	zq_cnn_batchnorm_b_a_nchwc1(data, N, H, W, C, widthStep, sliceStep, imStep, b_data, a_data);
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::Eltwise_Sum(const std::vector<const ZQ_CNN_Tensor4D_NCHWC1*>& input, ZQ_CNN_Tensor4D_NCHWC1& output)
{
	int in_num = (int)input.size();
	if (in_num < 2)
		return false;
	for (int i = 0; i < in_num; i++)
	{
		if (input[i] == 0) return false;
	}
	int N = input[0]->GetN();
	int H = input[0]->GetH();
	int W = input[0]->GetW();
	int C = input[0]->GetC();
	for (int i = 1; i < in_num; i++)
	{
		if (N != input[i]->GetN() || H != input[i]->GetH() || W != input[i]->GetW() || C != input[i]->GetC())
			return false;
	}
	if (output.GetN() != N || output.GetH() != H || output.GetW() != W || output.GetC() != C)
		if (!output.ChangeSize(N, H, W, C, 0, 0))
		
return false;
	if (N <= 0 || H <= 0 || W <= 0 || C <= 0)
		return true;
	std::vector<const float*> in_tensor_data(in_num);
	std::vector<int> in_imStep(in_num), in_widthStep(in_num), in_sliceStep(in_num);
	float* out_data = output.GetFirstPixelPtr();
	for (int i = 0; i < in_num; i++)
	{
		in_tensor_data[i] = input[i]->GetFirstPixelPtr();
		in_imStep[i] = input[i]->GetImageStep();
		in_widthStep[i] = input[i]->GetWidthStep();
		in_sliceStep[i] = input[i]->GetSliceStep();
	}
	int out_imStep = output.GetImageStep(), out_widthStep = output.GetWidthStep(), out_sliceStep = output.GetSliceStep();
	
	zq_cnn_eltwise_sum_nchwc1(in_num, &in_tensor_data[0], N, H, W, C, &in_widthStep[0], &in_sliceStep[0], &in_imStep[0], 
		out_data, out_widthStep, out_sliceStep, out_imStep);

	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::Eltwise_SumWithWeight(const std::vector<const ZQ_CNN_Tensor4D_NCHWC1*>& input, const std::vector<float>& weight,
	ZQ_CNN_Tensor4D_NCHWC1& output)
{
	int in_num = (int)input.size();
	if (in_num < 2 || in_num != weight.size())
		return false;
	for (int i = 0; i < in_num; i++)
	{
		if (input[i] == 0) return false;
	}
	int N = input[0]->GetN();
	int H = input[0]->GetH();
	int W = input[0]->GetW();
	int C = input[0]->GetC();
	for (int i = 1; i < in_num; i++)
	{
		if (N != input[i]->GetN() || H != input[i]->GetH() || W != input[i]->GetW() || C != input[i]->GetC())
			return false;
	}
	if (output.GetN() != N || output.GetH() != H || output.GetW() != W || output.GetC() != C)
		if (!output.ChangeSize(N, H, W, C, 0, 0))
		
return false;
	if (N <= 0 || H <= 0 || W <= 0 || C <= 0)
		return true;
	std::vector<const float*> in_tensor_data(in_num);
	std::vector<int> in_imStep(in_num), in_widthStep(in_num), in_sliceStep(in_num);
	float* out_data = output.GetFirstPixelPtr();
	for (int i = 0; i < in_num; i++)
	{
		in_tensor_data[i] = input[i]->GetFirstPixelPtr();
		in_imStep[i] = input[i]->GetImageStep();
		in_widthStep[i] = input[i]->GetWidthStep();
		in_sliceStep[i] = input[i]->GetSliceStep();
	}
	int out_imStep = output.GetImageStep(), out_widthStep = output.GetWidthStep(), out_sliceStep = output.GetSliceStep();
	
	zq_cnn_eltwise_sum_with_weight_nchwc1(in_num, &in_tensor_data[0], &weight[0], N, H, W, C, 
		&in_widthStep[0], &in_sliceStep[0], &in_imStep[0], out_data, out_widthStep,	out_sliceStep, out_imStep);
	return true;
}


bool ZQ_CNN_Forward_SSEUtils_NCHWC::Eltwise_Mul(const std::vector<const ZQ_CNN_Tensor4D_NCHWC1*>& input, ZQ_CNN_Tensor4D_NCHWC1& output)
{
	int in_num = (int)input.size();
	if (in_num < 2)
		return false;
	for (int i = 0; i < in_num; i++)
	{
		if (input[i] == 0) return false;
	}
	int N = input[0]->GetN();
	int H = input[0]->GetH();
	int W = input[0]->GetW();
	int C = input[0]->GetC();
	for (int i = 1; i < in_num; i++)
	{
		if (N != input[i]->GetN() || H != input[i]->GetH() || W != input[i]->GetW() || C != input[i]->GetC())
			return false;
	}
	if (output.GetN() != N || output.GetH() != H || output.GetW() != W || output.GetC() != C)
		if (!output.ChangeSize(N, H, W, C, 0, 0))
		
return false;
	if (N <= 0 || H <= 0 || W <= 0 || C <= 0)
		return true;
	std::vector<const float*> in_tensor_data(in_num);
	std::vector<int> in_imStep(in_num), in_widthStep(in_num), in_sliceStep(in_num);
	float* out_data = output.GetFirstPixelPtr();
	for (int i = 0; i < in_num; i++)
	{
		in_tensor_data[i] = input[i]->GetFirstPixelPtr();
		in_imStep[i] = input[i]->GetImageStep();
		in_widthStep[i] = input[i]->GetWidthStep();
		in_sliceStep[i] = input[i]->GetSliceStep();
	}
	int out_imStep = output.GetImageStep(), out_widthStep = output.GetWidthStep(), out_sliceStep = output.GetSliceStep();
	
	zq_cnn_eltwise_mul_nchwc1(in_num, &in_tensor_data[0], N, H, W, C, &in_widthStep[0], &in_sliceStep[0], &in_imStep[0], 
		out_data, out_widthStep, out_sliceStep, out_imStep);
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::Eltwise_Max(const std::vector<const ZQ_CNN_Tensor4D_NCHWC1*>& input, ZQ_CNN_Tensor4D_NCHWC1& output)
{
	int in_num = (int)input.size();
	if (in_num < 2)
		return false;
	for (int i = 0; i < in_num; i++)
	{
		if (input[i] == 0) return false;
	}
	int N = input[0]->GetN();
	int H = input[0]->GetH();
	int W = input[0]->GetW();
	int C = input[0]->GetC();
	for (int i = 1; i < in_num; i++)
	{
		if (N != input[i]->GetN() || H != input[i]->GetH() || W != input[i]->GetW() || C != input[i]->GetC())
			return false;
	}
	if (output.GetN() != N || output.GetH() != H || output.GetW() != W || output.GetC() != C)
		if (!output.ChangeSize(N, H, W, C, 0, 0))
		
return false;
	if (N <= 0 || H <= 0 || W <= 0 || C <= 0)
		return true;
	std::vector<const float*> in_tensor_data(in_num);
	std::vector<int> in_imStep(in_num), in_widthStep(in_num), in_sliceStep(in_num);
	float* out_data = output.GetFirstPixelPtr();
	for (int i = 0; i < in_num; i++)
	{
		in_tensor_data[i] = input[i]->GetFirstPixelPtr();
		in_imStep[i] = input[i]->GetImageStep();
		in_widthStep[i] = input[i]->GetWidthStep();
		in_sliceStep[i] = input[i]->GetSliceStep();
	}
	int out_imStep = output.GetImageStep(), out_widthStep = output.GetWidthStep(), out_sliceStep = output.GetSliceStep();

	zq_cnn_eltwise_max_nchwc1(in_num, &in_tensor_data[0], N, H, W, C, &in_widthStep[0], &in_sliceStep[0], &in_imStep[0],
		out_data, out_widthStep, out_sliceStep, out_imStep);
	return true;
}

#if __ARM_NEON || (ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE)

bool ZQ_CNN_Forward_SSEUtils_NCHWC::ConvolutionPrePack(const ZQ_CNN_Tensor4D_NCHWC4& filters,
	ZQ_CNN_Tensor4D_NCHWC::Buffer& packedfilters)
{
	const float* data = filters.GetFirstPixelPtr();
	int N = filters.GetN();
	int H = filters.GetH();
	int W = filters.GetW();
	int C = filters.GetC();
	int widthStep = filters.GetWidthStep();
	int sliceStep = filters.GetSliceStep();
	int imStep = filters.GetImageStep();
	if (H == 1 && W == 1)
	{
#if __ARM_NEON && __ARM_NEON_ARMV8
		if (N <= 32)
		{
			zq_cnn_convolution_gemm_nchwc4_prepack8_other_kernel1x1(filters.GetFirstPixelPtr(), N, H, W, C, widthStep, sliceStep, imStep,
				(void**)&(packedfilters.data), &(packedfilters.len));
		}
		else
#endif
		{
			zq_cnn_convolution_gemm_nchwc4_prepack4_kernel1x1(filters.GetFirstPixelPtr(), N, H, W, C, widthStep, sliceStep, imStep,
				(void**)&(packedfilters.data), &(packedfilters.len));
		}
	}
	else if (H == 3 && W == 3 && C <= 4)
	{
#if __ARM_NEON && __ARM_NEON_ARMV8
		if (C == 3)
		{
			zq_cnn_convolution_gemm_nchwc4_prepack8_other_kernel3x3_C3(filters.GetFirstPixelPtr(), N, H, W, C, widthStep, sliceStep, imStep,
				(void**)&(packedfilters.data), &(packedfilters.len));
	    }
		else
#endif
		{
			zq_cnn_convolution_gemm_nchwc4_prepack4_kernel3x3_C3C4(filters.GetFirstPixelPtr(), N, H, W, C, widthStep, sliceStep, imStep,
				(void**)&(packedfilters.data), &(packedfilters.len));
		}
	}
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::InnerProductPrePack(const ZQ_CNN_Tensor4D_NCHWC4& filters,
	ZQ_CNN_Tensor4D_NCHWC::Buffer& packedfilters)
{
	const float* data = filters.GetFirstPixelPtr();
	int N = filters.GetN();
	int H = filters.GetH();
	int W = filters.GetW();
	int C = filters.GetC();
	int widthStep = filters.GetWidthStep();
	int sliceStep = filters.GetSliceStep();
	int imStep = filters.GetImageStep();
#if __ARM_NEON && __ARM_NEON_ARMV8
	zq_cnn_innerproduct_gemm_nchwc4_prepack8_other(filters.GetFirstPixelPtr(), N, H, W, C, widthStep, sliceStep, imStep,
		(void**)&(packedfilters.data), &(packedfilters.len));
#else
	zq_cnn_innerproduct_gemm_nchwc4_prepack4(filters.GetFirstPixelPtr(), N, H, W, C, widthStep, sliceStep, imStep,
		(void**)&(packedfilters.data), &(packedfilters.len));
#endif
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::InnerProductWithBias(ZQ_CNN_Tensor4D_NCHWC4& input,
	const ZQ_CNN_Tensor4D_NCHWC::Buffer& packedfilters, int filter_N,
	const ZQ_CNN_Tensor4D_NCHWC4& bias,
	ZQ_CNN_Tensor4D_NCHWC4& output, void** buffer, __int64* buffer_len)
{
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	int bias_C = bias.GetC();
	int need_H = 1;
	int need_W = 1;
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| need_H < 0 || need_W < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}

	// 审计修复 2026-10-06（附录 IR.2）：packed 重载原来**没有** `filter_N != bias_C` 校验，
	// 而**同文件同名的 unpacked 重载有**（`InnerProductWithBias` :2412 等 4 处）—— 一份对四份错。
	// 内核按 `zq_mm_load_ps(bias + out_c)` 整宽读 bias：bias 只有 `ceil(bias_C/align)*align`
	// 个 float；`bias_C < filter_N` 且 `bias_C % align == 0` 时，最后一组会**读过 bias 缓冲末尾**。
	// packed 版同时也没有 `filter_C != in_C`（packed 路径拿不到 filter_C，
	// 故只补 bias 这一条）。
	// x86 上 Convolution 的 packed 重载被 `#if __ARM_NEON` 包着不可达，
	// 但 **InnerProduct 的 packed 重载在 x86 上真的会被调用**
	// （`ZQ_CNN_Layer_NCHWC.h:2299/2326` 没有 `#if`），所以这是活的洞。

	if (filter_N != bias_C)

		return false;


	int need_N = in_N;

	int need_C = filter_N;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr();
	float* out_firstPixelData = output.GetFirstPixelPtr();
	const float* bias_firstPixelData = bias.GetFirstPixelPtr();

#if __ARM_NEON && __ARM_NEON_ARMV8
	zq_cnn_innerproduct_gemm_nchwc4_packed8_other_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C,
		in_widthStep, in_sliceStep, in_imStep, (const float*)(packedfilters.data),
		out_firstPixelData, need_N, need_H, need_W, need_C,
		out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, buffer, buffer_len);
#else
	zq_cnn_innerproduct_gemm_nchwc4_packed4_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C,
		in_widthStep, in_sliceStep, in_imStep, (const float*)(packedfilters.data),
		out_firstPixelData, need_N, need_H, need_W, need_C,
		out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, buffer, buffer_len);
#endif
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::InnerProductWithBiasPReLU(ZQ_CNN_Tensor4D_NCHWC4& input,
	const ZQ_CNN_Tensor4D_NCHWC::Buffer& packedfilters, int filter_N,
	const ZQ_CNN_Tensor4D_NCHWC4& bias, const ZQ_CNN_Tensor4D_NCHWC4& slope,
	ZQ_CNN_Tensor4D_NCHWC4& output,
	void** buffer, __int64* buffer_len)
{
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	int bias_C = bias.GetC();
	int need_H = 1;
	int need_W = 1;
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| need_H < 0 || need_W < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}

	// 审计修复 2026-10-06（附录 IR.2）：packed 重载原来**没有** `filter_N != bias_C` 校验，
	// 而**同文件同名的 unpacked 重载有**（`InnerProductWithBias` :2412 等 4 处）—— 一份对四份错。
	// 内核按 `zq_mm_load_ps(bias + out_c)` 整宽读 bias：bias 只有 `ceil(bias_C/align)*align`
	// 个 float；`bias_C < filter_N` 且 `bias_C % align == 0` 时，最后一组会**读过 bias 缓冲末尾**。
	// packed 版同时也没有 `filter_C != in_C`（packed 路径拿不到 filter_C，
	// 故只补 bias 这一条）。
	// x86 上 Convolution 的 packed 重载被 `#if __ARM_NEON` 包着不可达，
	// 但 **InnerProduct 的 packed 重载在 x86 上真的会被调用**
	// （`ZQ_CNN_Layer_NCHWC.h:2299/2326` 没有 `#if`），所以这是活的洞。

	if (filter_N != bias_C)

		return false;


	int need_N = in_N;

	int need_C = filter_N;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr();
	float* out_firstPixelData = output.GetFirstPixelPtr();
	const float* bias_firstPixelData = bias.GetFirstPixelPtr();
	const float* slope_firstPixelData = slope.GetFirstPixelPtr();

#if __ARM_NEON && _ARM_NEON_ARMV8
	zq_cnn_innerproduct_gemm_nchwc4_packed8_other_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C,
		in_widthStep, in_sliceStep, in_imStep, (const float*)(packedfilters.data),
		out_firstPixelData, need_N, need_H, need_W, need_C,
		out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData, buffer, buffer_len);
#else
	zq_cnn_innerproduct_gemm_nchwc4_packed4_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C,
		in_widthStep, in_sliceStep, in_imStep, (const float*)(packedfilters.data),
		out_firstPixelData, need_N, need_H, need_W, need_C,
		out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData, buffer, buffer_len);
#endif
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::InnerProductWithPReLU(ZQ_CNN_Tensor4D_NCHWC4& input,
	const ZQ_CNN_Tensor4D_NCHWC::Buffer& packedfilters, int filter_N,
	const ZQ_CNN_Tensor4D_NCHWC4& slope,
	ZQ_CNN_Tensor4D_NCHWC4& output, void** buffer, __int64* buffer_len)
{
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	int need_H = 1;
	int need_W = 1;
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| need_H < 0 || need_W < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}


	int need_N = in_N;

	int need_C = filter_N;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr();
	float* out_firstPixelData = output.GetFirstPixelPtr();
	const float* slope_firstPixelData = slope.GetFirstPixelPtr();

#if __ARM_NEON && __ARM_NEON_ARMV8
	zq_cnn_innerproduct_gemm_nchwc4_packed8_other(in_firstPixelData, in_N, in_H, in_W, in_C,
		in_widthStep, in_sliceStep, in_imStep, (const float*)(packedfilters.data),
		out_firstPixelData, need_N, need_H, need_W, need_C,
		out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
#else
	zq_cnn_innerproduct_gemm_nchwc4_packed4(in_firstPixelData, in_N, in_H, in_W, in_C,
		in_widthStep, in_sliceStep, in_imStep, (const float*)(packedfilters.data),
		out_firstPixelData, need_N, need_H, need_W, need_C,
		out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
#endif
	zq_cnn_prelu_nchwc4(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep, slope_firstPixelData);
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::InnerProduct(ZQ_CNN_Tensor4D_NCHWC4& input,
	const ZQ_CNN_Tensor4D_NCHWC::Buffer& packedfilters, int filter_N,
	ZQ_CNN_Tensor4D_NCHWC4& output, void** buffer, __int64* buffer_len)
{
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	int need_H = 1;
	int need_W = 1;
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| need_H < 0 || need_W < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}


	int need_N = in_N;

	int need_C = filter_N;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr();
	float* out_firstPixelData = output.GetFirstPixelPtr();
#if __ARM_NEON && __ARM_NEON_ARMV8
	zq_cnn_innerproduct_gemm_nchwc4_packed8_other(in_firstPixelData, in_N, in_H, in_W, in_C,
		in_widthStep, in_sliceStep, in_imStep, (const float*)(packedfilters.data),
		out_firstPixelData, need_N, need_H, need_W, need_C,
		out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
#else
	zq_cnn_innerproduct_gemm_nchwc4_packed4(in_firstPixelData, in_N, in_H, in_W, in_C,
		in_widthStep, in_sliceStep, in_imStep, (const float*)(packedfilters.data),
		out_firstPixelData, need_N, need_H, need_W, need_C,
		out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
#endif
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::ConvolutionWithBias(ZQ_CNN_Tensor4D_NCHWC4& input,
	const ZQ_CNN_Tensor4D_NCHWC::Buffer& packedfilters, int filter_N, int filter_H, int filter_W, int filter_C,
	const ZQ_CNN_Tensor4D_NCHWC4& bias, int strideH, int strideW, int dilation_H, int dilation_W, int padH, int padW,
	ZQ_CNN_Tensor4D_NCHWC4& output, void** buffer, __int64* buffer_len)
{
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	/*static int count = 0;
	if (in_H != 56 || in_W != 56 && count != 0)
		return false;
	count++;*/
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	int bias_C = bias.GetC();
	// 审计修复 2026-10-02（附录 BE.2 / BH）：strideH/strideW 来自**模型文件**
	// （不可信输入），下面这行是**整数除法** —— 除以 0 在 x86 上是 idiv，
	// 直接 SIGFPE、进程死。附录 BE 只修了 NCHW 那 7 处 wrapper，
	// NCHWC 这边是同一份代码复制过去的，同样要修。
	if (strideH <= 0 || strideW <= 0)
		return false;
	int need_H = (in_H - (filter_H - 1)*dilation_H - 1 + (padH << 1)) / strideH + 1;
	int need_W = (in_W - (filter_W - 1)*dilation_W - 1 + (padW << 1)) / strideW + 1;
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| need_H < 0 || need_W < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}
	
	// 审计修复 2026-10-06（附录 IR.2）：packed 重载原来**没有** `filter_N != bias_C` 校验，
	// 而**同文件同名的 unpacked 重载有**（`InnerProductWithBias` :2412 等 4 处）—— 一份对四份错。
	// 内核按 `zq_mm_load_ps(bias + out_c)` 整宽读 bias：bias 只有 `ceil(bias_C/align)*align`
	// 个 float；`bias_C < filter_N` 且 `bias_C % align == 0` 时，最后一组会**读过 bias 缓冲末尾**。
	// packed 版同时也没有 `filter_C != in_C`（packed 路径拿不到 filter_C，
	// 故只补 bias 这一条）。
	// x86 上 Convolution 的 packed 重载被 `#if __ARM_NEON` 包着不可达，
	// 但 **InnerProduct 的 packed 重载在 x86 上真的会被调用**
	// （`ZQ_CNN_Layer_NCHWC.h:2299/2326` 没有 `#if`），所以这是活的洞。

	if (filter_N != bias_C)

		return false;


	int need_N = in_N;

	int need_C = filter_N;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	if (padH != 0 || padW != 0)
	{
		if (!input.Padding(padW, padH, 0))
			return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr() - padH*in_widthStep - padW * 4;
	float* out_firstPixelData = output.GetFirstPixelPtr();
	const float* bias_firstPixelData = bias.GetFirstPixelPtr();

	if (filter_H == 1 && filter_W == 1)
	{
#if __ARM_NEON && __ARM_NEON_ARMV8 
		if (need_C <= 32)
		{
			zq_cnn_convolution_gemm_nchwc4_packedM8N8_other_kernel1x1_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, (const float*)(packedfilters.data),
				out_firstPixelData, need_N, need_H, need_W, need_C,
				out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, buffer, buffer_len);
		}
		else
#endif
		{
			zq_cnn_convolution_gemm_nchwc4_packedM4N4_kernel1x1_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, (const float*)(packedfilters.data),
				out_firstPixelData, need_N, need_H, need_W, need_C,
				out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, buffer, buffer_len);
		}
	}
	else if (filter_H == 3 && filter_W == 3 && in_C <= 4)
	{
#if __ARM_NEON && __ARM_NEON_ARMV8
		if (in_C == 3)
		{
			zq_cnn_convolution_gemm_nchwc4_packedM8N8_other_kernel3x3_C3_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, (const float*)(packedfilters.data), filter_H, filter_W,
				strideH, strideW, dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C,
				out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, buffer, buffer_len);
		}
		else
			return false;
#else
		return false;
#endif
	}
	else
		return false;
	return true;
}


bool ZQ_CNN_Forward_SSEUtils_NCHWC::ConvolutionWithBiasPReLU(ZQ_CNN_Tensor4D_NCHWC4& input,
	const ZQ_CNN_Tensor4D_NCHWC::Buffer& packedfilters, int filter_N, int filter_H, int filter_W, int filter_C,
	const ZQ_CNN_Tensor4D_NCHWC4& bias, const ZQ_CNN_Tensor4D_NCHWC4& slope, int strideH, int strideW,
	int dilation_H, int dilation_W, int padH, int padW, ZQ_CNN_Tensor4D_NCHWC4& output,
	void** buffer, __int64* buffer_len)
{
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	/*static int count = 0;
	if (in_H != 56 || in_W != 56 && count != 0)
		return false;
	count++;*/
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	int bias_C = bias.GetC();
	// 审计修复 2026-10-02（附录 BE.2 / BH）：strideH/strideW 来自**模型文件**
	// （不可信输入），下面这行是**整数除法** —— 除以 0 在 x86 上是 idiv，
	// 直接 SIGFPE、进程死。附录 BE 只修了 NCHW 那 7 处 wrapper，
	// NCHWC 这边是同一份代码复制过去的，同样要修。
	if (strideH <= 0 || strideW <= 0)
		return false;
	int need_H = (in_H - (filter_H - 1)*dilation_H - 1 + (padH << 1)) / strideH + 1;
	int need_W = (in_W - (filter_W - 1)*dilation_W - 1 + (padW << 1)) / strideW + 1;
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| need_H < 0 || need_W < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}

	// 审计修复 2026-10-06（附录 IR.2）：packed 重载原来**没有** `filter_N != bias_C` 校验，
	// 而**同文件同名的 unpacked 重载有**（`InnerProductWithBias` :2412 等 4 处）—— 一份对四份错。
	// 内核按 `zq_mm_load_ps(bias + out_c)` 整宽读 bias：bias 只有 `ceil(bias_C/align)*align`
	// 个 float；`bias_C < filter_N` 且 `bias_C % align == 0` 时，最后一组会**读过 bias 缓冲末尾**。
	// packed 版同时也没有 `filter_C != in_C`（packed 路径拿不到 filter_C，
	// 故只补 bias 这一条）。
	// x86 上 Convolution 的 packed 重载被 `#if __ARM_NEON` 包着不可达，
	// 但 **InnerProduct 的 packed 重载在 x86 上真的会被调用**
	// （`ZQ_CNN_Layer_NCHWC.h:2299/2326` 没有 `#if`），所以这是活的洞。

	if (filter_N != bias_C)

		return false;


	int need_N = in_N;

	int need_C = filter_N;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	if (padH != 0 || padW != 0)
	{
		if (!input.Padding(padW, padH, 0))
			return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr() - padH*in_widthStep - padW * 4;
	float* out_firstPixelData = output.GetFirstPixelPtr();
	const float* bias_firstPixelData = bias.GetFirstPixelPtr();
	const float* slope_firstPixelData = slope.GetFirstPixelPtr();

	if (filter_H == 1 && filter_W == 1)
	{
#if __ARM_NEON && __ARM_NEON_ARMV8
		if (need_C <= 32)
		{
			zq_cnn_convolution_gemm_nchwc4_packedM8N8_other_kernel1x1_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, (const float*)(packedfilters.data),
				out_firstPixelData, need_N, need_H, need_W, need_C,
				out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData, buffer, buffer_len);
		}
		else
#endif
		{
			zq_cnn_convolution_gemm_nchwc4_packedM4N4_kernel1x1_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, (const float*)(packedfilters.data),
				out_firstPixelData, need_N, need_H, need_W, need_C,
				out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData, buffer, buffer_len);
		}
	}
	else if (filter_H == 3 && filter_W == 3 && in_C <= 4)
	{
#if __ARM_NEON && __ARM_NEON_ARMV8
		if (in_C == 3)
		{
			zq_cnn_convolution_gemm_nchwc4_packedM8N8_other_kernel3x3_C3_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, (const float*)(packedfilters.data), filter_H, filter_W,
				strideH, strideW, dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C,
				out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData, buffer, buffer_len);
		}
		else
			return false;
#else
		return false;
#endif
	}
	else
		return false;
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::ConvolutionWithPReLU(ZQ_CNN_Tensor4D_NCHWC4& input,
	const ZQ_CNN_Tensor4D_NCHWC::Buffer& packedfilters, int filter_N, int filter_H, int filter_W, int filter_C,
	const ZQ_CNN_Tensor4D_NCHWC4& slope, int strideH, int strideW, int dilation_H, int dilation_W, int padH, int padW,
	ZQ_CNN_Tensor4D_NCHWC4& output, void** buffer, __int64* buffer_len)
{
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	/*static int count = 0;
	if (in_H != 56 || in_W != 56 && count != 0)
		return false;
	count++;*/
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	// 审计修复 2026-10-02（附录 BE.2 / BH）：strideH/strideW 来自**模型文件**
	// （不可信输入），下面这行是**整数除法** —— 除以 0 在 x86 上是 idiv，
	// 直接 SIGFPE、进程死。附录 BE 只修了 NCHW 那 7 处 wrapper，
	// NCHWC 这边是同一份代码复制过去的，同样要修。
	if (strideH <= 0 || strideW <= 0)
		return false;
	int need_H = (in_H - (filter_H - 1)*dilation_H - 1 + (padH << 1)) / strideH + 1;
	int need_W = (in_W - (filter_W - 1)*dilation_W - 1 + (padW << 1)) / strideW + 1;
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| need_H < 0 || need_W < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}


	int need_N = in_N;

	int need_C = filter_N;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	if (padH != 0 || padW != 0)
	{
		if (!input.Padding(padW, padH, 0))
			return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr() - padH*in_widthStep - padW * 4;
	float* out_firstPixelData = output.GetFirstPixelPtr();
	const float* slope_firstPixelData = slope.GetFirstPixelPtr();

	if (filter_H == 1 && filter_W == 1)
	{
#if __ARM_NEON && __ARM_NEON_ARMV8
		if (need_C <= 32)
		{
			zq_cnn_convolution_gemm_nchwc4_packedM8N8_other_kernel1x1(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, (const float*)(packedfilters.data),
				out_firstPixelData, need_N, need_H, need_W, need_C,
				out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
		}
		else
#endif
		{
			zq_cnn_convolution_gemm_nchwc4_packedM4N4_kernel1x1(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, (const float*)(packedfilters.data),
				out_firstPixelData, need_N, need_H, need_W, need_C,
				out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
		}
	}
	else if (filter_H == 3 && filter_W == 3 && in_C <= 4)
	{
#if __ARM_NEON && __ARM_NEON_ARMV8
		if (in_C == 3)
		{
			zq_cnn_convolution_gemm_nchwc4_packedM8N8_other_kernel3x3_C3(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, (const float*)(packedfilters.data), filter_H, filter_W,
				strideH, strideW, dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C,
				out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
		}
		else
			return false;
#else
		return false;
#endif
	}
	else
		return false;
	zq_cnn_prelu_nchwc4(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep, slope_firstPixelData);
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::Convolution(ZQ_CNN_Tensor4D_NCHWC4& input,
	const ZQ_CNN_Tensor4D_NCHWC::Buffer& packedfilters, int filter_N, int filter_H, int filter_W, int filter_C,
	int strideH, int strideW,
	int dilation_H, int dilation_W, int padH, int padW,
	ZQ_CNN_Tensor4D_NCHWC4& output, void** buffer, __int64* buffer_len)
{
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	/*static int count = 0;
	if (in_H != 56 || in_W != 56 && count != 0)
	return false;
	count++;*/
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	// 审计修复 2026-10-02（附录 BE.2 / BH）：strideH/strideW 来自**模型文件**
	// （不可信输入），下面这行是**整数除法** —— 除以 0 在 x86 上是 idiv，
	// 直接 SIGFPE、进程死。附录 BE 只修了 NCHW 那 7 处 wrapper，
	// NCHWC 这边是同一份代码复制过去的，同样要修。
	if (strideH <= 0 || strideW <= 0)
		return false;
	int need_H = (in_H - (filter_H - 1)*dilation_H - 1 + (padH << 1)) / strideH + 1;
	int need_W = (in_W - (filter_W - 1)*dilation_W - 1 + (padW << 1)) / strideW + 1;
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| need_H < 0 || need_W < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}


	int need_N = in_N;

	int need_C = filter_N;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	if (padH != 0 || padW != 0)
	{
		if (!input.Padding(padW, padH, 0))
			return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr() - padH*in_widthStep - padW * 4;
	float* out_firstPixelData = output.GetFirstPixelPtr();
	
	if (filter_H == 1 && filter_W == 1)
	{
#if __ARM_NEON && __ARM_NEON_ARMV8
		if (need_C <= 32)
		{
			zq_cnn_convolution_gemm_nchwc4_packedM8N8_other_kernel1x1(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, (const float*)(packedfilters.data),
				out_firstPixelData, need_N, need_H, need_W, need_C,
				out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
		}
		else
#endif
		{
			zq_cnn_convolution_gemm_nchwc4_packedM4N4_kernel1x1(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, (const float*)(packedfilters.data),
				out_firstPixelData, need_N, need_H, need_W, need_C,
				out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
		}
	}
	else if (filter_H == 3 && filter_W == 3 && in_C <= 4)
	{
#if __ARM_NEON && __ARM_NEON_ARMV8
		if (in_C == 3)
		{
			zq_cnn_convolution_gemm_nchwc4_packedM8N8_other_kernel3x3_C3(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, (const float*)(packedfilters.data), filter_H, filter_W,
				strideH, strideW, dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C,
				out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
		}
		else
			return false;
#else
		return false;
		
#endif
	}
	else
		return false;
	return true;
}


bool ZQ_CNN_Forward_SSEUtils_NCHWC::InnerProductWithBias(ZQ_CNN_Tensor4D_NCHWC4& input, const ZQ_CNN_Tensor4D_NCHWC4& filters,
	const ZQ_CNN_Tensor4D_NCHWC4& bias, ZQ_CNN_Tensor4D_NCHWC4& output, 
	void** buffer, __int64* buffer_len)
{
	double t1 = omp_get_wtime();
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	int filter_N = filters.GetN();
	int filter_H = filters.GetH();
	int filter_W = filters.GetW();
	int filter_C = filters.GetC();
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	int bias_C = bias.GetC();
	int need_H = 1;
	int need_W = 1;
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| need_H < 0 || need_W < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}
	if (filter_C != in_C || filter_N != bias_C)
		return false;

	int need_N = in_N;

	int need_C = filter_N;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int filter_sliceStep = filters.GetSliceStep();
	int filter_widthStep = filters.GetWidthStep();
	int filter_imStep = filters.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr();
	const float* filter_firstPixelData = filters.GetFirstPixelPtr();
	float* out_firstPixelData = output.GetFirstPixelPtr();
	const float* bias_firstPixelData = bias.GetFirstPixelPtr();


#if (ZQ_CNN_USE_BLAS_GEMM || ZQ_CNN_USE_MKL_GEMM || ZQ_CNN_USE_ZQ_GEMM)
	if (out_N >= 16 && filter_N >= 16)
	{
		zq_cnn_innerproduct_gemm_nchwc4_general_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
			filter_widthStep, filter_sliceStep, filter_imStep,
			out_firstPixelData, need_N, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, buffer, buffer_len);

	}
	else
#endif
	{
		if (4 * in_W == in_widthStep && in_widthStep*in_H == in_sliceStep
			&& 4 * in_W == filter_widthStep && filter_widthStep*in_H == filter_sliceStep
			&& 4 * out_W == out_widthStep && out_widthStep * out_H == out_sliceStep)
		{
			zq_cnn_innerproduct_nchwc4_noborder_with_bias(in_firstPixelData, in_N, in_H*in_W*in_C, filter_firstPixelData, filter_N,
				// 审计修复 2026-10-02（附录 BN.1）：传 out_sliceStep 时，N>1 的结果会
				// 互相覆盖（out 是 [N,1,1,K]，sliceStep 与 imStep 差 K 倍）。
				out_firstPixelData, out_imStep, bias_firstPixelData);
		}
		else
		{
			zq_cnn_innerproduct_gemm_nchwc4_general_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep,
				out_firstPixelData, need_N, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, buffer, buffer_len);
		}
	}

	double t2 = omp_get_wtime();
	//printf("utils:conv: %.3f ms\n", (t2 - t1) * 1000);
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::InnerProductWithBiasPReLU(ZQ_CNN_Tensor4D_NCHWC4& input, const ZQ_CNN_Tensor4D_NCHWC4& filters,
	const ZQ_CNN_Tensor4D_NCHWC4& bias, const ZQ_CNN_Tensor4D_NCHWC4& slope, ZQ_CNN_Tensor4D_NCHWC4& output,
	void** buffer, __int64* buffer_len)
{
	double t1 = omp_get_wtime();
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	int filter_N = filters.GetN();
	int filter_H = filters.GetH();
	int filter_W = filters.GetW();
	int filter_C = filters.GetC();
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	int bias_C = bias.GetC();
	int slope_C = slope.GetC();
	int need_H = 1;
	int need_W = 1;
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| need_H < 0 || need_W < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}
	if (filter_C != in_C || filter_N != bias_C || filter_N != slope_C)
		return false;

	int need_N = in_N;

	int need_C = filter_N;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int filter_sliceStep = filters.GetSliceStep();
	int filter_widthStep = filters.GetWidthStep();
	int filter_imStep = filters.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr();
	const float* filter_firstPixelData = filters.GetFirstPixelPtr();
	float* out_firstPixelData = output.GetFirstPixelPtr();
	const float* bias_firstPixelData = bias.GetFirstPixelPtr();
	const float* slope_firstPixelData = slope.GetFirstPixelPtr();


#if (ZQ_CNN_USE_BLAS_GEMM || ZQ_CNN_USE_MKL_GEMM || ZQ_CNN_USE_ZQ_GEMM)
	if (out_N >= 16 && filter_N >= 16)
	{
		zq_cnn_innerproduct_gemm_nchwc4_general_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
			filter_widthStep, filter_sliceStep, filter_imStep,
			out_firstPixelData, need_N, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData, buffer, buffer_len);

	}
	else
#endif
	{
		if (4 * in_W == in_widthStep && in_widthStep*in_H == in_sliceStep
			&& 4 * in_W == filter_widthStep && filter_widthStep*in_H == filter_sliceStep
			&& 4 * out_W == out_widthStep && out_widthStep * out_H == out_sliceStep)
		{
			zq_cnn_innerproduct_nchwc4_noborder_with_bias_prelu(in_firstPixelData, in_N, in_H*in_W*in_C, filter_firstPixelData, filter_N,
				// 审计修复 2026-10-02（附录 BN.1）：传 out_sliceStep 时，N>1 的结果会
				// 互相覆盖（out 是 [N,1,1,K]，sliceStep 与 imStep 差 K 倍）。
				out_firstPixelData, out_imStep, bias_firstPixelData, slope_firstPixelData);
		}
		else
		{
			zq_cnn_innerproduct_gemm_nchwc4_general_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep,
				out_firstPixelData, need_N, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData, buffer, buffer_len);
		}
	}
	double t2 = omp_get_wtime();
	//printf("utils:conv: %.3f ms\n", (t2 - t1) * 1000);
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::InnerProductWithPReLU(ZQ_CNN_Tensor4D_NCHWC4& input, const ZQ_CNN_Tensor4D_NCHWC4& filters,
	const ZQ_CNN_Tensor4D_NCHWC4& slope, ZQ_CNN_Tensor4D_NCHWC4& output, 
	void** buffer, __int64* buffer_len)
{
	double t1 = omp_get_wtime();
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	int filter_N = filters.GetN();
	int filter_H = filters.GetH();
	int filter_W = filters.GetW();
	int filter_C = filters.GetC();
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	int slope_C = slope.GetC();
	int need_H = 1;
	int need_W = 1;
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| need_H < 0 || need_W < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}
	if (filter_C != in_C || filter_N != slope_C)
		return false;

	int need_N = in_N;

	int need_C = filter_N;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int filter_sliceStep = filters.GetSliceStep();
	int filter_widthStep = filters.GetWidthStep();
	int filter_imStep = filters.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr();
	const float* filter_firstPixelData = filters.GetFirstPixelPtr();
	float* out_firstPixelData = output.GetFirstPixelPtr();
	const float* slope_firstPixelData = slope.GetFirstPixelPtr();


#if (ZQ_CNN_USE_BLAS_GEMM || ZQ_CNN_USE_MKL_GEMM || ZQ_CNN_USE_ZQ_GEMM)
	if (out_N >= 16 && filter_N >= 16)
	{
		zq_cnn_innerproduct_gemm_nchwc4_general(in_firstPixelData, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
			filter_widthStep, filter_sliceStep, filter_imStep,
			out_firstPixelData, need_N, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);

	}
	else
#endif
	{
		if (4 * in_W == in_widthStep && in_widthStep*in_H == in_sliceStep
			&& 4 * in_W == filter_widthStep && filter_widthStep*in_H == filter_sliceStep
			&& 4 * out_W == out_widthStep && out_widthStep * out_H == out_sliceStep)
		{
			zq_cnn_innerproduct_nchwc4_noborder(in_firstPixelData, in_N, in_H*in_W*in_C, filter_firstPixelData, filter_N,
				// 审计修复 2026-10-02（附录 BN.1）：传 out_sliceStep 时，N>1 的结果会
				// 互相覆盖（out 是 [N,1,1,K]，sliceStep 与 imStep 差 K 倍）。
				out_firstPixelData, out_imStep);
		}
		else
		{
			zq_cnn_innerproduct_gemm_nchwc4_general(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep,
				out_firstPixelData, need_N, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
		}
	}
	zq_cnn_prelu_nchwc4(out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, slope_firstPixelData);

	double t2 = omp_get_wtime();
	//printf("utils:conv: %.3f ms\n", (t2 - t1) * 1000);
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::InnerProduct(ZQ_CNN_Tensor4D_NCHWC4& input, const ZQ_CNN_Tensor4D_NCHWC4& filters, 
	ZQ_CNN_Tensor4D_NCHWC4& output, void** buffer, __int64* buffer_len)
{
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	int filter_N = filters.GetN();
	int filter_H = filters.GetH();
	int filter_W = filters.GetW();
	int filter_C = filters.GetC();
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	int need_H = 1;
	int need_W = 1;
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| need_H < 0 || need_W < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}
	if (filter_C != in_C)
		return false;

	int need_N = in_N;
	int need_C = filter_N;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int filter_sliceStep = filters.GetSliceStep();
	int filter_widthStep = filters.GetWidthStep();
	int filter_imStep = filters.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr();
	const float* filter_firstPixelData = filters.GetFirstPixelPtr();
	float* out_firstPixelData = output.GetFirstPixelPtr();


#if (ZQ_CNN_USE_BLAS_GEMM || ZQ_CNN_USE_MKL_GEMM || ZQ_CNN_USE_ZQ_GEMM)
	if (out_N >= 16 && filter_N >= 16)
	{
		zq_cnn_innerproduct_gemm_nchwc4_general(in_firstPixelData, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
			filter_widthStep, filter_sliceStep, filter_imStep,
			out_firstPixelData, need_N, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);

	}
	else
#endif
	{
		if (4 * in_W == in_widthStep && in_widthStep*in_H == in_sliceStep
			&& 4 * in_W == filter_widthStep && filter_widthStep*in_H == filter_sliceStep
			&& 4 * out_W == out_widthStep && out_widthStep * out_H == out_sliceStep)
		{
			zq_cnn_innerproduct_nchwc4_noborder(in_firstPixelData, in_N, in_H*in_W*in_C, filter_firstPixelData, filter_N,
				// 审计修复 2026-10-02（附录 BN.1）：传 out_sliceStep 时，N>1 的结果会
				// 互相覆盖（out 是 [N,1,1,K]，sliceStep 与 imStep 差 K 倍）。
				out_firstPixelData, out_imStep);
		}
		else
		{
			zq_cnn_innerproduct_gemm_nchwc4_general(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep,
				out_firstPixelData, need_N, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
		}
	}
	return true;
}


bool ZQ_CNN_Forward_SSEUtils_NCHWC::ConvolutionWithBias(ZQ_CNN_Tensor4D_NCHWC4& input, const ZQ_CNN_Tensor4D_NCHWC4& filters,
	const ZQ_CNN_Tensor4D_NCHWC4& bias, int strideH, int strideW, int dilation_H, int dilation_W, int padH, int padW,
	ZQ_CNN_Tensor4D_NCHWC4& output, void** buffer, __int64* buffer_len)
{
	double t1 = omp_get_wtime();
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	int filter_N = filters.GetN();
	int filter_H = filters.GetH();
	int filter_W = filters.GetW();
	int filter_C = filters.GetC();
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	int bias_C = bias.GetC();
	// 审计修复 2026-10-02（附录 BE.2 / BH）：strideH/strideW 来自**模型文件**
	// （不可信输入），下面这行是**整数除法** —— 除以 0 在 x86 上是 idiv，
	// 直接 SIGFPE、进程死。附录 BE 只修了 NCHW 那 7 处 wrapper，
	// NCHWC 这边是同一份代码复制过去的，同样要修。
	if (strideH <= 0 || strideW <= 0)
		return false;
	int need_H = (in_H - (filter_H - 1)*dilation_H - 1 + (padH << 1)) / strideH + 1;
	int need_W = (in_W - (filter_W - 1)*dilation_W - 1 + (padW << 1)) / strideW + 1;
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| need_H < 0 || need_W < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}
	if (filter_C != in_C || filter_N != bias_C)
		return false;

	int need_N = in_N;

	int need_C = filter_N;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	if (padH != 0 || padW != 0)
	{
		if (!input.Padding(padW, padH, 0))
			return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int filter_sliceStep = filters.GetSliceStep();
	int filter_widthStep = filters.GetWidthStep();
	int filter_imStep = filters.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr() - padH*in_widthStep - padW*4;
	const float* filter_firstPixelData = filters.GetFirstPixelPtr();
	float* out_firstPixelData = output.GetFirstPixelPtr();
	const float* bias_firstPixelData = bias.GetFirstPixelPtr();

	if (filter_H == 3 && filter_W == 3)
	{
		if (filter_C == 3)
		{
			zq_cnn_conv_no_padding_gemm_nchwc4_kernel3x3_C3_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, buffer, buffer_len);
		}
		else
		{
			zq_cnn_conv_no_padding_gemm_nchwc4_kernel3x3_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, buffer, buffer_len);
		}
	}
	else if (filter_H == 2 && filter_W == 2)
	{
		if (filter_C == 3)
		{
			zq_cnn_conv_no_padding_gemm_nchwc4_kernel2x2_C3_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, buffer, buffer_len);
		}
		else
		{
			zq_cnn_conv_no_padding_gemm_nchwc4_kernel2x2_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, buffer, buffer_len);
		}
	}
	else if (filter_H == 1 && filter_W == 1)
	{
		zq_cnn_conv_no_padding_gemm_nchwc4_kernel1x1_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
			filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
			out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, buffer, buffer_len);
	}
	else
	{
		zq_cnn_conv_no_padding_gemm_nchwc4_general_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
			filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
			out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, buffer, buffer_len);
	}

	double t2 = omp_get_wtime();
	//printf("utils:conv: %.3f ms\n", (t2 - t1) * 1000);
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::ConvolutionWithBiasPReLU(ZQ_CNN_Tensor4D_NCHWC4& input, const ZQ_CNN_Tensor4D_NCHWC4& filters,
	const ZQ_CNN_Tensor4D_NCHWC4& bias, const ZQ_CNN_Tensor4D_NCHWC4& slope, int strideH, int strideW,
	int dilation_H, int dilation_W, int padH, int padW, ZQ_CNN_Tensor4D_NCHWC4& output,
	void** buffer, __int64* buffer_len)
{
	double t1 = omp_get_wtime();
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	int filter_N = filters.GetN();
	int filter_H = filters.GetH();
	int filter_W = filters.GetW();
	int filter_C = filters.GetC();
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	int bias_C = bias.GetC();
	int slope_C = slope.GetC();
	// 审计修复 2026-10-02（附录 BE.2 / BH）：strideH/strideW 来自**模型文件**
	// （不可信输入），下面这行是**整数除法** —— 除以 0 在 x86 上是 idiv，
	// 直接 SIGFPE、进程死。附录 BE 只修了 NCHW 那 7 处 wrapper，
	// NCHWC 这边是同一份代码复制过去的，同样要修。
	if (strideH <= 0 || strideW <= 0)
		return false;
	int need_H = (in_H - (filter_H - 1)*dilation_H - 1 + (padH << 1)) / strideH + 1;
	int need_W = (in_W - (filter_W - 1)*dilation_W - 1 + (padW << 1)) / strideW + 1;
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| need_H < 0 || need_W < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}
	if (filter_C != in_C || filter_N != bias_C || filter_N != slope_C)
		return false;

	int need_N = in_N;

	int need_C = filter_N;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	if (padH != 0 || padW != 0)
	{
		if (!input.Padding(padW, padH, 0))
			return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int filter_sliceStep = filters.GetSliceStep();
	int filter_widthStep = filters.GetWidthStep();
	int filter_imStep = filters.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr() - padH*in_widthStep - padW*4;
	const float* filter_firstPixelData = filters.GetFirstPixelPtr();
	float* out_firstPixelData = output.GetFirstPixelPtr();
	const float* bias_firstPixelData = bias.GetFirstPixelPtr();
	const float* slope_firstPixelData = slope.GetFirstPixelPtr();

	if (filter_H == 3 && filter_W == 3)
	{
		if (filter_C == 3)
		{
			zq_cnn_conv_no_padding_gemm_nchwc4_kernel3x3_C3_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData, buffer, buffer_len);
		}
		else
		{
			zq_cnn_conv_no_padding_gemm_nchwc4_kernel3x3_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData, buffer, buffer_len);
		}
	}
	else if (filter_H == 2 && filter_W == 2)
	{
		if (filter_C == 3)
		{
			zq_cnn_conv_no_padding_gemm_nchwc4_kernel2x2_C3_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData, buffer, buffer_len);
		}
		else
		{
			zq_cnn_conv_no_padding_gemm_nchwc4_kernel2x2_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData, buffer, buffer_len);
		}
	}
	else if (filter_H == 1 && filter_W == 1)
	{
		zq_cnn_conv_no_padding_gemm_nchwc4_kernel1x1_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
			filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
			out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData, buffer, buffer_len);
	}
	else
	{
		zq_cnn_conv_no_padding_gemm_nchwc4_general_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
			filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
			out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData, buffer, buffer_len);
	}

	double t2 = omp_get_wtime();
	//printf("utils:conv: %.3f ms\n", (t2 - t1) * 1000);
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::ConvolutionWithPReLU(ZQ_CNN_Tensor4D_NCHWC4& input, const ZQ_CNN_Tensor4D_NCHWC4& filters,
	const ZQ_CNN_Tensor4D_NCHWC4& slope, int strideH, int strideW, int dilation_H, int dilation_W, int padH, int padW,
	ZQ_CNN_Tensor4D_NCHWC4& output, void** buffer, __int64* buffer_len)
{
	double t1 = omp_get_wtime();
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	int filter_N = filters.GetN();
	int filter_H = filters.GetH();
	int filter_W = filters.GetW();
	int filter_C = filters.GetC();
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	int slope_C = slope.GetC();
	// 审计修复 2026-10-02（附录 BE.2 / BH）：strideH/strideW 来自**模型文件**
	// （不可信输入），下面这行是**整数除法** —— 除以 0 在 x86 上是 idiv，
	// 直接 SIGFPE、进程死。附录 BE 只修了 NCHW 那 7 处 wrapper，
	// NCHWC 这边是同一份代码复制过去的，同样要修。
	if (strideH <= 0 || strideW <= 0)
		return false;
	int need_H = (in_H - (filter_H - 1)*dilation_H - 1 + (padH << 1)) / strideH + 1;
	int need_W = (in_W - (filter_W - 1)*dilation_W - 1 + (padW << 1)) / strideW + 1;
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| need_H < 0 || need_W < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}
	if (filter_C != in_C || filter_N != slope_C)
		return false;

	int need_N = in_N;

	int need_C = filter_N;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	if (padH != 0 || padW != 0)
	{
		if (!input.Padding(padW, padH, 0))
			return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int filter_sliceStep = filters.GetSliceStep();
	int filter_widthStep = filters.GetWidthStep();
	int filter_imStep = filters.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr() - padH*in_widthStep - padW*4;
	const float* filter_firstPixelData = filters.GetFirstPixelPtr();
	float* out_firstPixelData = output.GetFirstPixelPtr();
	const float* slope_firstPixelData = slope.GetFirstPixelPtr();


	if (filter_H == 3 && filter_W == 3)
	{
		if (filter_C == 3)
		{
			zq_cnn_conv_no_padding_gemm_nchwc4_kernel3x3_C3(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
		}
		else
		{
			zq_cnn_conv_no_padding_gemm_nchwc4_kernel3x3(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
		}
	}
	else if (filter_H == 2 && filter_W == 2)
	{
		if (filter_C == 3)
		{
			zq_cnn_conv_no_padding_gemm_nchwc4_kernel2x2_C3(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
		}
		else
		{
			zq_cnn_conv_no_padding_gemm_nchwc4_kernel2x2(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
		}
	}
	else if (filter_H == 1 && filter_W == 1)
	{
		zq_cnn_conv_no_padding_gemm_nchwc4_kernel1x1(in_firstPixelData, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
			filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
			out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
	}
	else
	{
		zq_cnn_conv_no_padding_gemm_nchwc4_general(in_firstPixelData, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
			filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
			out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
	}

	zq_cnn_prelu_nchwc4(out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, slope_firstPixelData);

	double t2 = omp_get_wtime();
	//printf("utils:conv: %.3f ms\n", (t2 - t1) * 1000);
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::Convolution(ZQ_CNN_Tensor4D_NCHWC4& input, const ZQ_CNN_Tensor4D_NCHWC4& filters, int strideH, int strideW,
	int dilation_H, int dilation_W, int padH, int padW,
	ZQ_CNN_Tensor4D_NCHWC4& output, void** buffer, __int64* buffer_len)
{
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	int filter_N = filters.GetN();
	int filter_H = filters.GetH();
	int filter_W = filters.GetW();
	int filter_C = filters.GetC();
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	// 审计修复 2026-10-02（附录 BE.2 / BH）：strideH/strideW 来自**模型文件**
	// （不可信输入），下面这行是**整数除法** —— 除以 0 在 x86 上是 idiv，
	// 直接 SIGFPE、进程死。附录 BE 只修了 NCHW 那 7 处 wrapper，
	// NCHWC 这边是同一份代码复制过去的，同样要修。
	if (strideH <= 0 || strideW <= 0)
		return false;
	int need_H = (in_H - (filter_H - 1)*dilation_H - 1 + (padH << 1)) / strideH + 1;
	int need_W = (in_W - (filter_W - 1)*dilation_W - 1 + (padW << 1)) / strideW + 1;
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| need_H < 0 || need_W < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}
	if (filter_C != in_C)
		return false;

	int need_N = in_N;
	int need_C = filter_N;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	if (padH != 0 || padW != 0)
	{
		if (!input.Padding(padW, padH, 0))
			return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int filter_sliceStep = filters.GetSliceStep();
	int filter_widthStep = filters.GetWidthStep();
	int filter_imStep = filters.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr() - padH*in_widthStep - padW*4;
	const float* filter_firstPixelData = filters.GetFirstPixelPtr();
	float* out_firstPixelData = output.GetFirstPixelPtr();

	if (filter_H == 3 && filter_W == 3)
	{
		if (filter_C == 3)
		{
			zq_cnn_conv_no_padding_gemm_nchwc4_kernel3x3_C3(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
		}
		else
		{
			zq_cnn_conv_no_padding_gemm_nchwc4_kernel3x3(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
		}
	}
	else if (filter_H == 2 && filter_W == 2)
	{
		if (filter_C == 3)
		{
			zq_cnn_conv_no_padding_gemm_nchwc4_kernel2x2_C3(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
		}
		else
		{
			zq_cnn_conv_no_padding_gemm_nchwc4_kernel2x2(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
		}
	}
	else if (filter_H == 1 && filter_W == 1)
	{
		zq_cnn_conv_no_padding_gemm_nchwc4_kernel1x1(in_firstPixelData, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
			filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
			out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
	}
	else
	{
		zq_cnn_conv_no_padding_gemm_nchwc4_general(in_firstPixelData, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
			filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
			out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
	}

	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::DepthwiseConvolutionWithBias(ZQ_CNN_Tensor4D_NCHWC4& input, const ZQ_CNN_Tensor4D_NCHWC4& filters,
	const ZQ_CNN_Tensor4D_NCHWC4& bias, int strideH, int strideW, int dilation_H, int dilation_W,
	int padH, int padW, ZQ_CNN_Tensor4D_NCHWC4& output)
{
	double t1 = omp_get_wtime();
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	int filter_N = filters.GetN();
	int filter_H = filters.GetH();
	int filter_W = filters.GetW();
	int filter_C = filters.GetC();
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	float bias_C = (float)bias.GetC();
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| (in_H - filter_H + (padH << 1)) < 0 || (in_W - filter_W + (padW << 1)) < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}
	if (filter_C != in_C || filter_N != 1)
		return false;

	int need_N = in_N;
	// 审计修复 2026-10-02（附录 BE.2 / BH）：strideH/strideW 来自**模型文件**
	// （不可信输入），下面这行是**整数除法** —— 除以 0 在 x86 上是 idiv，
	// 直接 SIGFPE、进程死。附录 BE 只修了 NCHW 那 7 处 wrapper，
	// NCHWC 这边是同一份代码复制过去的，同样要修。
	if (strideH <= 0 || strideW <= 0)
		return false;
	int need_H = (in_H - (filter_H - 1)*dilation_H - 1 + (padH << 1)) / strideH + 1;
	int need_W = (in_W - (filter_W - 1)*dilation_W - 1 + (padW << 1)) / strideW + 1;
	int need_C = in_C;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	if (padH != 0 || padW != 0)
	{
		if (!input.Padding(padW, padH, 0))
			return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int filter_sliceStep = filters.GetSliceStep();
	int filter_widthStep = filters.GetWidthStep();
	int filter_imStep = filters.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr() - padH*in_widthStep - padW*4;
	const float* filter_firstPixelData = filters.GetFirstPixelPtr();
	float* out_firstPixelData = output.GetFirstPixelPtr();
	const float* bias_firstPixelData = bias.GetFirstPixelPtr();

	if (filter_H == 3 && filter_W == 3)
	{
		if(strideH == 1 && strideW == 1 && dilation_H == 1 && dilation_W == 1)
		{
			zq_cnn_depthwise_conv_no_padding_nchwc4_kernel3x3_s1d1_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData);
		}
		else if (strideH == 2 && strideW == 2 && dilation_H == 1 && dilation_W == 1)
		{
			zq_cnn_depthwise_conv_no_padding_nchwc4_kernel3x3_s2d1_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData);
		}
		else
		{
			zq_cnn_depthwise_conv_no_padding_nchwc4_kernel3x3_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData);
		}
	}
	else if (filter_H == 2 && filter_W == 2)
	{
		if (strideH == 1 && strideW == 1 && dilation_H == 1 && dilation_W == 1)
		{
			zq_cnn_depthwise_conv_no_padding_nchwc4_kernel2x2_s1d1_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData);
		}
		else
		{
			zq_cnn_depthwise_conv_no_padding_nchwc4_kernel2x2_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData);
		}
	}
	else if (filter_H == 5 && filter_W == 5
		&& strideH == 1 && strideW == 1 && dilation_H == 1 && dilation_W == 1)
	{
		zq_cnn_depthwise_conv_no_padding_nchwc4_kernel5x5_s1d1_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
			filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
			dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData);
	}
	else
	{
		zq_cnn_depthwise_conv_no_padding_nchwc4_general_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
			filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
			dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData);
	}

	double t2 = omp_get_wtime();
	//printf("utils:conv: %.3f ms\n", (t2 - t1) * 1000);
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::DepthwiseConvolutionWithBiasPReLU(ZQ_CNN_Tensor4D_NCHWC4& input, const ZQ_CNN_Tensor4D_NCHWC4& filters,
	const ZQ_CNN_Tensor4D_NCHWC4& bias, const ZQ_CNN_Tensor4D_NCHWC4& prelu_slope, int strideH, int strideW, int dilation_H, int dilation_W,
	int padH, int padW, ZQ_CNN_Tensor4D_NCHWC4& output)
{
	double t1 = omp_get_wtime();
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	int filter_N = filters.GetN();
	int filter_H = filters.GetH();
	int filter_W = filters.GetW();
	int filter_C = filters.GetC();
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	float bias_C = (float)bias.GetC();
	float slope_C = (float)prelu_slope.GetC();
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| (in_H - filter_H + (padH << 1)) < 0 || (in_W - filter_W + (padW << 1)) < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}
	if (filter_C != in_C || filter_N != 1)
		return false;

	int need_N = in_N;
	// 审计修复 2026-10-02（附录 BE.2 / BH）：strideH/strideW 来自**模型文件**
	// （不可信输入），下面这行是**整数除法** —— 除以 0 在 x86 上是 idiv，
	// 直接 SIGFPE、进程死。附录 BE 只修了 NCHW 那 7 处 wrapper，
	// NCHWC 这边是同一份代码复制过去的，同样要修。
	if (strideH <= 0 || strideW <= 0)
		return false;
	int need_H = (in_H - (filter_H - 1)*dilation_H - 1 + (padH << 1)) / strideH + 1;
	int need_W = (in_W - (filter_W - 1)*dilation_W - 1 + (padW << 1)) / strideW + 1;
	int need_C = in_C;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	if (padH != 0 || padW != 0)
	{
		if (!input.Padding(padW, padH, 0))
			return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int filter_sliceStep = filters.GetSliceStep();
	int filter_widthStep = filters.GetWidthStep();
	int filter_imStep = filters.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr() - padH*in_widthStep - padW*4;
	const float* filter_firstPixelData = filters.GetFirstPixelPtr();
	float* out_firstPixelData = output.GetFirstPixelPtr();
	const float* bias_firstPixelData = bias.GetFirstPixelPtr();
	const float* slope_firstPixelData = prelu_slope.GetFirstPixelPtr();

	if (filter_H == 3 && filter_W == 3)
	{
		if (strideH == 1 && strideW == 1 && dilation_H == 1 && dilation_W == 1)
		{
			zq_cnn_depthwise_conv_no_padding_nchwc4_kernel3x3_s1d1_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData);
		}
		else if (strideH == 2 && strideW == 2 && dilation_H == 1 && dilation_W == 1)
		{
			zq_cnn_depthwise_conv_no_padding_nchwc4_kernel3x3_s2d1_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData);
		}
		else
		{
			zq_cnn_depthwise_conv_no_padding_nchwc4_kernel3x3_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData);
		}
	}
	else if (filter_H == 2 && filter_W == 2)
	{
		if (strideH == 1 && strideW == 1 && dilation_H == 1 && dilation_W == 1)
		{
			zq_cnn_depthwise_conv_no_padding_nchwc4_kernel2x2_s1d1_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData);
		}
		else
		{
			zq_cnn_depthwise_conv_no_padding_nchwc4_kernel2x2_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData);
		}
	}
	else if (filter_H == 5 && filter_W == 5
		&& strideH == 1 && strideW == 1 && dilation_H == 1 && dilation_W == 1)
	{
		zq_cnn_depthwise_conv_no_padding_nchwc4_kernel5x5_s1d1_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
			filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
			dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData);
	}
	else
	{
		zq_cnn_depthwise_conv_no_padding_nchwc4_general_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
			filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
			dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData);
	}

	double t2 = omp_get_wtime();
	//printf("utils:conv: %.3f ms\n", (t2 - t1) * 1000);
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::DepthwiseConvolution(ZQ_CNN_Tensor4D_NCHWC4& input, const ZQ_CNN_Tensor4D_NCHWC4& filters,
	int strideH, int strideW, int dilation_H, int dilation_W,
	int padH, int padW, ZQ_CNN_Tensor4D_NCHWC4& output)
{
	double t1 = omp_get_wtime();
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	int filter_N = filters.GetN();
	int filter_H = filters.GetH();
	int filter_W = filters.GetW();
	int filter_C = filters.GetC();
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| (in_H - filter_H + (padH << 1)) < 0 || (in_W - filter_W + (padW << 1)) < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}
	if (filter_C != in_C || filter_N != 1)
		return false;

	int need_N = in_N;
	// 审计修复 2026-10-02（附录 BE.2 / BH）：strideH/strideW 来自**模型文件**
	// （不可信输入），下面这行是**整数除法** —— 除以 0 在 x86 上是 idiv，
	// 直接 SIGFPE、进程死。附录 BE 只修了 NCHW 那 7 处 wrapper，
	// NCHWC 这边是同一份代码复制过去的，同样要修。
	if (strideH <= 0 || strideW <= 0)
		return false;
	int need_H = (in_H - (filter_H - 1)*dilation_H - 1 + (padH << 1)) / strideH + 1;
	int need_W = (in_W - (filter_W - 1)*dilation_W - 1 + (padW << 1)) / strideW + 1;
	int need_C = in_C;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	if (padH != 0 || padW != 0)
	{
		if (!input.Padding(padW, padH, 0))
			return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int filter_sliceStep = filters.GetSliceStep();
	int filter_widthStep = filters.GetWidthStep();
	int filter_imStep = filters.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr() - padH*in_widthStep - padW*4;
	const float* filter_firstPixelData = filters.GetFirstPixelPtr();
	float* out_firstPixelData = output.GetFirstPixelPtr();

	if (filter_H == 3 && filter_W == 3)
	{
		if (strideH == 1 && strideW == 1 && dilation_H == 1 && dilation_W == 1)
		{
			zq_cnn_depthwise_conv_no_padding_nchwc4_kernel3x3_s1d1(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
		}
		else if (strideH == 2 && strideW == 2 && dilation_H == 1 && dilation_W == 1)
		{
			zq_cnn_depthwise_conv_no_padding_nchwc4_kernel3x3_s2d1(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
		}
		else
		{
			zq_cnn_depthwise_conv_no_padding_nchwc4_kernel3x3(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
		}
	}
	else if (filter_H == 2 && filter_W == 2)
	{
		if (strideH == 1 && strideW == 1 && dilation_H == 1 && dilation_W == 1)
		{
			zq_cnn_depthwise_conv_no_padding_nchwc4_kernel2x2_s1d1(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
		}
		else
		{
			zq_cnn_depthwise_conv_no_padding_nchwc4_kernel2x2(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
		}
	}
	else if (filter_H == 5 && filter_W == 5
		&& strideH == 1 && strideW == 1 && dilation_H == 1 && dilation_W == 1)
	{
		zq_cnn_depthwise_conv_no_padding_nchwc4_kernel5x5_s1d1(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
			filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
			dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
	}
	else
	{
		zq_cnn_depthwise_conv_no_padding_nchwc4_general(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
			filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
			dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
	}

	double t2 = omp_get_wtime();
	//printf("utils:conv: %.3f ms\n", (t2 - t1) * 1000);
	return true;
}

void ZQ_CNN_Forward_SSEUtils_NCHWC::MaxPooling(ZQ_CNN_Tensor4D_NCHWC4 &input, ZQ_CNN_Tensor4D_NCHWC4 &output, int kernel_H, int kernel_W,
	int stride_H, int stride_W, int pad_H_top, int pad_H_bottom,
	int pad_W_left, int pad_W_right, bool global_pool)
{
	int in_N = input.GetN();
	int in_H0 = input.GetH();
	int in_W0 = input.GetW();
	int in_C = input.GetC();
	int need_W, need_H;
	int need_N = in_N;
	int need_C = in_C;
	if (global_pool)
	{
		need_H = 1;
		need_W = 1;
		kernel_H = in_H0;
		kernel_W = in_W0;
		stride_H = 1;
		stride_W = 1;
		// 审计 F2：global_pool 不需要 padding（NCHW 那一支也不加），
		// 显式清零，免得外层传进来的 pad 让 need/use 尺寸互相打架。
		pad_H_top = pad_H_bottom = pad_W_left = pad_W_right = 0;
	}
	else
	{
		need_W = (int)ceil((float)(in_W0 + pad_W_left + pad_W_right - kernel_W) / stride_W + 1);
		need_H = (int)ceil((float)(in_H0 + pad_H_top + pad_H_bottom - kernel_H) / stride_H + 1);
	}

	if (need_W <= 0 || need_H <= 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
			return;
		// 审计修复 2026-10-06（附录 IR.1）：这一行原来缺，NCHW 版（`ZQ_CNN_Forward_SSEUtils.h:1601`）有。
		// `ChangeSize(0,0,0,0,0,0)` 是**成功**的（把 output 置空并返回 true），
		// 所以少了它就会继续往下走到 `(in_H - kernel_H) % stride_H`：
		// `stride_H == 0` 时那是**整数 idiv 除零 -> SIGFPE**，进程直接死。
		// 触发：`stride_H <= 0 || stride_W <= 0`。
		// 经模型文件当前不可达（`ZQ_CNN_Layer_NCHWC.h:2184-2192` 的 ReadParam
		// 已有 `stride_H <= 0 -> return false`），但这 6 个是 public static，
		// 外部可直接调 —— 而 NCHW 版专门补的那行 `return ;` 就是这条路径的防线。
		// 顺带：即便 stride > 0，`need_H == 0`（in_H=3, kernel_H=5, stride_H=2）时
		// NCHWC 会把 output 置成 0 大小后**继续调内核**（内核的循环条件让它空转，
		// 实际不写内存），NCHW 则直接返回 —— 也是行为分叉。
		return;
	}

	// 审计 F2：`% stride == 0` 是**无 pad** 时的等价写法，有 pad 时不成立。
	// 直接说清它真正要表达的事：最后一个窗口不用裁边。
	const float* in_data = 0;
	int in_H = 0, in_W = 0;
	if (!zq_nchwc_pool_prepare_pad(input, pad_H_top, pad_H_bottom, pad_W_left, pad_W_right,
		in_data, in_H, in_W))
		return;
	bool suredivided = ((need_H - 1)*stride_H + kernel_H <= in_H)
		&& ((need_W - 1)*stride_W + kernel_W <= in_W);
	if (output.GetN() != need_N || output.GetH() != need_H || output.GetW() != need_W || output.GetC() != need_C)
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
		return;
	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* out_data = output.GetFirstPixelPtr();

	if (suredivided)
	{
		if (kernel_H == 2 && kernel_W == 2)
		{
			zq_cnn_maxpooling_nopadding_suredivided_nchwc4_kernel2x2(in_data, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, kernel_H, kernel_W, stride_H, stride_W,
				out_data, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
		}
		else if (kernel_H == 3 && kernel_W == 3)
		{
			zq_cnn_maxpooling_nopadding_suredivided_nchwc4_kernel3x3(in_data, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, kernel_H, kernel_W, stride_H, stride_W,
				out_data, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
		}
		else
		{
			zq_cnn_maxpooling_nopadding_suredivided_nchwc4_general(in_data, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, kernel_H, kernel_W, stride_H, stride_W,
				out_data, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
		}
	}
	else
	{
		zq_cnn_maxpooling_nopadding_nodivided_nchwc4_general(in_data, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, kernel_H, kernel_W, stride_H, stride_W,
			out_data, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
	}
}

void ZQ_CNN_Forward_SSEUtils_NCHWC::AVGPooling(ZQ_CNN_Tensor4D_NCHWC4 &input, ZQ_CNN_Tensor4D_NCHWC4 &output, int kernel_H, int kernel_W,
	int stride_H, int stride_W, int pad_H_top, int pad_H_bottom,
	int pad_W_left, int pad_W_right, bool global_pool)
{
	int in_N = input.GetN();
	int in_H0 = input.GetH();
	int in_W0 = input.GetW();
	int in_C = input.GetC();
	int need_W, need_H;
	int need_N = in_N;
	int need_C = in_C;
	if (global_pool)
	{
		need_H = 1;
		need_W = 1;
		kernel_H = in_H0;
		kernel_W = in_W0;
		stride_H = 1;
		stride_W = 1;
		// 审计 F2：global_pool 不需要 padding（NCHW 那一支也不加），
		// 显式清零，免得外层传进来的 pad 让 need/use 尺寸互相打架。
		pad_H_top = pad_H_bottom = pad_W_left = pad_W_right = 0;
	}
	else
	{
		need_W = (int)ceil((float)(in_W0 + pad_W_left + pad_W_right - kernel_W) / stride_W + 1);
		need_H = (int)ceil((float)(in_H0 + pad_H_top + pad_H_bottom - kernel_H) / stride_H + 1);
	}

	if (need_W <= 0 || need_H <= 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
			return;
		// 审计修复 2026-10-06（附录 IR.1）：这一行原来缺，NCHW 版（`ZQ_CNN_Forward_SSEUtils.h:1601`）有。
		// `ChangeSize(0,0,0,0,0,0)` 是**成功**的（把 output 置空并返回 true），
		// 所以少了它就会继续往下走到 `(in_H - kernel_H) % stride_H`：
		// `stride_H == 0` 时那是**整数 idiv 除零 -> SIGFPE**，进程直接死。
		// 触发：`stride_H <= 0 || stride_W <= 0`。
		// 经模型文件当前不可达（`ZQ_CNN_Layer_NCHWC.h:2184-2192` 的 ReadParam
		// 已有 `stride_H <= 0 -> return false`），但这 6 个是 public static，
		// 外部可直接调 —— 而 NCHW 版专门补的那行 `return ;` 就是这条路径的防线。
		// 顺带：即便 stride > 0，`need_H == 0`（in_H=3, kernel_H=5, stride_H=2）时
		// NCHWC 会把 output 置成 0 大小后**继续调内核**（内核的循环条件让它空转，
		// 实际不写内存），NCHW 则直接返回 —— 也是行为分叉。
		return;
	}

	// 审计 F2：`% stride == 0` 是**无 pad** 时的等价写法，有 pad 时不成立。
	// 直接说清它真正要表达的事：最后一个窗口不用裁边。
	const float* in_data = 0;
	int in_H = 0, in_W = 0;
	if (!zq_nchwc_pool_prepare_pad(input, pad_H_top, pad_H_bottom, pad_W_left, pad_W_right,
		in_data, in_H, in_W))
		return;
	bool suredivided = ((need_H - 1)*stride_H + kernel_H <= in_H)
		&& ((need_W - 1)*stride_W + kernel_W <= in_W);
	if (output.GetN() != need_N || output.GetH() != need_H || output.GetW() != need_W || output.GetC() != need_C)
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
		return;
	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* out_data = output.GetFirstPixelPtr();

	if (suredivided)
	{
		if (kernel_H == 2 && kernel_W == 2)
		{
			zq_cnn_avgpooling_nopadding_suredivided_nchwc4_kernel2x2(in_data, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, kernel_H, kernel_W, stride_H, stride_W,
				out_data, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
		}
		else if (kernel_H == 3 && kernel_W == 3)
		{
			zq_cnn_avgpooling_nopadding_suredivided_nchwc4_kernel3x3(in_data, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, kernel_H, kernel_W, stride_H, stride_W,
				out_data, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
		}
		else
		{
			zq_cnn_avgpooling_nopadding_suredivided_nchwc4_general(in_data, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, kernel_H, kernel_W, stride_H, stride_W,
				out_data, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
		}
	}
	else
	{
		zq_cnn_avgpooling_nopadding_nodivided_nchwc4_general(in_data, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, kernel_H, kernel_W, stride_H, stride_W,
			out_data, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
	}
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::AddBiasPReLU(ZQ_CNN_Tensor4D_NCHWC4 &input, const ZQ_CNN_Tensor4D_NCHWC4& bias, const ZQ_CNN_Tensor4D_NCHWC4& slope)
{
	int N = input.GetN();
	int H = input.GetH();
	int W = input.GetW();
	int C = input.GetC();
	if (N <= 0 || H <= 0 || W <= 0 || C <= 0)
		return true;
	if (bias.GetC() != C || slope.GetC() != C)
		return false;
	float* data = input.GetFirstPixelPtr();
	const float* bias_Data = bias.GetFirstPixelPtr();
	const float* slope_Data = slope.GetFirstPixelPtr();
	int imStep = input.GetImageStep();
	int widthStep = input.GetWidthStep();
	int sliceStep = input.GetSliceStep();
	bool sure_lessthan1 = true;
	for (int c = 0; c < C; c++)
	{
		if (slope_Data[c] > 1)
		{
			sure_lessthan1 = false;
			break;
		}
	}
	if (sure_lessthan1)
	{
		zq_cnn_addbias_prelu_nchwc4_sure_slope_lessthan1(data, N, H, W, C, widthStep, sliceStep, imStep, bias_Data, slope_Data);
	}
	else
	{
		zq_cnn_addbias_prelu_nchwc4(data, N, H, W, C, widthStep, sliceStep, imStep, bias_Data, slope_Data);
	}

	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::PReLU(ZQ_CNN_Tensor4D_NCHWC4 &input, const ZQ_CNN_Tensor4D_NCHWC4& slope)
{
	int N = input.GetN();
	int H = input.GetH();
	int W = input.GetW();
	int C = input.GetC();
	if (N <= 0 || H <= 0 || W <= 0 || C <= 0)
		return true;
	if (slope.GetC() != C)
		return false;
	float* data = input.GetFirstPixelPtr();
	const float* slope_Data = slope.GetFirstPixelPtr();
	int imStep = input.GetImageStep();
	int widthStep = input.GetWidthStep();
	int sliceStep = input.GetSliceStep();
	bool sure_lessthan1 = true;
	for (int c = 0; c < C; c++)
	{
		if (slope_Data[c] > 1)
		{
			sure_lessthan1 = false;
			break;
		}
	}
	if (sure_lessthan1)
	{
		zq_cnn_prelu_nchwc4_sure_slope_lessthan1(data, N, H, W, C, widthStep, sliceStep, imStep, slope_Data);
	}
	else
	{
		zq_cnn_prelu_nchwc4(data, N, H, W, C, widthStep, sliceStep, imStep, slope_Data);
	}

	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::ReLU(ZQ_CNN_Tensor4D_NCHWC4 &input, float slope)
{
	int N = input.GetN();
	int H = input.GetH();
	int W = input.GetW();
	int C = input.GetC();
	if (N <= 0 || H <= 0 || W <= 0 || C <= 0)
		return true;
	float* data = input.GetFirstPixelPtr();
	int imStep = input.GetImageStep();
	int widthStep = input.GetWidthStep();
	int sliceStep = input.GetSliceStep();

	zq_cnn_relu_nchwc4(data, N, H, W, C, widthStep, sliceStep, imStep, slope);

	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::Softmax(ZQ_CNN_Tensor4D_NCHWC4 &input, int axis)
{
	if (axis < 0 || axis >= 4)
		return false;
	int N = input.GetN();
	int H = input.GetH();
	int W = input.GetW();
	int C = input.GetC();
	if (N <= 0 || H <= 0 || W <= 0 || C <= 0)
		return true;
	float* data = input.GetFirstPixelPtr();
	int sliceStep = input.GetSliceStep();
	int widthStep = input.GetWidthStep();
	int imStep = input.GetImageStep();

	if (axis == 1)
		zq_cnn_softmax_nchwc4_C(data, N, H, W, C, widthStep, sliceStep, imStep);
	else
		return false;
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::BatchNorm_b_a(ZQ_CNN_Tensor4D_NCHWC4 &input, const ZQ_CNN_Tensor4D_NCHWC4& b, const ZQ_CNN_Tensor4D_NCHWC4& a)
{
	int N = input.GetN();
	int H = input.GetH();
	int W = input.GetW();
	int C = input.GetC();
	if (N <= 0 || H <= 0 || W <= 0 || C <= 0)
		return true;
	if (b.GetC() != C || a.GetC() != C)
		return false;
	float* data = input.GetFirstPixelPtr();
	const float* b_data = b.GetFirstPixelPtr();
	const float* a_data = a.GetFirstPixelPtr();
	int widthStep = input.GetWidthStep();
	int sliceStep = input.GetSliceStep();
	int imStep = input.GetImageStep();

	zq_cnn_batchnorm_b_a_nchwc4(data, N, H, W, C, widthStep, sliceStep, imStep, b_data, a_data);
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::Eltwise_Sum(const std::vector<const ZQ_CNN_Tensor4D_NCHWC4*>& input, ZQ_CNN_Tensor4D_NCHWC4& output)
{
	int in_num = (int)input.size();
	if (in_num < 2)
		return false;
	for (int i = 0; i < in_num; i++)
	{
		if (input[i] == 0) return false;
	}
	int N = input[0]->GetN();
	int H = input[0]->GetH();
	int W = input[0]->GetW();
	int C = input[0]->GetC();
	for (int i = 1; i < in_num; i++)
	{
		if (N != input[i]->GetN() || H != input[i]->GetH() || W != input[i]->GetW() || C != input[i]->GetC())
			return false;
	}
	if (output.GetN() != N || output.GetH() != H || output.GetW() != W || output.GetC() != C)
		if (!output.ChangeSize(N, H, W, C, 0, 0))
		
return false;
	if (N <= 0 || H <= 0 || W <= 0 || C <= 0)
		return true;
	std::vector<const float*> in_tensor_data(in_num);
	std::vector<int> in_imStep(in_num), in_widthStep(in_num), in_sliceStep(in_num);
	float* out_data = output.GetFirstPixelPtr();
	for (int i = 0; i < in_num; i++)
	{
		in_tensor_data[i] = input[i]->GetFirstPixelPtr();
		in_imStep[i] = input[i]->GetImageStep();
		in_widthStep[i] = input[i]->GetWidthStep();
		in_sliceStep[i] = input[i]->GetSliceStep();
	}
	int out_imStep = output.GetImageStep(), out_widthStep = output.GetWidthStep(), out_sliceStep = output.GetSliceStep();

	zq_cnn_eltwise_sum_nchwc4(in_num, &in_tensor_data[0], N, H, W, C, &in_widthStep[0], &in_sliceStep[0], &in_imStep[0],
		out_data, out_widthStep, out_sliceStep, out_imStep);

	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::Eltwise_SumWithWeight(const std::vector<const ZQ_CNN_Tensor4D_NCHWC4*>& input, const std::vector<float>& weight,
	ZQ_CNN_Tensor4D_NCHWC4& output)
{
	int in_num = (int)input.size();
	if (in_num < 2 || in_num != weight.size())
		return false;
	for (int i = 0; i < in_num; i++)
	{
		if (input[i] == 0) return false;
	}
	int N = input[0]->GetN();
	int H = input[0]->GetH();
	int W = input[0]->GetW();
	int C = input[0]->GetC();
	for (int i = 1; i < in_num; i++)
	{
		if (N != input[i]->GetN() || H != input[i]->GetH() || W != input[i]->GetW() || C != input[i]->GetC())
			return false;
	}
	if (output.GetN() != N || output.GetH() != H || output.GetW() != W || output.GetC() != C)
		if (!output.ChangeSize(N, H, W, C, 0, 0))
		
return false;
	if (N <= 0 || H <= 0 || W <= 0 || C <= 0)
		return true;
	std::vector<const float*> in_tensor_data(in_num);
	std::vector<int> in_imStep(in_num), in_widthStep(in_num), in_sliceStep(in_num);
	float* out_data = output.GetFirstPixelPtr();
	for (int i = 0; i < in_num; i++)
	{
		in_tensor_data[i] = input[i]->GetFirstPixelPtr();
		in_imStep[i] = input[i]->GetImageStep();
		in_widthStep[i] = input[i]->GetWidthStep();
		in_sliceStep[i] = input[i]->GetSliceStep();
	}
	int out_imStep = output.GetImageStep(), out_widthStep = output.GetWidthStep(), out_sliceStep = output.GetSliceStep();

	zq_cnn_eltwise_sum_with_weight_nchwc4(in_num, &in_tensor_data[0], &weight[0], N, H, W, C,
		&in_widthStep[0], &in_sliceStep[0], &in_imStep[0], out_data, out_widthStep, out_sliceStep, out_imStep);
	return true;
}


bool ZQ_CNN_Forward_SSEUtils_NCHWC::Eltwise_Mul(const std::vector<const ZQ_CNN_Tensor4D_NCHWC4*>& input, ZQ_CNN_Tensor4D_NCHWC4& output)
{
	int in_num = (int)input.size();
	if (in_num < 2)
		return false;
	for (int i = 0; i < in_num; i++)
	{
		if (input[i] == 0) return false;
	}
	int N = input[0]->GetN();
	int H = input[0]->GetH();
	int W = input[0]->GetW();
	int C = input[0]->GetC();
	for (int i = 1; i < in_num; i++)
	{
		if (N != input[i]->GetN() || H != input[i]->GetH() || W != input[i]->GetW() || C != input[i]->GetC())
			return false;
	}
	if (output.GetN() != N || output.GetH() != H || output.GetW() != W || output.GetC() != C)
		if (!output.ChangeSize(N, H, W, C, 0, 0))
		
return false;
	if (N <= 0 || H <= 0 || W <= 0 || C <= 0)
		return true;
	std::vector<const float*> in_tensor_data(in_num);
	std::vector<int> in_imStep(in_num), in_widthStep(in_num), in_sliceStep(in_num);
	float* out_data = output.GetFirstPixelPtr();
	for (int i = 0; i < in_num; i++)
	{
		in_tensor_data[i] = input[i]->GetFirstPixelPtr();
		in_imStep[i] = input[i]->GetImageStep();
		in_widthStep[i] = input[i]->GetWidthStep();
		in_sliceStep[i] = input[i]->GetSliceStep();
	}
	int out_imStep = output.GetImageStep(), out_widthStep = output.GetWidthStep(), out_sliceStep = output.GetSliceStep();

	zq_cnn_eltwise_mul_nchwc4(in_num, &in_tensor_data[0], N, H, W, C, &in_widthStep[0], &in_sliceStep[0], &in_imStep[0],
		out_data, out_widthStep, out_sliceStep, out_imStep);
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::Eltwise_Max(const std::vector<const ZQ_CNN_Tensor4D_NCHWC4*>& input, ZQ_CNN_Tensor4D_NCHWC4& output)
{
	int in_num = (int)input.size();
	if (in_num < 2)
		return false;
	for (int i = 0; i < in_num; i++)
	{
		if (input[i] == 0) return false;
	}
	int N = input[0]->GetN();
	int H = input[0]->GetH();
	int W = input[0]->GetW();
	int C = input[0]->GetC();
	for (int i = 1; i < in_num; i++)
	{
		if (N != input[i]->GetN() || H != input[i]->GetH() || W != input[i]->GetW() || C != input[i]->GetC())
			return false;
	}
	if (output.GetN() != N || output.GetH() != H || output.GetW() != W || output.GetC() != C)
		if (!output.ChangeSize(N, H, W, C, 0, 0))
		
return false;
	if (N <= 0 || H <= 0 || W <= 0 || C <= 0)
		return true;
	std::vector<const float*> in_tensor_data(in_num);
	std::vector<int> in_imStep(in_num), in_widthStep(in_num), in_sliceStep(in_num);
	float* out_data = output.GetFirstPixelPtr();
	for (int i = 0; i < in_num; i++)
	{
		in_tensor_data[i] = input[i]->GetFirstPixelPtr();
		in_imStep[i] = input[i]->GetImageStep();
		in_widthStep[i] = input[i]->GetWidthStep();
		in_sliceStep[i] = input[i]->GetSliceStep();
	}
	int out_imStep = output.GetImageStep(), out_widthStep = output.GetWidthStep(), out_sliceStep = output.GetSliceStep();

	zq_cnn_eltwise_max_nchwc4(in_num, &in_tensor_data[0], N, H, W, C, &in_widthStep[0], &in_sliceStep[0], &in_imStep[0],
		out_data, out_widthStep, out_sliceStep, out_imStep);
	return true;
}

#endif // __ARM_NEON || (ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE)



#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX

bool ZQ_CNN_Forward_SSEUtils_NCHWC::InnerProductWithBias(ZQ_CNN_Tensor4D_NCHWC8& input, const ZQ_CNN_Tensor4D_NCHWC8& filters,
	const ZQ_CNN_Tensor4D_NCHWC8& bias, ZQ_CNN_Tensor4D_NCHWC8& output,
	void** buffer, __int64* buffer_len)
{
	double t1 = omp_get_wtime();
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	int filter_N = filters.GetN();
	int filter_H = filters.GetH();
	int filter_W = filters.GetW();
	int filter_C = filters.GetC();
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	int bias_C = bias.GetC();
	int need_H = 1;
	int need_W = 1;
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| need_H < 0 || need_W < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}
	if (filter_C != in_C || filter_N != bias_C)
		return false;

	int need_N = in_N;

	int need_C = filter_N;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int filter_sliceStep = filters.GetSliceStep();
	int filter_widthStep = filters.GetWidthStep();
	int filter_imStep = filters.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr();
	const float* filter_firstPixelData = filters.GetFirstPixelPtr();
	float* out_firstPixelData = output.GetFirstPixelPtr();
	const float* bias_firstPixelData = bias.GetFirstPixelPtr();

	

#if (ZQ_CNN_USE_BLAS_GEMM || ZQ_CNN_USE_MKL_GEMM || ZQ_CNN_USE_ZQ_GEMM)
	if (out_N >= 16 && filter_N >= 16)
	{
		zq_cnn_innerproduct_gemm_nchwc8_general_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
			filter_widthStep, filter_sliceStep, filter_imStep,
			out_firstPixelData, need_N, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, buffer, buffer_len);

	}
	else
#endif
	{
		if (8 * in_W == in_widthStep && in_widthStep*in_H == in_sliceStep
			&& 8 * in_W == filter_widthStep && filter_widthStep*in_H == filter_sliceStep
			&& 8 * out_W == out_widthStep && out_widthStep * out_H == out_sliceStep)
		{
			zq_cnn_innerproduct_nchwc8_noborder_with_bias(in_firstPixelData, in_N, in_H*in_W*in_C, filter_firstPixelData, filter_N,
				// 审计修复 2026-10-02（附录 BN.1）：传 out_sliceStep 时，N>1 的结果会
				// 互相覆盖（out 是 [N,1,1,K]，sliceStep 与 imStep 差 K 倍）。
				out_firstPixelData, out_imStep, bias_firstPixelData);
		}
		else
		{
			zq_cnn_innerproduct_gemm_nchwc8_general_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep,
				out_firstPixelData, need_N, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, buffer, buffer_len);
		}
	}
	double t2 = omp_get_wtime();
	//printf("utils:conv: %.3f ms\n", (t2 - t1) * 1000);
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::InnerProductWithBiasPReLU(ZQ_CNN_Tensor4D_NCHWC8& input, const ZQ_CNN_Tensor4D_NCHWC8& filters,
	const ZQ_CNN_Tensor4D_NCHWC8& bias, const ZQ_CNN_Tensor4D_NCHWC8& slope, ZQ_CNN_Tensor4D_NCHWC8& output,
	void** buffer, __int64* buffer_len)
{
	double t1 = omp_get_wtime();
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	int filter_N = filters.GetN();
	int filter_H = filters.GetH();
	int filter_W = filters.GetW();
	int filter_C = filters.GetC();
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	int bias_C = bias.GetC();
	int slope_C = slope.GetC();
	int need_H = 1;
	int need_W = 1;
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| need_H < 0 || need_W < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}
	if (filter_C != in_C || filter_N != bias_C || filter_N != slope_C)
		return false;

	int need_N = in_N;

	int need_C = filter_N;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int filter_sliceStep = filters.GetSliceStep();
	int filter_widthStep = filters.GetWidthStep();
	int filter_imStep = filters.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr();
	const float* filter_firstPixelData = filters.GetFirstPixelPtr();
	float* out_firstPixelData = output.GetFirstPixelPtr();
	const float* bias_firstPixelData = bias.GetFirstPixelPtr();
	const float* slope_firstPixelData = slope.GetFirstPixelPtr();


#if (ZQ_CNN_USE_BLAS_GEMM || ZQ_CNN_USE_MKL_GEMM || ZQ_CNN_USE_ZQ_GEMM)
	if (out_N >= 16 && filter_N >= 16)
	{
		zq_cnn_innerproduct_gemm_nchwc8_general_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
			filter_widthStep, filter_sliceStep, filter_imStep,
			out_firstPixelData, need_N, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData, buffer, buffer_len);

	}
	else
#endif
	{
		if (8 * in_W == in_widthStep && in_widthStep*in_H == in_sliceStep
			&& 8 * in_W == filter_widthStep && filter_widthStep*in_H == filter_sliceStep
			&& 8 * out_W == out_widthStep && out_widthStep * out_H == out_sliceStep)
		{
			zq_cnn_innerproduct_nchwc8_noborder_with_bias_prelu(in_firstPixelData, in_N, in_H*in_W*in_C, filter_firstPixelData, filter_N,
				// 审计修复 2026-10-02（附录 BN.1）：传 out_sliceStep 时，N>1 的结果会
				// 互相覆盖（out 是 [N,1,1,K]，sliceStep 与 imStep 差 K 倍）。
				out_firstPixelData, out_imStep, bias_firstPixelData, slope_firstPixelData);
		}
		else
		{
			zq_cnn_innerproduct_gemm_nchwc8_general_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep,
				out_firstPixelData, need_N, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData, buffer, buffer_len);
		}
	}
	double t2 = omp_get_wtime();
	//printf("utils:conv: %.3f ms\n", (t2 - t1) * 1000);
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::InnerProductWithPReLU(ZQ_CNN_Tensor4D_NCHWC8& input, const ZQ_CNN_Tensor4D_NCHWC8& filters,
	const ZQ_CNN_Tensor4D_NCHWC8& slope, ZQ_CNN_Tensor4D_NCHWC8& output,
	void** buffer, __int64* buffer_len)
{
	double t1 = omp_get_wtime();
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	int filter_N = filters.GetN();
	int filter_H = filters.GetH();
	int filter_W = filters.GetW();
	int filter_C = filters.GetC();
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	int slope_C = slope.GetC();
	int need_H = 1;
	int need_W = 1;
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| need_H < 0 || need_W < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}
	if (filter_C != in_C || filter_N != slope_C)
		return false;

	int need_N = in_N;

	int need_C = filter_N;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int filter_sliceStep = filters.GetSliceStep();
	int filter_widthStep = filters.GetWidthStep();
	int filter_imStep = filters.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr();
	const float* filter_firstPixelData = filters.GetFirstPixelPtr();
	float* out_firstPixelData = output.GetFirstPixelPtr();
	const float* slope_firstPixelData = slope.GetFirstPixelPtr();


#if (ZQ_CNN_USE_BLAS_GEMM || ZQ_CNN_USE_MKL_GEMM || ZQ_CNN_USE_ZQ_GEMM)
	if (out_N >= 16 && filter_N >= 16)
	{
		zq_cnn_innerproduct_gemm_nchwc8_general(in_firstPixelData, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
			filter_widthStep, filter_sliceStep, filter_imStep,
			out_firstPixelData, need_N, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);

	}
	else
#endif
	{
		if (8 * in_W == in_widthStep && in_widthStep*in_H == in_sliceStep
			&& 8 * in_W == filter_widthStep && filter_widthStep*in_H == filter_sliceStep
			&& 8 * out_W == out_widthStep && out_widthStep * out_H == out_sliceStep)
		{
			zq_cnn_innerproduct_nchwc8_noborder(in_firstPixelData, in_N, in_H*in_W*in_C, filter_firstPixelData, filter_N,
				// 审计修复 2026-10-02（附录 BN.1）：传 out_sliceStep 时，N>1 的结果会
				// 互相覆盖（out 是 [N,1,1,K]，sliceStep 与 imStep 差 K 倍）。
				out_firstPixelData, out_imStep);
		}
		else
		{
			zq_cnn_innerproduct_gemm_nchwc8_general(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep,
				out_firstPixelData, need_N, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
		}
	}
	zq_cnn_prelu_nchwc8(out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, slope_firstPixelData);

	double t2 = omp_get_wtime();
	//printf("utils:conv: %.3f ms\n", (t2 - t1) * 1000);
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::InnerProduct(ZQ_CNN_Tensor4D_NCHWC8& input, const ZQ_CNN_Tensor4D_NCHWC8& filters,
	ZQ_CNN_Tensor4D_NCHWC8& output, void** buffer, __int64* buffer_len)
{
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	int filter_N = filters.GetN();
	int filter_H = filters.GetH();
	int filter_W = filters.GetW();
	int filter_C = filters.GetC();
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	int need_H = 1;
	int need_W = 1;
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| need_H < 0 || need_W < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}
	if (filter_C != in_C)
		return false;

	int need_N = in_N;
	int need_C = filter_N;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int filter_sliceStep = filters.GetSliceStep();
	int filter_widthStep = filters.GetWidthStep();
	int filter_imStep = filters.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr();
	const float* filter_firstPixelData = filters.GetFirstPixelPtr();
	float* out_firstPixelData = output.GetFirstPixelPtr();


#if (ZQ_CNN_USE_BLAS_GEMM || ZQ_CNN_USE_MKL_GEMM || ZQ_CNN_USE_ZQ_GEMM)
	if (out_N >= 16 && filter_N >= 16)
	{
		zq_cnn_innerproduct_gemm_nchwc8_general(in_firstPixelData, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
			filter_widthStep, filter_sliceStep, filter_imStep,
			out_firstPixelData, need_N, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);

	}
	else
#endif
	{
		if (8 * in_W == in_widthStep && in_widthStep*in_H == in_sliceStep
			&& 8 * in_W == filter_widthStep && filter_widthStep*in_H == filter_sliceStep
			&& 8 * out_W == out_widthStep && out_widthStep * out_H == out_sliceStep)
		{
			zq_cnn_innerproduct_nchwc8_noborder(in_firstPixelData, in_N, in_H*in_W*in_C, filter_firstPixelData, filter_N,
				// 审计修复 2026-10-02（附录 BN.1）：传 out_sliceStep 时，N>1 的结果会
				// 互相覆盖（out 是 [N,1,1,K]，sliceStep 与 imStep 差 K 倍）。
				out_firstPixelData, out_imStep);
		}
		else
		{
			zq_cnn_innerproduct_gemm_nchwc8_general(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep,
				out_firstPixelData, need_N, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
		}
	}
	return true;
}


bool ZQ_CNN_Forward_SSEUtils_NCHWC::ConvolutionWithBias(ZQ_CNN_Tensor4D_NCHWC8& input, const ZQ_CNN_Tensor4D_NCHWC8& filters,
	const ZQ_CNN_Tensor4D_NCHWC8& bias, int strideH, int strideW, int dilation_H, int dilation_W, int padH, int padW,
	ZQ_CNN_Tensor4D_NCHWC8& output, void** buffer, __int64* buffer_len)
{
	double t1 = omp_get_wtime();
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	int filter_N = filters.GetN();
	int filter_H = filters.GetH();
	int filter_W = filters.GetW();
	int filter_C = filters.GetC();
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	int bias_C = bias.GetC();
	// 审计修复 2026-10-02（附录 BE.2 / BH）：strideH/strideW 来自**模型文件**
	// （不可信输入），下面这行是**整数除法** —— 除以 0 在 x86 上是 idiv，
	// 直接 SIGFPE、进程死。附录 BE 只修了 NCHW 那 7 处 wrapper，
	// NCHWC 这边是同一份代码复制过去的，同样要修。
	if (strideH <= 0 || strideW <= 0)
		return false;
	int need_H = (in_H - (filter_H - 1)*dilation_H - 1 + (padH << 1)) / strideH + 1;
	int need_W = (in_W - (filter_W - 1)*dilation_W - 1 + (padW << 1)) / strideW + 1;
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| need_H < 0 || need_W < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}
	if (filter_C != in_C || filter_N != bias_C)
		return false;

	int need_N = in_N;

	int need_C = filter_N;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	if (padH != 0 || padW != 0)
	{
		if (!input.Padding(padW, padH, 0))
			return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int filter_sliceStep = filters.GetSliceStep();
	int filter_widthStep = filters.GetWidthStep();
	int filter_imStep = filters.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr() - padH*in_widthStep - padW * 8;
	const float* filter_firstPixelData = filters.GetFirstPixelPtr();
	float* out_firstPixelData = output.GetFirstPixelPtr();
	const float* bias_firstPixelData = bias.GetFirstPixelPtr();

	if (filter_H == 3 && filter_W == 3)
	{
		if (filter_C == 3)
		{
			zq_cnn_conv_no_padding_gemm_nchwc8_kernel3x3_C3_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, buffer, buffer_len);
		}
		else
		{
			zq_cnn_conv_no_padding_gemm_nchwc8_kernel3x3_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, buffer, buffer_len);
		}
	}
	else if (filter_H == 2 && filter_W == 2)
	{
		if (filter_C == 3)
		{
			zq_cnn_conv_no_padding_gemm_nchwc8_kernel2x2_C3_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, buffer, buffer_len);
		}
		else
		{
			zq_cnn_conv_no_padding_gemm_nchwc8_kernel2x2_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, buffer, buffer_len);
		}
	}
	else if (filter_H == 1 && filter_W == 1)
	{
		zq_cnn_conv_no_padding_gemm_nchwc8_kernel1x1_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
			filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
			out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, buffer, buffer_len);
	}
	else
	{
		zq_cnn_conv_no_padding_gemm_nchwc8_general_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
			filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
			out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, buffer, buffer_len);
	}

	double t2 = omp_get_wtime();
	//printf("utils:conv: %.3f ms\n", (t2 - t1) * 1000);
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::ConvolutionWithBiasPReLU(ZQ_CNN_Tensor4D_NCHWC8& input, const ZQ_CNN_Tensor4D_NCHWC8& filters,
	const ZQ_CNN_Tensor4D_NCHWC8& bias, const ZQ_CNN_Tensor4D_NCHWC8& slope, int strideH, int strideW,
	int dilation_H, int dilation_W, int padH, int padW, ZQ_CNN_Tensor4D_NCHWC8& output,
	void** buffer, __int64* buffer_len)
{
	double t1 = omp_get_wtime();
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	int filter_N = filters.GetN();
	int filter_H = filters.GetH();
	int filter_W = filters.GetW();
	int filter_C = filters.GetC();
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	int bias_C = bias.GetC();
	int slope_C = slope.GetC();
	// 审计修复 2026-10-02（附录 BE.2 / BH）：strideH/strideW 来自**模型文件**
	// （不可信输入），下面这行是**整数除法** —— 除以 0 在 x86 上是 idiv，
	// 直接 SIGFPE、进程死。附录 BE 只修了 NCHW 那 7 处 wrapper，
	// NCHWC 这边是同一份代码复制过去的，同样要修。
	if (strideH <= 0 || strideW <= 0)
		return false;
	int need_H = (in_H - (filter_H - 1)*dilation_H - 1 + (padH << 1)) / strideH + 1;
	int need_W = (in_W - (filter_W - 1)*dilation_W - 1 + (padW << 1)) / strideW + 1;
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| need_H < 0 || need_W < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}
	if (filter_C != in_C || filter_N != bias_C || filter_N != slope_C)
		return false;

	int need_N = in_N;

	int need_C = filter_N;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	if (padH != 0 || padW != 0)
	{
		if (!input.Padding(padW, padH, 0))
			return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int filter_sliceStep = filters.GetSliceStep();
	int filter_widthStep = filters.GetWidthStep();
	int filter_imStep = filters.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr() - padH*in_widthStep - padW * 8;
	const float* filter_firstPixelData = filters.GetFirstPixelPtr();
	float* out_firstPixelData = output.GetFirstPixelPtr();
	const float* bias_firstPixelData = bias.GetFirstPixelPtr();
	const float* slope_firstPixelData = slope.GetFirstPixelPtr();

	if (filter_H == 3 && filter_W == 3)
	{
		if (filter_C == 3)
		{
			zq_cnn_conv_no_padding_gemm_nchwc8_kernel3x3_C3_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData, buffer, buffer_len);
		}
		else
		{
			zq_cnn_conv_no_padding_gemm_nchwc8_kernel3x3_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData, buffer, buffer_len);
		}
	}
	else if (filter_H == 2 && filter_W == 2)
	{
		if (filter_C == 3)
		{
			zq_cnn_conv_no_padding_gemm_nchwc8_kernel2x2_C3_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData, buffer, buffer_len);
		}
		else
		{
			zq_cnn_conv_no_padding_gemm_nchwc8_kernel2x2_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData, buffer, buffer_len);
		}
	}
	else if (filter_H == 1 && filter_W == 1)
	{
		zq_cnn_conv_no_padding_gemm_nchwc8_kernel1x1_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
			filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
			out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData, buffer, buffer_len);
	}
	else
	{
		zq_cnn_conv_no_padding_gemm_nchwc8_general_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
			filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
			out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData, buffer, buffer_len);
	}

	double t2 = omp_get_wtime();
	//printf("utils:conv: %.3f ms\n", (t2 - t1) * 1000);
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::ConvolutionWithPReLU(ZQ_CNN_Tensor4D_NCHWC8& input, const ZQ_CNN_Tensor4D_NCHWC8& filters,
	const ZQ_CNN_Tensor4D_NCHWC8& slope, int strideH, int strideW, int dilation_H, int dilation_W, int padH, int padW,
	ZQ_CNN_Tensor4D_NCHWC8& output, void** buffer, __int64* buffer_len)
{
	double t1 = omp_get_wtime();
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	int filter_N = filters.GetN();
	int filter_H = filters.GetH();
	int filter_W = filters.GetW();
	int filter_C = filters.GetC();
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	int slope_C = slope.GetC();
	// 审计修复 2026-10-02（附录 BE.2 / BH）：strideH/strideW 来自**模型文件**
	// （不可信输入），下面这行是**整数除法** —— 除以 0 在 x86 上是 idiv，
	// 直接 SIGFPE、进程死。附录 BE 只修了 NCHW 那 7 处 wrapper，
	// NCHWC 这边是同一份代码复制过去的，同样要修。
	if (strideH <= 0 || strideW <= 0)
		return false;
	int need_H = (in_H - (filter_H - 1)*dilation_H - 1 + (padH << 1)) / strideH + 1;
	int need_W = (in_W - (filter_W - 1)*dilation_W - 1 + (padW << 1)) / strideW + 1;
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| need_H < 0 || need_W < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}
	if (filter_C != in_C || filter_N != slope_C)
		return false;

	int need_N = in_N;

	int need_C = filter_N;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	if (padH != 0 || padW != 0)
	{
		if (!input.Padding(padW, padH, 0))
			return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int filter_sliceStep = filters.GetSliceStep();
	int filter_widthStep = filters.GetWidthStep();
	int filter_imStep = filters.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr() - padH*in_widthStep - padW * 8;
	const float* filter_firstPixelData = filters.GetFirstPixelPtr();
	float* out_firstPixelData = output.GetFirstPixelPtr();
	const float* slope_firstPixelData = slope.GetFirstPixelPtr();


	if (filter_H == 3 && filter_W == 3)
	{
		if (filter_C == 3)
		{
			zq_cnn_conv_no_padding_gemm_nchwc8_kernel3x3_C3(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
		}
		else
		{
			zq_cnn_conv_no_padding_gemm_nchwc8_kernel3x3(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
		}
	}
	else if (filter_H == 2 && filter_W == 2)
	{
		if (filter_C == 3)
		{
			zq_cnn_conv_no_padding_gemm_nchwc8_kernel2x2_C3(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
		}
		else
		{
			zq_cnn_conv_no_padding_gemm_nchwc8_kernel2x2(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
		}
	}
	else if (filter_H == 1 && filter_W == 1)
	{
		zq_cnn_conv_no_padding_gemm_nchwc8_kernel1x1(in_firstPixelData, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
			filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
			out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
	}
	else
	{
		zq_cnn_conv_no_padding_gemm_nchwc8_general(in_firstPixelData, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
			filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
			out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
	}

	zq_cnn_prelu_nchwc8(out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, slope_firstPixelData);

	double t2 = omp_get_wtime();
	//printf("utils:conv: %.3f ms\n", (t2 - t1) * 1000);
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::Convolution(ZQ_CNN_Tensor4D_NCHWC8& input, const ZQ_CNN_Tensor4D_NCHWC8& filters, int strideH, int strideW,
	int dilation_H, int dilation_W, int padH, int padW,
	ZQ_CNN_Tensor4D_NCHWC8& output, void** buffer, __int64* buffer_len)
{
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	int filter_N = filters.GetN();
	int filter_H = filters.GetH();
	int filter_W = filters.GetW();
	int filter_C = filters.GetC();
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	// 审计修复 2026-10-02（附录 BE.2 / BH）：strideH/strideW 来自**模型文件**
	// （不可信输入），下面这行是**整数除法** —— 除以 0 在 x86 上是 idiv，
	// 直接 SIGFPE、进程死。附录 BE 只修了 NCHW 那 7 处 wrapper，
	// NCHWC 这边是同一份代码复制过去的，同样要修。
	if (strideH <= 0 || strideW <= 0)
		return false;
	int need_H = (in_H - (filter_H - 1)*dilation_H - 1 + (padH << 1)) / strideH + 1;
	int need_W = (in_W - (filter_W - 1)*dilation_W - 1 + (padW << 1)) / strideW + 1;
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| need_H < 0 || need_W < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}
	if (filter_C != in_C)
		return false;

	int need_N = in_N;
	int need_C = filter_N;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	if (padH != 0 || padW != 0)
	{
		if (!input.Padding(padW, padH, 0))
			return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int filter_sliceStep = filters.GetSliceStep();
	int filter_widthStep = filters.GetWidthStep();
	int filter_imStep = filters.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr() - padH*in_widthStep - padW * 8;
	const float* filter_firstPixelData = filters.GetFirstPixelPtr();
	float* out_firstPixelData = output.GetFirstPixelPtr();

	if (filter_H == 3 && filter_W == 3)
	{
		if (filter_C == 3)
		{
			zq_cnn_conv_no_padding_gemm_nchwc8_kernel3x3_C3(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
		}
		else
		{
			zq_cnn_conv_no_padding_gemm_nchwc8_kernel3x3(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
		}
	}
	else if (filter_H == 2 && filter_W == 2)
	{
		if (filter_C == 3)
		{
			zq_cnn_conv_no_padding_gemm_nchwc8_kernel2x2_C3(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
		}
		else
		{
			zq_cnn_conv_no_padding_gemm_nchwc8_kernel2x2(in_firstPixelData, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
				filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
				out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
		}
	}
	else if (filter_H == 1 && filter_W == 1)
	{
		zq_cnn_conv_no_padding_gemm_nchwc8_kernel1x1(in_firstPixelData, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
			filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
			out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
	}
	else
	{
		zq_cnn_conv_no_padding_gemm_nchwc8_general(in_firstPixelData, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, filter_firstPixelData, filter_N, filter_H, filter_W, filter_C,
			filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW, dilation_H, dilation_W,
			out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, buffer, buffer_len);
	}

	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::DepthwiseConvolutionWithBias(ZQ_CNN_Tensor4D_NCHWC8& input, const ZQ_CNN_Tensor4D_NCHWC8& filters,
	const ZQ_CNN_Tensor4D_NCHWC8& bias, int strideH, int strideW, int dilation_H, int dilation_W,
	int padH, int padW, ZQ_CNN_Tensor4D_NCHWC8& output)
{
	double t1 = omp_get_wtime();
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	int filter_N = filters.GetN();
	int filter_H = filters.GetH();
	int filter_W = filters.GetW();
	int filter_C = filters.GetC();
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	float bias_C = (float)bias.GetC();
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| (in_H - filter_H + (padH << 1)) < 0 || (in_W - filter_W + (padW << 1)) < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}
	if (filter_C != in_C || filter_N != 1)
		return false;

	int need_N = in_N;
	// 审计修复 2026-10-02（附录 BE.2 / BH）：strideH/strideW 来自**模型文件**
	// （不可信输入），下面这行是**整数除法** —— 除以 0 在 x86 上是 idiv，
	// 直接 SIGFPE、进程死。附录 BE 只修了 NCHW 那 7 处 wrapper，
	// NCHWC 这边是同一份代码复制过去的，同样要修。
	if (strideH <= 0 || strideW <= 0)
		return false;
	int need_H = (in_H - (filter_H - 1)*dilation_H - 1 + (padH << 1)) / strideH + 1;
	int need_W = (in_W - (filter_W - 1)*dilation_W - 1 + (padW << 1)) / strideW + 1;
	int need_C = in_C;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	if (padH != 0 || padW != 0)
	{
		if (!input.Padding(padW, padH, 0))
			return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int filter_sliceStep = filters.GetSliceStep();
	int filter_widthStep = filters.GetWidthStep();
	int filter_imStep = filters.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr() - padH*in_widthStep - padW * 8;
	const float* filter_firstPixelData = filters.GetFirstPixelPtr();
	float* out_firstPixelData = output.GetFirstPixelPtr();
	const float* bias_firstPixelData = bias.GetFirstPixelPtr();

	if (filter_H == 3 && filter_W == 3)
	{
		if (strideH == 1 && strideW == 1 && dilation_H == 1 && dilation_W == 1)
		{
			zq_cnn_depthwise_conv_no_padding_nchwc8_kernel3x3_s1d1_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData);
		}
		else if (strideH == 2 && strideW == 2 && dilation_H == 1 && dilation_W == 1)
		{
			zq_cnn_depthwise_conv_no_padding_nchwc8_kernel3x3_s2d1_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData);
		}
		else
		{
			zq_cnn_depthwise_conv_no_padding_nchwc8_kernel3x3_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData);
		}
	}
	else if (filter_H == 2 && filter_W == 2)
	{
		if (strideH == 1 && strideW == 1 && dilation_H == 1 && dilation_W == 1)
		{
			zq_cnn_depthwise_conv_no_padding_nchwc8_kernel2x2_s1d1_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData);
		}
		else
		{
			zq_cnn_depthwise_conv_no_padding_nchwc8_kernel2x2_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData);
		}
	}
	else if (filter_H == 5 && filter_W == 5
		&& strideH == 1 && strideW == 1 && dilation_H == 1 && dilation_W == 1)
	{
		zq_cnn_depthwise_conv_no_padding_nchwc8_kernel5x5_s1d1_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
			filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
			dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData);
	}
	else
	{
		zq_cnn_depthwise_conv_no_padding_nchwc8_general_with_bias(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
			filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
			dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData);
	}

	double t2 = omp_get_wtime();
	//printf("utils:conv: %.3f ms\n", (t2 - t1) * 1000);
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::DepthwiseConvolutionWithBiasPReLU(ZQ_CNN_Tensor4D_NCHWC8& input, const ZQ_CNN_Tensor4D_NCHWC8& filters,
	const ZQ_CNN_Tensor4D_NCHWC8& bias, const ZQ_CNN_Tensor4D_NCHWC8& prelu_slope, int strideH, int strideW, int dilation_H, int dilation_W,
	int padH, int padW, ZQ_CNN_Tensor4D_NCHWC8& output)
{
	double t1 = omp_get_wtime();
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	int filter_N = filters.GetN();
	int filter_H = filters.GetH();
	int filter_W = filters.GetW();
	int filter_C = filters.GetC();
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	float bias_C = (float)bias.GetC();
	float slope_C = (float)prelu_slope.GetC();
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| (in_H - filter_H + (padH << 1)) < 0 || (in_W - filter_W + (padW << 1)) < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}
	if (filter_C != in_C || filter_N != 1)
		return false;

	int need_N = in_N;
	// 审计修复 2026-10-02（附录 BE.2 / BH）：strideH/strideW 来自**模型文件**
	// （不可信输入），下面这行是**整数除法** —— 除以 0 在 x86 上是 idiv，
	// 直接 SIGFPE、进程死。附录 BE 只修了 NCHW 那 7 处 wrapper，
	// NCHWC 这边是同一份代码复制过去的，同样要修。
	if (strideH <= 0 || strideW <= 0)
		return false;
	int need_H = (in_H - (filter_H - 1)*dilation_H - 1 + (padH << 1)) / strideH + 1;
	int need_W = (in_W - (filter_W - 1)*dilation_W - 1 + (padW << 1)) / strideW + 1;
	int need_C = in_C;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	if (padH != 0 || padW != 0)
	{
		if (!input.Padding(padW, padH, 0))
			return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int filter_sliceStep = filters.GetSliceStep();
	int filter_widthStep = filters.GetWidthStep();
	int filter_imStep = filters.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr() - padH*in_widthStep - padW * 8;
	const float* filter_firstPixelData = filters.GetFirstPixelPtr();
	float* out_firstPixelData = output.GetFirstPixelPtr();
	const float* bias_firstPixelData = bias.GetFirstPixelPtr();
	const float* slope_firstPixelData = prelu_slope.GetFirstPixelPtr();

	if (filter_H == 3 && filter_W == 3)
	{
		if (strideH == 1 && strideW == 1 && dilation_H == 1 && dilation_W == 1)
		{
			zq_cnn_depthwise_conv_no_padding_nchwc8_kernel3x3_s1d1_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData);
		}
		else if (strideH == 2 && strideW == 2 && dilation_H == 1 && dilation_W == 1)
		{
			zq_cnn_depthwise_conv_no_padding_nchwc8_kernel3x3_s2d1_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData);
		}
		else
		{
			zq_cnn_depthwise_conv_no_padding_nchwc8_kernel3x3_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData);
		}
	}
	else if (filter_H == 2 && filter_W == 2)
	{
		if (strideH == 1 && strideW == 1 && dilation_H == 1 && dilation_W == 1)
		{
			zq_cnn_depthwise_conv_no_padding_nchwc8_kernel2x2_s1d1_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData);
		}
		else
		{
			zq_cnn_depthwise_conv_no_padding_nchwc8_kernel2x2_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData);
		}
	}
	else if (filter_H == 5 && filter_W == 5
		&& strideH == 1 && strideW == 1 && dilation_H == 1 && dilation_W == 1)
	{
		zq_cnn_depthwise_conv_no_padding_nchwc8_kernel5x5_s1d1_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
			filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
			dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData);
	}
	else
	{
		zq_cnn_depthwise_conv_no_padding_nchwc8_general_with_bias_prelu(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
			filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
			dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep, bias_firstPixelData, slope_firstPixelData);
	}

	double t2 = omp_get_wtime();
	//printf("utils:conv: %.3f ms\n", (t2 - t1) * 1000);
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::DepthwiseConvolution(ZQ_CNN_Tensor4D_NCHWC8& input, const ZQ_CNN_Tensor4D_NCHWC8& filters,
	int strideH, int strideW, int dilation_H, int dilation_W,
	int padH, int padW, ZQ_CNN_Tensor4D_NCHWC8& output)
{
	double t1 = omp_get_wtime();
	int in_N = input.GetN();
	int in_H = input.GetH();
	int in_W = input.GetW();
	int in_C = input.GetC();
	int filter_N = filters.GetN();
	int filter_H = filters.GetH();
	int filter_W = filters.GetW();
	int filter_C = filters.GetC();
	int out_N = output.GetN();
	int out_H = output.GetH();
	int out_W = output.GetW();
	int out_C = output.GetC();
	if (in_N <= 0 || in_H <= 0 || in_W <= 0 || in_C == 0
		|| (in_H - filter_H + (padH << 1)) < 0 || (in_W - filter_W + (padW << 1)) < 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
		
return false;
		return true;
	}
	if (filter_C != in_C || filter_N != 1)
		return false;

	int need_N = in_N;
	// 审计修复 2026-10-02（附录 BE.2 / BH）：strideH/strideW 来自**模型文件**
	// （不可信输入），下面这行是**整数除法** —— 除以 0 在 x86 上是 idiv，
	// 直接 SIGFPE、进程死。附录 BE 只修了 NCHW 那 7 处 wrapper，
	// NCHWC 这边是同一份代码复制过去的，同样要修。
	if (strideH <= 0 || strideW <= 0)
		return false;
	int need_H = (in_H - (filter_H - 1)*dilation_H - 1 + (padH << 1)) / strideH + 1;
	int need_W = (in_W - (filter_W - 1)*dilation_W - 1 + (padW << 1)) / strideW + 1;
	int need_C = in_C;
	if (out_N != need_N || out_H != need_H || out_W != need_W || out_C != need_C)
	{
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
return false;
	}

	if (padH != 0 || padW != 0)
	{
		if (!input.Padding(padW, padH, 0))
			return false;
	}

	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int filter_sliceStep = filters.GetSliceStep();
	int filter_widthStep = filters.GetWidthStep();
	int filter_imStep = filters.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* in_firstPixelData = input.GetFirstPixelPtr() - padH*in_widthStep - padW * 8;
	const float* filter_firstPixelData = filters.GetFirstPixelPtr();
	float* out_firstPixelData = output.GetFirstPixelPtr();

	if (filter_H == 3 && filter_W == 3)
	{
		if (strideH == 1 && strideW == 1 && dilation_H == 1 && dilation_W == 1)
		{
			zq_cnn_depthwise_conv_no_padding_nchwc8_kernel3x3_s1d1(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
		}
		else if (strideH == 2 && strideW == 2 && dilation_H == 1 && dilation_W == 1)
		{
			zq_cnn_depthwise_conv_no_padding_nchwc8_kernel3x3_s2d1(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
		}
		else
		{
			zq_cnn_depthwise_conv_no_padding_nchwc8_kernel3x3(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
		}
	}
	else if (filter_H == 2 && filter_W == 2)
	{
		if (strideH == 1 && strideW == 1 && dilation_H == 1 && dilation_W == 1)
		{
			zq_cnn_depthwise_conv_no_padding_nchwc8_kernel2x2_s1d1(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
		}
		else
		{
			zq_cnn_depthwise_conv_no_padding_nchwc8_kernel2x2(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
				filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
				dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
		}
	}
	else if (filter_H == 5 && filter_W == 5
		&& strideH == 1 && strideW == 1 && dilation_H == 1 && dilation_W == 1)
	{
		zq_cnn_depthwise_conv_no_padding_nchwc8_kernel5x5_s1d1(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
			filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
			dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
	}
	else
	{
		zq_cnn_depthwise_conv_no_padding_nchwc8_general(in_firstPixelData, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep,
			filter_firstPixelData, filter_N, filter_H, filter_W, filter_C, filter_widthStep, filter_sliceStep, filter_imStep, strideH, strideW,
			dilation_H, dilation_W, out_firstPixelData, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
	}

	double t2 = omp_get_wtime();
	//printf("utils:conv: %.3f ms\n", (t2 - t1) * 1000);
	return true;
}

void ZQ_CNN_Forward_SSEUtils_NCHWC::MaxPooling(ZQ_CNN_Tensor4D_NCHWC8 &input, ZQ_CNN_Tensor4D_NCHWC8 &output, int kernel_H, int kernel_W,
	int stride_H, int stride_W, int pad_H_top, int pad_H_bottom,
	int pad_W_left, int pad_W_right, bool global_pool)
{
	int in_N = input.GetN();
	int in_H0 = input.GetH();
	int in_W0 = input.GetW();
	int in_C = input.GetC();
	int need_W, need_H;
	int need_N = in_N;
	int need_C = in_C;
	if (global_pool)
	{
		need_H = 1;
		need_W = 1;
		kernel_H = in_H0;
		kernel_W = in_W0;
		stride_H = 1;
		stride_W = 1;
		// 审计 F2：global_pool 不需要 padding（NCHW 那一支也不加），
		// 显式清零，免得外层传进来的 pad 让 need/use 尺寸互相打架。
		pad_H_top = pad_H_bottom = pad_W_left = pad_W_right = 0;
	}
	else
	{
		need_W = (int)ceil((float)(in_W0 + pad_W_left + pad_W_right - kernel_W) / stride_W + 1);
		need_H = (int)ceil((float)(in_H0 + pad_H_top + pad_H_bottom - kernel_H) / stride_H + 1);
	}

	if (need_W <= 0 || need_H <= 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
			return;
		// 审计修复 2026-10-06（附录 IR.1）：这一行原来缺，NCHW 版（`ZQ_CNN_Forward_SSEUtils.h:1601`）有。
		// `ChangeSize(0,0,0,0,0,0)` 是**成功**的（把 output 置空并返回 true），
		// 所以少了它就会继续往下走到 `(in_H - kernel_H) % stride_H`：
		// `stride_H == 0` 时那是**整数 idiv 除零 -> SIGFPE**，进程直接死。
		// 触发：`stride_H <= 0 || stride_W <= 0`。
		// 经模型文件当前不可达（`ZQ_CNN_Layer_NCHWC.h:2184-2192` 的 ReadParam
		// 已有 `stride_H <= 0 -> return false`），但这 6 个是 public static，
		// 外部可直接调 —— 而 NCHW 版专门补的那行 `return ;` 就是这条路径的防线。
		// 顺带：即便 stride > 0，`need_H == 0`（in_H=3, kernel_H=5, stride_H=2）时
		// NCHWC 会把 output 置成 0 大小后**继续调内核**（内核的循环条件让它空转，
		// 实际不写内存），NCHW 则直接返回 —— 也是行为分叉。
		return;
	}

	// 审计 F2：`% stride == 0` 是**无 pad** 时的等价写法，有 pad 时不成立。
	// 直接说清它真正要表达的事：最后一个窗口不用裁边。
	const float* in_data = 0;
	int in_H = 0, in_W = 0;
	if (!zq_nchwc_pool_prepare_pad(input, pad_H_top, pad_H_bottom, pad_W_left, pad_W_right,
		in_data, in_H, in_W))
		return;
	bool suredivided = ((need_H - 1)*stride_H + kernel_H <= in_H)
		&& ((need_W - 1)*stride_W + kernel_W <= in_W);
	if (output.GetN() != need_N || output.GetH() != need_H || output.GetW() != need_W || output.GetC() != need_C)
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
		return;
	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* out_data = output.GetFirstPixelPtr();

	if (suredivided)
	{
		if (kernel_H == 2 && kernel_W == 2)
		{
			zq_cnn_maxpooling_nopadding_suredivided_nchwc8_kernel2x2(in_data, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, kernel_H, kernel_W, stride_H, stride_W,
				out_data, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
		}
		else if (kernel_H == 3 && kernel_W == 3)
		{
			zq_cnn_maxpooling_nopadding_suredivided_nchwc8_kernel3x3(in_data, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, kernel_H, kernel_W, stride_H, stride_W,
				out_data, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
		}
		else
		{
			zq_cnn_maxpooling_nopadding_suredivided_nchwc8_general(in_data, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, kernel_H, kernel_W, stride_H, stride_W,
				out_data, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
		}
	}
	else
	{
		zq_cnn_maxpooling_nopadding_nodivided_nchwc8_general(in_data, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, kernel_H, kernel_W, stride_H, stride_W,
			out_data, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
	}
}

void ZQ_CNN_Forward_SSEUtils_NCHWC::AVGPooling(ZQ_CNN_Tensor4D_NCHWC8 &input, ZQ_CNN_Tensor4D_NCHWC8 &output, int kernel_H, int kernel_W,
	int stride_H, int stride_W, int pad_H_top, int pad_H_bottom,
	int pad_W_left, int pad_W_right, bool global_pool)
{
	int in_N = input.GetN();
	int in_H0 = input.GetH();
	int in_W0 = input.GetW();
	int in_C = input.GetC();
	int need_W, need_H;
	int need_N = in_N;
	int need_C = in_C;
	if (global_pool)
	{
		need_H = 1;
		need_W = 1;
		kernel_H = in_H0;
		kernel_W = in_W0;
		stride_H = 1;
		stride_W = 1;
		// 审计 F2：global_pool 不需要 padding（NCHW 那一支也不加），
		// 显式清零，免得外层传进来的 pad 让 need/use 尺寸互相打架。
		pad_H_top = pad_H_bottom = pad_W_left = pad_W_right = 0;
	}
	else
	{
		need_W = (int)ceil((float)(in_W0 + pad_W_left + pad_W_right - kernel_W) / stride_W + 1);
		need_H = (int)ceil((float)(in_H0 + pad_H_top + pad_H_bottom - kernel_H) / stride_H + 1);
	}

	if (need_W <= 0 || need_H <= 0)
	{
		if (!output.ChangeSize(0, 0, 0, 0, 0, 0))
			return;
		// 审计修复 2026-10-06（附录 IR.1）：这一行原来缺，NCHW 版（`ZQ_CNN_Forward_SSEUtils.h:1601`）有。
		// `ChangeSize(0,0,0,0,0,0)` 是**成功**的（把 output 置空并返回 true），
		// 所以少了它就会继续往下走到 `(in_H - kernel_H) % stride_H`：
		// `stride_H == 0` 时那是**整数 idiv 除零 -> SIGFPE**，进程直接死。
		// 触发：`stride_H <= 0 || stride_W <= 0`。
		// 经模型文件当前不可达（`ZQ_CNN_Layer_NCHWC.h:2184-2192` 的 ReadParam
		// 已有 `stride_H <= 0 -> return false`），但这 6 个是 public static，
		// 外部可直接调 —— 而 NCHW 版专门补的那行 `return ;` 就是这条路径的防线。
		// 顺带：即便 stride > 0，`need_H == 0`（in_H=3, kernel_H=5, stride_H=2）时
		// NCHWC 会把 output 置成 0 大小后**继续调内核**（内核的循环条件让它空转，
		// 实际不写内存），NCHW 则直接返回 —— 也是行为分叉。
		return;
	}

	// 审计 F2：`% stride == 0` 是**无 pad** 时的等价写法，有 pad 时不成立。
	// 直接说清它真正要表达的事：最后一个窗口不用裁边。
	const float* in_data = 0;
	int in_H = 0, in_W = 0;
	if (!zq_nchwc_pool_prepare_pad(input, pad_H_top, pad_H_bottom, pad_W_left, pad_W_right,
		in_data, in_H, in_W))
		return;
	bool suredivided = ((need_H - 1)*stride_H + kernel_H <= in_H)
		&& ((need_W - 1)*stride_W + kernel_W <= in_W);
	if (output.GetN() != need_N || output.GetH() != need_H || output.GetW() != need_W || output.GetC() != need_C)
		if (!output.ChangeSize(need_N, need_H, need_W, need_C, 0, 0))
		
		return;
	int in_sliceStep = input.GetSliceStep();
	int in_widthStep = input.GetWidthStep();
	int in_imStep = input.GetImageStep();
	int out_sliceStep = output.GetSliceStep();
	int out_widthStep = output.GetWidthStep();
	int out_imStep = output.GetImageStep();
	float* out_data = output.GetFirstPixelPtr();

	if (suredivided)
	{
		if (kernel_H == 2 && kernel_W == 2)
		{
			zq_cnn_avgpooling_nopadding_suredivided_nchwc8_kernel2x2(in_data, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, kernel_H, kernel_W, stride_H, stride_W,
				out_data, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
		}
		else if (kernel_H == 3 && kernel_W == 3)
		{
			zq_cnn_avgpooling_nopadding_suredivided_nchwc8_kernel3x3(in_data, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, kernel_H, kernel_W, stride_H, stride_W,
				out_data, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
		}
		else
		{
			zq_cnn_avgpooling_nopadding_suredivided_nchwc8_general(in_data, in_N, in_H, in_W, in_C,
				in_widthStep, in_sliceStep, in_imStep, kernel_H, kernel_W, stride_H, stride_W,
				out_data, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
		}
	}
	else
	{
		zq_cnn_avgpooling_nopadding_nodivided_nchwc8_general(in_data, in_N, in_H, in_W, in_C,
			in_widthStep, in_sliceStep, in_imStep, kernel_H, kernel_W, stride_H, stride_W,
			out_data, need_N, need_H, need_W, need_C, out_widthStep, out_sliceStep, out_imStep);
	}
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::AddBiasPReLU(ZQ_CNN_Tensor4D_NCHWC8 &input, const ZQ_CNN_Tensor4D_NCHWC8& bias, const ZQ_CNN_Tensor4D_NCHWC8& slope)
{
	int N = input.GetN();
	int H = input.GetH();
	int W = input.GetW();
	int C = input.GetC();
	if (N <= 0 || H <= 0 || W <= 0 || C <= 0)
		return true;
	if (bias.GetC() != C || slope.GetC() != C)
		return false;
	float* data = input.GetFirstPixelPtr();
	const float* bias_Data = bias.GetFirstPixelPtr();
	const float* slope_Data = slope.GetFirstPixelPtr();
	int imStep = input.GetImageStep();
	int widthStep = input.GetWidthStep();
	int sliceStep = input.GetSliceStep();
	bool sure_lessthan1 = true;
	for (int c = 0; c < C; c++)
	{
		if (slope_Data[c] > 1)
		{
			sure_lessthan1 = false;
			break;
		}
	}
	if (sure_lessthan1)
	{
		zq_cnn_addbias_prelu_nchwc8_sure_slope_lessthan1(data, N, H, W, C, widthStep, sliceStep, imStep, bias_Data, slope_Data);
	}
	else
	{
		zq_cnn_addbias_prelu_nchwc8(data, N, H, W, C, widthStep, sliceStep, imStep, bias_Data, slope_Data);
	}

	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::PReLU(ZQ_CNN_Tensor4D_NCHWC8 &input, const ZQ_CNN_Tensor4D_NCHWC8& slope)
{
	int N = input.GetN();
	int H = input.GetH();
	int W = input.GetW();
	int C = input.GetC();
	if (N <= 0 || H <= 0 || W <= 0 || C <= 0)
		return true;
	if (slope.GetC() != C)
		return false;
	float* data = input.GetFirstPixelPtr();
	const float* slope_Data = slope.GetFirstPixelPtr();
	int imStep = input.GetImageStep();
	int widthStep = input.GetWidthStep();
	int sliceStep = input.GetSliceStep();
	bool sure_lessthan1 = true;
	for (int c = 0; c < C; c++)
	{
		if (slope_Data[c] > 1)
		{
			sure_lessthan1 = false;
			break;
		}
	}
	if (sure_lessthan1)
	{
		zq_cnn_prelu_nchwc8_sure_slope_lessthan1(data, N, H, W, C, widthStep, sliceStep, imStep, slope_Data);
	}
	else
	{
		zq_cnn_prelu_nchwc8(data, N, H, W, C, widthStep, sliceStep, imStep, slope_Data);
	}

	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::ReLU(ZQ_CNN_Tensor4D_NCHWC8 &input, float slope)
{
	int N = input.GetN();
	int H = input.GetH();
	int W = input.GetW();
	int C = input.GetC();
	if (N <= 0 || H <= 0 || W <= 0 || C <= 0)
		return true;
	float* data = input.GetFirstPixelPtr();
	int imStep = input.GetImageStep();
	int widthStep = input.GetWidthStep();
	int sliceStep = input.GetSliceStep();

	zq_cnn_relu_nchwc8(data, N, H, W, C, widthStep, sliceStep, imStep, slope);

	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::Softmax(ZQ_CNN_Tensor4D_NCHWC8 &input, int axis)
{
	if (axis < 0 || axis >= 4)
		return false;
	int N = input.GetN();
	int H = input.GetH();
	int W = input.GetW();
	int C = input.GetC();
	if (N <= 0 || H <= 0 || W <= 0 || C <= 0)
		return true;
	float* data = input.GetFirstPixelPtr();
	int sliceStep = input.GetSliceStep();
	int widthStep = input.GetWidthStep();
	int imStep = input.GetImageStep();

	if (axis == 1)
		zq_cnn_softmax_nchwc8_C(data, N, H, W, C, widthStep, sliceStep, imStep);
	else
		return false;
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::BatchNorm_b_a(ZQ_CNN_Tensor4D_NCHWC8 &input, const ZQ_CNN_Tensor4D_NCHWC8& b, const ZQ_CNN_Tensor4D_NCHWC8& a)
{
	int N = input.GetN();
	int H = input.GetH();
	int W = input.GetW();
	int C = input.GetC();
	if (N <= 0 || H <= 0 || W <= 0 || C <= 0)
		return true;
	if (b.GetC() != C || a.GetC() != C)
		return false;
	float* data = input.GetFirstPixelPtr();
	const float* b_data = b.GetFirstPixelPtr();
	const float* a_data = a.GetFirstPixelPtr();
	int widthStep = input.GetWidthStep();
	int sliceStep = input.GetSliceStep();
	int imStep = input.GetImageStep();

	zq_cnn_batchnorm_b_a_nchwc8(data, N, H, W, C, widthStep, sliceStep, imStep, b_data, a_data);
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::Eltwise_Sum(const std::vector<const ZQ_CNN_Tensor4D_NCHWC8*>& input, ZQ_CNN_Tensor4D_NCHWC8& output)
{
	int in_num = (int)input.size();
	if (in_num < 2)
		return false;
	for (int i = 0; i < in_num; i++)
	{
		if (input[i] == 0) return false;
	}
	int N = input[0]->GetN();
	int H = input[0]->GetH();
	int W = input[0]->GetW();
	int C = input[0]->GetC();
	for (int i = 1; i < in_num; i++)
	{
		if (N != input[i]->GetN() || H != input[i]->GetH() || W != input[i]->GetW() || C != input[i]->GetC())
			return false;
	}
	if (output.GetN() != N || output.GetH() != H || output.GetW() != W || output.GetC() != C)
		if (!output.ChangeSize(N, H, W, C, 0, 0))
		
return false;
	if (N <= 0 || H <= 0 || W <= 0 || C <= 0)
		return true;
	std::vector<const float*> in_tensor_data(in_num);
	std::vector<int> in_imStep(in_num), in_widthStep(in_num), in_sliceStep(in_num);
	float* out_data = output.GetFirstPixelPtr();
	for (int i = 0; i < in_num; i++)
	{
		in_tensor_data[i] = input[i]->GetFirstPixelPtr();
		in_imStep[i] = input[i]->GetImageStep();
		in_widthStep[i] = input[i]->GetWidthStep();
		in_sliceStep[i] = input[i]->GetSliceStep();
	}
	int out_imStep = output.GetImageStep(), out_widthStep = output.GetWidthStep(), out_sliceStep = output.GetSliceStep();

	zq_cnn_eltwise_sum_nchwc8(in_num, &in_tensor_data[0], N, H, W, C, &in_widthStep[0], &in_sliceStep[0], &in_imStep[0],
		out_data, out_widthStep, out_sliceStep, out_imStep);

	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::Eltwise_SumWithWeight(const std::vector<const ZQ_CNN_Tensor4D_NCHWC8*>& input, const std::vector<float>& weight,
	ZQ_CNN_Tensor4D_NCHWC8& output)
{
	int in_num = (int)input.size();
	if (in_num < 2 || in_num != weight.size())
		return false;
	for (int i = 0; i < in_num; i++)
	{
		if (input[i] == 0) return false;
	}
	int N = input[0]->GetN();
	int H = input[0]->GetH();
	int W = input[0]->GetW();
	int C = input[0]->GetC();
	for (int i = 1; i < in_num; i++)
	{
		if (N != input[i]->GetN() || H != input[i]->GetH() || W != input[i]->GetW() || C != input[i]->GetC())
			return false;
	}
	if (output.GetN() != N || output.GetH() != H || output.GetW() != W || output.GetC() != C)
		if (!output.ChangeSize(N, H, W, C, 0, 0))
		
return false;
	if (N <= 0 || H <= 0 || W <= 0 || C <= 0)
		return true;
	std::vector<const float*> in_tensor_data(in_num);
	std::vector<int> in_imStep(in_num), in_widthStep(in_num), in_sliceStep(in_num);
	float* out_data = output.GetFirstPixelPtr();
	for (int i = 0; i < in_num; i++)
	{
		in_tensor_data[i] = input[i]->GetFirstPixelPtr();
		in_imStep[i] = input[i]->GetImageStep();
		in_widthStep[i] = input[i]->GetWidthStep();
		in_sliceStep[i] = input[i]->GetSliceStep();
	}
	int out_imStep = output.GetImageStep(), out_widthStep = output.GetWidthStep(), out_sliceStep = output.GetSliceStep();

	zq_cnn_eltwise_sum_with_weight_nchwc8(in_num, &in_tensor_data[0], &weight[0], N, H, W, C,
		&in_widthStep[0], &in_sliceStep[0], &in_imStep[0], out_data, out_widthStep, out_sliceStep, out_imStep);
	return true;
}


bool ZQ_CNN_Forward_SSEUtils_NCHWC::Eltwise_Mul(const std::vector<const ZQ_CNN_Tensor4D_NCHWC8*>& input, ZQ_CNN_Tensor4D_NCHWC8& output)
{
	int in_num = (int)input.size();
	if (in_num < 2)
		return false;
	for (int i = 0; i < in_num; i++)
	{
		if (input[i] == 0) return false;
	}
	int N = input[0]->GetN();
	int H = input[0]->GetH();
	int W = input[0]->GetW();
	int C = input[0]->GetC();
	for (int i = 1; i < in_num; i++)
	{
		if (N != input[i]->GetN() || H != input[i]->GetH() || W != input[i]->GetW() || C != input[i]->GetC())
			return false;
	}
	if (output.GetN() != N || output.GetH() != H || output.GetW() != W || output.GetC() != C)
		if (!output.ChangeSize(N, H, W, C, 0, 0))
		
return false;
	if (N <= 0 || H <= 0 || W <= 0 || C <= 0)
		return true;
	std::vector<const float*> in_tensor_data(in_num);
	std::vector<int> in_imStep(in_num), in_widthStep(in_num), in_sliceStep(in_num);
	float* out_data = output.GetFirstPixelPtr();
	for (int i = 0; i < in_num; i++)
	{
		in_tensor_data[i] = input[i]->GetFirstPixelPtr();
		in_imStep[i] = input[i]->GetImageStep();
		in_widthStep[i] = input[i]->GetWidthStep();
		in_sliceStep[i] = input[i]->GetSliceStep();
	}
	int out_imStep = output.GetImageStep(), out_widthStep = output.GetWidthStep(), out_sliceStep = output.GetSliceStep();

	zq_cnn_eltwise_mul_nchwc8(in_num, &in_tensor_data[0], N, H, W, C, &in_widthStep[0], &in_sliceStep[0], &in_imStep[0],
		out_data, out_widthStep, out_sliceStep, out_imStep);
	return true;
}

bool ZQ_CNN_Forward_SSEUtils_NCHWC::Eltwise_Max(const std::vector<const ZQ_CNN_Tensor4D_NCHWC8*>& input, ZQ_CNN_Tensor4D_NCHWC8& output)
{
	int in_num = (int)input.size();
	if (in_num < 2)
		return false;
	for (int i = 0; i < in_num; i++)
	{
		if (input[i] == 0) return false;
	}
	int N = input[0]->GetN();
	int H = input[0]->GetH();
	int W = input[0]->GetW();
	int C = input[0]->GetC();
	for (int i = 1; i < in_num; i++)
	{
		if (N != input[i]->GetN() || H != input[i]->GetH() || W != input[i]->GetW() || C != input[i]->GetC())
			return false;
	}
	if (output.GetN() != N || output.GetH() != H || output.GetW() != W || output.GetC() != C)
		if (!output.ChangeSize(N, H, W, C, 0, 0))
		
return false;
	if (N <= 0 || H <= 0 || W <= 0 || C <= 0)
		return true;
	std::vector<const float*> in_tensor_data(in_num);
	std::vector<int> in_imStep(in_num), in_widthStep(in_num), in_sliceStep(in_num);
	float* out_data = output.GetFirstPixelPtr();
	for (int i = 0; i < in_num; i++)
	{
		in_tensor_data[i] = input[i]->GetFirstPixelPtr();
		in_imStep[i] = input[i]->GetImageStep();
		in_widthStep[i] = input[i]->GetWidthStep();
		in_sliceStep[i] = input[i]->GetSliceStep();
	}
	int out_imStep = output.GetImageStep(), out_widthStep = output.GetWidthStep(), out_sliceStep = output.GetSliceStep();

	zq_cnn_eltwise_max_nchwc8(in_num, &in_tensor_data[0], N, H, W, C, &in_widthStep[0], &in_sliceStep[0], &in_imStep[0],
		out_data, out_widthStep, out_sliceStep, out_imStep);
	return true;
}

#endif // ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX