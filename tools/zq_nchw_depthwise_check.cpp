/* NCHW（layers_c）depthwise 卷积门禁 —— 附录 CP
 *
 * 为什么是这一族
 * --------------
 * 附录 CO 之后，附录 CQ 的结构扫描指出 `layers_c/` 里 **18 个家族同时存在**
 * 「.c 里手写的标量实现」与「_raw.h 里模板化的 SIMD 实现」——
 * CN 与 CO 查出的两处缺陷都是**手写那份对、模板那份错**。
 * 这一族是剩下里最大的：`zq_cnn_depthwise_convolution_32f_align_c.c`
 * 单个 TU 就有 **159 个真实符号**，**此前一道数值门禁都没有**。
 *
 * 结构（逐条从 `nm` 的符号表与 `ZQ_CNN_Forward_SSEUtils.cpp` 的分派链读出来的）
 * ---------------------------------------------------------------------------
 *   基础名 4 种：general / kernel2x2 / kernel3x3 / kernel5x5
 *   外加一串**通道数特化** `_C4` / `_C8` / … / `_C512` 与 `_Cdiv16/32/64`，
 *   分派条件是 `padded_C == n` 或 `padded_C % n == 0`，
 *   其中 `padded_C = (in_C + 3) >> 2 << 2`（补到 4 的倍数）
 *   align0 只有 `general` 三个符号
 *   **159 个符号共享同一套 28 / 29 / 30 参数签名**（`_C4` 之类的后缀来自
 *   头里的 `#if` 预处理链，**不是**额外的参数）
 *
 *   两条顺带发现：
 *   ① 159 个里有 **12 个只有定义没有头里声明**（`_C12` / `_C48` 那几个），
 *      外部调不到，本门禁不覆盖；
 *   ② `kernel5x5_Cdiv32` 在分派器里**被注释掉了**
 *      （`/*if (padded_C % 32 == 0)`），即 x86 上是死代码 —— 与附录 CD 同类。
 *
 * 本门禁覆盖**公开头里声明的全部 147 个符号**。核心价值是
 * **让通道数特化版本与通用版本对同一份参考值互相校验**：
 * 若 `_C64` 与 `kernel3x3` 对同一组输入给出不同结果，
 * 那么**不论参考写得对不对，其中至少有一个是错的**。
 *
 * 语义（depthwise：每个通道一个 filter，不跨通道混合）
 *   out(n,k,oh,ow) = bias[k] + Σ_{fh,fw} in(n,k,oh*S+fh*D, ow*S+fw*D) * filt(k,fh,fw)
 *   with_bias_prelu 再过一遍 max(0,x) + slope[k]*min(0,x)
 *
 * NCHW 布局：`offset(n,c,h,w) = n*sliceStep + h*widthStep + w*pixelStep + c`，
 * **pixelStep 就是 C**（没有 NCHWC 那种补齐）—— 附录 CO.5 记过我在这里栽过三次。
 */
#include "zq_check_child.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <cmath>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include "ZQCNN/layers_c/zq_cnn_depthwise_convolution_32f_align_c.h"

typedef void (*F28)(const float* in, int N, int H, int W, int C,
                     int ps, int ws, int ss,
                     const float* f, int fN, int fH, int fW, int fC,
                     int fps, int fws, int fss, int sH, int sW, int dH, int dW,
                     float* out, int oN, int oH, int oW, int oC, int ops, int ows, int oss);
typedef void (*F29)(const float* in, int N, int H, int W, int C,
                     int ps, int ws, int ss,
                     const float* f, int fN, int fH, int fW, int fC,
                     int fps, int fws, int fss, int sH, int sW, int dH, int dW,
                     float* out, int oN, int oH, int oW, int oC, int ops, int ows, int oss,
                     const float* bias);
typedef void (*F30)(const float* in, int N, int H, int W, int C,
                     int ps, int ws, int ss,
                     const float* f, int fN, int fH, int fW, int fC,
                     int fps, int fws, int fss, int sH, int sW, int dH, int dW,
                     float* out, int oN, int oH, int oW, int oC, int ops, int ows, int oss,
                     const float* bias, const float* slope);

struct Entry { void* fn; int act; int align; int need; };

static const Entry g_entries[] = {
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align0_general), 0, 1, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align0_general_with_bias), 1, 1, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align0_general_with_bias_prelu), 2, 1, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_general), 0, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_general_with_bias), 1, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_general_with_bias_prelu), 2, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2), 0, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C128), 0, 4, 128 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C128_with_bias), 1, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C128_with_bias_prelu), 2, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C16), 0, 4, 16 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C16_with_bias), 1, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C16_with_bias_prelu), 2, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C24), 0, 4, 24 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C24_with_bias), 1, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C24_with_bias_prelu), 2, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C256), 0, 4, 256 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C256_with_bias), 1, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C256_with_bias_prelu), 2, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C32), 0, 4, 32 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C32_with_bias), 1, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C32_with_bias_prelu), 2, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C4), 0, 4, 4 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C4_with_bias), 1, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C4_with_bias_prelu), 2, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C64), 0, 4, 64 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C64_with_bias), 1, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C64_with_bias_prelu), 2, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C8), 0, 4, 8 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C8_with_bias), 1, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C8_with_bias_prelu), 2, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_Cdiv32), 0, 4, -1032 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_Cdiv32_with_bias), 1, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_Cdiv32_with_bias_prelu), 2, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_with_bias), 1, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_with_bias_prelu), 2, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3), 0, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C128), 0, 4, 128 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C128_with_bias), 1, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C128_with_bias_prelu), 2, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C16), 0, 4, 16 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C16_with_bias), 1, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C16_with_bias_prelu), 2, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C24), 0, 4, 24 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C24_with_bias), 1, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C24_with_bias_prelu), 2, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C256), 0, 4, 256 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C256_with_bias), 1, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C256_with_bias_prelu), 2, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C32), 0, 4, 32 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C32_with_bias), 1, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C32_with_bias_prelu), 2, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C4), 0, 4, 4 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C4_with_bias), 1, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C4_with_bias_prelu), 2, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C64), 0, 4, 64 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C64_with_bias), 1, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C64_with_bias_prelu), 2, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C8), 0, 4, 8 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C8_with_bias), 1, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C8_with_bias_prelu), 2, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_Cdiv32), 0, 4, -1032 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_Cdiv32_with_bias), 1, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_Cdiv32_with_bias_prelu), 2, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_with_bias), 1, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_with_bias_prelu), 2, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel5x5), 0, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel5x5_Cdiv16), 0, 4, -1016 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel5x5_Cdiv16_with_bias), 1, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel5x5_Cdiv16_with_bias_prelu), 2, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel5x5_Cdiv32), 0, 4, -1032 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel5x5_Cdiv32_with_bias), 1, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel5x5_Cdiv32_with_bias_prelu), 2, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel5x5_with_bias), 1, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel5x5_with_bias_prelu), 2, 4, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_general), 0, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_general_with_bias), 1, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_general_with_bias_prelu), 2, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2), 0, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C128), 0, 8, 128 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C128_with_bias), 1, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C128_with_bias_prelu), 2, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C16), 0, 8, 16 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C16_with_bias), 1, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C16_with_bias_prelu), 2, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C24), 0, 8, 24 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C24_with_bias), 1, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C24_with_bias_prelu), 2, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C256), 0, 8, 256 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C256_with_bias), 1, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C256_with_bias_prelu), 2, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C32), 0, 8, 32 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C32_with_bias), 1, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C32_with_bias_prelu), 2, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C512), 0, 8, 512 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C512_with_bias), 1, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C512_with_bias_prelu), 2, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C64), 0, 8, 64 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C64_with_bias), 1, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C64_with_bias_prelu), 2, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C8), 0, 8, 8 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C8_with_bias), 1, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C8_with_bias_prelu), 2, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_Cdiv64), 0, 8, -1064 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_Cdiv64_with_bias), 1, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_Cdiv64_with_bias_prelu), 2, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_with_bias), 1, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_with_bias_prelu), 2, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3), 0, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C128), 0, 8, 128 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C128_with_bias), 1, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C128_with_bias_prelu), 2, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C16), 0, 8, 16 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C16_with_bias), 1, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C16_with_bias_prelu), 2, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C24), 0, 8, 24 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C24_with_bias), 1, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C24_with_bias_prelu), 2, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C256), 0, 8, 256 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C256_with_bias), 1, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C256_with_bias_prelu), 2, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C32), 0, 8, 32 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C32_with_bias), 1, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C32_with_bias_prelu), 2, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C512), 0, 8, 512 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C512_with_bias), 1, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C512_with_bias_prelu), 2, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C64), 0, 8, 64 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C64_with_bias), 1, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C64_with_bias_prelu), 2, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C8), 0, 8, 8 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C8_with_bias), 1, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C8_with_bias_prelu), 2, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_Cdiv64), 0, 8, -1064 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_Cdiv64_with_bias), 1, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_Cdiv64_with_bias_prelu), 2, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_with_bias), 1, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_with_bias_prelu), 2, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel5x5), 0, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel5x5_Cdiv32), 0, 8, -1032 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel5x5_Cdiv32_with_bias), 1, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel5x5_Cdiv32_with_bias_prelu), 2, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel5x5_Cdiv64), 0, 8, -1064 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel5x5_Cdiv64_with_bias), 1, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel5x5_Cdiv64_with_bias_prelu), 2, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel5x5_with_bias), 1, 8, -1 },
  { reinterpret_cast<void*>(zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel5x5_with_bias_prelu), 2, 8, -1 },
};

static const char* g_name[] = {

  "zq_cnn_depthwise_conv_no_padding_32f_align0_general",
  "zq_cnn_depthwise_conv_no_padding_32f_align0_general_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align0_general_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_general",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_general_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_general_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C128",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C128_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C128_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C16",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C16_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C16_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C24",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C24_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C24_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C256",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C256_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C256_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C32",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C32_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C32_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C4",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C4_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C4_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C64",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C64_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C64_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C8",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C8_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_C8_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_Cdiv32",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_Cdiv32_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_Cdiv32_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel2x2_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C128",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C128_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C128_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C16",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C16_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C16_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C24",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C24_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C24_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C256",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C256_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C256_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C32",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C32_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C32_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C4",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C4_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C4_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C64",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C64_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C64_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C8",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C8_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_C8_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_Cdiv32",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_Cdiv32_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_Cdiv32_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel3x3_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel5x5",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel5x5_Cdiv16",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel5x5_Cdiv16_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel5x5_Cdiv16_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel5x5_Cdiv32",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel5x5_Cdiv32_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel5x5_Cdiv32_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel5x5_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align128bit_kernel5x5_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_general",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_general_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_general_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C128",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C128_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C128_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C16",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C16_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C16_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C24",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C24_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C24_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C256",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C256_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C256_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C32",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C32_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C32_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C512",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C512_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C512_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C64",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C64_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C64_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C8",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C8_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_C8_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_Cdiv64",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_Cdiv64_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_Cdiv64_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel2x2_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C128",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C128_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C128_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C16",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C16_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C16_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C24",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C24_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C24_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C256",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C256_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C256_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C32",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C32_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C32_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C512",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C512_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C512_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C64",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C64_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C64_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C8",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C8_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_C8_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_Cdiv64",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_Cdiv64_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_Cdiv64_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel3x3_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel5x5",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel5x5_Cdiv32",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel5x5_Cdiv32_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel5x5_Cdiv32_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel5x5_Cdiv64",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel5x5_Cdiv64_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel5x5_Cdiv64_with_bias_prelu",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel5x5_with_bias",
  "zq_cnn_depthwise_conv_no_padding_32f_align256bit_kernel5x5_with_bias_prelu",
};
static const int N_ENTRY = (int)(sizeof(g_entries) / sizeof(g_entries[0]));

#define RES_FILE "/tmp/zq_dw_res.txt"
static const double TOL = 1e-5;

static float val(int seed, int idx)
{
    unsigned int x = (unsigned int)((unsigned int)seed * 2654435761u + (unsigned int)idx * 40503u);
    x ^= x >> 13; x *= 1274126177u; x ^= x >> 16;
    return (float)((int)(x % 2001) - 1000) * 0.001f;
}

struct Case { int entry, C, S, D, H, W; };

// 该入口要求的 padded_C：-1 = 任意；-1000-n = padded_C % n == 0；正数 = 必须等于
// **注意先剥掉激活动作后缀** —— 第一版直接对全名做 _C(\d+)$ 匹配，
// 而  是以  结尾的，于是特化全没解析出来，
// 拿 C=8/16 去喂「只支持 C=4」的入口，报出 32 个假红。
static int parse_need(const char* n)
{
    char base[256];
    size_t L = strlen(n);
    const char* suf[2] = { "_with_bias_prelu", "_with_bias" };
    for (int s = 0; s < 2; s++) {
        size_t sl = strlen(suf[s]);
        if (L > sl && strcmp(n + L - sl, suf[s]) == 0) { L -= sl; break; }
    }
    if (L >= sizeof(base)) L = sizeof(base) - 1;
    memcpy(base, n, L); base[L] = 0;
    const char* p = strstr(base, "_Cdiv");
    if (p) return -1000 - atoi(p + 5);
    p = strrchr(base, 'C');
    if (p && p > base && isdigit((unsigned char)p[1])) return atoi(p + 1);
    return -1;
}

static void run_one(const Case& c)
{
    const Entry& e = g_entries[c.entry];
    const int need_c = parse_need(g_name[c.entry]);   // 表里存的 need 不用，以名字为准
    const int C = c.C, S = c.S, D = c.D, H = c.H, W = c.W, N = 1;
    const char* nm = g_name[c.entry];
    const int kH = strstr(nm, "kernel2x2") ? 2 : strstr(nm, "kernel3x3") ? 3 :
                   strstr(nm, "kernel5x5") ? 5 : 3;      // general 用 3x3
    const int eH = (kH - 1) * D + 1, eW = eH;
    const int oH = (H - eH) / S + 1, oW = (W - eW) / S + 1;
    if (oH <= 0 || oW <= 0) return;

    // depthwise 的 filter 布局（从 raw.h 的索引式读出来的，**别按参数名推**）：
    //   cur_filter_c_ptr = cur_filter_pix_ptr + kc * zq_mm_align_size
    //   cur_filter_pix_ptr += filter_pixelStep   （列）
    //   cur_filter_row_ptr += filter_widthStep    （行）
    //   **filter_sliceStep 一次都没用到**
    // 也就是说通道在最内层、步进是**写死的 SIMD 宽度**，
    // 而不是 filter_pixelStep（第四次「参数名不是语义」——见 CJ.2 / CH.2 / CI.3）。
    const int cpad = (C + e.align - 1) / e.align * e.align;   // 每通道核的浮点跨度
    const int fps = cpad, fws = fps * kH, fss = fws * kH;
    const int ps = C, ws = ps * W, ss = ws * H;
    const int ops = C, ows = ops * oW, oss = ows * oH;

    const size_t n_in = (size_t)N * ss, n_f = (size_t)fss, n_o = (size_t)oss;
    // **in / flt / out / bias / slope 五个缓冲区全都要 32 字节对齐** ——
    // align128bit 用 _mm_load_ps（要 16）、align256bit 用 _mm256_load_ps（要 32），
    // 而 std::vector<float> 只给 16。第一版只对齐了后三个，子进程全部 SEGV
    //（ASan 对 SEGV 是 _exit(1)，门禁报成没跑完而不是崩——附录 CJ.4 的那条）。
    std::vector<float> in_m(n_in + 8), flt_m(n_f + 8), out_m(n_o + 8),
                          bias_m(C + 8), slope_m(C + 8);
    float* in  = (float*)(((size_t)in_m.data()   + 31) / 32 * 32);
    float* flt = (float*)(((size_t)flt_m.data()  + 31) / 32 * 32);
    float* out = (float*)(((size_t)out_m.data()  + 31) / 32 * 32);
    float* bp  = (float*)(((size_t)bias_m.data() + 31) / 32 * 32);
    float* sp  = (float*)(((size_t)slope_m.data()+ 31) / 32 * 32);
    for (size_t i = 0; i < n_in; i++) in[i] = val(1, (int)i);
    for (size_t i = 0; i < n_f; i++) flt[i] = val(2, (int)i);
    for (int k = 0; k < C; k++) { bp[k] = val(3, k) * 0.25f; sp[k] = 0.1f + 0.01f * (k % 7); }
    for (size_t i = 0; i < n_o; i++) out[i] = -12345.0f;

    if (e.act == 0)
        ((F28)e.fn)(in,  N, H, W, C, ps, ws, ss, flt, 1, kH, kH, C, fps, fws, fss,
                    S, S, D, D, out, N, oH, oW, C, ops, ows, oss);
    else if (e.act == 1)
        ((F29)e.fn)(in,  N, H, W, C, ps, ws, ss, flt, 1, kH, kH, C, fps, fws, fss,
                    S, S, D, D, out, N, oH, oW, C, ops, ows, oss, bp);
    else
        ((F30)e.fn)(in,  N, H, W, C, ps, ws, ss, flt, 1, kH, kH, C, fps, fws, fss,
                    S, S, D, D, out, N, oH, oW, C, ops, ows, oss, bp, sp);

    long n_ok = 0, n_bad = 0; double worst = 0.0;
    for (int n = 0; n < N; n++)
        for (int k = 0; k < C; k++)
            for (int oh = 0; oh < oH; oh++)
                for (int ow = 0; ow < oW; ow++) {
                    double sum = (e.act >= 1) ? bp[k] : 0.0, sc = 0.0;
                    for (int fh = 0; fh < kH; fh++)
                        for (int fw = 0; fw < kH; fw++) {
                            double a = in[(size_t)n * ss + (oh * S + fh * D) * ws + (ow * S + fw * D) * ps + k];
                            double f = flt[(size_t)fh * fws + (size_t)fw * fps + k];   // filter_N==1 -> 无 k 这一维
                            sum += a * f; sc += a * a * f * f;
                        }
                    if (e.act == 2 && sum < 0) sum *= sp[k];
                    double got = out[(size_t)n * oss + oh * ows + ow * ops + k];
                    double den = sqrt(sc); if (den < 1e-30) den = 1.0;
                    double be = fabs(got - sum) / den;
                    if (be > TOL) n_bad++; else n_ok++;
                    if (be > worst) worst = be;
                }
    FILE* f = fopen(RES_FILE, "w");
    if (f) { fprintf(f, "%ld %ld %.6e\n", n_ok, n_bad, worst); fclose(f); }
}

static int g_case = 0, g_ok = 0, g_bad = 0, g_crash = 0;

static void one(const Case& c)
{
    g_case++;
    remove(RES_FILE);
    pid_t pid = fork();
    if (pid == 0) {
        zq_child_silence_stderr();
        run_one(c);
        _exit(0);
    }
    int st = 0; waitpid(pid, &st, 0);
    long ok = 0, bad = 0; double worst = 0;
    int have = 0;
    FILE* f = fopen(RES_FILE, "r");
    if (f) { have = (fscanf(f, "%ld %ld %lf", &ok, &bad, &worst) == 3); fclose(f); }
    char tag[160];
    snprintf(tag, sizeof(tag), "C=%d %dx%d k=%d S=%d", c.C, c.H, c.W,
             (strstr(g_name[c.entry], "kernel2x2") ? 2 : strstr(g_name[c.entry], "kernel3x3") ? 3 :
              strstr(g_name[c.entry], "kernel5x5") ? 5 : 3), c.S);
    if (!have) { g_crash++; printf("  %-52s %s  没跑完（退出码 %d）\n", g_name[c.entry], tag, WEXITSTATUS(st)); return; }
    if (WIFSIGNALED(st)) { g_crash++; printf("  %-52s %s  CRASH(信号 %d)\n", g_name[c.entry], tag, WTERMSIG(st)); return; }
    if (bad > 0) { g_bad++; printf("  %-52s %s  FAIL %ld/%ld 格错, 最差 %.3e\n", g_name[c.entry], tag, bad, ok + bad, worst); }
    else { g_ok++; }
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("NCHW depthwise：公开头里声明的 %d 个入口全测\n", N_ENTRY);
    printf("（符号表共 159 个，其中 12 个只有定义没有声明，外部调不到，不覆盖）\n");
    printf("内核名全部写全、走函数指针表；判据：后向误差 + 逐格统计\n");
    printf("**每个入口只喂符合它自己契约的 in_C**（_C<n> 要 padded_C==n，_Cdiv<n> 要 padded_C%%n==0）\n");
    printf("out / bias / slope 缓冲区一律 32 字节对齐（附录 CJ.4、CO.5 各记过一次）\n\n");

    for (int e = 0; e < N_ENTRY; e++) {
        const int need = parse_need(g_name[e]);
        int Cs[3], nc = 0;
        if (need < -1000) {              // KIND_DIVn
            const int d = -1000 - need;
            Cs[nc++] = d * 2;            // 补到 4/8 的倍数后仍能被 d 整除
        } else if (need < 0) {           // 通用入口：几个不同的 C 都跑
            Cs[nc++] = 8; Cs[nc++] = 16; Cs[nc++] = 4;
        } else {
            Cs[nc++] = need;
        }
        for (int ci = 0; ci < nc; ci++) {
            int CC = Cs[ci];
            const int al = g_entries[e].align;
            if (al == 4) CC = (CC + 3) / 4 * 4;
            if (al == 8) CC = (CC + 7) / 8 * 8;
            for (int sd = 0; sd < 2; sd++) {
                Case c; memset(&c, 0, sizeof(c));
                c.entry = e; c.C = CC; c.S = (sd ? 2 : 1); c.D = 1; c.H = 14; c.W = 15;
                one(c);
            }
        }
    }
    printf("\n共 %d 个用例：全对 %d，有错 %d，崩溃/搭建失败 %d\n", g_case, g_ok, g_bad, g_crash);
    if (g_bad || g_crash)
        printf("**每一项在下结论之前都要先用独立复现对一遍**（附录 CA.3）。\n");
    return (g_bad || g_crash) ? 1 : 0;
}
