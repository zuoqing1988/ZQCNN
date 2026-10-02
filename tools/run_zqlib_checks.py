#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跑 tools/ 下所有 zq_*_check.cpp（第三方头库的独立回归测试）。

为什么要这个入口
----------------
附录 W/X/Y/Z/AA 一路下来的做法是：给每个「之前说没法验证」的第三方头写一个
脱离主工程的最小测试，用 ASan + LeakSanitizer 跑。到 AA 为止已经有 4 个，
但每个都要手敲一条编译命令 —— 没人会记得在每次改动后都跑一遍，于是它们会
慢慢变成「写过一次就没再跑过」的死文件。

这个脚本把它们统一起来：
  * 自动发现 tools/zq_*_check.cpp
  * gcc -O1 -g -fsanitize=address -I3rdparty/include/ZQlib
  * 逐个跑，任何一个非 0 退出就整体失败

用法:
    python tools/run_zqlib_checks.py            # 全部
    python tools/run_zqlib_checks.py mergesort  # 只跑名字里含这个串的
    python tools/run_zqlib_checks.py --list     # 只列出来

注意: 这些测试验的是**第三方头库**（3rdparty/include/ZQlib），与 ZQCNN 的
主工程构建无关，所以放在 tools/ 而不是 CMake 里。主工程的验证仍然是
tools/run_sample_regression.sh + tools/check_line_endings.py +
tools/check_text_encoding.py。
"""

import argparse
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WSL_DIST = 'Ubuntu-20.04'
INC = '/mnt/d/ZQCNN/3rdparty/include/ZQlib'


def run_wsl(script):
    # 必须喂字节: text=True 会在 Windows 上把 \n 变成 \r\n,
    # bash 于是看到 `set +\r` / `cd dir\r` 直接不跑（见附录 X.5）。
    p = subprocess.run('wsl -d %s -- bash -s' % WSL_DIST, shell=True,
                       input=script.encode('utf-8'), capture_output=True)
    return ((p.stdout or b'').decode('utf-8', 'replace')
            + (p.stderr or b'').decode('utf-8', 'replace'))


# 少数测试不是"纯 ZQlib 头"，还要编主工程的内核 .c。
# 一律用 **gcc** 编 .c（用 g++ 会把 C 的 braced-init 判成 narrowing 直接报错，
# 见附录 AU.2 踩过的坑），链接时再加 -mavx2 -mfma。
EXTRA_SOURCES = {
    'zq_lrn': [
        'gcc -O1 -g -mavx2 -mfma -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/math/zq_sse_mathfun.c -o $WDIR/zq_lrn_sse.o',
        'gcc -O1 -g -mavx2 -mfma -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/math/zq_avx_mathfun.c -o $WDIR/zq_lrn_avx.o',
    ],
    # zq_eltwise：同 zq_bns，两个 math .c 只是因为它们定义了 log/exp 等 SIMD 辅助。
    'zq_eltwise': [
        'gcc -O1 -g -mavx2 -mfma -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/math/zq_sse_mathfun.c -o $WDIR/zq_eltwise_sse.o',
        'gcc -O1 -g -mavx2 -mfma -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/math/zq_avx_mathfun.c -o $WDIR/zq_eltwise_avx.o',
    ],
    # zq_pool：只 include 了内核的 .c，不含 math/zq_*_mathfun.c，所以 EXTRA_SOURCES
    # 是空的 —— 留个空表项是为了"以后要加时知道该加哪儿"。
    'zq_pool': [],
    # zq_facedb 要 ZQlibFaceID + OpenCV 的头。OpenCV 路径从主工程的 CMakeCache 里取，
    # 取不到就编不过 —— 所以这里先探一下，探不到就把这个测试整体跳过并说明原因
    # （与 tools/probe_zqlib_headers_msvc.py 对那 14 个 C1083 的处理同一思路）。
    'zq_facedb': [
        'OCV=$(sed -n "s/^OpenCV_DIR:PATH=//p" $R/build_x64/CMakeCache.txt 2>/dev/null)',
        'if [ -z "$OCV" ]; then echo "NOOPENCV" > $WDIR/zq_facedb.skip; '
        '  else for m in core imgproc imgcodecs highgui; do '
        '  printf "#include <%s.h>\n" $m > $WDIR/probe_$m.cpp; '
        '  g++ -fsyntax-only -I$OCV/include -I$OCV/../../modules/$m/include '
        '      $WDIR/probe_$m.cpp 2>/dev/null || echo "NOOPENCV" > $WDIR/zq_facedb.skip; '
        'done; fi',
    ],
    # zq_facedb2（附录 BL）要**同样**那套 OpenCV 头 —— ZQ_FaceDatabase.h 通过
    # ZQ_FaceRecognizerSphereFace.h -> ZQ_FaceRecognizer.h 间接用到 cv::Mat。
    # 所以直接复用 zq_facedb 那段探测，不重写一遍（两处探测一旦漂移，
    # 就会出现"facedb 能跑、facedb2 编不过"而没人知道为什么）。
    'zq_facedb2': [
        'OCV=$(sed -n "s/^OpenCV_DIR:PATH=//p" $R/build_x64/CMakeCache.txt 2>/dev/null)',
        'if [ -z "$OCV" ]; then echo "NOOPENCV" > $WDIR/zq_facedb2.skip; '
        '  else for m in core imgproc imgcodecs highgui; do '
        '  printf "#include <%s.h>\n" $m > $WDIR/probe2_$m.cpp; '
        '  g++ -fsyntax-only -I$OCV/include -I$OCV/../../modules/$m/include '
        '      $WDIR/probe2_$m.cpp 2>/dev/null || echo "NOOPENCV" > $WDIR/zq_facedb2.skip; '
        'done; fi',
    ],
    # zq_innerproduct 要链上 ZQ_GEMM 的三个 TU，其中 zq_gemm_32f_align_c.c
    # 单独一个就要编 5 分钟以上。默认不跑（--with-slow 才跑），理由写在这里。
    'zq_innerproduct': [
        # 内核自己也要编 —— 只链 ZQ_GEMM 那三个会 undefined reference
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_c/zq_cnn_innerproduct_gemm_32f_align_c.c -o $WDIR/zq_ipgemm.o',
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQ_GEMM/math/zq_gemm_32f_align_c.c -o $WDIR/zq_gemm_align.o',
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQ_GEMM/math/zq_gemm_32f_align_c_asm.c -o $WDIR/zq_gemm_asm.o',
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQ_GEMM/math/zq_gemm_32f_auto.c -o $WDIR/zq_gemm_auto.o',
    ],
    # zq_gemm_shape（附录 BN.2）只需要 ZQ_GEMM 那三个 TU —— 测试直接调
    # zq_gemm_32f_AnoTrans_Btrans_auto，不经过任何 ZQCNN 的层。
    # 它编译慢（见 SLOW），默认不自动跑，要 --with-slow。
    'zq_gemm_shape': [
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQ_GEMM/math/zq_gemm_32f_align_c.c -o $WDIR/zq_shape_gemm_align.o',
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQ_GEMM/math/zq_gemm_32f_align_c_asm.c -o $WDIR/zq_shape_gemm_asm.o',
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQ_GEMM/math/zq_gemm_32f_auto.c -o $WDIR/zq_shape_gemm_auto.o',
    ],
    # zq_nchwc_ip（附录 BN）与 zq_innerproduct 共用 ZQ_GEMM 那四个 TU
    # （含编 5 分钟以上的 zq_gemm_32f_align_c.c），所以同样默认不跑。
    # 额外要编两个 layers_nchwc 的 .c + 一个 ZQ_CNN_Tensor4D_NCHWC.cpp：
    #   * ZQ_CNN_Tensor4D_NCHWC.cpp 用到 layers_nchwc/zq_cnn_resize_nchwc.h 里的
    #     6 个 ResizeBilinear 内核，不编它就是 6 个 undefined reference
    #   * 测试**用真实的 ZQ_CNN_Tensor4D_NCHWC1/4/8 类**分配与填充张量，
    #     这样"我以为的 NCHWC 布局"这个变量根本不存在（见 BN 的说明）
    'zq_nchwc_ip': [
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_nchwc/zq_cnn_innerproduct_gemm_nchwc.c -o $WDIR/zq_nchwc_ip.o',        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_nchwc/zq_cnn_resize_nchwc.c -o $WDIR/zq_nchwc_resize.o',
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_c/zq_cnn_innerproduct_gemm_32f_align_c.c -o $WDIR/zq_nchwcip_kernel.o',
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQ_GEMM/math/zq_gemm_32f_align_c.c -o $WDIR/zq_nchwcip_gemm_align.o',
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQ_GEMM/math/zq_gemm_32f_align_c_asm.c -o $WDIR/zq_nchwcip_gemm_asm.o',
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQ_GEMM/math/zq_gemm_32f_auto.c -o $WDIR/zq_nchwcip_gemm_auto.o',
        # ZQ_CNN_Tensor4D_NCHWC.cpp 是 C++，必须用 g++ 编
        'g++ -O1 -g -mavx2 -mfma -fopenmp -c -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Tensor4D_NCHWC.cpp -o $WDIR/zq_nchwcip_tensor.o',
    ],
    # zq_nchwc_conv（附录 BS）与 zq_nchwc_ip 共用同一批 TU，只是把
    # layers_nchwc/zq_cnn_innerproduct_gemm_nchwc.c 换成 convolution 那个 .c。
    'zq_nchwc_conv': [
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_nchwc/zq_cnn_convolution_gemm_nchwc.c -o $WDIR/zq_nchwcv.o',
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_nchwc/zq_cnn_resize_nchwc.c -o $WDIR/zq_nchwcv_resize.o',
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQ_GEMM/math/zq_gemm_32f_align_c.c -o $WDIR/zq_nchwcv_gemm_align.o',
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQ_GEMM/math/zq_gemm_32f_align_c_asm.c -o $WDIR/zq_nchwcv_gemm_asm.o',
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQ_GEMM/math/zq_gemm_32f_auto.c -o $WDIR/zq_nchwcv_gemm_auto.o',
        'g++ -O1 -g -mavx2 -mfma -fopenmp -c -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Tensor4D_NCHWC.cpp -o $WDIR/zq_nchwcv_tensor.o',
    ],
    # zq_nchwc_conv8（附录 CB）是 align=8 那一族，与 zq_nchwc_conv 编的是**同一批 TU**，
    # 只是把 .o 换个名字落一份，省得两个测试各编一次 5 分钟的 zq_gemm_32f_align_c.c。
    'zq_nchwc_conv8': [
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_nchwc/zq_cnn_convolution_gemm_nchwc.c -o $WDIR/zq_nchwcv8.o',
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_nchwc/zq_cnn_resize_nchwc.c -o $WDIR/zq_nchwcv8_resize.o',
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQ_GEMM/math/zq_gemm_32f_align_c.c -o $WDIR/zq_nchwcv8_gemm_align.o',
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQ_GEMM/math/zq_gemm_32f_align_c_asm.c -o $WDIR/zq_nchwcv8_gemm_asm.o',
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQ_GEMM/math/zq_gemm_32f_auto.c -o $WDIR/zq_nchwcv8_gemm_auto.o',
        'g++ -O1 -g -mavx2 -mfma -fopenmp -c -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Tensor4D_NCHWC.cpp -o $WDIR/zq_nchwcv8_tensor.o',
    ],
    # zq_nchw_conv（附录 CE）：**NCHW 卷积**是 x86 上的**主生产路径**
    # （NCHWC 只是可选的 SIMD 变体），而 21 个 zq_*_check.cpp 里唯独没有测它。
    # CB 刚在 NCHWC 那一族的相邻两个函数之间找出三处独立缺陷 ——
    # 同一个族的另一个成员出问题一点也不奇怪，所以补上这道门禁。
    # 不需要 ZQ_CNN_Tensor4D_NCHWC.cpp：测试自己用紧凑 NCHW 下标填 std::vector，
    # 不经过任何张量类（与 CB 那两道门禁"用真实张量类"的做法不同，这里刻意简化）。
    'zq_nchw_conv': [
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_c/zq_cnn_convolution_gemm_32f_align_c.c -o $WDIR/zq_nchwcv.o',
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQ_GEMM/math/zq_gemm_32f_align_c.c -o $WDIR/zq_nchwcv_gemm_align.o',
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQ_GEMM/math/zq_gemm_32f_align_c_asm.c -o $WDIR/zq_nchwcv_gemm_asm.o',
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQ_GEMM/math/zq_gemm_32f_auto.c -o $WDIR/zq_nchwcv_gemm_auto.o',
    ],
    # zq_nchwc_depthwise（附录 CF）：**NCHWC 深度可分离卷积**。
    # model/ 下有 284 个 DepthwiseConvolution 层，而 22 个 zq_*_check.cpp 一个都没覆盖它。
    # 这一族与普通卷积是两套完全不同的代码（不走 gemm，是手写的
    # 「每个通道一个 filter」SIMD 展开），所以 CB/CE 修的那些问题对它一概无效。
    # 好消息：它**不依赖 ZQ_GEMM**，所以三个 TU 全部很快，编一次几秒钟。
    'zq_nchwc_depthwise': [
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_nchwc/zq_cnn_depthwise_convolution_nchwc.c -o $WDIR/zq_dw.o',
        # ZQ_CNN_Tensor4D_NCHWC.cpp 要用 layers_nchwc/zq_cnn_resize_nchwc.h 里的
        # ResizeBilinear 内核，不编它就是 undefined reference（与 zq_nchwc_ip 同理）
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_nchwc/zq_cnn_resize_nchwc.c -o $WDIR/zq_dw_resize.o',
        'g++ -O1 -g -mavx2 -mfma -fopenmp -c -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Tensor4D_NCHWC.cpp -o $WDIR/zq_dw_tensor.o',
    ],
    # zq_nchwc_act（附录 CG）：NCHWC 激活层。**卷积的 with_bias_prelu 三个变体
    # 内部就直接调 prelu**（zq_mm_fmadd_ps(slope_v, min(0,x), max(0,x))），
    # 它一旦错，CB 修好的那十几个卷积入口会一起错，而之前没有独立的门禁盯它。
    # 15 个入口 = addbias / prelu / prelu_sure / addbias_prelu / addbias_prelu_sure
    #             × NCHWC1/4/8（符号表已用 nm 核实）
    'zq_nchwc_act': [
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_nchwc/zq_cnn_prelu_nchwc.c -o $WDIR/zq_act_prelu.o',
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_nchwc/zq_cnn_addbias_nchwc.c -o $WDIR/zq_act_addbias.o',
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_nchwc/zq_cnn_resize_nchwc.c -o $WDIR/zq_act_resize.o',
        'g++ -O1 -g -mavx2 -mfma -fopenmp -c -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Tensor4D_NCHWC.cpp -o $WDIR/zq_act_tensor.o',
    ],
    # zq_nchwc_elt_relu（附录 CH）：NCHWC 的 relu 与 eltwise。
    # 两者都是**逐元素**运算、语义没有歧义（不像 batchnormscale 那种逐通道归一化
    # 数学），所以参考实现几乎不可能写错 —— 门禁一旦变红，结论可信。
    # 15 个入口 = relu + eltwise{sum,max,mul,sum_with_weight} x NCHWC1/4/8
    'zq_nchwc_elt_relu': [
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_nchwc/zq_cnn_relu_nchwc.c -o $WDIR/zq_er_relu.o',
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_nchwc/zq_cnn_eltwise_nchwc.c -o $WDIR/zq_er_elt.o',
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_nchwc/zq_cnn_resize_nchwc.c -o $WDIR/zq_er_resize.o',
        'g++ -O1 -g -mavx2 -mfma -fopenmp -c -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Tensor4D_NCHWC.cpp -o $WDIR/zq_er_tensor.o',
    ],
    # zq_nchwc_pool（附录 CI）：NCHWC pooling 的 24 个入口。
    # pooling 是每个 CNN 都用的层，而 NCHWC 这一族之前一道门禁都没有
    # （NCHW 那一族有 zq_pool_check）。CE/CF/CG/CH 连着四轮都印证了：
    # 同一个功能的两个变体，测了的那份全对、没测的那份全错。
    # 符号表用 nm 取（24 个），不是按头里的声明数 —— 头里有重复声明。
    'zq_nchwc_pool': [
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_nchwc/zq_cnn_pooling_nchwc.c -o $WDIR/zq_pool8.o',
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_nchwc/zq_cnn_resize_nchwc.c -o $WDIR/zq_pool8_resize.o',
        'g++ -O1 -g -mavx2 -mfma -fopenmp -c -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Tensor4D_NCHWC.cpp -o $WDIR/zq_pool8_tensor.o',
    ],
    # zq_nchwc_bn（附录 CJ）：NCHWC batchnormscale 的 12 个入口
    # （scale / batchnorm_b_a / batchnorm_mean_var / mean_var_scale_bias × NCHWC1/4/8）。
    # 这一族的**参数名与直觉相反**（batchnorm_b_a 的第一个参数是乘数、第二个是加数），
    # 所以参考实现是逐条从源码抄下来的，不是按名字推的。
    'zq_nchwc_bn': [
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_nchwc/zq_cnn_batchnormscale_nchwc.c -o $WDIR/zq_bn.o',
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_nchwc/zq_cnn_resize_nchwc.c -o $WDIR/zq_bn_resize.o',
        'g++ -O1 -g -mavx2 -mfma -fopenmp -c -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Tensor4D_NCHWC.cpp -o $WDIR/zq_bn_tensor.o',
    ],
    # zq_nchwc_softmax（附录 CK）：NCHWC softmax 的 5 个入口（沿 C/H/W 三个轴）。
    # softmax 的输出是概率，判据用它自己当尺度；C 特意跑「是 align 倍数」与
    # 「不是 align 倍数」两种，让内核自己的对齐尾循环必须被走到。
    'zq_nchwc_softmax': [
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_nchwc/zq_cnn_softmax_nchwc.c -o $WDIR/zq_sm.o',
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_nchwc/zq_cnn_resize_nchwc.c -o $WDIR/zq_sm_resize.o',
        'g++ -O1 -g -mavx2 -mfma -fopenmp -c -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Tensor4D_NCHWC.cpp -o $WDIR/zq_sm_tensor.o',
    ],
    # zq_nchwc_resize（附录 CL）：NCHWC resize 的 6 个入口
    # （with / without_safeborder × NCHWC1/4/8）。两者的唯一区别是**边界处理**：
    # without 会把 x0/x1、y0/y1 钳到图内，with 不钳（要求调用方保证安全边界）。
    # 门禁用两种配置：cfgA 两个变体都安全（期望相同）、cfgB 只有 without 能跑（验钳位）。
    'zq_nchwc_resize': [
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_nchwc/zq_cnn_resize_nchwc.c -o $WDIR/zq_rz.o',
        'g++ -O1 -g -mavx2 -mfma -fopenmp -c -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Tensor4D_NCHWC.cpp -o $WDIR/zq_rz_tensor.o',
    ],
    # zq_nchw_resize（附录 CN）：**NCHW 的 resize / remap**，x86 上的**主生产路径**
    # （ZQ_CNN_Tensor4D.cpp 直接调用），此前一道数值门禁都没有。
    # 附录 CL 在它的 NCHWC 兄弟里查出过一处越界读（y 钳位漏 -1），
    # 而本文件的钳位字面量是对的（附录 CM 已核对）—— 但**算术与 map 语义仍无覆盖**。
    # 15 个真实符号（nm 核实；头里是 34 个声明，含重复）：
    #   resize_nn / resize_with_safeborder / resize_without_safeborder
    #   remap_without_safeborder / remap_without_safeborder_fillval   各 x align0/128/256
    # 末尾那个 sample_align_type 决定要不要半像素平移，名字里没有提示，两种都测。
    'zq_nchw_resize': [
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_c/zq_cnn_resize_32f_align_c.c -o $WDIR/zq_rzn.o',
        'g++ -O1 -g -mavx2 -mfma -fopenmp -c -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Tensor4D.cpp -o $WDIR/zq_rzn_tensor.o',
    ],
    # zq_nchw_act（附录 CO）：NCHW 的激活与归一化层，38 个真实符号。
    # 这一批是「手写标量 + raw 头模板」双实现家族里语义最确定的一批（附录 CP 的扫描结论），
    # 而 CN 证明过：这种结构下「手写那份对、模板那份错」是会发生的。
    # 带分支的都跑两个：relu 的 slope==0、dropout 的 scale==1 提前返回、scale 的 bias==NULL。
    'zq_nchw_act': [
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_c/zq_cnn_relu_32f_align_c.c -o $WDIR/zq_nact_relu.o',
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_c/zq_cnn_prelu_32f_align_c.c -o $WDIR/zq_nact_prelu.o',
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_c/zq_cnn_addbias_32f_align_c.c -o $WDIR/zq_nact_addbias.o',
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_c/zq_cnn_dropout_32f_align_c.c -o $WDIR/zq_nact_dropout.o',
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_c/zq_cnn_softmax_32f_align_c.c -o $WDIR/zq_nact_softmax.o',
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_c/zq_cnn_batchnormscale_32f_align_c.c -o $WDIR/zq_nact_bn.o',
    ],
    # zq_nchw_depthwise（附录 CP）：NCHW depthwise，159 个真实符号 / 33 个基础名，
    # 此前零覆盖。门禁取代表性子集（通用 + 基础 k + 各通道数特化），
    # 核心价值是让**特化版本与通用版本对同一份参考值互相校验**。
    'zq_nchw_depthwise': [
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_c/zq_cnn_depthwise_convolution_32f_align_c.c -o $WDIR/zq_dwnchw.o',
    ],
    # zq_nchw_scalop（附录 CQ）：NCHW scalaroperation，38 个 32f 真实符号
    # （7 运算 x 2 形式 x 3 对齐，pow 只有 align0）。全是逐元素二元运算，此前零覆盖。
    # **rminus = s - in、rdiv = s / in 是反向运算**（.c 里 vsubq_f32(y,x) / vdivq_f32(y,x)），
    # 按名字推会整个搞反 —— 门禁的标量特意取负数来暴露这一点。
    'zq_nchw_scalop': [
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_c/zq_cnn_scalaroperation_32f_align_c.c -o $WDIR/zq_scalop.o',
    ],
    # zq_nchw_sqrtnrm（附录 CS）：NCHW 的 sqrt(1) + normalize(5) 共 6 个 32f 入口。
    # normalize 是 **L2 归一化、不减均值**；每个入口都跑一次**全零输入** ——
    # 那正是附录 CR 修掉的 eps 缺失会算出 NaN 的场景。
    'zq_nchw_sqrtnrm': [
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_c/zq_cnn_sqrt_32f_align_c.c -o $WDIR/zq_sn_sqrt.o',
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_c/zq_cnn_normalize_32f_align_c.c -o $WDIR/zq_sn_nrm.o',
    ],
    # zq_bns 登记在这里是为了让 EXTRA_SOURCES 覆盖到它；它已在正常回归里。
    'zq_bns': [
        'gcc -O1 -g -mavx2 -mfma -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/math/zq_sse_mathfun.c -o $WDIR/zq_bns_sse.o',
        'gcc -O1 -g -mavx2 -mfma -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/math/zq_avx_mathfun.c -o $WDIR/zq_bns_avx.o',
    ],
}
EXTRA_LINK = {'zq_innerproduct': ' $WDIR/zq_ipgemm.o $WDIR/zq_gemm_align.o $WDIR/zq_gemm_asm.o $WDIR/zq_gemm_auto.o',
              'zq_nchwc_conv': (' $WDIR/zq_nchwcv.o $WDIR/zq_nchwcv_resize.o '
                                '$WDIR/zq_nchwcv_gemm_align.o $WDIR/zq_nchwcv_gemm_asm.o '
                                '$WDIR/zq_nchwcv_gemm_auto.o $WDIR/zq_nchwcv_tensor.o'),
              'zq_nchwc_conv8': (' $WDIR/zq_nchwcv8.o $WDIR/zq_nchwcv8_resize.o '
                                 '$WDIR/zq_nchwcv8_gemm_align.o $WDIR/zq_nchwcv8_gemm_asm.o '
                                 '$WDIR/zq_nchwcv8_gemm_auto.o $WDIR/zq_nchwcv8_tensor.o'),
              'zq_nchw_conv': (' $WDIR/zq_nchwcv.o $WDIR/zq_nchwcv_gemm_align.o '
                               '$WDIR/zq_nchwcv_gemm_asm.o $WDIR/zq_nchwcv_gemm_auto.o'),
              'zq_nchwc_depthwise': (' $WDIR/zq_dw.o $WDIR/zq_dw_resize.o $WDIR/zq_dw_tensor.o'),
              'zq_nchwc_act': (' $WDIR/zq_act_prelu.o $WDIR/zq_act_addbias.o '
                               '$WDIR/zq_act_resize.o $WDIR/zq_act_tensor.o'),
              'zq_nchwc_elt_relu': (' $WDIR/zq_er_relu.o $WDIR/zq_er_elt.o '
                                    '$WDIR/zq_er_resize.o $WDIR/zq_er_tensor.o'),
              'zq_nchwc_pool': (' $WDIR/zq_pool8.o $WDIR/zq_pool8_resize.o $WDIR/zq_pool8_tensor.o'),
              'zq_nchwc_bn': (' $WDIR/zq_bn.o $WDIR/zq_bn_resize.o $WDIR/zq_bn_tensor.o'),
              'zq_nchwc_softmax': (' $WDIR/zq_sm.o $WDIR/zq_sm_resize.o $WDIR/zq_sm_tensor.o'),
              'zq_nchwc_resize': (' $WDIR/zq_rz.o $WDIR/zq_rz_tensor.o'),
              'zq_nchw_resize': (' $WDIR/zq_rzn.o $WDIR/zq_rzn_tensor.o'),
              'zq_nchw_sqrtnrm': ' $WDIR/zq_sn_sqrt.o $WDIR/zq_sn_nrm.o',
              'zq_nchw_scalop': ' $WDIR/zq_scalop.o',
              'zq_nchw_depthwise': ' $WDIR/zq_dwnchw.o',
              'zq_nchw_act': (' $WDIR/zq_nact_relu.o $WDIR/zq_nact_prelu.o '
                              '$WDIR/zq_nact_addbias.o $WDIR/zq_nact_dropout.o '
                              '$WDIR/zq_nact_softmax.o $WDIR/zq_nact_bn.o'),
              'zq_gemm_shape': (' $WDIR/zq_shape_gemm_align.o $WDIR/zq_shape_gemm_asm.o '
                                '$WDIR/zq_shape_gemm_auto.o'),
              'zq_nchwc_ip': (' $WDIR/zq_nchwc_ip.o $WDIR/zq_nchwc_resize.o '
                              '$WDIR/zq_nchwcip_kernel.o $WDIR/zq_nchwcip_gemm_align.o '
                              '$WDIR/zq_nchwcip_gemm_asm.o $WDIR/zq_nchwcip_gemm_auto.o '
                              '$WDIR/zq_nchwcip_tensor.o'),
              'zq_lrn': ' $WDIR/zq_lrn_sse.o $WDIR/zq_lrn_avx.o',
              'zq_pool': '',
              'zq_bns': ' $WDIR/zq_bns_sse.o $WDIR/zq_bns_avx.o',
              'zq_eltwise': ' $WDIR/zq_eltwise_sse.o $WDIR/zq_eltwise_avx.o'}
EXTRA_INC = {'zq_facedb': ' -I$R -I$R/ZQCNN -I$R/ZQCNN/3rdparty/include/ZQlib',
             'zq_facedb2': ' -I$R -I$R/ZQCNN -I$R/ZQCNN/3rdparty/include/ZQlib',
             'zq_innerproduct': ' -I$R/ZQCNN -I$R/ZQ_GEMM',
             'zq_gemm_shape': ' -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
             'zq_nchwc_ip': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
             'zq_nchwc_conv': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
             'zq_nchwc_conv8': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
             'zq_nchw_conv': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
             'zq_nchwc_depthwise': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
             'zq_nchwc_act': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
             'zq_nchwc_elt_relu': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
             'zq_nchwc_pool': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
             'zq_nchwc_bn': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
             'zq_nchwc_softmax': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
             'zq_nchwc_resize': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
             'zq_nchw_resize': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
             'zq_nchw_sqrtnrm': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
             'zq_nchw_scalop': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
             'zq_nchw_depthwise': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
             'zq_nchw_act': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
             'zq_lrn': ' -I$R/ZQCNN -I$R/ZQ_GEMM',
             'zq_pool': ' -I$R/ZQCNN -I$R/ZQ_GEMM',
             'zq_bns': ' -I$R/ZQCNN -I$R/ZQ_GEMM',
             'zq_eltwise': ' -I$R/ZQCNN -I$R/ZQ_GEMM'}
# 测内核的测试自己也 include 了那个 .c，所以**主 TU 也要带 -mavx2 -mfma**，
# 否则 _mm256_set1_ps 这些 always_inline 内建会报
# "target specific option mismatch"（2026-10-02 实测）。
EXTRA_CXXFLAGS = {'zq_facedb': ' -mavx2 -mfma -fopenmp',
                  'zq_facedb2': ' -mavx2 -mfma -fopenmp',
                  'zq_innerproduct': ' -mavx2 -mfma -fopenmp',
                  'zq_gemm_shape': ' -mavx2 -mfma -fopenmp',
                  'zq_nchwc_ip': ' -mavx2 -mfma -fopenmp',
                  'zq_nchwc_conv': ' -mavx2 -mfma -fopenmp',
                  'zq_nchwc_conv8': ' -mavx2 -mfma -fopenmp',
                  'zq_nchw_conv': ' -mavx2 -mfma -fopenmp',
                  'zq_nchwc_depthwise': ' -mavx2 -mfma -fopenmp',
                  'zq_nchwc_act': ' -mavx2 -mfma -fopenmp',
                  'zq_nchwc_elt_relu': ' -mavx2 -mfma -fopenmp',
                  'zq_nchwc_pool': ' -mavx2 -mfma -fopenmp',
                  'zq_nchwc_bn': ' -mavx2 -mfma -fopenmp',
                  'zq_nchwc_softmax': ' -mavx2 -mfma -fopenmp',
                  'zq_nchwc_resize': ' -mavx2 -mfma -fopenmp',
                  'zq_nchw_resize': ' -mavx2 -mfma -fopenmp',
                  'zq_nchw_sqrtnrm': ' -mavx2 -mfma -fopenmp',
                  'zq_nchw_scalop': ' -mavx2 -mfma -fopenmp',
                  'zq_nchw_depthwise': ' -mavx2 -mfma -fopenmp',
                  'zq_nchw_act': ' -mavx2 -mfma -fopenmp',
                  'zq_lrn': ' -mavx2 -mfma',
                  'zq_pool': ' -mavx2 -mfma', 'zq_bns': ' -mavx2 -mfma',
                  'zq_eltwise': ' -mavx2 -mfma'}


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument('filter', nargs='?', default='')
    ap.add_argument('--list', action='store_true')
    # 明确不自动跑、但保留在仓库里的测试。
    # 每一个都要写清理由 —— "跑不过所以不跑"和"它是已知的未修项所以不跑"
    # 是两件完全不同的事，混起来就成了"无法验证"那个自我实现的结论（附录 W）。
    # 2026-10-02 一度把 zq_bns 放在这里（理由是「主内核索引约定已知未决」）。
    # 核实之后那个理由不成立：NCHWC 的布局是 [n][c片][h][w][align]，
    # 三个内层步长（imStep / sliceStep / widthStep / align）**全对**。
    # 现在 99 个用例逐位精确（相对误差 0.00e+00），它回到正常回归里。
    #
    # 2026-10-02 追加的两个（附录 BN）。**这两个是「已定位、未修」的项**，
    # 不是「跑不过所以不跑」—— 区别必须写清楚，否则就成了"无法验证"那个
    # 自我实现的结论（附录 W）。
    # 2026-10-02：zq_gemm_shape 与 zq_nchwc_ip 曾在 SKIP 里（附录 BN.3 / BO），
    # 理由是"已定位未修"。附录 BP 把两处都修掉之后，它们从 SKIP 里移除了 ——
    # 移出之前先确认过它们**真的**是绿的（不是被我改坏成"全部跳过"的假绿，
    # 见附录 BP.5）。
    # 2026-10-02：zq_nchwc_conv 曾因 kernel2x2_C3 整支全错而登记在 SKIP
    # （理由见附录 BX）。附录 CB 把它定位成**三处互相独立的缺陷**并全部修掉：
    #   1) filter im2col 循环漏了 cp_dst_ptr += matrix_B_rows -> 所有 filter 叠写到同一处
    #   2) matrix_A_cols 用了 filter_H*filter_W*align_C，K 与 matrix_B_rows 不等
    #      -> A 的尾巴没初始化 + 越界读 B
    #   3) 没有 kernel3x3_C3 那样的 align>=4 / #else 分支 -> align=1 时把平面布局
    #      当成交错布局读
    # 修完 align=1/4/8 逐格全对，zq_nchwc_conv 与新登记的 zq_nchwc_conv8 都转绿，
    # **移出 SKIP 之前先确认过它们真的是绿的**（附录 CB.3，不是改坏成"全跳过"的假绿）。
    SKIP = {}
    ap.add_argument('--with-slow', action='store_true',
                    help='连那些编译特别慢的测试一起跑（zq_innerproduct 要链 ZQ_GEMM 的'
                         '三个 TU，其中 zq_gemm_32f_align_c.c 单个 >5 分钟）')
    ap.add_argument('--no-asan', action='store_true',
                    help='不带 sanitizer 编译（想先确认能不能编过时用）')
    ap.add_argument('--ubsan', action='store_true',
                    help='用 -fsanitize=undefined 代替 address：抓有符号溢出 / 移位越界 / '
                         '空指针解引用 / 未对齐访问等 ASan 看不见的行为（见附录 AS.2）')
    args = ap.parse_args()

    if args.ubsan and args.no_asan:
        print('--ubsan 和 --no-asan 不能同时给')
        return 1
    san = '' if args.no_asan else ('-fsanitize=undefined' if args.ubsan
                                   else '-fsanitize=address')

    srcs = sorted(glob.glob(os.path.join(HERE, 'zq_*_check.cpp')))
    srcs = [s for s in srcs if args.filter in os.path.basename(s)]
    # 除非显式点名（args.filter 命中），否则跳过 SKIP 里的那几个，并**把理由打出来**。
    skipped = []
    kept = []
    SLOW = {'zq_gemm_shape': '要链 ZQ_GEMM 的三个 TU（zq_gemm_32f_align_c.c 单个 >5 分钟）',
            'zq_nchwc_ip': '要链 ZQ_GEMM 的四个 TU（含编 5 分钟以上的 zq_gemm_32f_align_c.c）',
            'zq_nchwc_conv': '同上（同一个编 5 分钟的 zq_gemm_32f_align_c.c）',
            'zq_nchwc_conv8': '同上（同一个编 5 分钟的 zq_gemm_32f_align_c.c）',
            'zq_nchw_conv': '要编两个大 TU（zq_cnn_convolution_gemm_32f_align_c.c 与 '
                            'zq_gemm_32f_align_c.c，各 5 分钟以上）',
            'zq_facedb': '要 OpenCV 头 + OpenCV 路径探测，ASan 下编一次约 3 分钟',
            'zq_facedb2': '同上（同一套 OpenCV 头探测）；另外用例 5 要跑 8 轮 x 8 线程',
            'zq_innerproduct': '要链 ZQ_GEMM 的三个 TU，编译 >5 分钟；'
                               '用 --with-slow 才跑'}
    if not args.with_slow:
        skipped_slow = []
        kept2 = []
        for s in srcs:
            tag2 = os.path.splitext(os.path.basename(s))[0][:-6]
            if tag2 in SLOW:
                skipped_slow.append((tag2, SLOW[tag2]))
            else:
                kept2.append(s)
        for tag2, why in skipped_slow:
            print('跳过（慢）%s: %s' % (tag2, why))
        srcs = kept2

    for s in srcs:
        tag = os.path.splitext(os.path.basename(s))[0][:-6]
        if tag in SKIP and tag not in args.filter:
            skipped.append((tag, SKIP[tag]))
        else:
            kept.append(s)
    for tag, why in skipped:
        print('跳过 %s: %s\n' % (tag, why))
    srcs = kept
    if not srcs:
        print('no zq_*_check.cpp matches %r' % args.filter)
        return 1

    print('找到 %d 个测试:' % len(srcs))
    for s in srcs:
        print('   %s' % os.path.basename(s))
    if args.list:
        return 0

    lines = ['set +e',
             'R=/mnt/d/ZQCNN',
             'WDIR=/tmp/zqchecks',
             'cd $WDIR && rm -rf * && mkdir -p $WDIR']
    for s in srcs:
        fname = os.path.basename(s)              # zq_xxx_check.cpp
        stem = fname[:-4]                        # zq_xxx_check
        tag = stem[:-6] if stem.endswith('_check') else stem
        for extra in EXTRA_SOURCES.get(tag, []):
            lines.append(extra)
        lines.append(
            "if g++ -O1 -g %s%s -I%s%s /mnt/d/ZQCNN/tools/%s%s -o %s "
            "2> %s.build.log; then echo 'B|%s|OK|'; else "
            "echo \"B|%s|BUILD_FAIL|$(grep -m1 -i error: %s.build.log | tr -d '\\r')\"; fi"
            % ('' if args.no_asan else san,
               EXTRA_CXXFLAGS.get(tag, ''),
               INC, EXTRA_INC.get(tag, ''), fname, EXTRA_LINK.get(tag, ''), tag,
               tag, tag, tag, tag))
        # 两套 sanitizer 的失败口径不同，分开写：
        #   ASan  -> 断言自己打的 "FAIL" 行数 + 进程非 0（越界/释放后使用会直接 abort）
        #   UBSan -> "runtime error:" 行数。**不要指望 rc**：不加
        #            -fno-sanitize-recover=all 的话 UBSan 只打一行就继续跑，rc 恒为 0，
        #            那一栏永远是 0 等于没查（2026-10-02 实测）。
        if args.ubsan:
            lines.append(
                "if [ -x ./%s ]; then UBSAN_OPTIONS=print_stacktrace=1 ./%s > %s.out 2>&1; "
                "echo \"R|%s|$?|$(grep -c 'runtime error:' %s.out)|"
                "$(grep -cE 'FAIL' %s.out)\"; fi"
                % (tag, tag, tag, tag, tag, tag))
        else:
            lines.append(
                "if [ -x ./%s ]; then ASAN_OPTIONS=detect_leaks=1 ./%s > %s.out 2>&1; "
                "echo \"R|%s|$?|$(grep -cE 'FAIL' %s.out)|0\"; fi"
                % (tag, tag, tag, tag, tag))
    lines.append('echo R|__END__|0|0')
    out = run_wsl('\n'.join(lines))

    build_fail, results = [], []
    for line in out.splitlines():
        if line.startswith('B|'):
            _, name, st, msg = (line.split('|', 3) + [''])[:4]
            if st != 'OK':
                build_fail.append((name, msg.strip()))
        elif line.startswith('R|') and '__END__' not in line:
            parts = line.split('|')
            if len(parts) >= 5:
                results.append((parts[1], parts[2], parts[3], parts[4]))

    print()
    nfail = 0
    for name, rc, nsan, nassert in results:
        ok = (rc == '0' and nsan == '0' and nassert == '0')
        if not ok:
            nfail += 1
        why = []
        if rc != '0':
            why.append('rc=%s' % rc)
        if nsan != '0':
            why.append('%s 条 sanitizer 报错' % nsan)
        if nassert != '0':
            why.append('%s 条断言失败' % nassert)
        print('%-34s %s' % (name, 'PASS' if ok else 'FAIL (%s)' % ', '.join(why)))
    for name, msg in build_fail:
        nfail += 1
        print('%-34s BUILD FAIL: %s' % (name, msg))

    print('\n%d/%d 通过' % (len(results) + len(build_fail) - nfail,
                           len(results) + len(build_fail)))
    if nfail:
        # 把失败的输出打出来，否则只知道失败不知道失败在哪。
        # UBSan 的栈可能落在最后 30 行之外（前面一堆正常运行日志），所以给到 80 行。
        for name, rc, nsan, nassert in results:
            if rc != '0' or nsan != '0' or nassert != '0':
                detail = run_wsl("cat /tmp/zqchecks/%s.out 2>/dev/null | tail -80" % name)
                print('\n===== %s =====\n%s' % (name, detail))
    return 1 if nfail else 0


if __name__ == '__main__':
    sys.exit(main())
