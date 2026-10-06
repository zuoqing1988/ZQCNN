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
import re
import shutil
import subprocess
import sys
import time

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
    # zq_resize_align（附录 ID）：`ResizeBilinearRect` 在 Align0 / Align128bit /
    # Align256bit 三种对齐上的**一致性**。用的就是 zq_nchw_resize 那两个 TU，
    # 只是编成另一份 .o（两道门禁不能共用同一个 .o：并行跑会互相覆盖）。
    'zq_resize_align': [
        'gcc -O1 -g -mavx2 -mfma $SAN -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_c/zq_cnn_resize_32f_align_c.c -o $WDIR/zq_rza.o',
        'g++ -O1 -g -mavx2 -mfma $SAN -fopenmp -c -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Tensor4D.cpp -o $WDIR/zq_rza_tensor.o',
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
    # zq_nchw_conv_free（附录 CU）：NCHW no_padding gemm 的**内存归属**。
    # 与 zq_nchw_conv 共用同一份实现 TU（zq_cnn_convolution_gemm_32f_align_c.c）。
    # **只需要 zq_cvf.o 一个 TU**。GEMM 由门禁自己桩掉（见 zq_nchw_conv_free_check.cpp
    # 顶部「GEMM 桩」一节）：这道门禁只问「释放的是不是自己的内存」「函数跑没跑」，
    # 数值由 zq_nchw_conv 负责。带上 ZQ_GEMM 那三个 TU 的话，光编译就要 7 分钟
    # （zq_gemm_32f_align_c.c 单个就 >5 分钟），门禁就只能挂 --with-slow；
    # 桩掉之后编一次 3.6 秒，于是它能进**每次**回归。
    'zq_nchw_conv_free': [
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_c/zq_cnn_convolution_gemm_32f_align_c.c -o $WDIR/zq_cvf.o',
    ],
    # zq_nchw_deconv（附录 CX）：NCHW 的转置卷积，7 个 32f 入口。
    # **仓内零调用方**（ZQ_CNN_Layer.h 里一处 deconv 都没有，model/ 下也没有），
    # 所以它是唯一一块"跑样本永远发现不了"的代码 —— 门禁是它唯一的防线。
    'zq_nchw_deconv': [
        # **实现 TU 必须带 $SAN**（见 EXTRA_SOURCES 处的说明）：这道门禁的判据
        # 就是"ASan 有没有报越界读"，库不插桩的话它永远报不出来。
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_c/zq_cnn_deconvolution_32f_align_c.c -o $WDIR/zq_dec.o',
    ],
    # zq_tile（附录 DD）：ZQ_CNN_Tensor4D::Tile。
    # **必须编 ZQ_CNN_Tensor4D.cpp**（Tile 是基类里的虚函数，绕不开真实张量对象），
    # 但那个 TU 带 ASan 编一次只要约 3 秒（实测），所以能进每次回归。
    'zq_tile': [
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Tensor4D.cpp -o $WDIR/zq_t4d.o',
        # ZQ_CNN_Tensor4D.cpp 里的 ResizeNearest / Remap 方法要调 resize/remap 内核，
        # 链接期就得带上（哪怕这道门禁一个 resize 都没跑）。
        # remap 那几个符号在**同一个** zq_cnn_resize_32f_align_c.c 里 ——
        # 第一版按名字猜成两个文件，第二个直接不存在，链接报
        # "zq_tile_rm.o: No such file or directory"。
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_c/zq_cnn_resize_32f_align_c.c -o $WDIR/zq_tile_rz.o',
    ],
    # zq_roi（附录 DX）：ZQ_CNN_Tensor4D::ROI 的边界检查。
    # 与 zq_tile 同理，必须用真实张量对象 → 要编 ZQ_CNN_Tensor4D.cpp + resize 内核。
    'zq_roi': [
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Tensor4D.cpp -o $WDIR/zq_roi_t4d.o',
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_c/zq_cnn_resize_32f_align_c.c -o $WDIR/zq_roi_rz.o',
    ],
    # zq_tensorop（附录 EE）：ZQ_CNN_Tensor4D 的就地运算
    # （FlipX / FlipY / AddScalar / MulScalar）。与 zq_tile / zq_roi 同理，必须用真实张量对象。
    'zq_tensorop': [
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Tensor4D.cpp -o $WDIR/zq_tensorop_t4d.o',
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_c/zq_cnn_resize_32f_align_c.c -o $WDIR/zq_tensorop_rz.o',
    ],
    # zq_layerwire（附录 EC）：UNUSED 层类型的**接线**。
    # 要编 ZQ_CNN_Layer.h（10455 行）+ ZQ_CNN_Tensor4D.cpp + resize 内核。
    'zq_layerwire': [
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Tensor4D.cpp -o $WDIR/zq_layerwire_t4d.o',
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_c/zq_cnn_resize_32f_align_c.c -o $WDIR/zq_layerwire_rz.o',
        # **不编 ZQ_CNN_Forward_SSEUtils.cpp**（附录 EC.1）：它一个 TU 引用半个库
        # （addbias / prelu / avgpooling / batchnorm / conv / conv_gemm …），
        # 最后会拖进 zq_cnn_convolution_gemm_32f_align_c.c（单编 5 分钟以上），
        # 只能挂进 SLOW 集合。门禁里自己定义那十几个 static 方法当**记录桩**。
    ],
    # zq_concat_alias（附录 EN）：Concat 的 top/bottom **跨下标别名**。
    # 走真的 ZQ_CNN_Net::LoadFrom -> _check_connect，所以**必须**编 ZQ_CNN_Net.h，
    # 而那会把 45 个 ZQ_CNN_Forward_SSEUtils 辅助函数拖成未定义符号 ——
    # tools/zq_net_fwd_tripwires.h（由 tools/gen_net_fwd_tripwires.py 从链接器
    # 的未定义符号表生成）提供**绊线**桩，不需要编那个 5 分钟的 Forward_SSEUtils.cpp。
    # 依赖的 TU 与 zq_layerwire 完全相同。
    'zq_concat_alias': [
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Tensor4D.cpp -o $WDIR/zq_concalias_t4d.o',
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_c/zq_cnn_resize_32f_align_c.c -o $WDIR/zq_concalias_rz.o',
    ],
    # zq_convparam（附录 EO.7）：卷积 kernel/dilate 整数溢出守卫（EM 的门禁）。
    # **EM.5 当年记的"要拖 2215 个符号、拖不进快速门禁"是错的** ——
    # `new ZQ_CNN_Layer_Convolution()` 实际只拖出 6 个未定义符号，
    # 且全是 ZQ_CNN_Forward_SSEUtils 的辅助函数，正是 zq_net_fwd_tripwires.h
    # （44 个绊线）覆盖的那一族。所以这道门禁在**快速通道**，不进 SLOW。
    'zq_convparam': [
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Tensor4D.cpp -o $WDIR/zq_convparam_t4d.o',
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_c/zq_cnn_resize_32f_align_c.c -o $WDIR/zq_convparam_rz.o',
    ],
    # zq_model_params（附录 FD）：每个随仓库 .zqparams 都必须走完
    # ReadParam 与 _check_connect —— 也就是**附录 EN 改的那一段**。
    # 权重是 66 MB / 27 个 .nchwbin，进不了快速通道；这里给一个**故意不存在**的
    # 权重路径，于是 LoadFrom 只会停在 "failed to open"，而任何参数级拒绝
    # （unknown blob / changes shape but declares top == bottom / missing ...）
    # 都会以不同的消息被抓出来 —— 两种失败都返回 false，只有区分消息才能说明
    # "是权重没找到"而不是"模型被拒"。
    'zq_model_params': [
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Tensor4D.cpp -o $WDIR/zq_modelparams_t4d.o',
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_c/zq_cnn_resize_32f_align_c.c -o $WDIR/zq_modelparams_rz.o',
    ],
    # zq_weight_tail（附录 GZ）：`.zqparams` 声明的层尺寸加总 == `.nchwbin` 长度。
    # 与 zq_model_params 守的是**同一件事**，但走的是另一条路：它在库**外面**，
    # 用库自己的 SaveModel 回存一遍比长度，**一行 warning 文本都不依赖**。
    # 留两条的理由见附录 GZ.4 —— 少任何一条，剩下的那条就可能被改措辞、
    # 改重定向悄悄弄失效，而失效的方式是"变绿"，不会有人发现。
    'zq_weight_tail': [
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Tensor4D.cpp -o $WDIR/zq_weighttail_t4d.o',
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_c/zq_cnn_resize_32f_align_c.c -o $WDIR/zq_weighttail_rz.o',
    ],
    # zq_loadbuffer（附录 HA）：`LoadFromBuffer` 路径。它原来**零行为覆盖** ——
    # 36 个 `LoadBinary_NCHW(buffer,…)` 重载全在门禁之外，只有一个 Linux sample
    # 顺带跑到过，而那条路径的"字节不够"检查与文件路径**不是同一份代码**。
    'zq_loadbuffer': [
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Tensor4D.cpp -o $WDIR/zq_loadbuf_t4d.o',
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_c/zq_cnn_resize_32f_align_c.c -o $WDIR/zq_loadbuf_rz.o',
    ],
    # zq_weight_roundtrip（附录 HB）：`zq_weight_tail` **只比长度**，
    # 而 SaveBinary_NCHW 与 LoadBinary_NCHW 是两份独立实现 ——
    # 布局理解不一致时长度照样相等、内容已经错了。这道门禁比**内容**。
    'zq_weight_roundtrip': [
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Tensor4D.cpp -o $WDIR/zq_wrt_t4d.o',
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_c/zq_cnn_resize_32f_align_c.c -o $WDIR/zq_wrt_rz.o',
    ],
    # zq_nchwc_roundtrip（附录 HC）：**NCHWC 那一侧**的往返。
    # zq_weight_tail / zq_weight_roundtrip / zq_loadbuffer 三道都只跑
    # ZQ_CNN_Net（NCHW）。而 ZQ_CNN_Net_NCHWC 是一份独立的模板实现，
    # 有自己的一套 LoadBinary_NCHW / SaveBinary_NCHW 与 _prepack()，往返零覆盖。
    # zq_nchwc_variants（附录 HD）：ZQ_CNN_Net_NCHWC 的**三个**张量变体。
    # 生产里只实例化过 NCHWC4，NCHWC1/NCHWC8 的**前向实现**却存在 ——
    # 「有人写了运行时代码却没有任何调用方」。这道门禁逐个变体分别编一遍。
    # zq_nchwc_v1（附录 HD）：ZQ_CNN_Net_NCHWC 的第 1 个张量变体。
    # 三者共用同一份实现（zq_nchwc_variants_body.h），这里只差一个 -D。
    # 生产里只实例化过 NCHWC4；另两个的**前向实现**却存在 ——
    # "有人写了运行时代码却没有任何调用方"。
    'zq_nchwc_v1': [
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_c/zq_cnn_resize_32f_align_c.c -o $WDIR/zq_nchwcv1_rz.o',
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_nchwc/zq_cnn_resize_nchwc.c -o $WDIR/zq_nchwcv1_rzn.o',
        'g++ -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Tensor4D.cpp -o $WDIR/zq_nchwcv1_t4d.o',
        'g++ -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Tensor4D_NCHWC.cpp -o $WDIR/zq_nchwcv1_tn.o',
        'g++ -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Net_NCHWC.cpp -o $WDIR/zq_nchwcv1_net.o',
    ],
    # zq_nchwc_v4（附录 HD）：ZQ_CNN_Net_NCHWC 的第 4 个张量变体。
    # 三者共用同一份实现（zq_nchwc_variants_body.h），这里只差一个 -D。
    # 生产里只实例化过 NCHWC4；另两个的**前向实现**却存在 ——
    # "有人写了运行时代码却没有任何调用方"。
    'zq_nchwc_v4': [
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_c/zq_cnn_resize_32f_align_c.c -o $WDIR/zq_nchwcv4_rz.o',
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_nchwc/zq_cnn_resize_nchwc.c -o $WDIR/zq_nchwcv4_rzn.o',
        'g++ -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Tensor4D.cpp -o $WDIR/zq_nchwcv4_t4d.o',
        'g++ -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Tensor4D_NCHWC.cpp -o $WDIR/zq_nchwcv4_tn.o',
        'g++ -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Net_NCHWC.cpp -o $WDIR/zq_nchwcv4_net.o',
    ],
    # zq_nchwc_v8（附录 HD）：ZQ_CNN_Net_NCHWC 的第 8 个张量变体。
    # 三者共用同一份实现（zq_nchwc_variants_body.h），这里只差一个 -D。
    # 生产里只实例化过 NCHWC4；另两个的**前向实现**却存在 ——
    # "有人写了运行时代码却没有任何调用方"。
    'zq_nchwc_v8': [
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_c/zq_cnn_resize_32f_align_c.c -o $WDIR/zq_nchwcv8_rz.o',
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_nchwc/zq_cnn_resize_nchwc.c -o $WDIR/zq_nchwcv8_rzn.o',
        'g++ -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Tensor4D.cpp -o $WDIR/zq_nchwcv8_t4d.o',
        'g++ -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Tensor4D_NCHWC.cpp -o $WDIR/zq_nchwcv8_tn.o',
        'g++ -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Net_NCHWC.cpp -o $WDIR/zq_nchwcv8_net.o',
    ],
    'zq_nchwc_roundtrip': [
        # **两个** resize 内核都要：NCHWC 的 net 也会引用 NCHW 那个张量类，
        # 只编 zq_cnn_resize_nchwc.c 会在链接期报一串
        # undefined reference to zq_cnn_resize_with_safeborder_32f_align0（照 zq_nchwc_net）。
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_c/zq_cnn_resize_32f_align_c.c -o $WDIR/zq_nchwcwz_rz.o',
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_nchwc/zq_cnn_resize_nchwc.c -o $WDIR/zq_nchwcwz_rzn.o',
        'g++ -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Tensor4D.cpp -o $WDIR/zq_nchwcwz_t4d.o',
        'g++ -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Tensor4D_NCHWC.cpp -o $WDIR/zq_nchwcwz_tn.o',
        'g++ -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Net_NCHWC.cpp -o $WDIR/zq_nchwcwz_net.o',
    ],
    # zq_nchwc_net（附录 FB）：`ZQ_CNN_Net_NCHWC` 的 net 级就地守卫。
    # EN 修就地守卫时**同时改了两份** Net（ZQ_CNN_Net.h 与 ZQ_CNN_Net_NCHWC.h，
    # 各自独立的拷贝），但两边覆盖极不对等：NCHW 那份有 zq_concat_alias 走真的
    # LoadFrom 验，**NCHWC 这份只有编译覆盖**（被主工程 CMake 编过），
    # 行为上零门禁 —— 也就是说那处镜像改动从来没有被任何东西验证过。
    # 与 ES.2 同一个形状，只是更隐蔽：它**编得过**，所以"能编过"这道轴也照不到。
    # zq_unusedlayers（附录 GE）：剩下三个 UNUSED 层的 ReadParam。
    # DeConvolution 已在 zq_convparam 里；这三个的守卫**全在 ReadParam**，
    # 而 ReadParam 不碰 Forward —— 所以不需要把绊线换成记录桩，
    # 附录 EY.4 记的「需要双模式桩」那个理由只对 Forward 成立。
    'zq_unusedlayers': [
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Tensor4D.cpp -o $WDIR/zq_unusedlayers_t4d.o',
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_c/zq_cnn_resize_32f_align_c.c -o $WDIR/zq_unusedlayers_rz.o',
    ],
    'zq_nchwc_net': [
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Tensor4D.cpp -o $WDIR/zq_nchwcnet_t4d.o',
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_c/zq_cnn_resize_32f_align_c.c -o $WDIR/zq_nchwcnet_rz.o',
        # **没有** ZQ_CNN_Layer.cpp —— 层是纯头文件（类体内定义），
        # 所以只需要编 Net_NCHWC 这一个 .cpp。
        'g++ -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Net_NCHWC.cpp -o $WDIR/zq_nchwcnet_net.o',
        'g++ -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Tensor4D_NCHWC.cpp -o $WDIR/zq_nchwcnet_tensor.o',
        # NCHWC 张量类的 Resize* 要调 zq_cnn_resize_*_nchwc{1,4,8}，链接期必须带上。
        # 注意那个 .c 要用 **gcc** 编（g++ 会把它判成 narrowing，附录 AU.2）。
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_nchwc/zq_cnn_resize_nchwc.c -o $WDIR/zq_nchwcnet_rzn.o',
    ],
    # zq_facegroup（附录 EI）：ZQlibFaceID 的文件读入行为。
    # 这是 ZQlibFaceID 里**唯一不需要外部库**的一组（其余头都 include 了
    # OpenCV / ncnn / SeetaFace，本机没有 Linux 库，链不过），所以也是
    # 唯一能真正在 Linux 上跑行为门禁的地方。头文件本身，无需 EXTRA_SOURCES。
    'zq_facegroup': [],
    # zq_lfw_eval（附录 IH）：`ZQ_FaceIDPrecisionEvaluation::EvaluationOnLFW`。
    # 这是 ZQlibFaceID 里**唯一** include 了 OpenCV、因而"零行为门禁"的头，
    # 而它的逻辑量恰恰是同目录里最大的（解析 list / 抽特征 / 留一法定阈值 / FAR-TAR）。
    # 靠 tools/opencv_stub/opencv2/opencv.hpp（只给 Mat / imread / flip 三样）解开。
    # 头文件本身，无需 EXTRA_SOURCES。
    'zq_lfw_eval': [],
    # zq_nchwc_tensor（附录 EA）：ZQ_CNN_Tensor4D_NCHWC **自己那批方法**
    # （Convert 族 / Permute / Flatten / Reshape）。该类被 9 道门禁当数据容器用，
    # 但自己的方法一个门禁都没有 —— 与 DX.6 在基类上发现的缺口同一个形状。
    # 只需编 ZQ_CNN_Tensor4D_NCHWC.cpp。
    'zq_nchwc_tensor': [
        # ZQ_CNN_Tensor4D_NCHWC.cpp 里的 Resize* 方法要调 NCHWC 的 resize 内核，
        # 链接期就得带上（哪怕这道门禁一个 resize 都没跑）—— 照 zq_nchwc_resize 的写法。
        # 注意那个 .c 要用 **gcc** 编（g++ 会把它判成 narrowing，附录 AU.2）。
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_nchwc/zq_cnn_resize_nchwc.c -o $WDIR/zq_nchwctensor_rz.o',
        'g++ -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Tensor4D_NCHWC.cpp -o $WDIR/zq_nchwctensor.o',
    ],
    # zq_convert（附录 DZ）：ZQ_CNN_Tensor4D 的 Convert 族。
    # 与 zq_tile / zq_roi / zq_reshape 同理，必须用真实张量对象。
    'zq_convert': [
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Tensor4D.cpp -o $WDIR/zq_convert_t4d.o',
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_c/zq_cnn_resize_32f_align_c.c -o $WDIR/zq_convert_rz.o',
    ],
    # zq_reshape（附录 DY）：ZQ_CNN_Tensor4D::Reshape_NCHW / Flatten_NCHW。
    # 与 zq_tile / zq_roi 同理，必须用真实张量对象 → 编 Tensor4D.cpp + resize 内核。
    # 独立对象文件（不与 zq_tile 共用）：这道门禁的判据是**形状算错**，
    # 共用 .o 会让"某个 .o 没编出来"和"某个用例红"混在同一条日志里。
    'zq_reshape': [
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/ZQ_CNN_Tensor4D.cpp -o $WDIR/zq_reshape_t4d.o',
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_c/zq_cnn_resize_32f_align_c.c -o $WDIR/zq_reshape_rz.o',
    ],
    # zq_lstm（附录 IX）：给 **NCHW 的 LSTM_TF 内核**补一份带**独立参考实现**的门禁。
    # 此前只有 SamplesZQCNN/SampleLSTMTFCalib 那个**标定装置**（附录 IN），
    # 它回答「哪个张量能改变输出」，**不回答「算得对不对」** —— 没有参考实现，
    # 就没有任何一个数字可以判成败（IN.5 当时因此留下两条未解释的观察）。
    # 这里 10 组：(C, hidden, W, N) x 正/反向 x 共用 buffer 的复用分支。
    'zq_lstm': [
        'gcc -O1 -g -mavx2 -mfma $SAN -c -I$R/ZQCNN -I$R/ZQCNN/layers_c '
        '-I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_c/zq_cnn_lstm_32f_align_c.c -o $WDIR/zq_lstm_ref.o',
    ],
    # zq_nchw_lstm（附录 CW）：NCHW 的 LSTM。3 个 32f 入口
    # （align0_general 在 .c 里，align128/256 在 _raw.h 里，宏式声明）。
    'zq_nchw_lstm': [
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_c/zq_cnn_lstm_32f_align_c.c -o $WDIR/zq_lstm.o',
    ],
    # zq_nchw_reduction（附录 CT）：NCHW 的 sum/mean 两个 32f 入口，
    # 5 行（keepdims==0 + axis 0..3）。**axis 的约定是 (N,C,H,W)**，
    # 不是循环嵌套顺序 —— 见 ZQ_CNN_Forward_SSEUtils.h 里 out_dims[4]={N,C,H,W}。
    'zq_nchw_reduction': [
        'gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_c/zq_cnn_reduction_32f_align_c.c -o $WDIR/zq_red.o',
    ],
    # zq_deconv（附录 IW）：转置卷积的内核一直**零覆盖** ——
    # 它是 UNUSED 层（没有任何随仓模型跑得到），而三个 general 内核
    # （align0 / align128bit / align256bit）加起来 800 多行循环。
    # 整条 `ZQ_CNN_Forward_SSEUtils.cpp` 一个 TU 编一次 5 分钟以上（附录 EC.1），
    # 常规门禁都不编它；这里只链**内核那个小 TU**，11 个形状 x 3 条路 = 33 组。
    # $SAN 必须带上：判据里"没有越界读"这一条只有插桩后才算数。
    'zq_deconv': [
        'gcc -O1 -g -mavx2 -mfma $SAN -c -I$R/ZQCNN -I$R/ZQCNN/layers_c '
        '-I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/layers_c/zq_cnn_deconvolution_32f_align_c.c -o $WDIR/zq_deconv.o',
    ],
    # zq_bns 登记在这里是为了让 EXTRA_SOURCES 覆盖到它；它已在正常回归里。
    'zq_bns': [
        'gcc -O1 -g -mavx2 -mfma -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/math/zq_sse_mathfun.c -o $WDIR/zq_bns_sse.o',
        'gcc -O1 -g -mavx2 -mfma -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include '
        '$R/ZQCNN/math/zq_avx_mathfun.c -o $WDIR/zq_bns_avx.o',
    ],
}


def _glob_extra(pattern, out_prefix, inc='-I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
                cc='gcc', opt='-O1', tail='-mavx2 -mfma -fopenmp'):
    """按 glob 展开成一串「编一个 TU」的命令。

    为什么需要它：NCHWC 前向要链上 layers_nchwc 的**全部** .c，
    NCHW 前向要链上 layers_c 的**全部** .c —— 手写清单会漏，
    而链接期的表现是「报一屏 undefined reference」，症状离原因很远
    （2026-10-06 为 `zq_padtype` 连补三版：补 Forward、补 layers_c、
    补 mathfun 都还是红的，第四版才发现 128 位那份在**另一个文件**里）。
    **"补一个报一屏、补两个还报一屏"就是"依赖是整族的"的信号。**

    第一版把 `.c` 从**源路径**里也去掉了（只该从对象名里去），
    拼出来的是 `gcc ... $R/ZQCNN/ZQCNN_layers_c_zq_cnn_addbias_32f_align_c` ——
    没有扩展名、没有目录。22 条命令全错，而门禁报出来的是 **0/0**，
    一个字都没说（见下面的 `assert` 守卫：源文件必须真的存在）。
    """
    import glob as _glob
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.normpath(os.path.join(here, '..'))
    out = []
    for f in sorted(_glob.glob(os.path.join(root, pattern))):
        if not os.path.isfile(f):
            continue
        rel = os.path.relpath(f, root).replace('\\', '/')      # 保留 .c
        stem = rel.replace('/', '_')[:-2]                      # 对象名去掉 .c
        if not os.path.isfile(f):
            raise AssertionError('glob 出来的源文件不存在: %s' % f)
        out.append('%s %s -g $SAN %s -c %s $R/%s -o $WDIR/%s_%s.o'
                   % (cc, opt, tail, inc, rel, out_prefix, stem))
    if not out:
        raise AssertionError('glob %r 一个文件都没匹配到 —— 路径写错了，'
                             '不是"没有需要编的 TU"' % pattern)
    return out


# zq_nchwc_batch（附录 IV）：NCHWC 的 **batch 维不变性**门禁。
# 判据不是"和手写参考比"，而是 out[n](N=2) 必须逐位等于 out[n](N=1)。
# 它不需要任何参考实现，却能覆盖"batch 维步进写错"这一整族 ——
# 附录 IU.1 / IU.2 两条缺陷都是这一族，且都**不越界、不崩、sanitizer 全静默**。
EXTRA_SOURCES['zq_nchwc_batch'] = (
    _glob_extra('ZQCNN/layers_nchwc/*.c', 'nb', cc='gcc') + [
        # ZQ_GEMM 的三个 TU：GEMM 版卷积在链接期就要符号。
        # `zq_gemm_32f_align_c.c` 在 -O1 下单个 TU 编 5 分钟以上（AGENTS.md
        # 「推不动的时候就把排除了什么记下来」那一条），所以这里用 -O0 —— 只求跑通。
        'gcc -O0 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM '
        '-I$R/3rdparty/include $R/ZQ_GEMM/math/zq_gemm_32f_align_c.c -o $WDIR/nb_gemm_align.o',
        'gcc -O0 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM '
        '-I$R/3rdparty/include $R/ZQ_GEMM/math/zq_gemm_32f_align_c_asm.c -o $WDIR/nb_gemm_asm.o',
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM '
        '-I$R/3rdparty/include $R/ZQ_GEMM/math/zq_gemm_32f_auto.c -o $WDIR/nb_gemm_auto.o',
        'g++ -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R -I$R/ZQCNN -I$R/ZQ_GEMM '
        '-I$R/3rdparty/include $R/ZQCNN/ZQ_CNN_Tensor4D_NCHWC.cpp -o $WDIR/nb_tensor.o',
        'g++ -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R -I$R/ZQCNN -I$R/ZQ_GEMM '
        '-I$R/3rdparty/include $R/ZQCNN/ZQ_CNN_Forward_SSEUtils_NCHWC.cpp -o $WDIR/nb_fwd.o',
    ])

# zq_nchwc_poolpad（附录 F2）：NCHWC 池化在 pad != 0 时是否符合「零填充后池化」。
# 判据是**独立参考**而不是 NCHW —— 因为实测 NCHW 那一支自己的窗口起点
# 就没退到 -pad 行（见该文件头的装置），拿它当参考等于把缺陷固化成标准。
# 只需要 NCHWC 那一侧的 TU。
# zq_nchwc_padreject（附录 DH）：NCHWC 卷积/深度卷积必须**拒载**它表达不了的
# padding 键（pad_type / 非对称 pad），而不是 warning 一行就按 pad=0 算。
# 走完整 ZQ_CNN_Net_NCHWC::LoadFrom，所以要带上 Net_NCHWC 那个 TU。
EXTRA_SOURCES['zq_nchwc_padreject'] = (
    _glob_extra('ZQCNN/layers_nchwc/*.c', 'pr', cc='gcc') + [
        'g++ -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R -I$R/ZQCNN -I$R/ZQ_GEMM '
        '-I$R/3rdparty/include $R/ZQCNN/ZQ_CNN_Tensor4D_NCHWC.cpp -o $WDIR/pr_tensor.o',
        'g++ -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R -I$R/ZQCNN -I$R/ZQ_GEMM '
        '-I$R/3rdparty/include $R/ZQCNN/ZQ_CNN_Net_NCHWC.cpp -o $WDIR/pr_net.o',
        # 层的 Forward 是**头文件内联**的，Net_NCHWC 引用了
        # ZQ_CNN_Forward_SSEUtils_NCHWC 的全部实现 —— 不带它就是满屏 undefined reference。
        'g++ -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R -I$R/ZQCNN -I$R/ZQ_GEMM '
        '-I$R/3rdparty/include $R/ZQCNN/ZQ_CNN_Forward_SSEUtils_NCHWC.cpp -o $WDIR/pr_fwd.o',
        # GEMM 的三个 TU：Forward_SSEUtils_NCHWC 里的卷积/全连接要符号。
        # zq_gemm_32f_align_c 在 -O1 下单个 TU 要编 5 分钟以上，所以用 -O0 ——
        # 这里只求跑通，不测性能。
        'gcc -O0 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM '
        '-I$R/3rdparty/include $R/ZQ_GEMM/math/zq_gemm_32f_align_c.c -o $WDIR/pr_ga.o',
        'gcc -O0 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM '
        '-I$R/3rdparty/include $R/ZQ_GEMM/math/zq_gemm_32f_align_c_asm.c -o $WDIR/pr_gb.o',
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM '
        '-I$R/3rdparty/include $R/ZQ_GEMM/math/zq_gemm_32f_auto.c -o $WDIR/pr_gc.o',
    ])

EXTRA_SOURCES['zq_nchwc_poolpad'] = (
    _glob_extra('ZQCNN/layers_nchwc/*.c', 'pp', cc='gcc') + [
        'gcc -O0 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM '
        '-I$R/3rdparty/include $R/ZQ_GEMM/math/zq_gemm_32f_align_c.c -o $WDIR/pp_gemm.o',
        'gcc -O0 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM '
        '-I$R/3rdparty/include $R/ZQ_GEMM/math/zq_gemm_32f_align_c_asm.c -o $WDIR/pp_gemm_asm.o',
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM '
        '-I$R/3rdparty/include $R/ZQ_GEMM/math/zq_gemm_32f_auto.c -o $WDIR/pp_gemm_auto.o',
        'g++ -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R -I$R/ZQCNN -I$R/ZQ_GEMM '
        '-I$R/3rdparty/include $R/ZQCNN/ZQ_CNN_Tensor4D_NCHWC.cpp -o $WDIR/pp_tensor.o',
        'g++ -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R -I$R/ZQCNN -I$R/ZQ_GEMM '
        '-I$R/3rdparty/include $R/ZQCNN/ZQ_CNN_Forward_SSEUtils_NCHWC.cpp -o $WDIR/pp_fwd.o',
    ])

# zq_padtype（附录 IV.1）：NCHW 层 `pad_type` 的输出尺寸与 padding 之和。
# 只实例化层、走 ReadParam -> SetBottomDim -> GetTopDim 这条**生产路径**，
# 不做数值对拍 —— 这一层测的是**契约**。
EXTRA_SOURCES['zq_padtype'] = (
    _glob_extra('ZQCNN/layers_c/*.c', 'pt', cc='gcc') + [
        # math 里的两个：lrn 要 zq_mm128/256_exp_ps / log_ps，
        # 128 位那份在 zq_sse_mathfun.c、256 位那份在 zq_avx_mathfun.c —— **两个文件**。
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM '
        '-I$R/3rdparty/include $R/ZQCNN/math/zq_avx_mathfun.c -o $WDIR/pt_mathfun.o',
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM '
        '-I$R/3rdparty/include $R/ZQCNN/math/zq_sse_mathfun.c -o $WDIR/pt_ssefun.o',
        'gcc -O0 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM '
        '-I$R/3rdparty/include $R/ZQ_GEMM/math/zq_gemm_32f_align_c.c -o $WDIR/pt_gemm.o',
        'gcc -O0 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM '
        '-I$R/3rdparty/include $R/ZQ_GEMM/math/zq_gemm_32f_align_c_asm.c -o $WDIR/pt_gemm_asm.o',
        'gcc -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM '
        '-I$R/3rdparty/include $R/ZQ_GEMM/math/zq_gemm_32f_auto.c -o $WDIR/pt_gemm_auto.o',
        'g++ -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R -I$R/ZQCNN -I$R/ZQ_GEMM '
        '-I$R/3rdparty/include $R/ZQCNN/ZQ_CNN_Tensor4D.cpp -o $WDIR/pt_t4d.o',
        # 层的 Forward 是**头文件内联**的，哪怕一次都不调用也要链上它
        'g++ -O1 -g $SAN -mavx2 -mfma -fopenmp -c -I$R -I$R/ZQCNN -I$R/ZQ_GEMM '
        '-I$R/3rdparty/include $R/ZQCNN/ZQ_CNN_Forward_SSEUtils.cpp -o $WDIR/pt_fwd.o',
    ])
EXTRA_LINK = {'zq_resize_align': ' $WDIR/zq_rza.o $WDIR/zq_rza_tensor.o',
              'zq_innerproduct': ' $WDIR/zq_ipgemm.o $WDIR/zq_gemm_align.o $WDIR/zq_gemm_asm.o $WDIR/zq_gemm_auto.o',
              # zq_scale（附录 IB）：直接链**内核那个小 TU**。
              # `ZQ_CNN_Forward_SSEUtils.cpp` 一个 TU 引用半个库、-O1 编一次 5 分钟以上
              # （附录 EC.1），所以现成的 sanitizer 门禁都**不编它**（用绊线桩）。
              # 但 `Scale` 的实现本身就在 layers_c/zq_cnn_batchnormscale_32f_align_c.c 里，
              # 直接把它链进来就行 —— 越界读发生在**内核里**，不需要整条前向链。
              # 这个 TU 里的四个逐通道内核（scale / batchnorm_b_a /
              # batchnormscale_mean_var_scale_bias，附录 IB / ID）都在这一个门禁里。
              'zq_scale': ' $R/ZQCNN/layers_c/zq_cnn_batchnormscale_32f_align_c.c',
              # zq_deconv（附录 IW）：内核那个小 TU，和 zq_scale 同一路理由。
              'zq_deconv': ' $WDIR/zq_deconv.o',
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
              # 附录 IV 新增两道。链接对象用通配：对象名是按源文件名生成的
              # （见 _glob_extra），手写清单漏一个就是一条 undefined reference。
              'zq_nchwc_batch': ' $WDIR/nb_*.o',
              'zq_nchwc_poolpad': ' $WDIR/pp_*.o',
              'zq_nchwc_padreject': ' $WDIR/pr_*.o',
              'zq_padtype': ' $WDIR/pt_*.o',
              'zq_nchw_resize': (' $WDIR/zq_rzn.o $WDIR/zq_rzn_tensor.o'),
              'zq_nchw_sqrtnrm': ' $WDIR/zq_sn_sqrt.o $WDIR/zq_sn_nrm.o',
              'zq_nchw_reduction': ' $WDIR/zq_red.o',
              'zq_nchw_lstm': ' $WDIR/zq_lstm.o',
              'zq_lstm': ' $WDIR/zq_lstm_ref.o',
              'zq_roi': ' $WDIR/zq_roi_t4d.o $WDIR/zq_roi_rz.o',
              'zq_reshape': ' $WDIR/zq_reshape_t4d.o $WDIR/zq_reshape_rz.o',
              'zq_convert': ' $WDIR/zq_convert_t4d.o $WDIR/zq_convert_rz.o',
              'zq_nchwc_tensor': ' $WDIR/zq_nchwctensor.o $WDIR/zq_nchwctensor_rz.o',
              'zq_layerwire': ' $WDIR/zq_layerwire_t4d.o $WDIR/zq_layerwire_rz.o',
              'zq_concat_alias': ' $WDIR/zq_concalias_t4d.o $WDIR/zq_concalias_rz.o',
              'zq_convparam': ' $WDIR/zq_convparam_t4d.o $WDIR/zq_convparam_rz.o',
              'zq_nchwc_net': (' $WDIR/zq_nchwcnet_t4d.o $WDIR/zq_nchwcnet_rz.o '
                                '$WDIR/zq_nchwcnet_net.o $WDIR/zq_nchwcnet_tensor.o '
                                '$WDIR/zq_nchwcnet_rzn.o'),
              'zq_model_params': ' $WDIR/zq_modelparams_t4d.o $WDIR/zq_modelparams_rz.o',
              'zq_weight_tail': ' $WDIR/zq_weighttail_t4d.o $WDIR/zq_weighttail_rz.o',
              'zq_loadbuffer': ' $WDIR/zq_loadbuf_t4d.o $WDIR/zq_loadbuf_rz.o',
              'zq_weight_roundtrip': ' $WDIR/zq_wrt_t4d.o $WDIR/zq_wrt_rz.o',
              'zq_nchwc_v1': (' $WDIR/zq_nchwcv1_t4d.o $WDIR/zq_nchwcv1_tn.o '
                              '$WDIR/zq_nchwcv1_net.o $WDIR/zq_nchwcv1_rz.o '
                              '$WDIR/zq_nchwcv1_rzn.o'),
              'zq_nchwc_v4': (' $WDIR/zq_nchwcv4_t4d.o $WDIR/zq_nchwcv4_tn.o '
                              '$WDIR/zq_nchwcv4_net.o $WDIR/zq_nchwcv4_rz.o '
                              '$WDIR/zq_nchwcv4_rzn.o'),
              'zq_nchwc_v8': (' $WDIR/zq_nchwcv8_t4d.o $WDIR/zq_nchwcv8_tn.o '
                              '$WDIR/zq_nchwcv8_net.o $WDIR/zq_nchwcv8_rz.o '
                              '$WDIR/zq_nchwcv8_rzn.o'),
              'zq_nchwc_roundtrip': (' $WDIR/zq_nchwcwz_t4d.o $WDIR/zq_nchwcwz_tn.o '
                                    '$WDIR/zq_nchwcwz_net.o $WDIR/zq_nchwcwz_rz.o '
                                    '$WDIR/zq_nchwcwz_rzn.o'),
              'zq_unusedlayers': ' $WDIR/zq_unusedlayers_t4d.o $WDIR/zq_unusedlayers_rz.o',
              'zq_tensorop': ' $WDIR/zq_tensorop_t4d.o $WDIR/zq_tensorop_rz.o',
              'zq_facegroup': '',
              'zq_lfw_eval': '',
              'zq_tile': ' $WDIR/zq_t4d.o $WDIR/zq_tile_rz.o',
              'zq_nchw_deconv': ' $WDIR/zq_dec.o',
              # -ldl 必须**放在源文件之后**：Ubuntu 20.04 默认 --as-needed，
              # 放在前面会被当成"当时没人需要 libdl"而丢掉（门禁里 dlsym(RTLD_NEXT) 用到）。
              'zq_nchw_conv_free': ' $WDIR/zq_cvf.o -ldl',
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
EXTRA_INC = {'zq_resize_align': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
             'zq_facedb': ' -I$R -I$R/ZQCNN -I$R/ZQCNN/3rdparty/include/ZQlib',
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
             'zq_nchwc_batch': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
             'zq_nchwc_poolpad': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
             'zq_nchwc_padreject': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
             'zq_padtype': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
             'zq_nchw_resize': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
             'zq_nchw_sqrtnrm': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
             'zq_nchw_reduction': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
             'zq_nchw_lstm': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
             'zq_lstm': ' -I$R/ZQCNN -I$R/ZQCNN/layers_c -I$R/ZQ_GEMM',
             'zq_roi': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
             'zq_reshape': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
             'zq_convert': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
             'zq_nchwc_tensor': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
             'zq_layerwire': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
             'zq_concat_alias': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
             'zq_convparam': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
             'zq_nchwc_net': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
             'zq_tensorop': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include -I$R/tools',
             'zq_unusedlayers': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include -I$R/tools',
             'zq_model_params': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include -I$R/tools',
             'zq_weight_tail': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include -I$R/tools',
             'zq_loadbuffer': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include -I$R/tools',
             'zq_weight_roundtrip': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include -I$R/tools',
             'zq_nchwc_v1': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include -I$R/tools',
             'zq_nchwc_v4': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include -I$R/tools',
             'zq_nchwc_v8': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include -I$R/tools',
             'zq_nchwc_roundtrip': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include -I$R/tools',
             'zq_facegroup': ' -I$R -I$R/ZQCNN -I$R/ZQlibFaceID -I$R/ZQ_GEMM -I$R/3rdparty/include',
             # **opencv_stub 放在最后**：它要抢在系统 OpenCV 前面被 #include <opencv2/opencv.hpp> 命中。
             # 本机 WSL 没装 OpenCV，所以顺序无所谓，但写成"永远优先桩"是刻意的 ——
             # 万一哪天装了 OpenCV，这道门禁必须在**同一套语义**下永远可复现。
             'zq_lfw_eval': (' -I$R -I$R/ZQCNN -I$R/ZQlibFaceID -I$R/ZQ_GEMM -I$R/3rdparty/include '
                             '-I$R/tools/opencv_stub'),
             'zq_tile': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
             'zq_nchw_deconv': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
             'zq_nchw_conv_free': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
             'zq_nchw_scalop': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
             'zq_nchw_depthwise': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
             'zq_nchw_act': ' -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include',
             'zq_lrn': ' -I$R/ZQCNN -I$R/ZQ_GEMM',
             'zq_pool': ' -I$R/ZQCNN -I$R/ZQ_GEMM',
             'zq_bns': ' -I$R/ZQCNN -I$R/ZQ_GEMM',
             'zq_deconv': ' -I$R/ZQCNN -I$R/ZQCNN/layers_c -I$R/ZQ_GEMM',
             'zq_eltwise': ' -I$R/ZQCNN -I$R/ZQ_GEMM'}
# 测内核的测试自己也 include 了那个 .c，所以**主 TU 也要带 -mavx2 -mfma**，
# 否则 _mm256_set1_ps 这些 always_inline 内建会报
# "target specific option mismatch"（2026-10-02 实测）。
EXTRA_CXXFLAGS = {'zq_resize_align': ' -mavx2 -mfma -fopenmp',
             'zq_scale': ' -mavx2 -mfma',
                  # zq_deconv：头里那几个 always_inline 内建，缺了会 target mismatch
                  'zq_deconv': ' -mavx2 -mfma',
                  'zq_facedb': ' -mavx2 -mfma -fopenmp',
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
                  'zq_nchwc_batch': ' -mavx2 -mfma -fopenmp',
                  'zq_nchwc_poolpad': ' -mavx2 -mfma -fopenmp',
                  'zq_nchwc_padreject': ' -mavx2 -mfma -fopenmp',
                  'zq_padtype': ' -mavx2 -mfma -fopenmp',
                  'zq_nchw_resize': ' -mavx2 -mfma -fopenmp',
                  'zq_nchw_sqrtnrm': ' -mavx2 -mfma -fopenmp',
                  'zq_nchw_reduction': ' -mavx2 -mfma -fopenmp',
                  'zq_nchw_lstm': ' -mavx2 -mfma -fopenmp',
                  'zq_lstm': ' -mavx2 -mfma',
                  'zq_roi': ' -mavx2 -mfma -fopenmp',
                  'zq_reshape': ' -mavx2 -mfma -fopenmp',
                  'zq_convert': ' -mavx2 -mfma -fopenmp',
                  'zq_nchwc_tensor': ' -mavx2 -mfma -fopenmp',
                  'zq_layerwire': ' -mavx2 -mfma -fopenmp',
                  'zq_concat_alias': ' -mavx2 -mfma -fopenmp',
                  'zq_convparam': ' -mavx2 -mfma -fopenmp',
                  'zq_nchwc_net': ' -mavx2 -mfma -fopenmp',
                  'zq_tensorop': ' -mavx2 -mfma -fopenmp',
                  'zq_unusedlayers': ' -mavx2 -mfma -fopenmp',
                  'zq_model_params': ' -mavx2 -mfma -fopenmp',
                  'zq_weight_tail': ' -mavx2 -mfma -fopenmp',
                  'zq_loadbuffer': ' -mavx2 -mfma -fopenmp',
                  'zq_weight_roundtrip': ' -mavx2 -mfma -fopenmp',
                  'zq_nchwc_v1': ' -mavx2 -mfma -fopenmp',
                  'zq_nchwc_v4': ' -mavx2 -mfma -fopenmp',
                  'zq_nchwc_v8': ' -mavx2 -mfma -fopenmp',
                  'zq_nchwc_roundtrip': ' -mavx2 -mfma -fopenmp',
                  'zq_facegroup': ' -mavx2 -mfma -fopenmp',
                  'zq_lfw_eval': ' -mavx2 -mfma -fopenmp',
                  'zq_tile': ' -mavx2 -mfma -fopenmp',
                  'zq_nchw_deconv': ' -mavx2 -mfma -fopenmp',
                  # **-fno-sanitize=address 必须排在 harness 加的 -fsanitize=address 之后**
                  # （EXTRA_CXXFLAGS 正是接在 san 后面拼的）。这道门禁要用 free 拦截器
                  # 记录"谁被释放了"，而 ASan 运行时自己也要调 free —— 在它初始化完成前
                  # 把 free 抢过来，一调用就段错误（附录 CU.9）。
                  'zq_nchw_conv_free': ' -mavx2 -mfma -fopenmp -fno-sanitize=address',
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
    # UBSan 默认是**可恢复**的：打一行 "runtime error:" 之后继续跑，进程照样 rc=0。
    # 那意味着一个门禁里出了 20 处 UB 也只会报 1 条，而且"跑完了"与"干净"长得一样。
    # 加 -fno-sanitize-recover=all 让**第一处 UB 直接终止进程**，
    # 于是"结果文件读不出来 = 失败"那条守卫（附录 CJ.4）就能真正生效，
    # 定位也只需看第一条 stderr。2026-10-02 之前这条轴没加，等于没查。
    #
    # alignment 检查**保持开启**：这一族内核全是手写 SIMD（_mm_load_ps 等），
    # 未对齐访问是 ASan 看不见、而 UBSan 看得见的一类 —— 值可能算对，但那是
    # "在 x86 上碰巧能跑"，换个平台或开 -march 更高的目标就可能变慢或崩。
    san = '' if args.no_asan else (
        '-fsanitize=undefined -fno-sanitize-recover=all'
        if args.ubsan else '-fsanitize=address')

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

    # **工作目录必须每次运行唯一**（附录 EB.1）。原来是固定的 `/tmp/zqchecks`
    # 且开头 `rm -rf *` —— 任何两次并发运行都会互相摧毁：
    # 一次正在链接，另一次把 `.o` 全删了，于是报出一个
    # `BUILD FAIL: g++: error: .../zq_dwnchw.o: No such file`，
    # **看起来像被测代码坏了，其实是两个进程在抢同一个目录**。
    # 2026-10-03 亲历：审计 harness 的 B 阶段把本脚本作为子进程调起，
    # 我同时手工跑了几次单门禁 —— 审计那边 `zq_bns` 就这样红了，
    # 而它单跑 PASS。
    # 用 pid + 时间戳做唯一名，并保证**本轮结束时不删别人的目录**。
    run_id = '%d_%d' % (os.getpid(), int(time.time()))
    wdir = '/tmp/zqchecks_%s' % run_id
    lines = ['set +e',
             'R=/mnt/d/ZQCNN',
             'WDIR=%s' % wdir,
             # **必须先 mkdir 再 cd**：唯一目录名每轮都是新的、初始并不存在。
             # 原来写的是 `cd $WDIR && rm -rf * && mkdir -p $WDIR`，
             # 靠的是 `/tmp/zqchecks` 早已存在才没出事 ——
             # 换成唯一名之后 `cd` 直接失败、`&&` 把 mkdir 短路掉，
             # 于是所有编译都写到不存在的路径上，**每一道门禁都 BUILD FAIL 且消息为空**。
             'mkdir -p $WDIR && cd $WDIR && rm -rf ./*']
    for s in srcs:
        fname = os.path.basename(s)              # zq_xxx_check.cpp
        stem = fname[:-4]                        # zq_xxx_check
        tag = stem[:-6] if stem.endswith('_check') else stem
        extras = EXTRA_SOURCES.get(tag, [])
        extra_ok = 'EXTRA_OK=1;'
        first_log = []
        for ei, extra in enumerate(extras):
            # $SAN = 本轮实际用的 sanitizer 旗标（ASan 时是 -fsanitize=address，
            # --ubsan 时是 -fsanitize=undefined，--no-asan 时是空串）。
            # **实现 TU 必须带 sanitizer 编译**，否则里面的普通 load 不插桩，
            # 越界读一个元素也不会有人报 —— 2026-10-02 在 zq_nchw_deconv 上栽过：
            # EXTRA_SOURCES 不带 sanitizer 时，那道"专治越界读"的门禁对着
            # 一份回退了修复的库**依然全绿**。凡是判据依赖 sanitizer 的 tag 都写 $SAN。
            #
            # **必须把这一步的失败并进同一条判定**（附录 DY.8）。原来这里是一条裸命令：
            # gcc 失败时 stderr 进总输出、`.o` 不生成，**后面链接时才报**
            #     g++: error: /tmp/zqchecks/zq_dwnchw.o: No such file or directory
            # 而"这个 .o 是谁编的、为什么没编出来"被埋在几百行滚动输出里 ——
            # 2026-10-03 全量跑就出现过一次这样的 BUILD FAIL（`zq_nchw_depthwise`），
            # 报错指向**链接器**，完全看不出是**被测库的编译**挂了。
            # 与 DY.5 同一个毛病：**错误出现在错误的地方**。
            elog = '$WDIR/%s_extra%d.log' % (tag, ei)
            first_log.append(elog)
            extra_ok += ' %s > %s 2>&1 || EXTRA_OK=0;' % (extra.replace('$SAN', san), elog)
        # 失败时把**所有**相关日志里的第一条 `error`/`fatal` 拼进消息 ——
        # 只报链接错误等于没报，只报 extra 错误又会漏掉门禁自身的编译错误。
        # **grep 一条都没命中时退回日志第一行**：gcc/g++ 报的未必含 error/fatal 两个词
        # （例：'want' was not declared in this scope 是 error，但
        #   "no matching function for call to ..." 这类未必），
        # 消息为空的话操作员只知道"BUILD FAIL"，等于没报（DY.8 的同一个毛病）。
        # 日志名**只给 tag**，不要再带 `.build.log` 后缀 —— 下面格式串里已经写了
        # `2> %s.build.log`。2026-10-03 我给这里传了一个已带后缀的名字，
        # 实际写出来的是 `zq_convert.build.log.build.log`，
        # 于是"取日志内容"那一步读到的是一个**空文件** ——
        # 改动本身带了 bug，症状是 BUILD FAIL 消息**永远为空**。
        # 与 DY.5 / DY.8 同源：**修复本身要单独验一次**。
        msg_logs = ' '.join(first_log + ['$WDIR/%s.build.log' % tag])
        firstline = ' '.join('head -1 %s' % x for x in (first_log + ['$WDIR/%s.build.log' % tag]))
        # `grep` 未命中时**不能靠 `||` 兜底**：命令替换里 `a | grep | tr || head -1`
        # 的 `||` 绑在整条管道的**最后一条命令**上，管道退出码取自 `tr`（恒 0），
        # `head -1` 永远不执行 —— 这正是消息为空的**第二个**原因。
        # 改成显式赋值：`M=<grep 结果>; [ -n "$M" ] || M=<日志第一行>`。
        lines.append(
            "if %s g++ -O1 -g %s%s -I%s%s /mnt/d/ZQCNN/tools/%s%s -o %s "
            "2> %s.build.log; then echo 'B|%s|OK|'; else "
            "M=$(cat %s 2>/dev/null | grep -m1 -iE 'error|fatal' | tr -d '\\r'); "
            "[ -n \"$M\" ] || M=$(%s 2>/dev/null | tr -d '\\r'); "
            "echo \"B|%s|BUILD_FAIL|$M\"; fi"
            % (extra_ok,
               '' if args.no_asan else san,
               EXTRA_CXXFLAGS.get(tag, ''),
               INC, EXTRA_INC.get(tag, ''), fname, EXTRA_LINK.get(tag, ''), tag,
               tag,
               tag, msg_logs, firstline, tag))
        # 两套 sanitizer 的失败口径不同，分开写：
        #   ASan  -> 断言自己打的 "FAIL" 行数 + 进程非 0（越界/释放后使用会直接 abort）
        #   UBSan -> "runtime error:" 行数。**不要指望 rc**：不加
        #            -fno-sanitize-recover=all 的话 UBSan 只打一行就继续跑，rc 恒为 0，
        #            那一栏永远是 0 等于没查（2026-10-02 实测）。
        # **ZQ_CHILD_ERR**（附录 CZ）：fork 型门禁的子进程把 stderr 重定向到
        # 这个文件，而不是 /dev/null —— 否则 sanitizer 的报告会被**一起吞掉**，
        # 现象是"知道门禁失败了、不知道它为什么失败"（附录 CY.4）。
        # 判定失败时由下面的代码把这个文件的前若干行打出来。
        # **必须用本轮的 $WDIR**（附录 EB.2）。附录 EB 把 WDIR 改成每轮唯一之后，
        # 这一行曾**漏改**，仍写死 '/tmp/zqchecks/' ——
        # 于是 (a) 子进程 stderr 落到一个没人读的目录，
        #     (b) 失败明细 `cat /tmp/zqchecks/<tag>.out` 永远读不到东西，
        # 输出只剩一行 "===== zq_facegroup =====" 后面空白。
        # **我当时只验了 BUILD FAIL 那条路径（做过变异测试），没验明细这条** ——
        # 正是本会话自己写进 AGENTS.md 的「修复本身要单独验一次」。
        ce = wdir + '/' + tag + '.child.err'
        if args.ubsan:
            lines.append(
                "if [ -x ./%s ]; then ZQ_CHILD_ERR=%s "
                "UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 ./%s > %s.out 2>&1; "
                "echo \"R|%s|$?|$(grep -c 'runtime error:' %s.out)|"
                "$(grep -cE 'FAIL' %s.out)\"; fi"
                % (tag, ce, tag, tag, tag, tag, tag))
        else:
            lines.append(
                "if [ -x ./%s ]; then ZQ_CHILD_ERR=%s ASAN_OPTIONS=detect_leaks=1 "
                "ZQ_MODEL_FULL_LOAD=1 ./%s > %s.out 2>&1; "
                "echo \"R|%s|$?|$(grep -cE 'FAIL' %s.out)|0\"; fi"
                % (tag, ce, tag, tag, tag, tag))
    lines.append('echo R|__END__|0|0')
    # 注意：**不要在这里 `rm -rf $WDIR`**（附录 EC.1）。
    # 清理放进 Python 侧、且**只在没有构建失败时**做 ——
    # 2026-10-03 我把清理写成脚本最后一行，于是"有门禁编译失败"时
    # 连 `*.build.log` 一起删了，**诊断证据当场消失**，
    # 查"为什么链接不过"只能从头再跑一遍。
    # 清理本身会毁掉证据 —— 与"报告被后一个用例擦掉"（DY.5）同一类。
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

    # 把各门禁的子进程 stderr 拉回本地（附录 CZ）。
    # WSL 里的 /tmp 与本地 TEMP 是两个文件系统，不拿回来就打不开。
    # **必须放在下面"打印失败详情"的循环之前** —— 第一版把它放在循环后面，
    # 于是第一轮跑完时文件还没到本地、什么都不打，第二轮才看得到；
    # 而回归通常是"跑一次就去看"，看到的正好是空的那一轮。
    #
    # **必须走 run_wsl（脚本经 stdin 送进 bash）**，不能用
    # subprocess.run('wsl ... bash -lc "..."', shell=True)：后者要过 cmd.exe，
    # 里面的 `;` `*` `2>/dev/null` 全会被 cmd 先解释一遍，第一版就是这么写的，
    # 结果本地目录空着、报告一个也没拉回来（而 WSL 侧其实已经写好了）。
    # 本地路径要先转成 /mnt/<盘符>/... 的形式 WSL 才认得。
    tmp = os.environ.get('TEMP', '.')
    cdir = os.path.join(tmp, 'zqchild_%s' % run_id)   # 本轮专属镜像目录
    dst_prefix = os.path.join(tmp, 'zqchild_')        # 报告的稳定落点
    try:
        m2 = re.match(r'([A-Za-z]):[\\/]+(.*)', tmp)
        if m2:
            wsl_tmp = '/mnt/%s/%s' % (m2.group(1).lower(), m2.group(2).replace('\\', '/'))
        else:
            wsl_tmp = tmp.replace('\\', '/')
        cdir_wsl = wsl_tmp.rstrip('/') + '/zqchild_%s' % run_id
        if not os.path.isdir(cdir):
            os.makedirs(cdir)
        # **源目录用本轮的 $WDIR**，不能再写死 /tmp/zqchecks ——
        # 那样会把**别的运行**的报告也一起拷过来。
        run_wsl('rm -f %s/*.child.err 2>/dev/null; '
                'cp %s/*.child.err %s/ 2>/dev/null; true'
                % (cdir_wsl, wdir, cdir_wsl))
        for fn in os.listdir(cdir):
            if fn.endswith('.child.err'):
                dst = dst_prefix + fn[:-len('.child.err')]
                with open(os.path.join(cdir, fn), 'rb') as a, open(dst, 'wb') as b:
                    b.write(a.read())
        # 报告已落到稳定路径，本轮镜像目录可以删了
        try:
            shutil.rmtree(cdir, ignore_errors=True)
        except (OSError, IOError):
            pass
    except (OSError, IOError):
        pass

    print()
    nfail = 0
    # 2026-10-04：两套 sanitizer 模式下这两个列的**口径是互换的**（见上面生成
    # `R|tag|rc|...|...` 那两行），而标签一直照着 UBSan 那套写。于是 ASan 轮里
    # 一个**普通的断言失败**会被报成"2 条 sanitizer 报错" —— 实测（附录 GZ.3）
    # 就是这么被当成"门禁自己崩了"查了半天的：它只是 `zq_model_params` 报了两行
    # `**FAIL**`，一条 sanitizer 报告都没有。标签要说它真正数的是什么。
    lbl_san = '条 sanitizer 报错' if args.ubsan else '条断言失败'
    lbl_assert = '条断言失败' if args.ubsan else '条 sanitizer 报错'
    for name, rc, nsan, nassert in results:
        ok = (rc == '0' and nsan == '0' and nassert == '0')
        if not ok:
            nfail += 1
        why = []
        if rc != '0':
            why.append('rc=%s' % rc)
        if nsan != '0':
            why.append('%s %s' % (nsan, lbl_san))
        if nassert != '0':
            why.append('%s %s' % (nassert, lbl_assert))
        print('%-34s %s' % (name, 'PASS' if ok else 'FAIL (%s)' % ', '.join(why)))
        if not ok:
            # 把子进程的 sanitizer 报告打出来（附录 CZ）。
            # 不打的话，一个 fork 型门禁因为 ASan/UBSan 报错而红时，
            # 操作员只看到"没跑完"，还得单独把二进制手工跑一遍才看得到原因。
            cpath = os.path.join(tmp, 'zqchild_' + name)
            try:
                if os.path.isfile(cpath) and os.path.getsize(cpath):
                    print('---- %s 子进程 sanitizer 报告（前 24 行）----' % name)
                    with open(cpath, encoding='utf-8', errors='replace') as f:
                        for i, line in enumerate(f):
                            if i >= 24:
                                print('   ...')
                                break
                            print('   ' + line.rstrip())
                    print('---- 报告结束 ----')
            except IOError:
                pass
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
                detail = run_wsl("cat %s/%s.out 2>/dev/null | tail -80" % (wdir, name))
                print('\n===== %s =====\n%s' % (name, detail))

    # 清理本轮的 WDIR —— 放在**最后**，且**只在全部通过时**做（附录 EC.1 / EB.3）。
    # 两次踩坑：
    #   ① 只判 `not build_fail` 不够 —— 门禁**运行**失败（不是构建失败）时
    #      `build_fail` 仍为空，于是目录被删掉，而上面那段失败明细
    #      `cat $WDIR/<tag>.out` 紧接着就读，读到的是空 ——
    #      输出只剩一行 `===== zq_facegroup =====` 后面什么都没有。
    #   ② 清理绝不能写成 `rm -rf /tmp/zqchecks*`：会删掉正在跑的别的运行。
    if not nfail and not build_fail:
        run_wsl('rm -rf %s 2>/dev/null; true' % wdir)
    return 1 if nfail else 0


if __name__ == '__main__':
    sys.exit(main())
