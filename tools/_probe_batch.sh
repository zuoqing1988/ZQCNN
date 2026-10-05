#!/bin/bash
set -e
R=/mnt/d/ZQCNN
W=/tmp/iu_batch
rm -rf "$W"
mkdir -p "$W"
cd "$W"
# NCHWC 前向要链上全部 layers_nchwc 内核 + ZQ_GEMM 的两个 TU。
# zq_gemm_32f_align_c.c 单独编在 -O1 下要 5 分钟以上（AGENTS.md「推不动的时候…」），
# 所以这里用 -O0，只求跑通，不测性能。
for f in $R/ZQCNN/layers_nchwc/*.c; do
  gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include "$f" -o "$(basename $f .c).o"
done
gcc -O0 -g -mavx2 -mfma -fopenmp -c -I$R/ZQ_GEMM -I$R/ZQCNN -I$R/3rdparty/include \
    $R/ZQ_GEMM/math/zq_gemm_32f_align_c.c -o gemm_align.o
gcc -O0 -g -mavx2 -mfma -fopenmp -c -I$R/ZQ_GEMM -I$R/ZQCNN -I$R/3rdparty/include \
    $R/ZQ_GEMM/math/zq_gemm_32f_align_c_asm.c -o gemm_asm.o
gcc -O1 -g -mavx2 -mfma -fopenmp -c -I$R/ZQ_GEMM -I$R/ZQCNN -I$R/3rdparty/include \
    $R/ZQ_GEMM/math/zq_gemm_32f_auto.c -o gemm_auto.o
g++ -O1 -g -mavx2 -mfma -fopenmp -c -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include \
    $R/ZQCNN/ZQ_CNN_Tensor4D_NCHWC.cpp -o tensor.o
g++ -O1 -g -mavx2 -mfma -fopenmp -c -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include \
    $R/ZQCNN/ZQ_CNN_Forward_SSEUtils_NCHWC.cpp -o fwd.o
g++ -O1 -g -mavx2 -mfma -fopenmp -I$R -I$R/ZQCNN -I$R/ZQ_GEMM -I$R/3rdparty/include \
    $R/tools/_batch_probe.cpp *.o -o batchcheck -fopenmp
set +e
./batchcheck
echo "RC=$?"