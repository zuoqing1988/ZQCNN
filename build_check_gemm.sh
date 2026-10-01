#!/bin/bash
set -e
mkdir -p /tmp/zbchk
cd /tmp/zbchk
rm -f ./*.o zbench
for f in /mnt/d/ZQCNN/ZQ_GEMM/math/*.c; do
  b=$(basename "$f" .c)
  gcc -c -O3 -mavx2 -mfma -fopenmp -I/mnt/d/ZQCNN/ZQ_GEMM/math -I/mnt/d/ZQCNN/ZQCNN "$f" -o "$b.o" 2>&1 | head -3
done
ls *.o
g++ -O3 -mavx2 -mfma -fopenmp -I/mnt/d/ZQCNN/ZQ_GEMM/math -I/mnt/d/ZQCNN/ZQCNN -o zbench /mnt/d/ZQCNN/SamplesZQBLAS/SampleGEMMCompare.cpp ./*.o -ldl -lm 2>&1 | head -10
ls -la zbench | head -2
