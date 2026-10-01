#!/bin/bash
# 跑一遍关键 sample 并把**去掉计时噪声**的输出存下来（A/B 用）。
#
# 为什么不能直接 diff tools/capture_sample_outputs.sh 的输出
# ------------------------------------------------------
# 每个 sample 都会打 `convert cost: 0.529 ms` / `stage 1: cost 1.481 ms` /
# `intr 24.01 GF/s` 这类计时行。同一份二进制连跑两次都不同（实测 MTCNN 的
# stage 1 在 1.48~1.51 ms 之间跳，GF/s 更是差 20% 以上），
# 直接 diff 只会得到几百行噪声，真正的数值差异被埋在里面。
#
# 所以这里把「所有时间/吞吐数字」替换成占位符再存。留下的
# `nms cost: <T> ms, (159-->24)` 里的 `(159-->24)` —— 那是**检测框数量**，
# 必须原样保留，它才是我们真正要比的东西。
#
# 顺带：连跑两次的结果必须**完全一致**（归一化之后），否则说明这条流水线上
# 还有别的非确定性（多线程归约顺序之类），A/B 也就没有意义了。
#
#   bash tools/capture_sample_outputs.sh <产物目录> <输出目录>

OUT_DIR=${1:-/mnt/d/ZQCNN/cmake-out-unix-x64/Release}
DEST=${2:-/tmp/sample_out}
mkdir -p "$DEST"
cd "$OUT_DIR" || exit 1

NORM='s/[0-9]+\.[0-9]+ ?ms/<T>ms/g;
      s/[0-9]+(\.[0-9]+)? ?ms/<T>ms/g;
      s/[0-9]+\.[0-9]+ ?s\b/<T>s/g;
      s/[0-9]+\.[0-9]+ GF\/s/<GF>GF\/s/g;
      s/[0-9]+\.[0-9]+e\+[0-9]+/<E>/g'

for e in SampleMTCNN SampleMTCNN_NCHWC4 SampleSSD SampleFaceDetectorMTCNN \
         SampleCascadeOnet SampleCascadeOnet_Interface SampleMTCNNLoadFromCode; do
  if [ -x "./$e" ]; then
    "./$e" 2>&1 | sed -E "$NORM" > "$DEST/$e.txt"
    echo "$e rc=${PIPESTATUS[0]} -> $DEST/$e.txt"
  else
    echo "$e MISSING"
  fi
done
# GEMM 对拍单独存：它的误差列是确定性的，只有 GF/s 需要归一化
if [ -x "./SampleGEMMAsmCompare" ]; then
  "./SampleGEMMAsmCompare" 2>&1 \
    | sed -E 's/[0-9]+\.[0-9]+ GF\/s/<GF>GF\/s/g; s/ +$//' \
    > "$DEST/SampleGEMMAsmCompare.txt"
  echo "SampleGEMMAsmCompare rc=${PIPESTATUS[0]} -> $DEST/SampleGEMMAsmCompare.txt"
fi
