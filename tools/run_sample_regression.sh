#!/bin/bash
# 跑一遍关键 sample，报告退出码与耗时。用于每次改动后的双平台回归。
#
#   tools/run_sample_regression.sh            # 在 WSL 里跑（脚本自身就是给 WSL 用的）
#   wsl -d Ubuntu-20.04 -- bash -c "bash /mnt/d/ZQCNN/tools/run_sample_regression.sh"
#
# Windows 侧直接进 cmake-out-win32-x64/release/Release 跑同名 exe。
# 所有 sample 的 namedWindow/imshow/waitKey 都已注释掉，不会阻塞。
# 跑不动的 sample 只会是 Model Zoo 权重/图片不在仓库，不是代码问题。

OUT_DIR=${1:-/mnt/d/ZQCNN/cmake-out-unix-x64/Release}
cd "$OUT_DIR" || exit 1
for e in SampleMTCNN SampleMTCNN_NCHWC4 SampleSSD SampleFaceDetectorMTCNN \
         SampleCascadeOnet SampleCascadeOnet_Interface SampleMTCNNLoadFromCode \
         SampleGEMMAsmCompare; do
  if [ -x "./$e" ]; then
    s=$(date +%s%N)
    "./$e" >/dev/null 2>&1
    rc=$?
    echo "$e rc=$rc $(( ($(date +%s%N)-s)/1000000 ))ms"
  else
    echo "$e MISSING"
  fi
done
