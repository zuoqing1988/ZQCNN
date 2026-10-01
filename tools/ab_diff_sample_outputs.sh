#!/bin/bash
# A/B 对比两个 sample 输出目录。第一个参数是 A，第二个是 B。
set -u
A=${1:?A dir}
B=${2:?B dir}
for f in SampleMTCNN SampleMTCNN_NCHWC4 SampleSSD SampleFaceDetectorMTCNN \
         SampleCascadeOnet SampleCascadeOnet_Interface SampleMTCNNLoadFromCode; do
  if diff -q "$A/$f.txt" "$B/$f.txt" >/dev/null 2>&1; then
    echo "$f  IDENTICAL"
  else
    echo "$f  DIFF:"
    diff "$A/$f.txt" "$B/$f.txt" | head -14
  fi
done
