#!/bin/bash
# 两个二进制**交替**跑 N 轮，比 total 耗时。
#
# 为什么要交替（AGENTS.md「构建规则」第 10 条）
# --------------------------------------------
# 「先把 A 跑完再跑 B」会让中间的睿频/温度漂移**系统性偏袒后跑的那个**。
# 本机实测：空对照（同一个文件跟它自己比）64 个形状有 20 个出现 >3% 的
# 假差异、最大的 10%。所以必须 ABABAB 交替，最后比每组的中位数。
set -u
cd /mnt/d/ZQCNN/cmake-out-unix-x64/Release || exit 1

A=${1:-/mnt/c/Users/Administrator/AppData/Local/Temp/binA_SampleMTCNN}
B=${2:-/mnt/c/Users/Administrator/AppData/Local/Temp/binB_SampleMTCNN}
N=${3:-7}
KEY=${4:-total}

rm -f /tmp/abtimes_A.txt /tmp/abtimes_B.txt
for i in $(seq 1 $N); do
  "$A" 2>&1 | grep -E "$KEY" | sed -E 's/.*[= ]([0-9]+\.[0-9]+) ?ms.*/\1/' | tail -1 >> /tmp/abtimes_A.txt
  "$B" 2>&1 | grep -E "$KEY" | sed -E 's/.*[= ]([0-9]+\.[0-9]+) ?ms.*/\1/' | tail -1 >> /tmp/abtimes_B.txt
done
python3 - <<'PY'
import statistics
a = [float(x) for x in open('/tmp/abtimes_A.txt') if x.strip()]
b = [float(x) for x in open('/tmp/abtimes_B.txt') if x.strip()]
if not a or not b:
    print('没取到样本: A=%d B=%d' % (len(a), len(b))); raise SystemExit(1)
ma, mb = statistics.median(a), statistics.median(b)
print('A (1 || 全走 GEMM) : n=%d  median=%.3f ms  min=%.3f  all=%s'
      % (len(a), ma, min(a), ' '.join('%.1f' % x for x in a)))
print('B (守卫恢复)       : n=%d  median=%.3f ms  min=%.3f  all=%s'
      % (len(b), mb, min(b), ' '.join('%.1f' % x for x in b)))
print('B/A 中位 = %.3f  (>1 表示 B 更慢；噪声下限约 7%%)'
      % (mb / ma if ma else float('nan')))
PY
