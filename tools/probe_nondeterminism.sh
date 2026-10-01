#!/bin/bash
# 附录 AW.6：同一二进制、同一输入，MTCNN 的 pre-NMS 候选框数每次不同。
# 这个脚本把**所有** nms 计数抓出来，跨多次运行比对，定位到底是哪一行在变。
#
#   bash tools/probe_nondeterminism.sh [轮数]
#
# 输出：每个 nms 序号 -> 出现过的不同取值个数 + 取值列表。
# 全部是 1 就说明这一轮没复现（可以加大轮数再看）。
cd /mnt/d/ZQCNN/cmake-out-unix-x64/Release || exit 1
N=${1:-8}
D=/tmp/nd_runs
rm -rf $D && mkdir -p $D
for i in $(seq 1 $N); do
  ./SampleMTCNNLoadFromCode 2>&1 | grep -oE '\([0-9]+-->[0-9]+\)' > "$D/run$i.txt"
done
python3 - <<'PY'
import glob, os, collections
D = '/tmp/nd_runs'
runs = []
for f in sorted(glob.glob(os.path.join(D, 'run*.txt'))):
    runs.append([l.strip() for l in open(f) if l.strip()])
if not runs:
    print('没抓到任何 nms 行'); raise SystemExit(1)
n = min(len(r) for r in runs)
print('共 %d 次运行，每个 %d 个 nms 行' % (len(runs), len(runs[0])))
for i in range(n):
    vals = collections.Counter(r[i] for r in runs)
    if len(vals) == 1:
        print('  nms[%2d]  恒定  %s' % (i, list(vals)[0]))
    else:
        print('  nms[%2d]  **变化** %s' % (i, dict(vals)))
PY
