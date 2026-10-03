#!/bin/bash
# 跑一遍关键 sample，报告**状态**、退出码与耗时。用于每次改动后的双平台回归。
#
#   tools/run_sample_regression.sh            # 在 WSL 里跑（脚本自身就是给 WSL 用的）
#   wsl -d Ubuntu-20.04 -- bash -c "bash /mnt/d/ZQCNN/tools/run_sample_regression.sh"
#
# Windows 侧直接进 cmake-out-win32-x64/release/Release 跑同名 exe。
# 所有 sample 的 namedWindow/imshow/waitKey 都已注释掉，不会阻塞。
# 跑不动的 sample 只会是 Model Zoo 权重/图片不在仓库，不是代码问题。
#
# ---------------------------------------------------------------------------
# 审计修复 2026-10-02（附录 BR.1）：原来这里把输出丢进 /dev/null、**只看退出码**。
# 问题是本仓库有一批 sample 是"平台桩"：它们打印一句
#     SampleFaceDetectorMTCNN only support windows
#     SampleCascadeOnet_Interface not support in linux
# 然后 return 0。于是 Windows 专用的 sample 在 Linux 上被记成 `rc=0`，
# 看起来"8 个 sample 全绿"，其实其中两个在 Linux 上什么都没做。
#
# 现在：输出留下来，分三种状态
#     OK    跑完了、有输出、rc=0
#     STUB  自己说了"这个平台不支持"（只报告，**不**判失败）
#     NOOUT 没有任何输出（**判失败** —— 什么都没打就退 0，多半是路径/权重没就位）
# 退出码：有 OK 之外的非 STUB 项（rc!=0 或 NOOUT）就非 0。
# ---------------------------------------------------------------------------
set -u
OUT_DIR=${1:-/mnt/d/ZQCNN/cmake-out-unix-x64/Release}
cd "$OUT_DIR" || exit 1

# 自己声明"本平台不支持"的那些措辞
STUB_RE='only support|not support|not supported|only supports'

# 检出数下界（2026-10-04 加，附录 GS）。
#
# 为什么需要：原来只看 rc==0 且输出非空。而"模型被悄悄弄坏"的典型症状
# **不是崩溃，是检出 0 张脸** —— 层接线错、权重数组顺序错、
# 某个 num_output 读偏，都会安静地给出 0。那样回归照样绿。
#
# 下界取"实测值的一半（向下取整，至少 1）"而不是精确值：
#   * 精确值太脆 —— 修好一个真实的漏检 bug，检出数会**上升**，精确匹配会误报；
#   * ">= 1" 太弱 —— 只检出 1 张脸（实测是 10 张）也能过。
# 下界能抓住"丢了一半以上"这一档，而这正是静默损坏的形态。
#
# 实测（2026-10-04，WSL gcc 9.4，仓库自带权重）：
#   SampleMTCNN=10   SampleMTCNN_NCHWC4=4   SampleMTCNNLoadFromCode=84
# 换模型或换测试图之后要**重新量一遍**这三个数并更新本表 ——
# 忘了更新的症状是"门禁突然红了"，而不是"门禁一直没在管这件事"。
#
# 注意 SampleMTCNN 与 SampleMTCNN_NCHWC4 **用的不是同一张图**
# （data/11.jpg vs data/4_320x240.jpg），所以 10 与 4 **不可直接比较**。
# 2026-10-04 我一度把这两个数当成"NCHWC4 少检出 60%"的缺陷，
# 核了输入路径才发现是两张图 —— 记在这里以免下次再犯。
detect_floor_case() {   # $1=sample 名 -> 期望下界；空 = 该 sample 不适用
  case "$1" in
    SampleMTCNN)             echo 5 ;;
    SampleMTCNN_NCHWC4)       echo 2 ;;
    SampleMTCNNLoadFromCode)  echo 42 ;;
    *)                        echo "" ;;
  esac
}

n_ok=0; n_stub=0; n_bad=0
for e in SampleMTCNN SampleMTCNN_NCHWC4 SampleSSD SampleFaceDetectorMTCNN \
         SampleCascadeOnet SampleCascadeOnet_Interface SampleMTCNNLoadFromCode \
         SampleGEMMAsmCompare; do
  if [ -x "./$e" ]; then
    s=$(date +%s%N)
    out=$("./$e" 2>&1); rc=$?
    t=$(( ($(date +%s%N)-s)/1000000 ))
    if printf '%s' "$out" | grep -qiE "$STUB_RE"; then
      st=STUB; n_stub=$((n_stub+1))
      # 桩也要把它自己说的那半句打出来，否则"STUB"仍然看不出是哪个平台不支持
      echo "$e $st rc=$rc ${t}ms  [$(printf '%s' "$out" | grep -iE "$STUB_RE" | head -1)]"
    elif [ -z "$out" ]; then
      st=NOOUT; n_bad=$((n_bad+1))
      echo "$e $st rc=$rc ${t}ms  <-- 一行输出都没有，多半是权重/路径没就位"
    elif [ "$rc" -ne 0 ]; then
      st=FAIL; n_bad=$((n_bad+1))
      echo "$e $st rc=$rc ${t}ms"
      printf '%s\n' "$out" | tail -5 | sed 's/^/      /'
    else
      # 检出数下界（见上面 detect_floor_case 的注释）
      floor=$(detect_floor_case "$e")
      found=""
      if [ -n "$floor" ]; then
        found=$(printf '%s' "$out" | sed -n 's/.*final found num: *\([0-9][0-9]*\).*/\1/p' | tail -1)
        if [ -z "$found" ]; then
          st=NOCOUNT; n_bad=$((n_bad+1))
          echo "$e $st rc=$rc ${t}ms  <-- 这个 sample 应当打印 'final found num:'，实际没有"
        elif [ "$found" -lt "$floor" ]; then
          st=FEWFACES; n_bad=$((n_bad+1))
          echo "$e $st rc=$rc ${t}ms  检出 $found 张 < 下界 $floor  <== 模型多半是静默坏了"
        else
          st=OK; n_ok=$((n_ok+1))
          echo "$e $st rc=$rc ${t}ms  检出 $found 张（下界 $floor）"
        fi
      else
        st=OK; n_ok=$((n_ok+1))
        echo "$e $st rc=$rc ${t}ms  $(printf '%s' "$out" | wc -l) 行输出"
      fi
    fi
  else
    st=MISSING; n_bad=$((n_bad+1))
    echo "$e $st"
  fi
done

echo "---- 真跑了的 $n_ok 个；本平台不支持的桩 $n_stub 个；问题 $n_bad 个 ----"
echo "（桩不算失败，但**不要**把它们算进「sample 全绿」里 —— 附录 BR.1 的原话："
echo "  修之前 8 个 rc 全 0，其中 2 个在 Linux 上只是打印了一句「不支持」。）"
[ "$n_bad" -eq 0 ] || exit 1
exit 0
