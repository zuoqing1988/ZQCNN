# Linux 侧全量 sample 扫描（Windows 版见 tools/sweep_samples_win.py）。
# 逐个运行，记录真实退出码；只报"既不是 0 也不是 1"的。
# 2026-10-21 首次全量扫描：无崩溃（4 个 rc=2 是输出 jpg 被 exec 过滤器误收，已修）。
cd /mnt/d/ZQCNN/cmake-out-unix-x64/Release || exit 1
ok=0; fail=0; to=0
for e in *; do
  # /mnt/d 是 Windows 挂载, 连输出图片都带 +x 位, 所以必须按后缀过滤
  [ -f "$e" ] || continue
  [ -x "$e" ] || continue
  case "$e" in *.*) continue ;; esac
  timeout 20 "./$e" >/dev/null 2>&1
  rc=$?
  case $rc in
    0)   ok=$((ok+1)) ;;
    1)   fail=$((fail+1)) ;;
    124) to=$((to+1));  echo "TIMEOUT        $e" ;;
    *)   echo "rc=$rc  $e" ;;
  esac
done
echo "---"
echo "rc=0: $ok   rc=1: $fail   TIMEOUT: $to"
