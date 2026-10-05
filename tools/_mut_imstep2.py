"""对 check_imstep_guard.py 做变异测试（**不动工作区**）。

把真实内核头复制到临时目录当 ZQCNN/ 的替身，在**副本**上把 imStep 改回 sliceStep，
验证门禁变红且**点名到具体行**；再把副本恢复、验证门禁回绿。

为什么不直接改工作区：AGENTS.md 第 8 条 —— 回归跑着的时候不要改生产文件。
v68 当时正在读同一批 .h，直接改会让它看到半改的树
（那一轮已经因为 glob 陷阱差点栽过一次，AGENTS.md 第 31 条）。
"""
import io
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = r"D:\ZQCNN"
GATE = os.path.join(ROOT, "tools", "check_imstep_guard.py")
GOOD = "n++, in_im_ptr += in_imStep, out_im_ptr += out_imStep)"
BAD = "n++, in_im_ptr += in_sliceStep, out_im_ptr += out_sliceStep)"

SRC_DIRS = [
    r"ZQCNN\layers_nchwc",
    r"ZQCNN\layers_c",
    r"ZQCNN\math",
    r"ZQ_GEMM\math",
]


def run_gate(root=None):
    args = [sys.executable, GATE]
    if root:
        args += ["--root", root]
    r = subprocess.run(args, capture_output=True, encoding="utf-8", errors="replace")
    return r.returncode, (r.stdout or "")


def patch(path, frm, to):
    with io.open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    n = 0
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("n++,") and frm in ln:
            lines[i] = ln.replace(frm, to)
            n += 1
    with io.open(path, "w", encoding="utf-8", newline="") as f:
        f.writelines(lines)
    return n


def main():
    tmp = tempfile.mkdtemp(prefix="zq_imstep_mut_")
    try:
        fake_zqcnn = os.path.join(tmp, "ZQCNN")
        os.makedirs(fake_zqcnn)
        for d in SRC_DIRS:
            src = os.path.join(ROOT, d)
            dst = os.path.join(tmp, d)
            if os.path.isdir(src):
                shutil.copytree(src, dst)
        if not os.path.isdir(os.path.join(tmp, "ZQ_GEMM")):
            os.makedirs(os.path.join(tmp, "ZQ_GEMM"))

        rc, out = run_gate(tmp)
        print("=== 基线（副本，未变异） RC=%d ===" % rc)
        print(out.strip())

        targets = [
            os.path.join(fake_zqcnn, "layers_nchwc", "zq_cnn_pooling_nchwc_raw.h"),
            os.path.join(fake_zqcnn, "layers_nchwc", "zq_cnn_resize_nchwc_raw.h"),
        ]
        for t in targets:
            rel = os.path.relpath(t, tmp)
            try:
                n = patch(t, GOOD, BAD)
                rc2, out2 = run_gate(tmp)
                print("\n=== 变异 %s（%d 处）RC=%d ===" % (rel, n, rc2))
                print(out2.strip())
            finally:
                patch(t, BAD, GOOD)
            rc3, out3 = run_gate(tmp)
            print("--- 还原后 RC=%d ---" % rc3)
            assert rc3 == 0, "还原之后仍然红，说明还原写错了"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n全部断言通过（finally 已清理临时目录）")


if __name__ == "__main__":
    main()