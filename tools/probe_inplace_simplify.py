#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""How do real .zqparams use the _is_inplace_safe layer types?

ZQ_CNN_Net::_simplify_inplace() only ever rewrites index 0:
    tops[i][0] = bottoms[i][0];
so for an in-place-safe layer that declares a top *different* from bottoms[0]
(or declares extra bottoms), the declared top is silently dropped and never
written -- downstream layers then read whatever was in that blob.

That is a malformed-model acceptance, not a memory-safety bug, so before
deciding whether to add a guard: how many real models do that?
Diagnostic only, writes nothing.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# must stay identical to ZQ_CNN_Net::_is_inplace_safe()
INPLACE_SAFE = {
    "relu", "relu6", "prelu",
    "batchnormscale", "batchnorm", "scale", "addbias",
}

TOP_RE = re.compile(r"\btop\s*=\s*(\S+)")
BOTTOM_RE = re.compile(r"\bbottom\s*=\s*(\S+)")
LAYER_RE = re.compile(r"top\s*=\s*(\S+)\s+bottoms?\s*=\s*(\S+)")
BARE_LAYER_RE = re.compile(r"^(Input|Convolution|DepthwiseConvolution|DeConvolution|"
                           r"BatchNormScale|BatchNorm|PReLU|ReLU|ReLU6|Scale|AddBias|"
                           r"Pooling|InnerProduct|Eltwise|Softmax|Concat|Flatten|Reshape|"
                           r"Permute|Normalize|Upsampling|PriorBox|PriorBoxText|"
                           r"DetectionOutput|DetectionOutput_MXNET|DetectionOuput|"
                           r"LSTM_TF|Reduction|UnaryOperation|Squeeze|Tile|Sqrt|ScalarOperation|"
                           r"Copy|MeanStdNorm|Dropout|Interp)\b")


def main():
    seen_dirs = set()
    files = []
    for base, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in (".git", "cmake-out-win32-x64",
                                                "cmake-out-unix-x64", "build_x64",
                                                "build", "__pycache__")]
        for n in sorted(names):
            if n.endswith(".zqparams"):
                # 同一个模型在 build 目录里有副本，按内容去重
                real = os.path.join("model", n)
                if os.path.exists(os.path.join(ROOT, real)):
                    files.append(os.path.join(ROOT, real))
                else:
                    files.append(os.path.join(base, n))
    files = sorted(set(files))

    multi_bottom = 0
    top_ne_bottom0 = 0
    safe_layers = 0
    hits = []
    for path in files:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for lineno, raw in enumerate(fh, 1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                m = BARE_LAYER_RE.match(line)
                if not m:
                    continue
                ltype = m.group(1)
                if ltype.lower() not in INPLACE_SAFE:
                    continue
                safe_layers += 1
                tops = TOP_RE.findall(line)
                bots = BOTTOM_RE.findall(line)
                if len(bots) > 1:
                    multi_bottom += 1
                    hits.append((os.path.relpath(path, ROOT), lineno, ltype,
                                 line[:90]))
                elif tops and bots and tops[0] != bots[0]:
                    top_ne_bottom0 += 1
                    hits.append((os.path.relpath(path, ROOT), lineno, ltype,
                                 line[:90]))

    print("scanned %d distinct .zqparams" % len(files))
    print("in-place-safe layers seen: %d" % safe_layers)
    print("  ... with MORE THAN ONE bottom : %d" % multi_bottom)
    print("  ... with top != bottoms[0]     : %d" % top_ne_bottom0)
    if hits:
        # 只打前 10 条：这门探针第一次跑时把 638 条全打了出来，82 KB，
        # 把 Bash 的输出上限顶爆、真正的结论被埋在中间看不见。
        print("\nhits (first 10 of %d):" % len(hits))
        for h in hits[:10]:
            print("   %s:%d  %s  %s" % h)
    return 0


if __name__ == "__main__":
    sys.exit(main())
