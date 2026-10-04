// `slice_model_weights.py` 用的最小探针：只做一次 LoadFrom，成 0 败非 0。
//
// 为什么需要它：定位某一层在权重文件里的字节区间，第一版想按层类型推 float 数，
// 差了 37 万字节 —— `.zqparams` 里**没有** `in_channels`（它是从 bottom blob 的
// 形状推出来的），静态形状推断等于把 `SetBottomDim` 重写一遍。
// 改成"截断二分支"之后，**完全不需要形状推断**（附录 HU.1）。
//
// 只用公开接口：`ZQ_CNN_Net::LoadFrom(param, weights)`。
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cstdio>
#include "ZQ_CNN_Tensor4D.h"
#include "ZQ_CNN_Net.h"
#include "zq_net_fwd_tripwires.h"
#include "zq_concat_getsize_real.h"

int main(int argc, char** argv)
{
    if (argc < 3) { printf("用法: %s <zqparams> <nchwbin>\n", argv[0]); return 2; }
    // 探针只需要"能不能加载"，不需要任何输出
    if (!freopen("/dev/null", "w", stdout)) return 2;
    if (!freopen("/dev/null", "w", stderr)) return 2;
    ZQ::ZQ_CNN_Net net;
    bool ok = net.LoadFrom(argv[1], argv[2]);
    return ok ? 0 : 1;
}
