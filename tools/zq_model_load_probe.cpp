/* 一次性探针：把 model/ 下每个 .zqparams 都过一遍真的 ZQ_CNN_Net::LoadFrom。
 *
 * 为什么需要它
 * ------------
 * 附录 EN 把 `_check_connect` 的就地守卫从"同一下标"改成"比全部组合"，
 * 又在 `_concat_NCHW` 里加了 output/input 别名检查。这两处都在**模型加载路径**上，
 * 所以"改动有没有误杀真实模型"必须实测，不能只靠静态统计
 * （附录 ED.2：`tools/probe_inplace_topbottom.py` 报的是 0 命中）。
 *
 * 这是**探针不是门禁**：它依赖 model/ 下的 .nchwbin（大文件），不适合进默认回归。
 * 日常覆盖靠 sample 回归。
 *
 * 用法：zq_model_load_probe <model_dir>
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include "ZQCNN/ZQ_CNN_Net.h"
#include "zq_net_fwd_tripwires.h"
// _concat_NCHW_get_size 是 44 个绊线之外**唯一**的排除项：Concat 的 LayerSetup
// 在加载期就合法调它。逐字照抄的实现与同步义务见那份文件的头注释。
#include "zq_concat_getsize_real.h"

using namespace ZQ;

int main(int argc, char** argv)
{
    const char* dir = (argc > 1) ? argv[1] : "model";
    std::vector<std::string> params;
    FILE* ls = (argc > 2) ? nullptr : nullptr;
    (void)ls;
    // 不引 dirent，直接由 shell 传文件列表进来更省事也更可移植
    for (int i = 2; i < argc; i++) params.push_back(argv[i]);
    std::sort(params.begin(), params.end());

    int ok = 0, bad = 0;
    for (size_t i = 0; i < params.size(); i++)
    {
        std::string p = params[i];
        std::string b = p;
        size_t dot = b.rfind('.');
        if (dot != std::string::npos) b = b.substr(0, dot);
        b += ".nchwbin";
        ZQ_CNN_Net net;
        bool loaded = net.LoadFrom(p, b);
        if (loaded) { ok++; printf("  OK   %s\n", p.c_str()); }
        else { bad++; printf("  FAIL %s\n", p.c_str()); }
    }
    printf("\n%d 个模型：加载成功 %d，失败 %d\n", (int)params.size(), ok, bad);
    return bad ? 1 : 0;
}
