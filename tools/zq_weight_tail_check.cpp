// 权重**字节数**契约门禁（附录 GZ）。
//
// 契约：`.zqparams` 里各层声明的尺寸加总，必须**等于** `.nchwbin` 的长度。
// 少字节会报错（离原因近），多字节**不会** —— `LoadFrom` 只查"字节不够"
// （附录 GZ.1），尾部多余的数据被静默忽略。
//
// 怎么量的：不看库自己打的那行 `warning:`，而是用**库自己的 `SaveModel`**
// 把刚读进来的权重再存一遍，比长度。
//
//     回存长度 == 文件长度  -> 配对
//     回存长度 <  文件长度  -> 文件尾部有一截从未被任何层读到
//
// 两条独立判据守同一件事（附录 GZ.4）：
//   - `zq_model_params` 用库内部那行 warning + ROUNDTRIP 行；
//   - 本门禁在**库外面**，一条 warning 文本都不依赖。
// 少任何一条，剩下的那条就可能被改措辞/改重定向悄悄弄失效。
//
// 2026-10-04 实测：27 个随仓 `.zqparams` 里 23 个有配套 `.nchwbin`，
// 其中 **21 个字节精确**；`det1-dw20-fast` 多 464 字节、
// `det1-dw20-plus` 多 12752 字节，两对都已修正（附录 GZ.5）。
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <dirent.h>
#include "ZQCNN/ZQ_CNN_Tensor4D.h"
#include "ZQCNN/ZQ_CNN_Net.h"
// 链接期需要的那批 ZQ_CNN_Forward_SSEUtils 符号，用现成的**绊线桩**顶上
// （与 zq_model_params_check.cpp 同一套，见那个文件头的说明）。
// 本门禁一个 Forward 都不跑，桩响了本身就是一条结论。
#include "zq_net_fwd_tripwires.h"
#include "zq_concat_getsize_real.h"

#define MODEL_DIR "/mnt/d/ZQCNN/model"

static long long file_size(const char* p)
{
    FILE* f = fopen(p, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long long n = ftell(f);
    fclose(f);
    return n;
}

// 单个模型：加载 -> 回存 -> 比长度。返回 0 通过。
static int check_one(const char* zqparams, const char* nchwbin, bool verbose)
{
    ZQ::ZQ_CNN_Net net;
    if (!net.LoadFrom(zqparams, nchwbin)) {
        // 加载失败不算"尾部多了字节"，但也不该悄悄放过：门禁要能看见。
        printf("        %s 加载失败（字节**不够**或参数被拒）\n", nchwbin);
        return 1;
    }
    char rt[512];
    snprintf(rt, sizeof(rt), "/tmp/zq_weight_tail_rt_%s.nchwbin",
             strrchr(nchwbin, '/') ? strrchr(nchwbin, '/') + 1 : nchwbin);
    remove(rt);
    if (!net.SaveModel(rt)) {
        printf("        回存失败\n");
        return 1;
    }
    long long ondisk = file_size(nchwbin);
    long long saved = file_size(rt);
    remove(rt);
    if (verbose) printf("        ondisk=%lld reachable=%lld tail=%lld\n",
                        ondisk, saved, ondisk - saved);
    return saved == ondisk ? 0 : 1;
}

int main(int argc, char** argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    // 单模型诊断模式：<zqparams> <nchwbin>，专供变异测试用
    if (argc >= 3) return check_one(argv[1], argv[2], true);

    printf("权重字节数契约门禁（附录 GZ）\n");
    printf("判据：用库自己的 SaveModel 回存一遍，回存长度必须等于 .nchwbin 长度。\n\n");

    DIR* d = opendir(MODEL_DIR);
    if (!d) {
        printf("  打开 %s 失败 —— 门禁不能在没有模型的情况下报通过\n", MODEL_DIR);
        return 1;
    }
    std::vector<std::string> params;
    struct dirent* e;
    while ((e = readdir(d)) != NULL) {
        std::string n = e->d_name;
        if (n.size() > 9 && n.compare(n.size() - 9, 9, ".zqparams") == 0)
            params.push_back(std::string(MODEL_DIR) + "/" + n);
    }
    closedir(d);
    if (params.empty()) {
        printf("  没有找到 .zqparams —— 同样是\"没检查\"，不是通过\n");
        return 1;
    }

    int ok = 0, bad = 0, skipped = 0;
    for (size_t i = 0; i < params.size(); i++) {
        std::string p = params[i];
        p = p.substr(0, p.size() - 9) + ".nchwbin";
        const char* base = strrchr(params[i].c_str(), '/');
        base = base ? base + 1 : params[i].c_str();
        if (access(p.c_str(), R_OK) != 0) {
            skipped++;
            printf("  %-34s 跳过（仓库里没有配套 .nchwbin）\n", base);
            continue;
        }
        long long ondisk = file_size(p.c_str());
        printf("  %-34s %lld 字节 ... ", base, ondisk);
        if (check_one(params[i].c_str(), p.c_str(), true) == 0) {
            ok++;
            printf("  OK（回存 %lld 字节，逐字节配对）\n", ondisk);
        } else {
            bad++;
            printf("  **FAIL**\n");
        }
    }
    printf("\n共 %d 个模型：字节精确 %d，不配对 %d，跳过 %d\n",
           (int)params.size(), ok, bad, skipped);
    return bad == 0 ? 0 : 1;
}
