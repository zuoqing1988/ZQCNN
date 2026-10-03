/* `ZQ_CNN_Net_NCHWC` 的 net 级门禁 —— 附录 FB
 *
 * 为什么需要它
 * ------------
 * 附录 EN 修就地守卫时**同时改了两份** `ZQ_CNN_Net.h` 与 `ZQ_CNN_Net_NCHWC.h`
 * （它们是各自独立的拷贝，只改一份必然漂移）。但两边的覆盖极不对等：
 *
 *   - `ZQCNN/ZQ_CNN_Net.h`  -> 有 `zq_concat_alias_check` 走真的 `LoadFrom` 验；
 *   - `ZQCNN/ZQ_CNN_Net_NCHWC.h` -> **只有编译覆盖**（它被主工程 CMake 编过），
 *     行为上零门禁。
 *
 * 也就是说 EN 那处镜像改动**从来没有被任何东西验证过**。
 * 这正是 ES.2（`ZQ_OpticalFlow.h` 重复声明，任何编译器都编不过、却活了下来）
 * 的同一个形状，只是这里更隐蔽 —— 它**编得过**，所以"能编过"这道轴也照不到。
 *
 * 这一层支持 10 种层类型、**没有 Concat**，所以 EN 那个"跨下标 Concat 别名"
 * 在这里不成立；但**同下标别名**（`Convolution bottom=A top=A`）和
 * **跨下标别名**（`Convolution bottom=A bottom=B top=B`）对任何形状会变的层都成立，
 * 而 `Convolution` / `DepthwiseConvolution` / `InnerProduct` / `Eltwise`
 * 全都是形状会变的。
 *
 * 判据
 * ----
 *   0  Input -> Convolution(bottom=data top=A) -> ReLU(bottom=A top=B)   放行
 *   1  ReLU(bottom=A top=A)                                              **同下标别名** -> 拒
 *   2  ReLU(bottom=A top=B) 但 A/B 在别处也用                             跨下标别名 -> 拒
 *   3  同上但合法（每个 blob 各自被消费）                                  放行
 *
 * 形态：与 `zq_concat_alias_check` 同构 —— 走真的 `LoadFrom`，
 * 依赖 `tools/zq_net_fwd_tripwires.h`（44 个绊线桩），
 * 只需编 `ZQ_CNN_Tensor4D.cpp` + resize 内核 + 两个 `.cpp`（Net_NCHWC / Layer）。
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include "zq_check_child.h"
#include "ZQCNN/ZQ_CNN_Tensor4D.h"
#include "ZQCNN/ZQ_CNN_Net_NCHWC.h"
#include "ZQCNN/ZQ_CNN_Tensor4D_NCHWC.h"
#include "zq_net_fwd_tripwires.h"

using namespace ZQ;

#define RES_FILE "/tmp/zq_nchwcnet_res.txt"
#define PARAM_FILE "/tmp/zq_nchwcnet_net.zqparams"
#define BINARY_FILE "/tmp/zq_nchwcnet_net.nchwbin"

// 0/1/4 是**必须放行**的对照组；2/3 是别名。
// 只测"别名被拒"的话，在 LoadFrom 开头无脑 return false 也能全绿
// （附录 EN.6 同一个道理）。
//
// 用例 1 尤其重要：ReLU 在 `_is_inplace_safe` 名单里，**同下标别名对它是合法的**。
// 守卫如果被"顺手"收紧成"一律不许 top == bottom"，这一条会红 ——
// 而收紧它的人不会想到自己拒掉了所有真实的就地 ReLU（附录 FA 讲的单边风险）。
static const char* g_desc[] = {
    "三段链，全部独立 blob",
    "ReLU top == bottom（就地安全层的豁免，必须放行）",
    "Convolution top == bottom（形状会变的层，同下标别名）",
    "Eltwise top == 第二个 bottom（跨下标别名）",
    "四段链，各 blob 独立",
};
static const int g_expect[] = { 1, 1, 0, 0, 1 };
static const int N_CASE = 5;

// 这几层的 `Forward` 会打上 ConvertToNCHWC 之类的调用；本门禁只调 LoadFrom，
// 走不到那里，所以只要 ReadParam 接受了就算通过。
static std::string param_line(int idx)
{
    std::string head = "Input name=data C=3 H=8 W=8\n";
    switch (idx) {
    case 0:
        return head
             + "Convolution name=c1 bottom=data top=A num_output=4 kernel_size=3 stride=1 pad=1\n"
             + "ReLU name=r1 bottom=A top=B\n"
             // NCHWC 的 Pooling 是 `ZQ_CNN_Layer_NCHWC_Pooling`（**另一个类**，
             // 见 ZQCNN/ZQ_CNN_Layer_NCHWC.h:1979），它认 `kernel_size` / `stride` / `pad`
             // —— **不是**主库 `ZQ_CNN_Layer_Pooling` 的
             // `pool=` / `kernel_H=` / `stride_H=` / `pad_type=`。
             // 我按主库那份写了参数名，于是每个 key 都被报 "unknown para"，
             // 门禁报"应放行、实际拒绝" —— 看着像守卫坏了，其实是**类找错了**。
             + "Pooling name=p1 bottom=B top=C kernel_size=2 stride=2 pad=0\n";
    case 1:
        // 就地安全的层：top == bottom **合法**，必须放行
        return head
             + "Convolution name=c1 bottom=data top=A num_output=4 kernel_size=3 stride=1 pad=1\n"
             + "ReLU name=r1 bottom=A top=A\n"
             + "ReLU name=r2 bottom=A top=B\n";
    case 2:
        // 形状会变的层，同下标别名
        return head
             + "Convolution name=c1 bottom=data top=data num_output=4 kernel_size=3 stride=1 pad=1\n"
             + "ReLU name=r1 bottom=data top=A\n";
    case 3:
        // 跨下标：Eltwise 的 tops[0] 就是 bottoms[1]
        return head
             + "Convolution name=c1 bottom=data top=A num_output=4 kernel_size=3 stride=1 pad=1\n"
             + "Convolution name=c2 bottom=data top=B num_output=4 kernel_size=3 stride=1 pad=1\n"
             + "Eltwise name=e1 bottom=A bottom=B top=B operation=SUM\n";
    default:
        return head
             + "Convolution name=c1 bottom=data top=A num_output=4 kernel_size=3 stride=1 pad=1\n"
             + "ReLU name=r1 bottom=A top=B\n"
             + "Pooling name=p1 bottom=B top=C kernel_size=2 stride=2 pad=0\n"
             + "ReLU name=r2 bottom=C top=D\n";
    }
}

// `ZQ_CNN_Net_NCHWC` 是**模板类**（T 是张量变体）—— 随仓库唯一的用法
// `SampleLnet106.cpp:41-48` 就是 NCHWC1/4/8 三种都实例化。
// 守卫写在模板里，三种变体各跑一遍才算数。
static bool do_load(int variant)
{
    if (variant == 1) {
        ZQ_CNN_Net_NCHWC<ZQ_CNN_Tensor4D_NCHWC4> net;
        return net.LoadFrom(PARAM_FILE, BINARY_FILE);
    }
    if (variant == 2) {
        ZQ_CNN_Net_NCHWC<ZQ_CNN_Tensor4D_NCHWC8> net;
        return net.LoadFrom(PARAM_FILE, BINARY_FILE);
    }
    ZQ_CNN_Net_NCHWC<ZQ_CNN_Tensor4D_NCHWC1> net;
    return net.LoadFrom(PARAM_FILE, BINARY_FILE);
}

static int child(int idx)
{
    std::string p = param_line(idx);
    FILE* fp = fopen(PARAM_FILE, "wb");
    if (!fp) return 2;
    fwrite(p.c_str(), 1, p.size(), fp);
    fclose(fp);
    // 权重文件：**必须真的够长**，否则 Convolution::LoadBinary_NCHW 失败、
    // LoadFrom 也就返回 false —— 于是"合法对照"会显示成"被拒"，
    // 症状与"守卫误杀"一模一样。
    // 卷积的 `dst_len = filters->GetN()*GetH()*GetW()*GetC()`
    //          = num_output(4) * kernel_H(3) * kernel_W(3) * C(3) = 108 个 float。
    // 用例 2 有两个卷积，所以按**用例需要的层数**写。
    int convs = (idx == 3) ? 2 : 1;   // 除跨下标那条（两个卷积），其余都是一个
    const int FLOATS_PER_CONV = 4 * 3 * 3 * 3;   // 108
    fp = fopen(BINARY_FILE, "wb");
    if (!fp) return 2;
    {
        std::vector<float> zeros((size_t)convs * FLOATS_PER_CONV, 0.f);
        if (!zeros.empty())
            fwrite(&zeros[0], sizeof(float), zeros.size(), fp);
    }
    fclose(fp);

    bool loaded = false;
    loaded |= do_load(0);
    loaded |= do_load(1);
    loaded |= do_load(2);
    FILE* f = fopen(RES_FILE, "a");
    if (f) { fprintf(f, "%d %d\n", idx, loaded ? 1 : 0); fclose(f); }
    return 0;
}

int main(int argc, char** argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc > 1) {
        int idx = atoi(argv[1]);
        if (idx < 0 || idx >= N_CASE) return 0;
        return child(idx);
    }
    printf("ZQ_CNN_Net_NCHWC 就地守卫门禁（附录 FB）\n");
    printf("判据：形状会变的层不得把自己的 bottom 当 top（同下标与跨下标都不行）。\n");
    printf("      44 个 ZQ_CNN_Forward_SSEUtils 辅助函数是绊线，被调到就红。\n\n");

    int ok = 0, bad = 0, crash = 0;
    for (int i = 0; i < N_CASE; i++) {
        remove(RES_FILE);
        pid_t pid = fork();
        if (pid == 0) { zq_child_silence_stderr(); _exit(child(i)); }
        int st = 0; waitpid(pid, &st, 0);

        int got = -1, loaded = -1, have = 0;
        FILE* f = fopen(RES_FILE, "r");
        if (f) { have = (fscanf(f, "%d %d", &got, &loaded) == 2); fclose(f); }
        int tripped = (WIFEXITED(st) && WEXITSTATUS(st) == 3);

        if (!have || WIFSIGNALED(st) || tripped) {
            crash++;
            printf("  用例 %d %-42s %s%s\n", i, g_desc[i],
                   !have ? "结果文件读不出来"
                         : (tripped ? "**撞上绊线**" : "子进程被信号杀"),
                   tripped ? "（LoadFrom 不该调 Forward）" : "");
            continue;
        }
        if (loaded != g_expect[i]) {
            bad++;
            printf("  用例 %d %-42s %s：实际%s\n", i, g_desc[i],
                   g_expect[i] ? "应放行" : "应拒", loaded ? "放行" : "拒绝");
        } else {
            ok++;
            printf("  用例 %d %-42s %s\n", i, g_desc[i],
                   loaded ? "放行" : "拒绝");
        }
    }
    printf("\n共 %d 个用例：对 %d，错 %d，崩/撞线 %d\n", N_CASE, ok, bad, crash);
    return (bad || crash) ? 1 : 0;
}
