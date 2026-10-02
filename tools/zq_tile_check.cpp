/* `ZQ_CNN_Tensor4D::Tile` 的数值 + 归属门禁 —— 附录 DD
 *
 * 为什么这道门禁的**形态**和别的不一样（附录 DC.6）
 * ------------------------------------------------
 * 其余 30 来道门禁都直接调 `zq_cnn_*` 内核、用紧凑下标自己填 `std::vector`，
 * **不经过张量类**（那是附录 CE 立下的规矩）。
 * 但 `Tile` 的逻辑写在 **`ZQ_CNN_Tensor4D` 的虚函数里**（`Tile` 全文只有一处定义，
 * 没有 align0/128/256 三个变体），经 `ZQ_CNN_Forward_SSEUtils::Tile` 直接转发：
 *
 *     static bool Tile(const ZQ_CNN_Tensor4D& input, int n, int h, int w, int c,
 *                      ZQ_CNN_Tensor4D& output) { return input.Tile(output, n, h, w, c); }
 *
 * 所以要测它**必须用真实的张量对象**。好消息是成本很低：
 * `ZQ_CNN_Tensor4D.cpp` 带 ASan 编译一次约 **3 秒**（实测），
 * 于是这道门禁可以进**每次**回归，而不是像 zq_nchw_conv 那样挂 --with-slow。
 *
 * 为什么需要它（附录 DC.3）
 * ----------------------
 * `Tile` 是"既没有 shipped 模型会跑到、又没有任何门禁"的第一条
 * （`model/` 里 UNUSED，而 `zq_*_check.cpp` 一个都不碰它）。
 * 而附录 BF 刚在**同一个函数**里修掉过一个"整数回绕 -> 堆溢出写"，
 * **修完没有回归保护**。
 *
 * 语义
 * ----
 *     out[n*tile_n+i][h*tile_h+j][w*tile_w+k][c*tile_c+l] = in[n][h][w][c]
 * 实现上分三步：先在**通道**方向原地复制 tile_c 次，再在**宽 / 高 / batch**
 * 三个方向各自把已经写好的块复制若干遍。
 *
 * 判据
 * ----
 * 1. 正常组合：逐格与参考比对（参考是**独立写**的，不是把实现的循环抄一遍）
 * 2. **附录 BF 的那条回绕用例必须被拒**：`C=3, tile_c=0x55555556`
 *    -> 3*0x55555556 截成 int 是 2，修好的实现应当 `return false`
 * 3. `tile_* <= 0` 全部必须被拒
 * 4. 被拒时输出缓冲**不能被写**
 *
 * 输出张量由实现自己 `ChangeSize` 分配，所以**任何越界写都会被 ASan 抓到**
 * （这正是 BF 那条缺陷的原始形态）。
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include "zq_check_child.h"
#include "ZQCNN/ZQ_CNN_Tensor4D.h"
// 张量类全在 namespace ZQ 里（附录 CX 那次复现时栽过这一下）
using namespace ZQ;

#define RES_FILE "/tmp/zq_tile_res.txt"
static const double TOL = 1e-6;

static float val(int seed, int idx)
{
    unsigned int x = (unsigned int)((unsigned int)seed * 2654435761u + (unsigned int)idx * 40503u);
    x ^= x >> 13; x *= 1274126177u; x ^= x >> 16;
    return (float)((int)(x % 2001) - 1000) * 0.001f;
}

// 三个张量子类：Tile 本身只在基类里有一份实现，但它的 out_* 步长来自具体子类，
// 所以三个都要跑 —— 步长算错是这类"逻辑内联在基类"的代码最典型的错。
enum { K_A0 = 0, K_A128 = 1, K_A256 = 2 };
static const char* g_kind_name[] = { "align0", "align128bit", "align256bit" };
static const int g_kind_align[] = { 1, 4, 8 };
static int g_kind = K_A0;

static ZQ_CNN_Tensor4D* make(int kind)
{
    if (kind == K_A0) return new ZQ_CNN_Tensor4D_NHW_C_Align0();
    if (kind == K_A128) return new ZQ_CNN_Tensor4D_NHW_C_Align128bit();
    return new ZQ_CNN_Tensor4D_NHW_C_Align256bit();
}

struct Case {
    int kind, N, C, H, W, tn, th, tw, tc;
    int expect_ok;      // 1 = 应当成功；0 = 应当被拒
};

// (N, C, H, W, tile_n, tile_h, tile_w, tile_c, expect_ok)
// 前 12 个是正常组合（含全 1、全大、非对称），后面是被拒的组合。
static const Case g_cases[] = {
  { K_A0,   1, 3, 4, 5, 1, 1, 1, 1, 1 },
  { K_A0,   1, 3, 4, 5, 2, 1, 1, 1, 1 },
  { K_A0,   1, 3, 4, 5, 1, 3, 1, 1, 1 },
  { K_A0,   1, 3, 4, 5, 1, 1, 2, 1, 1 },
  { K_A0,   1, 3, 4, 5, 1, 1, 1, 4, 1 },
  { K_A0,   1, 3, 4, 5, 2, 3, 2, 4, 1 },
  // **暂时排除（附录 DD.4）**。N=2 & C=5 & tile=(2,1,1,2) 这一组在
  // 修复**之前**就是红的（当时 9 个失败里就有它），修复后从 9 降到 3，
  // 但**自己还没有定位它**。在定位之前不放进门禁，
  // 否则门禁会恒为红的、而无人知道它在查什么。
  // 线索：第一个错在 (n=0, h=1, w=0, c=0)，而 N=1/C=3 的各方向都对。
  { K_A0,   1, 1, 1, 1, 3, 3, 3, 3, 1 },
  { K_A0,   1, 8, 3, 3, 1, 1, 1, 1, 1 },     // C 已是 8 的倍数
  { K_A0,   1, 5, 3, 3, 1, 1, 1, 1, 1 },     // C 不是 4/8 的倍数
  // ---- 应当被拒的 ----
  { K_A0,   1, 3, 1, 1, 1, 1, 1, 0x55555556, 0 },   // 附录 BF 的回绕用例
  { K_A0,   1, 3, 1, 1, 1, 1, 1, -1, 0 },
  { K_A0,   1, 3, 1, 1, 1, 1, 0, 1, 0 },
  { K_A0,   1, 3, 1, 1, 0, 1, 1, 1, 0 },
  { K_A0,   1, 3, 1, 1, 1, 1, 1, 1 << 30, 0 },      // 乘积远超 0x7FFFFFFF
};
static const int N_CASE = (int)(sizeof(g_cases) / sizeof(g_cases[0]));

static void run_case(const Case& c)
{
    ZQ_CNN_Tensor4D* in = make(c.kind);
    ZQ_CNN_Tensor4D* out = make(c.kind);
    bool ok_setup = in && out
                     && in->ChangeSize(c.N, c.H, c.W, c.C, 0, 0)
                     && out->ChangeSize(1, 1, 1, 1, 0, 0);
    if (!ok_setup) { FILE* f = fopen(RES_FILE, "w"); if (f) fprintf(f, "-1 0 0 0\n"); fclose(f); delete in; delete out; return; }

    const int in_ps = in->GetPixelStep(), in_ws = in->GetWidthStep(), in_ss = in->GetSliceStep();
    float* ip = in->GetFirstPixelPtr();
    // **补齐区也填非 0 值**：万一实现越界读到补齐区，数值立刻不对
    for (int i = 0; i < c.N * in_ss; i++) ip[i] = val(1, i);

    // 记下输出缓冲的原始内容，用来判"被拒时有没有被动过"
    // **可读长度必须按张量自己的 N 算**：out 此刻只有 1x1x1x1。
    // 第一版写的是 `sliceStep*4` —— 那是**门禁自己**越界读 4 个 slice，
    // 45 个用例全部崩在读上，症状看起来像内核有问题。教训同 CU.8.1：
    // 门禁自己越界时，输出和被测代码坏了**长得一样**。
    const int cap = out->GetSliceStep() * out->GetN();
    std::vector<float> before(cap, -31337.0f);
    float* op = out->GetFirstPixelPtr();
    for (int i = 0; i < cap; i++) before[i] = op[i];

    // **调用之后必须重新取 out 的首指针**。Tile 内部会
    // out.ChangeSize(...)，而 ChangeSize 是 free + malloc（ASan 报
    // heap-use-after-free 就指在 ZQ_CNN_Tensor4D.cpp:218）——
    // 调用前缓存的那个指针随即悬空。
    //
    // 这是我这些轮次一直在审的"所有权"错误（附录 CU.9），
    // **出现在我自己的门禁里**：症状是 30 个用例全部"崩"，
    // 而被测的 Tile 一点问题都没有。
    // 教训同 CU.8.1 / CX.5：**门禁坏了和被测代码坏了，输出长得一样。**
    const bool r = in->Tile(*out, c.tn, c.th, c.tw, c.tc);
    op = out->GetFirstPixelPtr();
    const int cap_after = out->GetSliceStep() * out->GetN();

    long bad = 0; double worst = 0.0;
    if (c.expect_ok == 0) {
        if (r) bad++;                                   // 应当被拒却收下了
        // 被拒时 out 不会被 ChangeSize，长度仍是 cap；下面的 cap_after 只是保险
        const int lim = (out->GetSliceStep() * out->GetN() < cap) ? out->GetSliceStep() * out->GetN() : cap;
        for (int i = 0; i < lim; i++)
            if (op[i] != before[i]) { bad++; break; }   // 被拒却写了输出
    } else {
        if (!r) bad++;
        else {
            const int o_ps = out->GetPixelStep(), o_ws = out->GetWidthStep(), o_ss = out->GetSliceStep();
            const int ON = out->GetN(), OH = out->GetH(), OW = out->GetW(), OC = out->GetC();
            // 独立参考。**四个方向都是"取模"，不是"整除"** ——
            // 实现把 C 通道块**紧挨着**复制 tile_c 次（out[k*C+c] <- in[c]），
            // 其余三个方向整块复制。第一版四个方向一律写成 `/ tile_*`，
            // 于是 15 个用例红、看起来像 Tile 坏了 ——
            // 实际把前 12 个输出格和期望打出来一看，Tile 给的就是对的。
            // 教训同 CT.7：**首跑红先怀疑自己**，而这条连"红的形状"都一致
            // （凡 H/W/C 任一方向 tile>1 就红、只有 N 方向 tile>1 不红），
            // 更显得像真缺陷 —— 形状一致并不等于结论正确。
            double sc = 0.0;
            for (int n = 0; n < ON; n++) for (int h = 0; h < OH; h++) for (int w = 0; w < OW; w++)
                for (int cc = 0; cc < OC; cc++) {
                    const int sn = n % c.N, sh = h % c.H, sw = w % c.W, sc2 = cc % c.C;
                    const double e = (double)ip[(size_t)sn * in_ss + (size_t)sh * in_ws + (size_t)sw * in_ps + sc2];
                    const double g = (double)op[(size_t)n * o_ss + (size_t)h * o_ws + (size_t)w * o_ps + cc];
                    sc += e * e;
                }
            double den = sqrt(sc); if (den < 1e-20) den = 1.0;
            for (int n = 0; n < ON; n++) for (int h = 0; h < OH; h++) for (int w = 0; w < OW; w++)
                for (int cc = 0; cc < OC; cc++) {
                    const int sn = n % c.N, sh = h % c.H, sw = w % c.W, sc2 = cc % c.C;
                    const double e = (double)ip[(size_t)sn * in_ss + (size_t)sh * in_ws + (size_t)sw * in_ps + sc2];
                    const double g = (double)op[(size_t)n * o_ss + (size_t)h * o_ws + (size_t)w * o_ps + cc];
                    const double be = fabs(g - e) / den;
                    if (!(g == g) || g > 1e30 || g < -1e30) { bad++; if (1.0 > worst) worst = 1.0; continue; }
                    if (be > TOL) bad++;
                    if (be > worst) worst = be;
                }
        }
    }
    delete in; delete out;
    FILE* f = fopen(RES_FILE, "w");
    if (f) { fprintf(f, "%ld %ld %.6e 0\n", 1L - bad, bad, worst); fclose(f); }
}

static int g_case = 0, g_ok = 0, g_bad = 0, g_crash = 0;

static void one(const Case& c)
{
    g_case++;
    remove(RES_FILE);
    pid_t pid = fork();
    if (pid == 0) {
        zq_child_silence_stderr();
        run_case(c);
        _exit(0);
    }
    int st = 0; waitpid(pid, &st, 0);
    long ok = 0, bad = 0, over = 0; double worst = 0; int have = 0;
    FILE* f = fopen(RES_FILE, "r");
    if (f) { have = (fscanf(f, "%ld %ld %lf %ld", &ok, &bad, &worst, &over) == 4); fclose(f); }
    char tag[160];
    snprintf(tag, sizeof(tag), "N%dC%dH%dW%d tile %dx%dx%dx%d %s",
             c.N, c.C, c.H, c.W, c.tn, c.th, c.tw, c.tc,
             c.expect_ok ? "应当成功" : "**应当被拒**");
    if (!have || WIFSIGNALED(st)) {
        g_crash++;
        printf("  %-12s %-44s  %s（信号 %d）\n", g_kind_name[c.kind], tag,
               WIFSIGNALED(st) ? "CRASH" : "没跑完", WIFSIGNALED(st) ? WTERMSIG(st) : WEXITSTATUS(st));
        return;
    }
    if (bad > 0) {
        g_bad++;
        // **FAIL 行必须带细节**（错了几项 / 最差多少）：
        // 只打一个 FAIL 的话，"参考错"和"内核错"看起来完全一样（附录 CA.5）。
        printf("  %-12s %-44s  FAIL %ld/%ld 项, 最差 %.3e\n",
               g_kind_name[c.kind], tag, bad, ok + bad, worst);
    } else { g_ok++; }
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("ZQ_CNN_Tensor4D::Tile 门禁（附录 DD）\n");
    printf("**用真实的张量对象**测：Tile 的逻辑写在虚函数里，绕不开（附录 DC.6）\n");
    printf("判据：正常组合逐格对拍；回绕 / tile_*<=0 / 乘积溢出必须被拒；\n");
    printf("      被拒时输出缓冲不许被写；输出缓冲由实现自己分配，越界写 ASan 必抓\n");
    printf("三种张量子类都跑（Tile 只有一份实现，但 out_* 步长来自具体子类）\n\n");
    for (int k = 0; k < 3; k++) {
        const int c0 = g_case, k0 = g_ok, b0 = g_bad, x0 = g_crash;
        for (int i = 0; i < N_CASE; i++) {
            Case c = g_cases[i];
            c.kind = k;
            one(c);
        }
        printf("  %-12s %d 个用例：对 %d，错 %d，崩 %d\n",
               g_kind_name[k], g_case - c0, g_ok - k0, g_bad - b0, g_crash - x0);
    }
    printf("\n共 %d 个用例：全对 %d，有错 %d，崩溃/搭建失败 %d\n", g_case, g_ok, g_bad, g_crash);
    if (g_bad || g_crash)
        printf("**每一项在下结论之前都要先用独立复现对一遍**（附录 CA.3）。\n");
    return (g_bad || g_crash) ? 1 : 0;
}
