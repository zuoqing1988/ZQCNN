/* `ZQ_CNN_Tensor4D` 的**就地运算**门禁 —— 附录 EE
 *
 * 为什么是这四个
 * --------------
 * 附录 DX 的覆盖探针在 EC 之后重跑（见报告）：基类 21 个有实体的方法里，
 * **8 个仍然没有任何门禁碰过**（占实体行数 129/684），
 * 其中实体 >= 5 行的 5 个是：
 *
 *     SaveToFile 34 / FlipY 24 / FlipX 23 / AddScalar 19 / MulScalar 19
 *
 * `SaveToFile` 每次回归都写文件，不适合进默认门禁（而且它的对拍要跟磁盘状态较劲）。
 * 剩下这四个是**纯就地运算**、无外部依赖、判据可以写得非常硬 —— 这一轮把它们收掉。
 *
 * 判据
 * ----
 * 1. **逐格对拍**，参考用**自己按步长公式取**的缓冲，不复用实现的循环：
 *      FlipX      out[n][h][w][c] == in[n][h][W-1-w][c]
 *      FlipY      out[n][h][w][c] == in[n][H-1-h][w][c]
 *      AddScalar  out[n][h][w][c] == in[n][h][w][c] + s
 *      MulScalar  out[n][h][w][c] == in[n][h][w][c] * s
 *    坐标的记法是 `(n, h, w, c)`，因为张量是 NHWC 对齐布局（`c` 最内层）。
 * 2. **做两次必须回到原样**（FlipX/FlipY 是对合；Add/Mul 则是可算的）——
 *    这一条能抓住"只翻了数据区的一半"这类错。
 * 3. **border 一圈必须原封不动**：这四个方法都只遍历 `n<H, w<W, c<C`，
 *    也就是**只动数据区**。这是有意的（padding 不该被翻），把它钉住，
 *    免得以后有人"顺手"把循环边界写大。
 * 4. **补齐区（align128/align256 里 C 补到 4/8 的那几格）也必须原封不动**。
 *
 * 形态：与 `zq_tile` / `zq_roi` 同（附录 DC.6）—— 逻辑写在成员函数里，
 * 绕不开真实张量对象，所以要编 `ZQ_CNN_Tensor4D.cpp` + resize 内核。
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include "zq_check_child.h"
#include "ZQCNN/ZQ_CNN_Tensor4D.h"

using namespace ZQ;

#define RES_FILE "/tmp/zq_tensorop_res.txt"

enum { K_A0 = 0, K_A128 = 1, K_A256 = 2 };
static const char* g_kind_name[] = { "align0", "align128bit", "align256bit" };

static ZQ_CNN_Tensor4D* make(int kind)
{
    if (kind == K_A0) return new ZQ_CNN_Tensor4D_NHW_C_Align0();
    if (kind == K_A128) return new ZQ_CNN_Tensor4D_NHW_C_Align128bit();
    return new ZQ_CNN_Tensor4D_NHW_C_Align256bit();
}

enum { OP_FLIPX = 0, OP_FLIPY, OP_ADDS, OP_MULS };
static const char* g_opname[] = { "FlipX", "FlipY", "AddScalar", "MulScalar" };
// note: 0 无事 / 1 搭建失败 / 2 返回值不对 / 3 数据错 / 4 二次运算没回到原样
//       / 5 border 或补齐区被动过
static const char* g_note[] = {
    "", "搭建失败", "**返回值与期望相反**", "**数据错**",
    "**做两次没回到原样**", "**border / 补齐区被动过**"
};

struct Case { int op; int N, C, H, W; float s; };

// 每格一个唯一值（base 32，n<8 / c,h,w<32 -> 最大 262143，float 里精确可表示）
static float val(int n, int c, int h, int w)
{
    return (float)(n * 32768 + c * 1024 + h * 32 + w) * 0.001f + 1.0f;
}
static const float PAD = -31337.0f;      // border / 补齐区的哨兵

// **偏移一律用 int64_t 有符号算完再转 size_t**。
// 写成 `(size_t)h * widthStep` 时，h 为负（border 那一圈）会回绕成 ~1.8e19 ——
// 索引飞到天外，第一版 48 个用例**全部**报 ASan 越界，而被测代码一个字节没动。
// 这是本会话第二次踩同一个坑（第一次在 zq_roi_check.cpp）。
static inline int64_t off(const ZQ_CNN_Tensor4D* t, int n, int c, int h, int w)
{
    return (int64_t)n * t->GetSliceStep() + (int64_t)h * t->GetWidthStep()
         + (int64_t)w * t->GetPixelStep() + c;
}

// firstPixelData 之前的 border 那一圈有多少个 float（= 整个分配的起点）
static inline int64_t base_off(const ZQ_CNN_Tensor4D* t)
{
    return -(int64_t)t->GetBorderH() * t->GetWidthStep()
           - (int64_t)t->GetBorderW() * t->GetPixelStep();
}

// 快照**整个分配**（含 border 与补齐区），索引与 off() 同坐标系。
// 第一版按 firstPixelData 起算、只申请 N*sliceStep，结果 border 那一圈
// （负偏移）根本落在向量之外 —— 又一次"门禁的参照系本身是错的"。
static void snapshot(const ZQ_CNN_Tensor4D* t, std::vector<float>& v)
{
    v.assign((size_t)t->GetN() * t->GetSliceStep(), 0.f);
    const float* start = t->GetFirstPixelPtr() + base_off(t);
    for (size_t i = 0; i < v.size(); i++) v[i] = start[i];
}

// 快照是按**整个分配**取的，而 `off()` 是相对 `firstPixelData` 的偏移（border 那一圈是负的）。
// base_off 本身就是"firstPixelData 之前有多少 float"（一个负数），
// 所以换到快照的坐标系要**减去**它：`bidx = off - base_off`。
//   数据格 (0,0,0,0)：off = 0，bidx = 0 + |base_off|  ✔ 落在快照里
//   左上角 border ：off = base_off，bidx = 0          ✔ 快照第 0 格就是它
// 写成 `off + base_off` 会**再减一次**，于是 border 的索引变成负数 ——
// 第一版 48 例全崩、第二版 48 例全"数据错"，两次都是这个符号。
static inline int64_t bidx(const ZQ_CNN_Tensor4D* t, int n, int c, int h, int w)
{
    return off(t, n, c, h, w) - base_off(t);
}

static void run_case(const Case& c, int kind)
{
    ZQ_CNN_Tensor4D* t = make(kind);
    long bad = 0, first = -1; int note = 0;

    if (!t->ChangeSize(c.N, c.H, c.W, c.C, 1, 1)) { note = 1; bad++; }
    else {
        // 填数据区（唯一值），border 与补齐区保持哨兵
        const int ps = t->GetPixelStep();
        float* p = t->GetFirstPixelPtr();
        for (int n = 0; n < c.N; n++)
            for (int h = 0; h < c.H; h++)
                for (int w = 0; w < c.W; w++) {
                    for (int cc = 0; cc < c.C; cc++) p[off(t, n, cc, h, w)] = val(n, cc, h, w);
                    for (int cc = c.C; cc < ps; cc++) p[off(t, n, cc, h, w)] = PAD;   // 补齐区
                }
        const int bH = t->GetBorderH(), bW = t->GetBorderW();
        for (int n = 0; n < c.N; n++)
            for (int h = -bH; h < c.H + bH; h++)
                for (int w = -bW; w < c.W + bW; w++) {
                    const bool in_data = (h >= 0 && h < c.H && w >= 0 && w < c.W);
                    if (!in_data)
                        for (int cc = 0; cc < ps; cc++) p[off(t, n, cc, h, w)] = PAD;  // border
                }

        std::vector<float> before;
        snapshot(t, before);

        // 判据 1：逐格对拍（参考用 before 里的原始值）
        bool r = true;
        switch (c.op) {
            case OP_FLIPX: r = t->FlipX(); break;
            case OP_FLIPY: r = t->FlipY(); break;
            case OP_ADDS: r = t->AddScalar(c.s); break;
            default:       r = t->MulScalar(c.s); break;
        }
        if (!r) { note = 2; bad++; }
        else {
            for (int n = 0; n < c.N && bad < 4; n++)
                for (int cc = 0; cc < c.C && bad < 4; cc++)
                    for (int h = 0; h < c.H && bad < 4; h++)
                        for (int w = 0; w < c.W && bad < 4; w++) {
                            const float x = before[bidx(t, n, cc, h, w)];
                            double want;
                            if (c.op == OP_FLIPX)      want = before[bidx(t, n, cc, h, c.W - 1 - w)];
                            else if (c.op == OP_FLIPY) want = before[bidx(t, n, cc, c.H - 1 - h, w)];
                            else if (c.op == OP_ADDS) want = (double)x + c.s;
                            else                       want = (double)x * c.s;
                            const double got = (double)p[off(t, n, cc, h, w)];
                            const double tol = 1e-4 * (1.0 + fabs(want));
                            if (!(fabs(got - want) <= tol)) {
                                if (first < 0) first = (long)off(t, n, cc, h, w);
                                if (!note) note = 3; bad++;
                            }
                        }
            // 判据 3/4：border 与补齐区必须原封不动
            if (!bad) {
                for (int n = 0; n < c.N && bad < 4; n++)
                    for (int h = -bH; h < c.H + bH && bad < 4; h++)
                        for (int w = -bW; w < c.W + bW && bad < 4; w++) {
                            const bool in_data = (h >= 0 && h < c.H && w >= 0 && w < c.W);
                            for (int cc = 0; cc < ps && bad < 4; cc++) {
                                const bool valid_ch = in_data && cc < c.C;
                                if (valid_ch) continue;
                                if (p[off(t, n, cc, h, w)] != PAD) {
                                    if (first < 0) first = (long)off(t, n, cc, h, w);
                                    if (!note) note = 5; bad++;
                                }
                            }
                        }
            }
            // 判据 2：做两次回到原样
            if (!bad) {
                bool r2 = true;
                switch (c.op) {
                    case OP_FLIPX: r2 = t->FlipX(); break;
                    case OP_FLIPY: r2 = t->FlipY(); break;
                    case OP_ADDS: r2 = t->AddScalar(-c.s); break;
                    default:       r2 = t->MulScalar(1.0f / c.s); break;
                }
                if (!r2) { note = 2; bad++; }
                else {
                    for (int n = 0; n < c.N && bad < 4; n++)
                        for (int cc = 0; cc < c.C && bad < 4; cc++)
                            for (int h = 0; h < c.H && bad < 4; h++)
                                for (int w = 0; w < c.W && bad < 4; w++) {
                                    const double want = (double)before[bidx(t, n, cc, h, w)];
                                    const double got = (double)p[off(t, n, cc, h, w)];
                                    const double tol = 1e-3 * (1.0 + fabs(want));
                                    if (!(fabs(got - want) <= tol)) {
                                        if (first < 0) first = (long)off(t, n, cc, h, w);
                                        if (!note) note = 4; bad++;
                                    }
                                }
                }
            }
        }
    }
    delete t;
    FILE* f = fopen(RES_FILE, "w");
    if (f) { fprintf(f, "%ld %ld %d %ld\n", 1L - bad, bad, note, first); fclose(f); }
}

static const Case g_cases[] = {
  { OP_FLIPX, 1, 3, 2, 2, 0.f },
  { OP_FLIPX, 1, 3, 2, 5, 0.f },
  { OP_FLIPX, 1, 8, 3, 3, 0.f },
  { OP_FLIPX, 2, 5, 2, 4, 0.f },
  { OP_FLIPX, 1, 3, 1, 7, 0.f },
  { OP_FLIPY, 1, 3, 2, 2, 0.f },
  { OP_FLIPY, 1, 3, 5, 2, 0.f },
  { OP_FLIPY, 1, 8, 3, 3, 0.f },
  { OP_FLIPY, 2, 5, 4, 2, 0.f },
  { OP_FLIPY, 1, 3, 1, 1, 0.f },
  { OP_ADDS,  1, 3, 2, 2, 1.5f },
  { OP_ADDS,  1, 8, 3, 3, -2.25f },
  { OP_ADDS,  2, 5, 2, 4, 0.0f },
  { OP_MULS,  1, 3, 2, 2, 2.0f },
  { OP_MULS,  1, 8, 3, 3, 0.5f },
  { OP_MULS,  2, 5, 2, 4, -1.0f },
};
static const int N_CASE = (int)(sizeof(g_cases) / sizeof(g_cases[0]));

static int g_case = 0, g_ok = 0, g_bad = 0, g_crash = 0;

static void one(const Case& c, int kind)
{
    g_case++;
    remove(RES_FILE);
    pid_t pid = fork();
    if (pid == 0) { zq_child_silence_stderr(); run_case(c, kind); _exit(0); }
    int st = 0; waitpid(pid, &st, 0);
    long ok = 0, bad = 0, first = -1; int note = 0, have = 0;
    FILE* f = fopen(RES_FILE, "r");
    if (f) { have = (fscanf(f, "%ld %ld %d %ld", &ok, &bad, &note, &first) == 4); fclose(f); }
    char tag[96];
    snprintf(tag, sizeof(tag), "%-10s N%dC%dH%dW%d s=%.3f", g_opname[c.op], c.N, c.C, c.H, c.W, c.s);
    if (!have) {
        g_crash++;
        printf("  %-12s %-36s  没跑完%s\n", g_kind_name[kind], tag,
               WIFSIGNALED(st) ? "（子进程被信号杀）" : "（结果文件读不出来）");
        return;
    }
    if (WIFSIGNALED(st)) {
        g_crash++;
        printf("  %-12s %-36s  CRASH（信号 %d）\n", g_kind_name[kind], tag, WTERMSIG(st));
        return;
    }
    if (bad > 0) {
        g_bad++;
        printf("  %-12s %-36s  %s %ld 项", g_kind_name[kind], tag, g_note[note], bad);
        if (note == 3 || note == 4 || note == 5) printf("，首个错格偏移 %ld", first);
        printf("\n");
    } else g_ok++;
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("ZQ_CNN_Tensor4D 就地运算门禁（附录 EE）\n");
    printf("为什么是这四个：覆盖探针在 EC 之后重跑，基类 21 个有实体的方法里 8 个仍无门禁，\n");
    printf("  其中 >= 5 行的是 SaveToFile 34 / FlipY 24 / FlipX 23 / AddScalar 19 / MulScalar 19。\n");
    printf("  SaveToFile 每次回归写文件、不适合进默认门禁；剩下这四个是纯就地运算、判据能写得很硬。\n");
    printf("判据：① 逐格对拍（参考用自己按步长公式取的缓冲）② 做两次回到原样\n");
    printf("      ③ **border 一圈必须原封不动**（这四个只遍历数据区，padding 不该被翻/被改）\n");
    printf("      ④ align128/align256 的**补齐区**也必须原封不动\n\n");
    for (int k = 0; k < 3; k++) {
        const int c0 = g_case, k0 = g_ok, b0 = g_bad, x0 = g_crash;
        for (int i = 0; i < N_CASE; i++) one(g_cases[i], k);
        printf("  %-12s %d 个用例：对 %d，错 %d，崩 %d\n",
               g_kind_name[k], g_case - c0, g_ok - k0, g_bad - b0, g_crash - x0);
    }
    printf("\n共 %d 个用例：全对 %d，有错 %d，崩溃/搭建失败 %d\n", g_case, g_ok, g_bad, g_crash);
    if (g_bad || g_crash)
        printf("**每一项在下结论之前都要先用独立复现对一遍**（附录 CA.3）。\n");
    return (g_bad || g_crash) ? 1 : 0;
}
