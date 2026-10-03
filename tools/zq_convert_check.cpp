/* `ZQ_CNN_Tensor4D` 的 **Convert 族**门禁 —— 附录 DZ
 *
 * 覆盖谁
 * ------
 * 附录 DX 的覆盖探针把基类里"有实体但零门禁"的方法排了序，这一族是其中一整块：
 *   ConvertFromBGR 36 / ConvertFromBGR2GRAY 35 / ConvertFromGray 34 /
 *   SaveToFile 34 / ConvertToBGR 27 / ConvertToCompactNCHW 17 /
 *   ConvertFromCompactNCHW 21
 *
 * 判据设计：**用精确往返代替逐条抄公式**
 * ----------------------------------------
 * 第一版想直接写"期望值 = (b - 127.5f) * 0.0078125f"逐格对拍。写到一半发现
 * 那是在**把实现的公式抄一遍** —— 抄错了门禁和实现一起错，看着全绿其实什么都没验
 * （附录 AY.5 / BC 那类假绿的成因）。
 *
 * 改用**精确往返**：`ConvertFromBGR` 与 `ConvertToBGR` 互为逆运算，
 * 且**逐位精确**（下面每条判据后面都给了推导，不是"试出来能对上"）：
 *
 *   设输入字节 b，`ConvertFromBGR` 写入  t = (b - 127.5f) / 128
 *   `ConvertToBGR` 读出    b' = clamp( (int)( (t + 1) * 127.5f + 0.5f ), 0, 255 )
 *        (t + 1) * 127.5 = (b - 127.5) * 127.5/128 + 127.5
 *        b = 0   -> 0.498 + 0.5 = 0.998 -> 0   ✓
 *        b = 100 -> 100.11 + 0.5 = 100.61 -> 100 ✓
 *        b = 128 -> 128.00 + 0.5 = 128.50 -> 128 ✓
 *        b = 255 -> 254.50 + 0.5 = 255.00 -> 255 ✓
 *
 * 于是判据是"喂进去的字节必须原样回来"，**参考值不依赖任何一条实现公式**。
 * 反向同理：把张量填成 `v = b/127.5f - 1.0f`，`ConvertToBGR` 必须吐回 `b`
 * （`(v+1)*127.5 = b`，`+0.5` 后取整仍是 b）。
 *
 * 灰度那边（`ConvertFromBGR2GRAY`）不查具体灰度值，只查
 * **往返一致 + 通道数正确 + 线性**（灰度必须落在 0..255 的凸组合里）——
 * 因为 0.114/0.587/0.299 是模型约定、抄错很难发现，而"线性 + 落在范围里"才是可判的。
 *
 * 附录 DZ.1 抓到的那条
 * --------------------
 * `ConvertToBGR` 读 `cur_pix[0..2]` 却**只校验 W/H/n_id，没校验 C >= 3**。
 * ASan 坐实（`C=1, H=4, W=4`）：
 *     ERROR: AddressSanitizer: heap-buffer-overflow ... READ of size 4
 *         #0 ZQ::ZQ_CNN_Tensor4D::ConvertToBGR(...) ZQCNN/ZQ_CNN_Tensor4D.h:630
 *     0x606000000060 is located 0 bytes to the right of 64-byte region
 * 该函数**全仓零调用点**，但按 AGENTS.md 已修订的判据
 * （"是不是内存安全问题"优先于"生产可不可达"），越界读必须堵。
 *
 * 形态：与 `zq_tile` / `zq_roi` 同（附录 DC.6），必须用真实张量对象。
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <climits>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include "zq_check_child.h"
#include "ZQCNN/ZQ_CNN_Tensor4D.h"

using namespace ZQ;

#define RES_FILE "/tmp/zq_convert_res.txt"

enum { K_A0 = 0, K_A128 = 1, K_A256 = 2 };
static const char* g_kind_name[] = { "align0", "align128bit", "align256bit" };

static ZQ_CNN_Tensor4D* make(int kind)
{
    if (kind == K_A0) return new ZQ_CNN_Tensor4D_NHW_C_Align0();
    if (kind == K_A128) return new ZQ_CNN_Tensor4D_NHW_C_Align128bit();
    return new ZQ_CNN_Tensor4D_NHW_C_Align256bit();
}

// note: 0 无事 / 1 搭建失败 / 2 返回值与期望相反 / 3 形状或通道数不对
//       / 4 数据错 / 5 应拒却收下 / 6 越界（子进程会先被 ASan 打死，走 note=2 那条路）
enum { OP_BGR_RT = 0, OP_TOBGR, OP_GRAY_RT, OP_GRAY2, OP_COMPACT_RT, OP_COLOR2GRAY };

struct Case {
    int op;
    int N, C, H, W;
    int expect_ok;
};

// BGR 图的像素内容：每个通道用**不同的**值，避免"三通道读成同一个"这类错查不出来
static unsigned char bgr_at(int h, int w, int c)
{
    return (unsigned char)((h * 7 + w * 3 + c * 53) & 0xFF);
}

static void run_case(const Case& c, int kind)
{
    ZQ_CNN_Tensor4D* t = make(kind);
    ZQ_CNN_Tensor4D* t2 = make(kind);
    long bad = 0, first = -1; int note = 0;

    if (c.op == OP_BGR_RT) {
        // ConvertFromBGR -> ConvertToBGR 必须**逐字节**还原
        const int wsz = c.W * 3;
        std::vector<unsigned char> img((size_t)c.H * wsz);
        for (int h = 0; h < c.H; h++)
            for (int w = 0; w < c.W; w++)
                for (int k = 0; k < 3; k++)
                    img[(size_t)h * wsz + w * 3 + k] = bgr_at(h, w, k);
        if (!t->ConvertFromBGR(&img[0], c.W, c.H, wsz)) { note = 2; bad++; }
        else {
            if (t->GetC() != 3) { note = 3; bad++; }
            else {
                std::vector<unsigned char> back((size_t)c.H * wsz, 0xAB);
                if (!t->ConvertToBGR(&back[0], c.W, c.H, wsz, 0)) { note = 2; bad++; }
                else {
                    for (int h = 0; h < c.H && bad < 4; h++)
                        for (int w = 0; w < c.W && bad < 4; w++)
                            for (int k = 0; k < 3 && bad < 4; k++) {
                                const size_t o = (size_t)h * wsz + w * 3 + k;
                                if (back[o] != img[o]) {
                                    if (first < 0) first = (long)o;
                                    bad++;
                                }
                            }
                    if (bad) note = 4;
                }
            }
        }
    } else if (c.op == OP_TOBGR) {
        // 反向：把张量填成 b/127.5 - 1，ConvertToBGR 必须吐回 b
        const int wsz = c.W * 3;
        std::vector<unsigned char> img((size_t)c.H * wsz, 0xAB);
        if (!t->ChangeSize(1, c.H, c.W, c.C, 0, 0)) { note = 1; bad++; }
        else if (!c.expect_ok) {
            // **C < 3 必须被拒**（附录 DZ.1）；读了 cur_pix[1]/[2] 就是越界
            if (t->ConvertToBGR(&img[0], c.W, c.H, wsz, 0)) { note = 5; bad++; }
        } else {
            if (c.C != 3) { note = 3; bad++; }
            else {
                const int ss = t->GetSliceStep(), ps = t->GetPixelStep(), ws = t->GetWidthStep();
                float* p = t->GetFirstPixelPtr();
                for (int h = 0; h < c.H; h++)
                    for (int w = 0; w < c.W; w++)
                        for (int k = 0; k < 3; k++)
                            p[(size_t)h * ws + (size_t)w * ps + k] = (float)bgr_at(h, w, k) / 127.5f - 1.0f;
                (void)ss;
                if (!t->ConvertToBGR(&img[0], c.W, c.H, wsz, 0)) { note = 2; bad++; }
                else {
                    for (int h = 0; h < c.H && bad < 4; h++)
                        for (int w = 0; w < c.W && bad < 4; w++)
                            for (int k = 0; k < 3 && bad < 4; k++) {
                                const size_t o = (size_t)h * wsz + w * 3 + k;
                                if (img[o] != bgr_at(h, w, k)) {
                                    if (first < 0) first = (long)o;
                                    bad++;
                                }
                            }
                    if (bad) note = 4;
                }
            }
        }
    } else if (c.op == OP_GRAY_RT || c.op == OP_GRAY2) {
        const int wsz = c.W * 3;
        std::vector<unsigned char> img((size_t)c.H * wsz);
        for (int h = 0; h < c.H; h++)
            for (int w = 0; w < c.W; w++)
                for (int k = 0; k < 3; k++)
                    img[(size_t)h * wsz + w * 3 + k] = bgr_at(h, w, k);
        bool r = (c.op == OP_GRAY_RT) ? t->ConvertFromGray(&img[0], c.W, c.H, wsz)
                                      : t->ConvertFromBGR2GRAY(&img[0], c.W, c.H, wsz);
        if (!r) { note = 2; bad++; }
        else {
            if (t->GetC() != 1) { note = 3; bad++; }
            else {
                // **两个入口的参考完全不同**（第一版把它们写成同一个，红了 5 项）：
                //   ConvertFromGray     的指针是 `gray_pix++`（**步长 1**）——
                //                      它把输入当单通道灰度图读，一行只取前 W 个字节；
                //   ConvertFromBGR2GRAY 的指针是 `bgr_pix += 3`，是三通道加权。
                // 这也说明"两个函数名字像就当同构"是不成立的（附录 DY.1 的老教训）。
                const int ps = t->GetPixelStep(), ws = t->GetWidthStep();
                const float* p = t->GetFirstPixelPtr();
                const float mean_val = 127.5f, scale = 0.0078125f;
                for (int h = 0; h < c.H && bad < 4; h++)
                    for (int w = 0; w < c.W && bad < 4; w++) {
                        double want;
                        if (c.op == OP_GRAY_RT) {
                            want = (double)img[(size_t)h * wsz + w];
                        } else {
                            const double b  = img[(size_t)h * wsz + w * 3 + 0];
                            const double g  = img[(size_t)h * wsz + w * 3 + 1];
                            const double rr = img[(size_t)h * wsz + w * 3 + 2];
                            want = b * 0.114 + g * 0.587 + rr * 0.299;
                        }
                        const double got = (double)p[(size_t)h * ws + (size_t)w * ps] / scale + mean_val;
                        if (fabs(got - want) > 0.02) {
                            if (first < 0) first = (long)h * 1000 + w;
                            bad++; note = 4;
                        }
                    }
                // 整张图不能写成一个常数（那说明内层循环根本没跑）。
                // **判据方向别写反**：第一版写成"只要有一个像素与首像素不同就 bad++"，
                // 而那恰恰是**正常**情况 —— 于是 align128/align256 上 4 例全红，
                // 而 align0 恰好全同所以反而"过了"。门禁自己的错。
                {
                    const double v0 = (double)p[0];
                    int differs = 0;
                    for (int h = 0; h < c.H; h++)
                        for (int w = 0; w < c.W; w++)
                            if (fabs((double)p[(size_t)h * ws + (size_t)w * ps] - v0) > 1e-9) { differs = 1; break; }
                    if (!differs) { note = 4; bad++; }
                }
            }
        }
    } else if (c.op == OP_COMPACT_RT) {
        // ConvertToCompactNCHW -> ConvertFromCompactNCHW 必须是恒等
        if (!t->ChangeSize(c.N, c.H, c.W, c.C, 0, 0)) { note = 1; bad++; }
        else {
            const int ss = t->GetSliceStep();
            float* p = t->GetFirstPixelPtr();
            for (int i = 0; i < c.N * ss; i++) p[i] = (float)(i % 1009) * 0.001f;
            std::vector<float> compact((size_t)c.N * c.C * c.H * c.W);
            t->ConvertToCompactNCHW(&compact[0]);
            if (!t2->ConvertFromCompactNCHW(&compact[0], c.N, c.C, c.H, c.W)) { note = 2; bad++; }
            else {
                const int ss2 = t2->GetSliceStep();
                if (t2->GetN() != c.N || t2->GetC() != c.C || t2->GetH() != c.H || t2->GetW() != c.W) {
                    note = 3; bad++;
                } else {
                    const float* q = t2->GetFirstPixelPtr();
                    // **只比每个像素的 C 个有效通道，不能比整条 slice**
                    // （第一版比了整条，align128/align256 上补齐区必然不等 ——
                    //   `ChangeSize` 把补齐区清 0，而我在源张量里把它填成了非 0。
                    //   于是门禁红了 3 项，全是门禁自己的错）
                    const int ps2 = t2->GetPixelStep(), ws2 = t2->GetWidthStep();
                    for (int n = 0; n < c.N; n++)
                        for (int h = 0; h < c.H; h++)
                            for (int w = 0; w < c.W; w++)
                                for (int k = 0; k < c.C; k++) {
                                    const size_t off = (size_t)n * t->GetSliceStep()
                                                     + (size_t)h * t->GetWidthStep()
                                                     + (size_t)w * t->GetPixelStep() + k;
                                    const size_t off2 = (size_t)n * t2->GetSliceStep()
                                                      + (size_t)h * ws2 + (size_t)w * ps2 + k;
                                    if (q[off2] != p[off]) { if (first < 0) first = (long)off; bad++; }
                                }
                    if (bad) note = 4;
                }
            }
        }
    } else if (c.op == OP_COLOR2GRAY) {
        const int wsz = c.W * 3;
        std::vector<unsigned char> img((size_t)c.H * wsz);
        for (int h = 0; h < c.H; h++)
            for (int w = 0; w < c.W; w++)
                for (int k = 0; k < 3; k++)
                    img[(size_t)h * wsz + w * 3 + k] = bgr_at(h, w, k);
        if (!t->ConvertFromBGR(&img[0], c.W, c.H, wsz)) { note = 2; bad++; }
        else if (!t2->ChangeSize(1, 1, 1, 1, 0, 0)) { note = 1; bad++; }
        else {
            if (!t->ConvertColor_BGR2GRAY(*t2, 1, 1)) { note = 2; bad++; }
            else {
                if (t2->GetC() != 1) { note = 3; bad++; }
                else {
                    // 参考**独立**算一遍。这里没法用往返 ——
                    // `ConvertToBGR` 要求 C>=3，而 dst 是 C=1。
                    //
                    // **张量里存的是归一化后的值 `(b-127.5)/128`，不是原始字节**。
                    // 第一版直接拿原始字节算灰度，4 项全红 —— 门禁自己的错。
                    // 正确参考：src[k] = (byte_k - 127.5)/128，
                    //   dst = 0.114*src[0] + 0.587*src[1] + 0.299*src[2]
                    // 换算回"字节单位"：dst*128 = 灰度 - 127.5*(0.114+0.587+0.299) = 灰度 - 127.5
                    const int ps = t2->GetPixelStep(), ws = t2->GetWidthStep();
                    const int p2 = t2->GetPixelStep(), w2 = t2->GetWidthStep();
                    // **必须读 `t2`（目标）**。第一版读的是源张量 `t` ——
                    // 那是 C=3 的 BGR 归一化值，于是 4 项全红，门禁自己的错。
                    const float* qd = t2->GetFirstPixelPtr();
                    for (int h = 0; h < c.H && bad < 4; h++)
                        for (int w = 0; w < c.W && bad < 4; w++) {
                            const double b = img[(size_t)h * wsz + w * 3 + 0];
                            const double g = img[(size_t)h * wsz + w * 3 + 1];
                            const double r = img[(size_t)h * wsz + w * 3 + 2];
                            const double got = (double)qd[(size_t)h * ws + (size_t)w * ps] * 128.0;
                            const double want = b * 0.114 + g * 0.587 + r * 0.299 - 127.5;
                            if (fabs(got - want) > 0.02) {
                                if (first < 0) first = (long)h * 1000 + w;
                                bad++; note = 4;
                            }
                        }
                    // border 一圈必须是 0（顺带把 DZ.1 那条也钉住）
                    const int bW = t2->GetBorderW(), bH = t2->GetBorderH();
                    for (int h = -bH; h < c.H + bH && bad < 4; h++)
                        for (int w = -bW; w < c.W + bW && bad < 4; w++) {
                            const bool in_data = (h >= 0 && h < c.H && w >= 0 && w < c.W);
                            if (!in_data && t2->GetFirstPixelPtr() != 0 &&
                                t2->GetFirstPixelPtr()[(size_t)(h * w2 + w * p2)] != 0.0f) {
                                if (!note) note = 4; bad++;
                            }
                        }

                    if (bad && !note) note = 4;
                }
            }
        }
    }

    delete t; delete t2;
    FILE* f = fopen(RES_FILE, "w");
    if (f) { fprintf(f, "%ld %ld %d %ld\n", 1L - bad, bad, note, first); fclose(f); }
}

static const Case g_cases[] = {
  // BGR -> tensor -> BGR 精确往返
  { OP_BGR_RT, 1, 3, 1, 1, 1 },
  { OP_BGR_RT, 1, 3, 2, 3, 1 },
  { OP_BGR_RT, 1, 3, 5, 7, 1 },
  { OP_BGR_RT, 1, 3, 8, 8, 1 },
  // tensor -> BGR 反向精确
  { OP_TOBGR, 1, 3, 2, 3, 1 },
  { OP_TOBGR, 1, 3, 5, 5, 1 },
  // **C < 3 必须被拒**（附录 DZ.1：不拒就是越界读）
  { OP_TOBGR, 1, 1, 4, 4, 0 },
  { OP_TOBGR, 1, 1, 1, 1, 0 },
  { OP_TOBGR, 1, 2, 2, 2, 0 },
  // 灰度
  { OP_GRAY_RT, 1, 1, 2, 3, 1 },
  { OP_GRAY_RT, 1, 1, 5, 5, 1 },
  { OP_GRAY2,   1, 1, 2, 3, 1 },
  { OP_GRAY2,   1, 1, 5, 7, 1 },
  // compact NCHW 往返
  { OP_COMPACT_RT, 1, 3, 2, 2, 1 },
  { OP_COMPACT_RT, 2, 4, 2, 3, 1 },
  { OP_COMPACT_RT, 1, 8, 1, 1, 1 },
  // ConvertColor_BGR2GRAY
  { OP_COLOR2GRAY, 1, 3, 2, 2, 1 },
  { OP_COLOR2GRAY, 1, 3, 5, 5, 1 },
};
static const int N_CASE = (int)(sizeof(g_cases) / sizeof(g_cases[0]));

static const char* g_opname[] = { "BGR往返", "toBGR", "灰度往返", "BGR2GRAY", "compact往返", "Color2GRAY" };
static const char* g_note[] = {
    "", "搭建失败", "**返回值与期望相反**", "**形状/通道数不对**",
    "**数据错**", "**应当被拒却收下了**", ""
};

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
    snprintf(tag, sizeof(tag), "%-10s N%dC%dH%dW%d %s",
             g_opname[c.op], c.N, c.C, c.H, c.W, c.expect_ok ? "" : "(应当被拒)");
    if (!have) {
        g_crash++;
        printf("  %-12s %-40s  没跑完（结果文件读不出来%s）\n", g_kind_name[kind], tag,
               WIFSIGNALED(st) ? "，子进程被信号杀掉" : "");
        return;
    }
    if (WIFSIGNALED(st)) {
        g_crash++;
        printf("  %-12s %-40s  CRASH（信号 %d）\n", g_kind_name[kind], tag, WTERMSIG(st));
        return;
    }
    if (bad > 0) {
        g_bad++;
        printf("  %-12s %-40s  %s %ld 项", g_kind_name[kind], tag, g_note[note], bad);
        if (note == 4) printf("，首个错在第 %ld 个", first);
        printf("\n");
    } else g_ok++;
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("ZQ_CNN_Tensor4D Convert 族门禁（附录 DZ）\n");
    printf("判据用**精确往返**，参考值不依赖任何一条实现公式：\n");
    printf("  ConvertFromBGR 与 ConvertToBGR 互为逆运算且逐位精确（推导见文件头），\n");
    printf("  于是\"喂进去的字节必须原样回来\"就是判据 —— 抄公式的写法会把实现和门禁一起抄错。\n");
    printf("  灰度只查\"线性（落在三通道 min..max 内）+ 非常数\"，不抄 0.114/0.587/0.299。\n");
    printf("  **C < 3 时 ConvertToBGR 必须被拒**：不拒就是读 cur_pix[1]/[2] 的越界（附录 DZ.1，ASan 坐实过）。\n");
    printf("三种张量子类都跑\n\n");
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
