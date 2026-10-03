/* `ZQ_CNN_Tensor4D::ROI` 的边界检查 + **border 几何**门禁 —— 附录 DX / DY.9
 *
 * 本文件有两段历史，第二段是 2026-10-03 补的
 * -------------------------------------------
 *
 * 【第一段：附录 DX】边界检查
 *
 * `ROI` 的拷贝与 border memset 我逐项核过，**都是对的**；
 * 有问题的是开头那一行边界检查：
 *
 *     if (off_x < 0 || off_y < 0 || off_x + width > W || off_y + height > H)
 *         return false;
 *
 * `off_x + width` 是 **int 加法**。`off_x` 来自 MTCNN 的 P-net 检测框输出
 * （`ZQ_CNN_MTCNN.h:615 / 621` 等处），**是数据/模型可控的**。
 * off_x 足够大时加法**回绕成负数** ⇒ `> W` 不成立 ⇒ **边界检查被整条绕过**，
 * 紧接着 `src_slice_ptr = GetFirstPixelPtr() + off_y*widthStep + off_x*pixelStep`
 * 就是一次**越界读**。
 *
 * UBSan 坐实（修之前）：
 *     ZQ_CNN_Tensor4D.h:74:40: runtime error: signed integer overflow:
 *         2147483645 + 8 cannot be represented in type 'int'
 *
 * 【第二段：附录 DY.9】非对称 border ⇒ 堆越界**写**
 *
 * 修完 DX 之后，DY.7 顺着查 borderW/borderH 的形参约定，发现
 * `ROI` / `ResizeBilinearRect` / `ConvertColor_BGR2GRAY` 内部都这么调：
 *
 *     dst.ChangeSize(N, height, width, C, dst_borderH, dst_borderW)   // ← W/H 传反
 *
 * 而 `ChangeSize` 的形参是 `(int N, int H, int W, int C, int borderW, int borderH)`，
 * **第 5 个是 borderW**。于是张量按**转置后**的 border 分配，
 * 紧接着的 border 清零却用**未转置的形参名**（`dstPixelStep*dst_borderW` 表水平、
 * `dstWidthStep*dst_borderH` 表垂直）—— **两边对不上，越界写**。
 *
 * ASan 坐实（修之前，`dst_borderH=3, dst_borderW=1`）：
 *     ==367469==ERROR: AddressSanitizer: heap-buffer-overflow ... WRITE of size 360
 *         #2 ZQ::ZQ_CNN_Tensor4D::ROI(...) ZQCNN/ZQ_CNN_Tensor4D.h:129
 *     0x617000000a80 is located 0 bytes to the right of 768-byte region
 *
 * **溢出条件是 `dst_borderH > dst_borderW`**（三处都一样）。
 * 全仓 25 处 ROI 调用点全传 `(0,0)`，所以 shipped 模型打不到 ——
 * 但这是公开 API，代价只有一个 token，**没有理由留着**。
 *
 * 判据（三条，缺一不可）
 * --------------------
 * 1. 该拒的必须被拒（DX 那一段，判据不变）
 * 2. 该成的必须成，且 **`dst.GetBorderW()` / `GetBorderH()` 必须等于调用方
 *    放进对应形参槽位的那个值** —— 这一条把"名字与行为一致"钉死，
 *    正是 DY.9 修的东西；没有它，"把两个参数交换回去"这种改法照样能通过
 * 3. **整个 dst 缓冲逐格核对**：数据区必须等于源 ROI 对应位置，
 *    **border 一圈必须是 0** —— 这一条专抓"清零范围算错"，
 *    而 memset 的上下界正是这类缺陷的高发处（附录 BF / DD.9 都栽在这里）
 *
 * 关于 stderr
 * ----------
 * 第一版这里写的是 `freopen("/dev/null", "w", stderr)`，**没用共享的
 * `zq_child_silence_stderr()`**，于是这道门禁的 sanitizer 报告被整个吞掉 ——
 * 与附录 DY.5（报告被后一个用例擦掉）同源。已改用共享头。
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include "zq_check_child.h"
#include "ZQCNN/ZQ_CNN_Tensor4D.h"

using namespace ZQ;

#define RES_FILE "/tmp/zq_roi_res.txt"

struct Case {
    const char* what;
    int srcH, srcW, srcC;
    int off_x, off_y, width, height;
    int borderH, borderW;   // **按 ROI 的形参名**：第 6 个是 borderH，第 7 个是 borderW
    int expect_ok;          // 1 = 应当成功；0 = 应当被拒
    int chk_geom;           // 1 = 成功后要核对 GetBorderW/H 与整块缓冲
};

static const Case g_cases[] = {
  // ---- 正常值：整图 / 子块 / 贴边 / 单像素 ----
  { "整图（偏移 0，尺寸 = 原图）",        8, 8, 3, 0, 0, 8, 8, 0, 0, 1, 1 },
  { "子块（偏移 2,2）",                    8, 8, 3, 2, 2, 3, 3, 0, 0, 1, 1 },
  { "贴边（偏移 5，宽 3，恰好到右边）",    8, 8, 3, 5, 0, 3, 8, 0, 0, 1, 1 },
  { "单像素",                              8, 8, 3, 7, 7, 1, 1, 0, 0, 1, 1 },
  // ---- 对称 border：全仓唯一在用的形态，**必须与修复前逐字节一致** ----
  { "对称 border 1,1",                     8, 8, 3, 1, 1, 4, 4, 1, 1, 1, 1 },
  { "对称 border 2,2",                     6, 6, 3, 0, 0, 6, 6, 2, 2, 1, 1 },
  // ---- **非对称 border** = 附录 DY.9 的触发条件，两个方向都要 ----
  { "**非对称 border** H=3 W=1（修复前越界）", 8, 8, 3, 1, 1, 4, 4, 3, 1, 1, 1 },
  { "**非对称 border** H=1 W=3",           8, 8, 3, 1, 1, 4, 4, 1, 3, 1, 1 },
  { "**非对称 border** H=4 W=1",           5, 5, 3, 0, 0, 5, 5, 4, 1, 1, 1 },
  { "**非对称 border** H=1 W=4",           5, 5, 3, 0, 0, 5, 5, 1, 4, 1, 1 },
  { "**非对称 border** C=8（对齐子类用）",  4, 4, 8, 1, 1, 2, 2, 3, 1, 1, 1 },
  // ---- 应当被拒的（判据同 DX）----
  { "正常越界（偏移 4 + 宽 8 > 8）",       8, 8, 3, 4, 0, 8, 8, 0, 0, 0, 0 },
  { "偏移越界（off_x = W）",               8, 8, 3, 8, 0, 1, 1, 0, 0, 0, 0 },
  { "负 width（原来会漏过检查）",          8, 8, 3, 0, 0, -1, 4, 0, 0, 0, 0 },
  { "负 height",                           8, 8, 3, 0, 0, 4, -1, 0, 0, 0, 0 },
  // **DX 修的那三个**：int 加法回绕
  { "**int 加法回绕** off_x = INT_MAX-2",  8, 8, 3, INT_MAX - 2, 0, 8, 8, 0, 0, 0, 0 },
  { "**int 加法回绕** off_y = INT_MAX-2",  8, 8, 3, 0, INT_MAX - 2, 8, 8, 0, 0, 0, 0 },
  { "**int 加法回绕** off_x = INT_MAX, width = INT_MAX", 8, 8, 3, INT_MAX, 0, INT_MAX, 8, 0, 0, 0, 0 },
  // 非对称 border **不能**让边界检查被绕过
  { "越界 + 非对称 border 仍须被拒",        8, 8, 3, 4, 0, 8, 8, 3, 1, 0, 0 },
};
static const int N_CASE = (int)(sizeof(g_cases) / sizeof(g_cases[0]));

// 源张量每格填 (线性下标 % 97) * 0.01f —— 纯线性、无空间结构，
// 这样 dst 的每一格都能反查它该等于源张量的哪一格。
static float src_val(int flat)
{
    return (float)(flat % 97) * 0.01f;
}

static void run_one(const Case& c)
{
    ZQ_CNN_Tensor4D_NHW_C_Align0* src = new ZQ_CNN_Tensor4D_NHW_C_Align0();
    ZQ_CNN_Tensor4D_NHW_C_Align0* dst = new ZQ_CNN_Tensor4D_NHW_C_Align0();
    long bad = 0;
    const char* note = "";
    if (!src->ChangeSize(1, c.srcH, c.srcW, c.srcC, 0, 0)) { bad++; note = "src 分配失败"; }
    else if (!dst->ChangeSize(1, 1, 1, 1, 0, 0)) { bad++; note = "dst 分配失败"; }
    else {
        const int ss = src->GetSliceStep();
        float* sp = src->GetFirstPixelPtr();
        for (int i = 0; i < ss; i++) sp[i] = src_val(i);

        const bool r = src->ROI(*dst, c.off_x, c.off_y, c.width, c.height, c.borderH, c.borderW);
        if (r != (c.expect_ok != 0)) { bad++; note = c.expect_ok ? "应当成功却被拒" : "应当被拒却收下"; }
        else if (r && c.chk_geom) {
            // 判据 2：形参名与行为必须一致
            if (dst->GetBorderW() != c.borderW || dst->GetBorderH() != c.borderH) {
                bad++;
                note = "GetBorderW/H 与传入的形参对不上";
            } else {
                // 判据 3：整块缓冲逐格核对（数据区 = 源 ROI，border = 0）
                const int ps = dst->GetPixelStep(), ws = dst->GetWidthStep();
                const int sps = src->GetPixelStep(), sws = src->GetWidthStep();
                const int bW = dst->GetBorderW(), bH = dst->GetBorderH();
                const int dW = dst->GetW(), dH = dst->GetH();
                const float* dp = dst->GetFirstPixelPtr();
                long wrong = 0;
                for (int h = -bH; h < dH + bH && wrong < 4; h++) {
                    for (int w = -bW; w < dW + bW && wrong < 4; w++) {
                        const bool in_data = (h >= 0 && h < dH && w >= 0 && w < dW);
                        // **先有符号算完再转 size_t**。写成 (size_t)h * ws 时，
                        // h 为负会回绕成 ~1.8e19，索引直接飞到天外 ——
                        // 第一版就这么写的，19 个用例**全部**报 ASan 越界，
                        // 症状与"被测代码坏了"一模一样（附录 CU.8.1 / CX.5 的老坑）。
                        const float got = dp[(size_t)(h * ws + w * ps)];
                        if (in_data) {
                            // 只比第 0 号通道：它足以判定"这一格搬对了没有"
                            const float w0 = src_val((h + c.off_y) * sws + (w + c.off_x) * sps);
                            if (got != w0) { wrong++; note = "数据区错"; }
                        } else {
                            if (got != 0.0f) { wrong++; note = "border 没清零"; }
                        }
                    }
                }
                if (wrong) bad += wrong;
            }
        }
    }
    delete src; delete dst;
    FILE* f = fopen(RES_FILE, "w");
    // **note 为空时必须写 "-"，不能写空串**：父进程用 `%95[^\n]` 读它，
    // 空串匹配不上、fscanf 返回 3 而不是 4，于是每个用例都被判成"没跑完"——
    // 19 个用例全红、还被打上"sanitizer 报错"的标签，而实际上**一条 sanitizer 报告都没有**。
    // 症状与"被测代码全坏了"一模一样（附录 CU.8.1 / CX.5 的老坑第三次）。
    if (f) { fprintf(f, "%ld %ld %.6e %s\n", 1L - bad, bad, 0.0, note[0] ? note : "-"); fclose(f); }
}

static int g_case = 0, g_ok = 0, g_bad = 0, g_crash = 0;

static void one(const Case& c)
{
    char tmp[32];
    g_case++;
    remove(RES_FILE);
    pid_t pid = fork();
    if (pid == 0) {
        zq_child_silence_stderr();      // **共享头**（附录 CZ/DY.5），不再自己 freopen /dev/null
        run_one(c);
        _exit(0);
    }
    int st = 0; waitpid(pid, &st, 0);
    long ok = 0, bad = 0, over = 0; double worst = 0; int have = 0;
    char note[96]; note[0] = 0;
    FILE* f = fopen(RES_FILE, "r");
    if (f) {
        have = (fscanf(f, "%ld %ld %lf %95[^\n]", &ok, &bad, &worst, note) == 4);
        fclose(f);
    }
    if (!have || WIFSIGNALED(st)) {
        g_crash++;
        // **不要在没证据的时候说"sanitizer 报错"**：结果文件缺失的原因可能是
        // 子进程自己没写（解析不匹配、提前 return、fopen 失败……），
        // 报成 sanitizer 会把排查方向直接带偏 —— 参见本文件里踩过的三次同类坑。
        printf("  %-44s  %s%s\n", c.what,
               WIFSIGNALED(st) ? "CRASH（信号 " : "没跑完（结果文件读不出来",
               WIFSIGNALED(st) ? (snprintf(tmp, sizeof(tmp), "%d）", WTERMSIG(st)), tmp)
                               : "，非信号终止）");
        return;
    }
    if (bad > 0) {
        g_bad++;
        printf("  %-44s  FAIL %ld 项  %s\n", c.what, bad, note);
    } else g_ok++;
}

// ---------------------------------------------------------------------
// 第二段：DY.9 的另外两个同源入口
// ---------------------------------------------------------------------
// 修复一共动了 51 处、覆盖三类方法（ROI / ResizeBilinearRect / ConvertColor_BGR2GRAY），
// 但 `zq_roi` 只测得到第一类 —— **另外两类修了却没有回归保护，等于下次还会坏**。
// 按本项目立下的规矩「先补覆盖，再谈改不改」，这里把另外两类也钉上。
//
// 判据同上：返回 true + GetBorderW/H 等于**同名形参** + (ConvertColor 额外) border 一圈是 0。
// 越界写由 ASan 兜底（子进程会直接 abort，判据 CJ.4 生效）。
#define RES_FILE2 "/tmp/zq_roi_res2.txt"

struct BCase {
    const char* what;
    int op;          // 0 = ConvertColor_BGR2GRAY，1 = ResizeBilinearRect
    int srcH, srcW, srcC;
    int bW, bH;      // **形参名**：ConvertColor 是 (dst_borderW, dst_borderH)
};

static const BCase g_bcases[] = {
  { "ConvertColor  对称 1,1",       0, 8, 8, 3, 1, 1 },
  { "ConvertColor  **非对称** W=1 H=3", 0, 8, 8, 3, 1, 3 },
  { "ConvertColor  **非对称** W=3 H=1", 0, 8, 8, 3, 3, 1 },
  { "ResizeRect    对称 1,1",       1, 8, 8, 3, 1, 1 },
  { "ResizeRect    **非对称** W=1 H=3", 1, 8, 8, 3, 1, 3 },
  { "ResizeRect    **非对称** W=3 H=1", 1, 8, 8, 3, 3, 1 },
  { "ResizeRect    **非对称** W=1 H=4（整图）", 1, 6, 6, 3, 1, 4 },
  // **标量重载**（int src_off_x/y, src_rect_w/h）：
  // 上面三组调的是 **vector 重载**（cpp:369/376），标量重载（cpp:262/269）是**另一处 ChangeSize 点**。
  // 第一次变异测试打在 262 行时门禁**没红**——那个"没红"代表的是**这个点没覆盖**、
  // 不代表"与它无关"。这两者必须区分开（附录 DA.2 立的规矩）。
  { "ResizeRect(标量)  对称 1,1", 2, 8, 8, 3, 1, 1 },
  { "ResizeRect(标量)  **非对称** W=1 H=3", 2, 8, 8, 3, 1, 3 },
  { "ResizeRect(标量)  **非对称** W=3 H=1", 2, 8, 8, 3, 3, 1 },
};
static const int N_BCASE = (int)(sizeof(g_bcases) / sizeof(g_bcases[0]));

static void run_bcase(const BCase& c)
{
    ZQ_CNN_Tensor4D_NHW_C_Align0* src = new ZQ_CNN_Tensor4D_NHW_C_Align0();
    ZQ_CNN_Tensor4D_NHW_C_Align0* dst = new ZQ_CNN_Tensor4D_NHW_C_Align0();
    long bad = 0;
    const char* note = "";
    if (!src->ChangeSize(1, c.srcH, c.srcW, c.srcC, 0, 0)) { bad++; note = "src 分配失败"; }
    else {
        const int ss = src->GetSliceStep();
        float* sp = src->GetFirstPixelPtr();
        for (int i = 0; i < ss; i++) sp[i] = (float)(i % 97) * 0.01f;
        bool r;
        if (c.op == 0) {
            r = src->ConvertColor_BGR2GRAY(*dst, c.bW, c.bH);
        } else if (c.op == 2) {
            r = src->ResizeBilinearRect(*dst, 4, 4, c.bW, c.bH, 0, 0, c.srcW, c.srcH,
                                        ZQ_CNN_Tensor4D::SAMPLE_ALIGN_CENTER);
        } else {
            std::vector<int> ox, oy, rw, rh;
            ox.push_back(0); oy.push_back(0); rw.push_back(c.srcW); rh.push_back(c.srcH);
            r = src->ResizeBilinearRect(*dst, 4, 4, c.bW, c.bH, ox, oy, rw, rh,
                                        ZQ_CNN_Tensor4D::SAMPLE_ALIGN_CENTER);
        }
        if (!r) { bad++; note = "应当成功却被拒"; }
        else if (dst->GetBorderW() != c.bW || dst->GetBorderH() != c.bH) {
            bad++; note = "GetBorderW/H 与传入的形参对不上";
        } else if (c.op == 0) {
            // ConvertColor 的 border 一圈必须是 0
            const int ps = dst->GetPixelStep(), ws = dst->GetWidthStep();
            const int bW = dst->GetBorderW(), bH = dst->GetBorderH();
            const int dW = dst->GetW(), dH = dst->GetH();
            const float* dp = dst->GetFirstPixelPtr();
            long wrong = 0;
            for (int h = -bH; h < dH + bH && wrong < 4; h++)
                for (int w = -bW; w < dW + bW && wrong < 4; w++) {
                    const bool in_data = (h >= 0 && h < dH && w >= 0 && w < dW);
                    if (!in_data && dp[(size_t)(h * ws + w * ps)] != 0.0f) { wrong++; note = "border 没清零"; }
                }
            if (wrong) bad += wrong;
        }
    }
    delete src; delete dst;
    FILE* f = fopen(RES_FILE2, "w");
    if (f) { fprintf(f, "%ld %ld %s\n", 1L - bad, bad, note[0] ? note : "-"); fclose(f); }
}

static void one_b(const BCase& c)
{
    g_case++;
    remove(RES_FILE2);
    pid_t pid = fork();
    if (pid == 0) { zq_child_silence_stderr(); run_bcase(c); _exit(0); }
    int st = 0; waitpid(pid, &st, 0);
    long ok = 0, bad = 0; int have = 0;
    char note[96]; note[0] = 0;
    FILE* f = fopen(RES_FILE2, "r");
    if (f) { have = (fscanf(f, "%ld %ld %95[^\n]", &ok, &bad, note) == 3); fclose(f); }
    if (!have || WIFSIGNALED(st)) {
        g_crash++;
        printf("  %-44s  没跑完%s\n", c.what, WIFSIGNALED(st) ? "（信号终止）" : "（结果文件读不出来）");
        return;
    }
    if (bad > 0) { g_bad++; printf("  %-44s  FAIL %ld 项  %s\n", c.what, bad, note); }
    else g_ok++;
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("ZQ_CNN_Tensor4D border 几何门禁（附录 DX + DY.9）\n");
    printf("第一段 ROI：判据 1 该拒的必须被拒（`off_x + width` 是 int 加法，off_x 来自 MTCNN 检测框、\n");
    printf("        数据可控，回绕成负数后 `> W` 不成立 ⇒ 边界检查被整条绕过 ⇒ 越界读。UBSan 坐实过。\n");
    printf("        判据 2 该成的必须成，且 GetBorderW()/GetBorderH() 必须等于**同名形参**。\n");
    printf("        判据 3 整块 dst 逐格核对：数据区 = 源 ROI，border 一圈必须是 0。\n");
    printf("第二段 ConvertColor_BGR2GRAY / ResizeBilinearRect：同样的几何判据。\n");
    printf("  这三处原来都这么调 ChangeSize(..., dst_borderH, dst_borderW) —— 第 5 形参却是 borderW，\n");
    printf("  张量按转置后的 border 分配、memset 按未转置的形参名清零 ⇒ **堆越界写**（H > W 时）。\n");
    printf("  ASan 坐实过：ROI 在 ZQ_CNN_Tensor4D.h:129、ConvertColor 在 :514、\n");
    printf("  ResizeBilinearRect 在 ZQ_CNN_Tensor4D.cpp:447。修复共 51 处调用点。\n");
    printf("对称 border 用例的结果必须与修复前**逐字节一致**（全仓 25 处 ROI 调用点都传对称值）。\n\n");
    for (int i = 0; i < N_CASE; i++) one(g_cases[i]);
    for (int i = 0; i < N_BCASE; i++) one_b(g_bcases[i]);
    printf("\n共 %d 个用例：全对 %d，有错 %d，崩溃/搭建失败 %d\n", g_case, g_ok, g_bad, g_crash);
    if (g_bad || g_crash)
        printf("**每一项在下结论之前都要先用独立复现对一遍**（附录 CA.3）。\n");
    return (g_bad || g_crash) ? 1 : 0;
}
