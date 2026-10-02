/* NCHWC resize 门禁 —— 附录 CL
 *
 * 覆盖面：6 个入口（nm 核实）
 *   zq_cnn_resize_with_safeborder_nchwc{1,4,8}
 *   zq_cnn_resize_without_safeborder_nchwc{1,4,8}
 *
 * 两者的**唯一**区别是边界处理（`zq_cnn_resize_nchwc_raw.h`）：
 *   without_safeborder : x0/x1 钳到 [0, in_W-1]，y0/y1 钳到 [0, in_H-1]（复制边界）
 *   with_safeborder    : **不钳**，直接按坐标读 —— 要求调用方保证安全边界
 *
 * 坐标与加权（从源码抄的）：
 *   w_step = 1/out_W * src_W;   h_step = 1/out_H * src_H
 *   coord_x_ini = 0.5*w_step - 0.5 + in_off_x
 *   coord_x(w)  = coord_x_ini + w*w_step
 *   x0 = floor(coord_x);  sx = coord_x - x0;  x1 = x0 + 1        （y 同理）
 *   v00 = in[y0][x0];  r0 = v00 + (in[y0][x1] - v00)*sx
 *   v10 = in[y1][x0];  r1 = v10 + (in[y1][x1] - v10)*sx
 *   out = r0 + (r1 - r0)*sy
 *
 * 两种配置（`with_safeborder` 在 B 配置下会越界读，**只给 without 跑**）：
 *   A  降采样、off 从 0 开始、rect 严格落在图内
 *      —— w_step >= 1 时 coord_x_ini >= 0、末尾 x1 仍在图内，**两个变体都安全**，
 *         期望结果相同（标准双线性）
 *   B  上采样、rect 顶到图的另一侧，末尾 x1 会超出 in_W
 *      —— 专门用来验 `without_safeborder` 的钳位
 *
 * 沿用 CB/CE/CF/CG/CH/CI/CJ/CK：名字写全走函数指针表、逐格统计、
 * 每用例 fork 子进程并**显式判"没读到结果文件"= 失败**（CJ.4）、
 * 用真实的 ZQ_CNN_Tensor4D_NCHWC{1,4,8}。
 */
#include "zq_check_child.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include "ZQCNN/ZQ_CNN_Tensor4D_NCHWC.h"
#include "ZQCNN/layers_nchwc/zq_cnn_resize_nchwc.h"

typedef void (*FN)(const float* in_data, int in_N, int in_H, int in_W, int in_C,
                   int in_ws, int in_ss, int in_is,
                   int in_off_x, int in_off_y, int in_rect_width, int in_rect_height,
                   float* out_data, int out_H, int out_W,
                   int out_ws, int out_ss, int out_is);

static const char* g_name[2] = { "without_safeborder", "with_safeborder" };

struct Entry { FN fn; int safe; int align; };

// 6 行平铺，**每个内核名写全**
static const Entry g_entries[6] = {
  { zq_cnn_resize_without_safeborder_nchwc1, 0, 1 },
  { zq_cnn_resize_without_safeborder_nchwc4, 0, 4 },
  { zq_cnn_resize_without_safeborder_nchwc8, 0, 8 },
  { zq_cnn_resize_with_safeborder_nchwc1,    1, 1 },
  { zq_cnn_resize_with_safeborder_nchwc4,    1, 4 },
  { zq_cnn_resize_with_safeborder_nchwc8,    1, 8 },
};
static const int N_ENTRY = 6;

#define RES_FILE "/tmp/zq_rz_res.txt"
static const double TOL = 1e-5;

static float val(int seed, int idx)
{
    unsigned int x = (unsigned int)((unsigned int)seed * 2654435761u + (unsigned int)idx * 40503u);
    x ^= x >> 13; x *= 1274126177u; x ^= x >> 16;
    return (float)((int)(x % 2001) - 1000) * 0.001f;
}

// 0 = 配置 A（降采样、两个变体都安全）；1 = 配置 B（只有 without 能跑）
struct Case { int entry, cfg, N, inH, inW, C, offX, offY, rectW, rectH, outH, outW; };

// 取输入像素，**不自己钳位** —— 钳位是内核做的事，必须由 x0/x1 的钳位来体现，
// 否则参考里出现两次钳位，变异掉其中一处是察觉不到的（第一版就踩了这个）。
static double in_at(const std::vector<float>& in, int C, int H, int W, int n, int c, int h, int w)
{
    if (h < 0 || h > H - 1 || w < 0 || w > W - 1) return -12345.0;   // 越界：给一个绝不会被算对的哨兵
    return in[(((size_t)n * C + c) * H + h) * W + w];
}

typedef void (*RUNNER)(const Case&);
template <class TEN>
static void run_one(const Case& c)
{
    const Entry& e = g_entries[c.entry];
    const int A = TEN().GetAlignSize();
    const int N = c.N, H = c.inH, W = c.inW, C = c.C;

    std::vector<float> in((size_t)N * C * H * W);
    for (size_t i = 0; i < in.size(); i++) in[i] = val(1, (int)i);
    TEN ti, to;
    if (!ti.ChangeSize(N, H, W, C, 0, 0)) return;
    if (!to.ChangeSize(N, c.outH, c.outW, C, 0, 0)) return;
    if (!ti.ConvertFromCompactNCHW(&in[0], N, C, H, W)) return;
    std::vector<float> zero((size_t)N * to.GetImageStep(), 0.0f);
    memcpy(to.GetFirstPixelPtr(), &zero[0], zero.size() * sizeof(float));

    e.fn(ti.GetFirstPixelPtr(), N, H, W, C,
         ti.GetWidthStep(), ti.GetSliceStep(), ti.GetImageStep(),
         c.offX, c.offY, c.rectW, c.rectH,
         to.GetFirstPixelPtr(), c.outH, c.outW,
         to.GetWidthStep(), to.GetSliceStep(), to.GetImageStep());

    // ---- 参考：标准双线性（with_safeborder 在配置 B 下不跑，所以这里一律用钳位版）----
    const float src_W = (float)c.rectW, src_H = (float)c.rectH;
    const float w_step = 1.0f / (float)c.outW * src_W;
    const float h_step = 1.0f / (float)c.outH * src_H;
    const float coord_x_ini = 0.5f * w_step - 0.5f + (float)c.offX;
    const float coord_y_ini = 0.5f * h_step - 0.5f + (float)c.offY;

    const int ows = to.GetWidthStep(), oss = to.GetSliceStep(), ois = to.GetImageStep();
    long n_ok = 0, n_bad = 0; double worst = 0.0;
    for (int n = 0; n < N; n++)
        for (int h = 0; h < c.outH; h++)
            for (int w = 0; w < c.outW; w++) {
                const float cx = coord_x_ini + w * w_step;
                const float cy = coord_y_ini + h * h_step;
                int x0 = (int)floorf(cx); float sx = cx - floorf(cx);
                int y0 = (int)floorf(cy); float sy = cy - floorf(cy);
                int x1 = x0 + 1, y1 = y0 + 1;
                if (!e.safe) {      // without_safeborder：坐标被钳到图内
                    if (x0 < 0) x0 = 0; if (x0 > W - 1) x0 = W - 1;
                    if (x1 < 0) x1 = 0; if (x1 > W - 1) x1 = W - 1;
                    if (y0 < 0) y0 = 0; if (y0 > H - 1) y0 = H - 1;
                    if (y1 < 0) y1 = 0; if (y1 > H - 1) y1 = H - 1;
                }
                for (int ch = 0; ch < C; ch++) {
                    double v00 = in_at(in, C, H, W, n, ch, y0, x0);
                    double v01 = in_at(in, C, H, W, n, ch, y0, x1);
                    double v10 = in_at(in, C, H, W, n, ch, y1, x0);
                    double v11 = in_at(in, C, H, W, n, ch, y1, x1);
                    double r0 = v00 + (v01 - v00) * sx;
                    double r1 = v10 + (v11 - v10) * sx;
                    double y = r0 + (r1 - r0) * sy;
                    double got = to.GetFirstPixelPtr()[n * ois + (ch / A) * oss + h * ows + w * A + (ch % A)];
                    double den = fabs(y) > 1.0 ? fabs(y) : 1.0;
                    double be = fabs(got - y) / den;
                    if (be > TOL) n_bad++; else n_ok++;
                    if (be > worst) worst = be;
                }
            }
    FILE* f = fopen(RES_FILE, "w");
    if (f) { fprintf(f, "%ld %ld %.6e\n", n_ok, n_bad, worst); fclose(f); }
}

static int g_case = 0, g_ok = 0, g_bad = 0, g_crash = 0;

static void one(const Case& c, RUNNER r)
{
    g_case++;
    remove(RES_FILE);
    pid_t pid = fork();
    if (pid == 0) {
        zq_child_silence_stderr();
        r(c);
        _exit(0);
    }
    int st = 0; waitpid(pid, &st, 0);
    long ok = 0, bad = 0; double worst = 0;
    int have = 0;
    FILE* f = fopen(RES_FILE, "r");
    if (f) { have = (fscanf(f, "%ld %ld %lf", &ok, &bad, &worst) == 3); fclose(f); }
    char nm[40], tag[96];
    snprintf(nm, sizeof(nm), "nchwc%d %s", g_entries[c.entry].align, g_name[g_entries[c.entry].safe]);
    snprintf(tag, sizeof(tag), "cfg%s in=%dx%d rect=%d,%d %dx%d -> out=%dx%d",
             c.cfg ? "B" : "A", c.inH, c.inW, c.offX, c.offY, c.rectW, c.rectH, c.outH, c.outW);
    if (!have) { g_crash++; printf("  %-28s %s  没跑完（子进程没写结果文件，退出码 %d）\n", nm, tag, WEXITSTATUS(st)); return; }
    if (WIFSIGNALED(st)) { g_crash++; printf("  %-28s %s  CRASH(信号 %d)\n", nm, tag, WTERMSIG(st)); return; }
    if (bad > 0) { g_bad++; printf("  %-28s %s  FAIL %ld/%ld 格错, 最差 %.3e\n", nm, tag, bad, ok + bad, worst); }
    else { g_ok++; }
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("NCHWC resize：6 个入口（with / without_safeborder × NCHWC1/4/8）\n");
    printf("内核名全部写全、走函数指针表（nm 核实）；用真实 ZQ_CNN_Tensor4D_NCHWC{1,4,8}\n");
    printf("判据：逐格后向误差。两种配置：\n");
    printf("  cfgA 降采样、off 从 0 起、rect 严格在图内 —— 两个变体都安全，期望相同\n");
    printf("  cfgB 上采样、rect 顶到另一侧，末尾 x1 会越界 —— 只给 without_safeborder 跑（验钳位）\n\n");

    RUNNER r[3] = { &run_one<ZQ::ZQ_CNN_Tensor4D_NCHWC1>,
                    &run_one<ZQ::ZQ_CNN_Tensor4D_NCHWC4>,
                    &run_one<ZQ::ZQ_CNN_Tensor4D_NCHWC8> };
    for (int e = 0; e < N_ENTRY; e++) {
        const int ai = (g_entries[e].align == 1) ? 0 : (g_entries[e].align == 4 ? 1 : 2);
        const int c0 = g_case, k0 = g_ok, b0 = g_bad, x0 = g_crash;
        const int A = g_entries[e].align;
        for (int j = 0; j < 2; j++) {
            Case c; memset(&c, 0, sizeof(c));
            c.entry = e; c.N = 1; c.inH = 16; c.inW = 16; c.C = A;
            if (j == 0) {   // cfgA：降采样，rect 严格在图内
                c.cfg = 0; c.offX = 2; c.offY = 3; c.rectW = 8; c.rectH = 6; c.outH = 6; c.outW = 8;
            } else {
                // cfgB：让**钳位真正影响结果**。要点有两个：
                //   ① w_step 必须是**非整数**，否则 sx 恒为 0，而 x1 是被 sx 加权的 ——
                //      钳不钳 x1 都一样（第一版用 out==in、w_step==1 就是这么白测的）
                //   ② 坐标要真的走出图外
                // 取 W=H=16、rect=20（比图大）、out=16：w_step=1.25，coord_x(15)=18.875
                // -> x0=18 > 15，without_safeborder 会把它钳到 15，且 sx=0.875 != 0。
                // with_safeborder 在这个配置下会越界读，所以只给 without 跑。
                c.cfg = 1; c.offX = 0; c.offY = 0; c.rectW = 20; c.rectH = 20; c.outH = 16; c.outW = 16;
            }
            // with_safeborder 在 cfgB 下会越界读，只跑 cfgA
            if (g_entries[e].safe && c.cfg == 1) continue;
            one(c, r[ai]);
        }
        printf("  nchwc%d %-22s  %d 个用例：对 %d，错 %d，崩 %d\n",
               A, g_name[g_entries[e].safe], g_case - c0, g_ok - k0, g_bad - b0, g_crash - x0);
    }
    printf("\n共 %d 个用例：全对 %d，有错 %d，崩溃/搭建失败 %d\n", g_case, g_ok, g_bad, g_crash);
    if (g_bad || g_crash)
        printf("**每一项在下结论之前都要先用独立复现对一遍**（附录 CA.3）。\n");
    return (g_bad || g_crash) ? 1 : 0;
}
