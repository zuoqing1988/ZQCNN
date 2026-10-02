/* NCHW（layers_c）resize / remap 门禁 —— 附录 CN
 *
 * 为什么要有这个门禁
 * ------------------
 * 附录 CL 在 **NCHWC** 那一族的 resize 里查出一处越界读（y 的钳位漏了 -1），
 * 而它的对照物 **NCHW** 那一族（`zq_cnn_resize_32f_align_c*`）写的是对的。
 * 但 NCHW 这一族 —— 也就是 x86 上的**主生产路径**、被
 * `ZQ_CNN_Tensor4D.cpp` 直接调用 —— **此前一道数值门禁都没有**。
 * 附录 CM 只能确认它的**钳位字面量**与 NCHWC 修好之后一致，
 * **算术正确性、map 语义、fillval 判据仍然全靠"看着像对的"**。
 *
 * 覆盖面：15 个真实符号（`nm` 核实；头里是 34 个**声明**，含重复）
 *   resize_nn / resize_with_safeborder / resize_without_safeborder   各 x align0/128/256
 *   remap_without_safeborder / remap_without_safeborder_fillval      各 x align0/128/256
 *
 * 语义（逐条从 `zq_cnn_resize_32f_align_c_raw.h` 读出来的）
 * -----------------------------------------------------------
 *  resize_*   矩形 resize。坐标原点由**末尾那个 `sample_align_type`** 决定：
 *        == 1 : coord_ini = in_off（**不做**半像素平移）
 *        != 1 : coord_ini = 0.5*step - 0.5 + in_off（半像素中心）
 *     两条分支都要测 —— 名字里没有任何提示。
 *  resize_nn  x_nn = (int)(coord_x + 0.5f)，再钳到 [0, n-1]（四舍五入，不是截断）
 *  remap_*    逐输出像素从 map_x/map_y 取坐标，双线性；坐标钳到 [0, n-1]
 *  remap_*_fillval  额外一步：**用未钳位的坐标**判
 *        `coord_y >= 0 && coord_y <= in_H-1 && coord_x >= 0 && coord_x <= in_W-1`，
 *        不满足就整像素写 `fillval`（连通道一起）
 *
 * NCHW 布局：`offset(n,c,h,w) = n*sliceStep + h*widthStep + w*pixelStep + c`
 * （见 AGENTS.md「NCHW 与 NCHWC 的步长语义」）
 *
 * 沿用 CB~CM：名字写全走函数指针表、逐格统计、每用例 fork 子进程并
 * **显式判"没读到结果文件" = 失败**（CJ.4）、用真实的 ZQ_CNN_Tensor4D 类。
 */
#include "zq_check_child.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include "ZQCNN/ZQ_CNN_Tensor4D.h"
#include "zq_check_alloc.h"
#include "ZQCNN/layers_c/zq_cnn_resize_32f_align_c.h"

// resize 三个变体：19 个参数，末尾是 sample_align_type
typedef void (*FN_RESIZE)(const float* in, int N, int H, int W, int C,
                          int pixelStep, int widthStep, int sliceStep,
                          int off_x, int off_y, int rect_w, int rect_h,
                          float* out, int out_H, int out_W,
                          int out_pixelStep, int out_widthStep, int out_sliceStep,
                          int sample_align_type);
// remap：16 个参数
typedef void (*FN_REMAP)(const float* in, int N, int H, int W, int C,
                         int pixelStep, int widthStep, int sliceStep,
                         const float* map_x, const float* map_y,
                         float* out, int out_H, int out_W,
                         int out_pixelStep, int out_widthStep, int out_sliceStep);
// remap_fillval：17 个参数，末尾是 fillval
typedef void (*FN_REMAP_FILL)(const float* in, int N, int H, int W, int C,
                              int pixelStep, int widthStep, int sliceStep,
                              const float* map_x, const float* map_y,
                              float* out, int out_H, int out_W,
                              int out_pixelStep, int out_widthStep, int out_sliceStep,
                              float fillval);

enum { K_NN = 0, K_WITH = 1, K_WITHOUT = 2, K_REMAP = 3, K_REMAP_FILL = 4 };
static const char* g_kind_name[5] = { "resize_nn", "resize_with_safeborder",
                                       "resize_without_safeborder", "remap", "remap_fillval" };

struct Entry { void* fn; int kind; int align; };

// 15 行平铺，**每个内核名写全**（nm 核实过的 15 个符号）
static const Entry g_entries[15] = {
  { (void*)zq_cnn_resize_nn_32f_align0, K_NN, 1 },
  { (void*)zq_cnn_resize_nn_32f_align128bit, K_NN, 4 },
  { (void*)zq_cnn_resize_nn_32f_align256bit, K_NN, 8 },
  { (void*)zq_cnn_resize_with_safeborder_32f_align0, K_WITH, 1 },
  { (void*)zq_cnn_resize_with_safeborder_32f_align128bit, K_WITH, 4 },
  { (void*)zq_cnn_resize_with_safeborder_32f_align256bit, K_WITH, 8 },
  { (void*)zq_cnn_resize_without_safeborder_32f_align0, K_WITHOUT, 1 },
  { (void*)zq_cnn_resize_without_safeborder_32f_align128bit, K_WITHOUT, 4 },
  { (void*)zq_cnn_resize_without_safeborder_32f_align256bit, K_WITHOUT, 8 },
  { (void*)zq_cnn_remap_without_safeborder_32f_align0, K_REMAP, 1 },
  { (void*)zq_cnn_remap_without_safeborder_32f_align128bit, K_REMAP, 4 },
  { (void*)zq_cnn_remap_without_safeborder_32f_align256bit, K_REMAP, 8 },
  { (void*)zq_cnn_remap_without_safeborder_fillval_32f_align0, K_REMAP_FILL, 1 },
  { (void*)zq_cnn_remap_without_safeborder_fillval_32f_align128bit, K_REMAP_FILL, 4 },
  { (void*)zq_cnn_remap_without_safeborder_fillval_32f_align256bit, K_REMAP_FILL, 8 },
};
static const int N_ENTRY = 15;

#define RES_FILE "/tmp/zq_rzn_res.txt"
static const double TOL = 1e-5;
static const float FILLVAL = -7.25f;      // 刻意取一个不可能与输入撞上的值

static float val(int seed, int idx)
{
    unsigned int x = (unsigned int)((unsigned int)seed * 2654435761u + (unsigned int)idx * 40503u);
    x ^= x >> 13; x *= 1274126177u; x ^= x >> 16;
    return (float)((int)(x % 2001) - 1000) * 0.001f;
}

// 0 = resize（跑两种 sample_align_type）；1 = remap（map 全在图内）；
// 2 = remap（map 有一半在图外，验钳位 / fillval）
struct Case { int entry, cfg; };

static inline double at(const float* p, int slice, int width, int pixel, int h, int w, int c)
{
    return p[slice + h * width + w * pixel + c];
}
static inline int cl(int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); }

typedef void (*RUNNER)(const Case&);
static void run_one(const Case& c)
{
    const Entry& e = g_entries[c.entry];
    const int A = e.align;
    const int N = 1, H = 16, W = 16, C = A;
    const int oH = 8, oW = 8;

    // ---- 紧凑数据 (n,c,h,w) ----
    const size_t nin = (size_t)N * C * H * W;
    float* in = zq_alloc_f32(nin);
    if (!in) return;
    for (size_t i = 0; i < nin; i++) in[i] = val(1, (int)i);

    // NCHW 张量：pixelStep = C（这里 C 恰是 align 的倍数，不额外补齐）
    const int in_pixelStep = C, in_widthStep = in_pixelStep * W, in_sliceStep = in_widthStep * H;
    const int out_pixelStep = C, out_widthStep = out_pixelStep * oW, out_sliceStep = out_widthStep * oH;
    float* out = zq_alloc_f32((size_t)N * out_sliceStep);
    float* mapx = zq_alloc_f32((size_t)oH * oW);
    float* mapy = zq_alloc_f32((size_t)oH * oW);
    if (!out || !mapx || !mapy) {
        if (in) zq_free_f32(in); if (out) zq_free_f32(out);
        if (mapx) zq_free_f32(mapx); if (mapy) zq_free_f32(mapy);
        return;
    }
    for (size_t i = 0; i < (size_t)N * out_sliceStep; i++) out[i] = -12345.0f;
    for (size_t i = 0; i < (size_t)oH * oW; i++) { mapx[i] = 0.f; mapy[i] = 0.f; }

    const bool is_remap = (e.kind == K_REMAP || e.kind == K_REMAP_FILL);
    // 三个 resize 变体用的矩形；with_safeborder 不钳位，所以给一个图内的矩形
    int offX = 2, offY = 3, rectW = 8, rectH = 6;
    float w_step = 1.0f / oW * rectW, h_step = 1.0f / oH * rectH;

    for (int h = 0; h < oH; h++)
        for (int w = 0; w < oW; w++) {
            float cx, cy;
            if (c.cfg == 2) {
                // 一半在图内、一半在图外：验 without 的钳位与 fillval 的判据
                cx = (h < oH / 2) ? (0.25f + w * 0.9f) : (-3.5f + w * 1.4f);
                cy = (w < oW / 2) ? (0.75f + h * 1.1f) : (H + 2.5f - h * 0.8f);
            } else {
                cx = 0.5f * w_step - 0.5f + offX + w * w_step;
                cy = 0.5f * h_step - 0.5f + offY + h * h_step;
            }
            mapx[h * oW + w] = cx; mapy[h * oW + w] = cy;
        }

    if (is_remap) {
        if (e.kind == K_REMAP_FILL)
            ((FN_REMAP_FILL)e.fn)(in, N, H, W, C, in_pixelStep, in_widthStep, in_sliceStep,
                                 mapx, mapy, out, oH, oW,
                                 out_pixelStep, out_widthStep, out_sliceStep, FILLVAL);
        else
            ((FN_REMAP)e.fn)(in, N, H, W, C, in_pixelStep, in_widthStep, in_sliceStep,
                            mapx, mapy, out, oH, oW,
                            out_pixelStep, out_widthStep, out_sliceStep);
    } else {
        ((FN_RESIZE)e.fn)(in, N, H, W, C, in_pixelStep, in_widthStep, in_sliceStep,
                          offX, offY, rectW, rectH, out, oH, oW,
                          out_pixelStep, out_widthStep, out_sliceStep,
                          /* sample_align_type */ (c.cfg == 1) ? 1 : 0);
    }

    // ---- 参考 ----
    long n_ok = 0, n_bad = 0; double worst = 0.0;
    for (int h = 0; h < oH; h++)
        for (int w = 0; w < oW; w++) {
            double exp[C > 0 ? 32 : 1];
            if (C > 32) return;
            double den_scale = 1.0;
            if (is_remap) {
                const float cx = mapx[h * oW + w], cy = mapy[h * oW + w];
                if (e.kind == K_REMAP_FILL) {
                    bool inside = (cy >= 0 && cy <= H - 1 && cx >= 0 && cx <= W - 1);
                    if (!inside) { for (int cc = 0; cc < C; cc++) exp[cc] = FILLVAL; den_scale = 1e4; goto done; }
                }
                int x0 = (int)floorf(cx), y0 = (int)floorf(cy);
                float sx = cx - floorf(cx), sy = cy - floorf(cy);
                int x1 = x0 + 1, y1 = y0 + 1;
                x0 = cl(x0, W - 1); x1 = cl(x1, W - 1);
                y0 = cl(y0, H - 1); y1 = cl(y1, H - 1);
                for (int cc = 0; cc < C; cc++) {
                    double v00 = at(in, 0, in_widthStep, in_pixelStep, y0, x0, cc);
                    double v01 = at(in, 0, in_widthStep, in_pixelStep, y0, x1, cc);
                    double v10 = at(in, 0, in_widthStep, in_pixelStep, y1, x0, cc);
                    double v11 = at(in, 0, in_widthStep, in_pixelStep, y1, x1, cc);
                    double r0 = v00 + (v01 - v00) * sx, r1 = v10 + (v11 - v10) * sx;
                    exp[cc] = r0 + (r1 - r0) * sy;
                }
            } else {
                float cx, cy;
                if (c.cfg == 1) { cx = offX + w * w_step; cy = offY + h * h_step; }
                else { cx = 0.5f * w_step - 0.5f + offX + w * w_step;
                       cy = 0.5f * h_step - 0.5f + offY + h * h_step; }
                if (e.kind == K_NN) {
                    int xn = cl((int)(cx + 0.5f), W - 1), yn = cl((int)(cy + 0.5f), H - 1);
                    for (int cc = 0; cc < C; cc++) exp[cc] = at(in, 0, in_widthStep, in_pixelStep, yn, xn, cc);
                } else {
                    int x0 = (int)floorf(cx), y0 = (int)floorf(cy);
                    float sx = cx - floorf(cx), sy = cy - floorf(cy);
                    int x1 = x0 + 1, y1 = y0 + 1;
                    if (e.kind == K_WITHOUT) {     // with_safeborder 不钳
                        x0 = cl(x0, W - 1); x1 = cl(x1, W - 1);
                        y0 = cl(y0, H - 1); y1 = cl(y1, H - 1);
                    }
                    for (int cc = 0; cc < C; cc++) {
                        double v00 = at(in, 0, in_widthStep, in_pixelStep, y0, x0, cc);
                        double v01 = at(in, 0, in_widthStep, in_pixelStep, y0, x1, cc);
                        double v10 = at(in, 0, in_widthStep, in_pixelStep, y1, x0, cc);
                        double v11 = at(in, 0, in_widthStep, in_pixelStep, y1, x1, cc);
                        double r0 = v00 + (v01 - v00) * sx, r1 = v10 + (v11 - v10) * sx;
                        exp[cc] = r0 + (r1 - r0) * sy;
                    }
                }
            }
        done:
            for (int cc = 0; cc < C; cc++) {
                double y = exp[cc];
                double got = out[h * out_widthStep + w * out_pixelStep + cc];
                double den = (fabs(y) > 1.0 ? fabs(y) : 1.0) * den_scale;
                double be = fabs(got - y) / den;
                if (be > TOL) n_bad++; else n_ok++;
                if (be > worst) worst = be;
            }
        }
    FILE* f = fopen(RES_FILE, "w");
    if (f) { fprintf(f, "%ld %ld %.6e\n", n_ok, n_bad, worst); fclose(f); }
    zq_free_f32(in); zq_free_f32(out); zq_free_f32(mapx); zq_free_f32(mapy);
}

static bool is_remap_kind(int k) { return (k == K_REMAP || k == K_REMAP_FILL); }

static int g_case = 0, g_ok = 0, g_bad = 0, g_crash = 0;

static void one(const Case& c)
{
    g_case++;
    remove(RES_FILE);
    pid_t pid = fork();
    if (pid == 0) {
        zq_child_silence_stderr();
        run_one(c);
        _exit(0);
    }
    int st = 0; waitpid(pid, &st, 0);
    long ok = 0, bad = 0; double worst = 0;
    int have = 0;
    FILE* f = fopen(RES_FILE, "r");
    if (f) { have = (fscanf(f, "%ld %ld %lf", &ok, &bad, &worst) == 3); fclose(f); }
    char nm[44], tag[48];
    snprintf(nm, sizeof(nm), "align%d %s", g_entries[c.entry].align, g_kind_name[g_entries[c.entry].kind]);
    const char* cfg;
    if (is_remap_kind(g_entries[c.entry].kind))
        cfg = (c.cfg == 2) ? "map 一半出图" : "map 全在图内";
    else
        cfg = (c.cfg == 1) ? "sample_align_type=1" : "半像素中心";
    snprintf(tag, sizeof(tag), "%s", cfg);
    if (!have) { g_crash++; printf("  %-40s %s  没跑完（退出码 %d）\n", nm, tag, WEXITSTATUS(st)); return; }
    if (WIFSIGNALED(st)) { g_crash++; printf("  %-40s %s  CRASH(信号 %d)\n", nm, tag, WTERMSIG(st)); return; }
    if (bad > 0) { g_bad++; printf("  %-40s %s  FAIL %ld/%ld 格错, 最差 %.3e\n", nm, tag, bad, ok + bad, worst); }
    else { g_ok++; }
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("NCHW resize / remap：15 个真实符号（nm 核实；头里是 34 个声明，含重复）\n");
    printf("  resize_nn / resize_with_safeborder / resize_without_safeborder\n");
    printf("  remap_without_safeborder / remap_without_safeborder_fillval   各 x align0/128/256\n");
    printf("内核名全部写全、走函数指针表；判据：逐格后向误差\n");
    printf("**末尾的 sample_align_type 决定要不要半像素平移**（名字里没有任何提示），两种都跑\n");
    printf("remap 的两种配置：map 全在图内 / 一半出图（验 without 的钳位与 fillval 的判据）\n\n");

    for (int e = 0; e < N_ENTRY; e++) {
        const int c0 = g_case, k0 = g_ok, b0 = g_bad, x0 = g_crash;
        const bool rm = is_remap_kind(g_entries[e].kind);
        for (int j = 0; j < 2; j++) {
            Case c; memset(&c, 0, sizeof(c));
            c.entry = e;
            c.cfg = rm ? (j == 0 ? 0 : 2) : j;      // resize: 0/1 两种对齐方式；remap: 图内/出图
            one(c);
        }
        printf("  align%d %-28s  %d 个用例：对 %d，错 %d，崩 %d\n",
               g_entries[e].align, g_kind_name[g_entries[e].kind],
               g_case - c0, g_ok - k0, g_bad - b0, g_crash - x0);
    }
    printf("\n共 %d 个用例：全对 %d，有错 %d，崩溃/搭建失败 %d\n", g_case, g_ok, g_bad, g_crash);
    if (g_bad || g_crash)
        printf("**每一项在下结论之前都要先用独立复现对一遍**（附录 CA.3）。\n");
    return (g_bad || g_crash) ? 1 : 0;
}
