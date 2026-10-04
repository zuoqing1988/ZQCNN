// `thread_num` 扫描，**每次检测放在独立进程里**（附录 HT）。
//
// 这是 HS 的加强版。HS 版把 4 种 `thread_num` 各在**同一个进程**里跑一次，
// 附录 HS.3 记了它的不足：**race 是非确定性的，一次跑不到不等于没有**。
// 中途试过在同一进程里"重复 8 次"，那版自己出了问题（`bad++` 却没有任何
// FAIL 打印），已回滚 —— 结论是**同进程重复抓不住 race**，
// 因为第一次留下的堆布局会影响后面的每一次。
//
// 这里的做法：**每一次检测都是一个独立进程**。
//   * 进程间不共享任何地址空间 —— 堆布局、分配器状态、`static` 变量
//     **每次都从同样的初始状态重来**，而这正是 race 最容易暴露的形态
//     （同一份代码在不同的堆布局下走不同的分支）。
//   * 顺带把"进程崩了"也变成可观测的：子进程非 0 退出会被记成一次失败，
//     而在同一进程里崩溃会让整轮扫描一起没掉。
//
// 模式：
//   父模式（无参）：对 thread_num = 1/2/4/8 各 spawn REPS 个子进程，
//                   把每个子进程写出的结果文件读回来**逐字节**比。
//   子模式（两参）：`<thread_num> <结果文件路径>` —— 跑一次检测，
//                   把 `n <每个框的 5 个量>` 写进那个文件，退出。
//
// **为什么结果走文件而不是 stdout**：`Init` / `Find` 可能往 stdout 写东西，
// 库自己的输出混进来就分不清"结果变了"和"库多打了一行"。
// 走文件则只有我们自己写的内容可比（与 `tools/zq_model_params_check.cpp`
// 那个 `OUT_FILE` 的做法一样）。
//
// 不做 GUI（AGENTS.md：无头环境下 namedWindow/imshow/waitKey 一律注释掉）。
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "ZQ_CNN_MTCNN.h"
#include "ZQ_CNN_CompileConfig.h"

#if ZQ_CNN_USE_BLAS_GEMM
#include <openblas/cblas.h>
#elif ZQ_CNN_USE_MKL_GEMM
#include <mkl.h>
#endif

#if defined(_WIN32)
#define P1 "model/det1-dw20-fast.zqparams"
#define M1 "model/det1-dw20-fast.nchwbin"
#define P2 "model/det2-dw24-fast.zqparams"
#define M2 "model/det2-dw24-fast.nchwbin"
#define P3 "model/det3-dw48-fast.zqparams"
#define M3 "model/det3-dw48-fast.nchwbin"
#define P4 "model/det5-dw64-v3s.zqparams"
#define M4 "model/det5-dw64-v3s.nchwbin"
#define IMG "data/11.jpg"
#else
#define P1 "../../model/det1-dw20-fast.zqparams"
#define M1 "../../model/det1-dw20-fast.nchwbin"
#define P2 "../../model/det2-dw24-fast.zqparams"
#define M2 "../../model/det2-dw24-fast.nchwbin"
#define P3 "../../model/det3-dw48-fast.zqparams"
#define M3 "../../model/det3-dw48-fast.nchwbin"
#define P4 "../../model/det5-dw64-v3s.zqparams"
#define M4 "../../model/det5-dw64-v3s.nchwbin"
#define IMG "../../data/11.jpg"
#endif

#if defined(_WIN32)
#include <windows.h>
#endif

static const int REPS = 6;

// **自己的可执行文件路径**。
// 2026-10-04 实测踩过：用 `argv[0]` 当命令**是不够的** ——
// 从 Git Bash 里用 `./SampleMTCNNThreadSweep.exe` 启动时 `argv[0]` 就是
// `./SampleMTCNNThreadSweep.exe`，而 `system()` 在 Windows 上走的是
// `cmd.exe /c`，**cmd 不认 `./`** —— 于是 24 个子进程**全部**没启动，整轮报红
// （父进程的"没写出结果文件"守卫正确抓住了，没有假通过）。
// Linux 侧没事，因为 POSIX shell 认 `./`。
// 所以 Windows 用 `GetModuleFileNameA` 取绝对路径。
static void self_path(char* buf, int n, const char* argv0)
{
#if defined(_WIN32)
    GetModuleFileNameA(NULL, buf, n);
#else
    snprintf(buf, n, "%s", argv0);
#endif
}

// 读回子进程写的结果文件；读不到返回 false（那本身就是一种失败）
static bool read_result(const char* path, std::string& out)
{
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    char buf[8192];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = 0;
    out = buf;
    // 去掉行尾差异（Windows 会写 \r\n）
    while (!out.empty() && (out[out.size() - 1] == '\n' || out[out.size() - 1] == '\r'))
        out.erase(out.size() - 1);
    return true;
}

int main(int argc, char** argv)
{
    // ---------------- 子模式：跑一次检测，把结果写文件 ----------------
    if (argc >= 3) {
        int tn = atoi(argv[1]);
        const char* outp = argv[2];
        FILE* o = fopen(outp, "wb");
        if (!o) return 2;
        cv::Mat img = cv::imread(IMG, 1);
        if (img.empty()) { fprintf(o, "ERR no-image\n"); fclose(o); return 2; }
        if (img.channels() == 1) cv::cvtColor(img, img, CV_GRAY2BGR);
        ZQ::ZQ_CNN_MTCNN mtcnn;
        mtcnn.TurnOffShowDebugInfo();
        if (!mtcnn.Init(P1, M1, P2, M2, P3, M3, tn, false, P4, M4)) {
            fprintf(o, "ERR init\n");
            fclose(o);
            return 3;
        }
        mtcnn.SetPara(img.cols, img.rows, 80, 0.5, 0.6, 0.8, 0.4, 0.5, 0.5, 0.709, 3, 20, 4, false);
        std::vector<ZQ::ZQ_CNN_BBox> bb;
        if (!mtcnn.Find(img.data, img.cols, img.rows, img.step[0], bb)) {
            fprintf(o, "ERR find\n");
            fclose(o);
            return 4;
        }
        // %.9g：足以把 float32 的每一位差异显出来（float32 最多 9 位有效数字）
        fprintf(o, "n=%d", (int)bb.size());
        for (size_t i = 0; i < bb.size(); i++) {
            fprintf(o, " [%.9g %.9g %.9g %.9g %.9g]", bb[i].score,
                    (double)bb[i].col1, (double)bb[i].row1,
                    (double)bb[i].col2, (double)bb[i].row2);
        }
        fclose(o);
        return 0;
    }

    // ---------------- 父模式：spawn 各 thread_num 各 REPS 次，逐字节比 ----------------
    printf("MTCNN thread_num 扫描，**每次检测一个独立进程**（附录 HT）\n");
    printf("判据：thread_num = 1/2/4/8 各 %d 个独立进程，结果必须**逐字节相同**。\n", REPS);
    printf("      （进程间不共享地址空间 —— 堆布局与分配器状态每次从头来，\n");
    printf("        这正是 race 最容易暴露的形态；同进程重复抓不住。）\n\n");

    const int ts[] = { 1, 2, 4, 8 };
    char exebuf[1024];
    self_path(exebuf, sizeof(exebuf), argv[0]);
    const char* exe = exebuf;
    char cmd[1024], res[256];
    std::string base;
    int bad = 0, ran = 0, spawn_fail = 0, ndiff = 0;

    for (size_t i = 0; i < sizeof(ts) / sizeof(ts[0]); i++) {
        for (int rep = 0; rep < REPS; rep++) {
            // 结果文件写在**当前工作目录**，不是 /tmp。
            // 2026-10-04 实测：Windows 上写 `/tmp/...` 时 24 个子进程**全部**
            // 没写出文件（父进程正确报成"没写出结果文件"而不是假通过 ——
            // 那个守卫起作用了），但整轮仍然是红的。
            // 当前目录两边都行，而且本 sample 本来就要在**产物目录**里跑。
            snprintf(res, sizeof(res), "zq_sweep_t%d_r%d.txt", ts[i], rep);
            remove(res);
#if defined(_WIN32)
            // **不要自己加引号**。MSVC 的 system() 已经把整条命令包在一层引号里
            // 交给 `cmd.exe /c`，字符串里再写 `\"` 会被二次转义，
            // cmd 于是把 `\"D:\...exe\"` 整个当成命令名 -> "'...exe' 不是内部或外部命令"，rc=1。
            // 2026-10-04 实测踩过：24 个子进程**全部**没启动。
            // 所以 Windows 侧用**不带引号**的路径，并且对含空格的路径**显式拒绝**，
            // 而不是静默跑出一个错的命令。
            if (strchr(exe, ' ')) {
                printf("  **FAIL** 可执行文件路径含空格，Windows 的 system() 这里处理不了：%s\n", exe);
                printf("        请从不含空格的目录运行。\n");
                return 1;
            }
            snprintf(cmd, sizeof(cmd), "%s %d %s >nul 2>&1", exe, ts[i], res);
#else
            snprintf(cmd, sizeof(cmd), "'%s' %d '%s' >/dev/null 2>&1", exe, ts[i], res);
#endif
            int rc = system(cmd);
            std::string got;
            if (!read_result(res, got)) {
                printf("  **FAIL** thread_num=%d rep=%d : 子进程**没写出结果文件**（system rc=%d）\n"
                       "        命令 : %s\n        exe  : %s\n"
                       "        结果 : %s\n",
                       ts[i], rep, rc, cmd, exe, res);
                bad++;
                spawn_fail++;
                continue;
            }
            ran++;
            if (i == 0 && rep == 0) {
                base = got;
                printf("  基准 thread_num=1 rep=0 : %s\n", got.c_str());
                continue;
            }
            if (got != base) {
                printf("  **FAIL** thread_num=%d rep=%d 结果与基准不同：\n"
                       "        基准 : %s\n        本次 : %s\n",
                       ts[i], rep, base.c_str(), got.c_str());
                bad++;
                ndiff++;
            } else if (rep == 0) {
                printf("  thread_num=%d rep=0 : 与基准逐字节相同\n", ts[i]);
            }
        }
    }

    printf("\n  跑了 %d 个独立进程（%d 个没写出结果文件），与基准不一致 %d 次\n",
           ran + spawn_fail, spawn_fail, ndiff);
    for (size_t i = 0; i < sizeof(ts) / sizeof(ts[0]); i++)
        for (int rep = 0; rep < REPS; rep++) {
            snprintf(res, sizeof(res), "zq_sweep_t%d_r%d.txt", ts[i], rep);
            remove(res);          // 别把临时文件留在产物目录里
        }
    printf("%s\n", bad == 0
           ? "THREAD SWEEP OK（跨进程、跨 thread_num 全部逐字节相同）"
           : "THREAD SWEEP FAILED");
    return bad == 0 ? 0 : 1;
}
