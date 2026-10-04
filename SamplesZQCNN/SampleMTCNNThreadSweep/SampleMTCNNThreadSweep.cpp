// `thread_num` 扫描：换线程数，检测结果**必须完全一致**（附录 HS）。
//
// 为什么要有这个
// --------------
// `ZQ_CNN_MTCNN::Init` 会按 `thread_num` 建**thread_num 份独立的网**，
// 每一份都用**融合参数**加载：
//     pnet[i].LoadFrom(pnet_param, pnet_model, /*merge_bn=*/true, 1e-9, true)
// 而 `_merge_bn` / `_merge_prelu` 会**分配张量、`delete` 被吃掉的 BN 层**。
//
// **而 `thread_num` 从来没有被任何门禁扫过**：
//   * `SampleMTCNN` / `SampleMTCNNfromlist` 都写 `int thread_num = 0;`
//     —— 而 `Init` 里第一件事是 `thread_num = __max(1, thread_num);`
//     ⇒ **0 被悄悄变成 1，单线程路径**；
//   * `SampleMTCNNGesture` 是 3，但那是一个 sample 里的固定值。
//
// 风险有两类，本 sample 测的是**结果层面**那一类：
//   1. 几份网之间共享了可变状态（`static` 缓冲、`ignore_small_value` 之类的
//      静态量、或层对象被复用），于是**多份网的结果会互相影响**；
//   2. 融合在某一份上做了与别的份不一致的事。
// 二者的**外部表现都是同一个**：换 `thread_num` 会换检测结果。
//
// 判据：同一张图，`thread_num = 1` 与 `1/2/4/8` 的检出**个数与每一个框的
// 坐标、分数**必须**逐位相同**。不同就是缺陷。
//
// **已知的不足**（如实记在附录 HS.3）：这里每个 `thread_num` 只跑**一次**，
// 而 race 是**非确定性**的 —— 一次跑不到不等于没有。
// 中途试过"每个 thread_num 重复 8 次"，但那版自己出了问题
// （`bad++` 了却没有打印任何 FAIL 行），已回滚，**不发半成品**。
// 要提高灵敏度，正确做法是把 `run_once` 拆到独立进程里各跑若干次再比
// —— 那样 race 也逃不掉（进程级隔离 + 比对输出）。
//
// 不做 GUI（AGENTS.md：无头环境下 namedWindow/imshow/waitKey 一律注释掉），
// 本 sample 只读图、只打数字。
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

// 一次检测的完整结果：框数 + 每个框的分数与坐标
struct R {
    int n;
    std::vector<float> v;   // 每个框 5 个 float：score, col1, row1, col2, row2
    std::string to_str() const
    {
        char buf[64];
        std::string s;
        snprintf(buf, sizeof(buf), "n=%d", n);
        s = buf;
        for (size_t i = 0; i + 5 <= v.size(); i += 5) {
            snprintf(buf, sizeof(buf), " [%.6g %.6g %.6g %.6g %.6g]",
                     v[i], v[i + 1], v[i + 2], v[i + 3], v[i + 4]);
            s += buf;
        }
        return s;
    }
};

static R run_once(const cv::Mat& img, int thread_num)
{
    R r;
    r.n = -1;
    ZQ::ZQ_CNN_MTCNN mtcnn;
    mtcnn.TurnOffShowDebugInfo();
    if (!mtcnn.Init(P1, M1, P2, M2, P3, M3, thread_num, false, P4, M4)) {
        printf("    thread_num=%d : Init 失败\n", thread_num);
        return r;
    }
    mtcnn.SetPara(img.cols, img.rows, 80, 0.5, 0.6, 0.8, 0.4, 0.5, 0.5, 0.709, 3, 20, 4, false);
    std::vector<ZQ::ZQ_CNN_BBox> thirdBbox;
    if (!mtcnn.Find(img.data, img.cols, img.rows, img.step[0], thirdBbox)) {
        printf("    thread_num=%d : Find 失败\n", thread_num);
        return r;
    }
    r.n = (int)thirdBbox.size();
    r.v.reserve(thirdBbox.size() * 5);
    for (size_t i = 0; i < thirdBbox.size(); i++) {
        const ZQ::ZQ_CNN_BBox& b = thirdBbox[i];
        r.v.push_back(b.score);
        r.v.push_back((float)b.col1);
        r.v.push_back((float)b.row1);
        r.v.push_back((float)b.col2);
        r.v.push_back((float)b.row2);
    }
    return r;
}

int main()
{
    printf("MTCNN thread_num 扫描（附录 HS）\n");
    printf("判据：同一张图，thread_num = 1/2/4/8 的检出个数与每个框的坐标、分数"
           "必须**逐位相同**。\n");
    printf("      （ZQ_CNN_MTCNN 会按 thread_num 建多份**各自带 merge** 的网；\n");
    printf("        而 SampleMTCNN 写的是 thread_num=0，被 __max(1,x) 悄悄变成 1。）\n");
    printf("已知不足：每个 thread_num 只跑一次，抓不到非确定性的 race（见附录 HS.3）。\n\n");

    cv::Mat img = cv::imread(IMG, 1);
    if (img.empty()) {
        printf("  读不到图 %s —— 不能空跑\n", IMG);
        return 1;
    }
    if (img.channels() == 1) cv::cvtColor(img, img, CV_GRAY2BGR);

    const int ts[] = { 1, 2, 4, 8 };
    std::vector<R> res;
    for (size_t i = 0; i < sizeof(ts) / sizeof(ts[0]); i++) {
        R r = run_once(img, ts[i]);
        printf("  thread_num=%d : %s\n", ts[i], r.to_str().c_str());
        res.push_back(r);
    }

    int bad = 0;
    for (size_t i = 1; i < res.size(); i++) {
        if (res[i].n < 0) { bad++; continue; }          // 上面已经报过原因了
        if (res[i].n != res[0].n) {
            printf("  **FAIL** thread_num=%d 检出 %d 个，thread_num=1 是 %d 个"
                   " —— **换线程数换了结果**\n", ts[i], res[i].n, res[0].n);
            bad++;
        } else if (res[i].v != res[0].v) {
            for (size_t k = 0; k + 5 <= res[i].v.size(); k++) {
                if (res[i].v[k] != res[0].v[k]) {
                    printf("  **FAIL** thread_num=%d 与 1 的第 %zu 个框的第 %zu 个量不同："
                           "%.9g vs %.9g\n", ts[i], k / 5, k % 5, res[i].v[k], res[0].v[k]);
                    break;
                }
            }
            bad++;
        }
    }
    printf("\n%s\n", bad == 0
           ? "THREAD SWEEP OK（4 种 thread_num 的检出逐位相同）"
           : "THREAD SWEEP FAILED（换线程数换了结果）");
    return bad == 0 ? 0 : 1;
}
