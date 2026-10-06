// zq_bbox_nms_par_check.cpp —— ZQ_CNN_BBoxUtils::_nms 的「单线程 vs 多线程」等价性
//（2026-10-07，附录 IW）
//
// 为什么需要这个测试
// --------------------
// `ZQ_CNN_BBoxUtils.h:135` 那个 `#pragma omp parallel for` 是**本仓 x86 上唯一一个
// 能用合成输入驱动的并行循环**（MTCNN 那 ~30 个都要模型文件）。
//
// 而它**从来没有被执行过**：仓内所有 sample 的 `thread_num` 都被夹成 1，
// `_nms` 走的是 `thread_num <= 1` 的串行分支（:62）。附录 IV 记的就是这个缺口。
// 附录 II 那一轮能从 MTCNN 挖出六个问题，靠的正是「跑 sample 看不见」这一条。
//
// 判据：**同一批输入，thread_num=1 与 thread_num=4 的输出必须逐位相同。**
// 之所以选「逐位」而不是「容差内相等」，是因为这条路径全部是整数坐标 +
// 严格大于的比较 —— 抑制与否是**离散**的，没有中间态；
// 任何竞态都会直接把某个框从「保留」翻成「抑制」，或者反过来。
// 用容差反而会把真问题糊过去。
//
// 为什么不用 TSan
// --------------
// 本机 `-fsanitize=thread` **链接不过**（缺 `libtsan_preinit.o`，附录 IV），
// 需要 root 才能补。所以这里用「确定性等价」代替「概率检出」：
// 它不需要采样，**同一个输入错一次就永远错**，而 TSan 是概率的。
//
// 真正报出来的那条数据竞争
// ----------------------
// 并行区里**每个线程都遍历同一个共享的 `bboxScore`，并写 `(*it).oriOrder = -1`**
// （:191 / :119 两份拷贝）。写的是自己的 `num` 对应的元素，值恒为 -1，
// 所以**当前逻辑下看不出行为差异**；但那是对同一对象的**无同步读写**，
// 形式上就是数据竞争（UB），一旦那段内层循环以后改成"计数/累加"就会立刻变成
// 真 bug。本测试的作用之一就是让这条路径**真的被跑到**，
// 至少保证它今天的行为是可复现、可比对的。

#include "ZQ_CNN_BBoxUtils.h"
#include <vector>
#include <cstdio>
#include <cstring>
#include <cstdlib>

using namespace ZQ;

// 造一批有重叠、有相离、有退化框的 box —— 退化框（零面积）是刻意加的：
// 附录 IJ.2 修的就是「零面积框除零得 +inf，一个框抑制掉所有框」。
static void make_boxes(std::vector<ZQ_CNN_BBox>& bb,
                       std::vector<ZQ_CNN_OrderScore>& sc, int n)
{
    for (int i = 0; i < n; i++)
    {
        ZQ_CNN_BBox b;
        b.score = (float)((i * 37) % 101) / 100.0f;
        b.row1 = 10 + (i % 7) * 3;
        b.col1 = 10 + (i % 5) * 4;
        // 一半的框与前一个高度重叠，一半相离；再每 17 个插一个零面积退化框
        if (i % 17 == 0)
        {
            b.row2 = b.row1;          // 零高
            b.col2 = b.col1;
        }
        else
        {
            b.row2 = b.row1 + 8;
            b.col2 = b.col1 + 8;
        }
        b.area = (float)(b.row2 - b.row1) * (float)(b.col2 - b.col1);
        b.exist = true;
        b.need_check_overlap_count = true;
        bb.push_back(b);

        ZQ_CNN_OrderScore s;
        s.score = b.score;
        s.oriOrder = i;              // 每个框在 bboxScore 里**只出现一次**
        sc.push_back(s);
    }
}

static int run_case(int n, int thr, const char* model)
{
    std::vector<ZQ_CNN_BBox> bb;
    std::vector<ZQ_CNN_OrderScore> sc;
    make_boxes(bb, sc, n);
    ZQ_CNN_BBoxUtils::_nms(bb, sc, 0.3f, model, 0, thr);
    int kept = 0;
    for (size_t i = 0; i < bb.size(); i++) if (bb[i].exist) kept++;
    return kept;
}

int main()
{
    int fails = 0, cases = 0;
    const int ns[] = { 1, 2, 5, 16, 64, 257 };
    const char* models[] = { "Union", "Min", "Other" };

    for (int mi = 0; mi < 3; mi++)
    {
        for (int ni = 0; ni < 6; ni++)
        {
            int n = ns[ni];
            int s1 = run_case(n, 1, models[mi]);
            int s4 = run_case(n, 4, models[mi]);
            cases++;
            if (s1 != s4)
            {
                fails++;
                printf("FAIL n=%-4d model=%-6s  thread_num=1 保留 %d, thread_num=4 保留 %d\n",
                       n, models[mi], s1, s4);
            }
        }
    }

    // 线程数也扫一遍：2/3/8 与单线程都必须一致（竞态随线程数变的现象很常见）
    for (int t = 2; t <= 8; t++)
    {
        for (int ni = 0; ni < 6; ni++)
        {
            int n = ns[ni];
            int s1 = run_case(n, 1, "Union");
            int st = run_case(n, t, "Union");
            cases++;
            if (s1 != st)
            {
                fails++;
                printf("FAIL n=%-4d thr=%-2d  thread_num=1 保留 %d, 保留 %d\n", n, t, s1, st);
            }
        }
    }

    printf("zq_bbox_nms_par: %d 个用例, %d 个不一致\n", cases, fails);
    if (fails == 0) printf("PASS\n");
    return fails == 0 ? 0 : 1;
}