/* Concat 自别名（top 与某个 bottom 同名、但不是**下标相同**的那一个）的复现 —— 附录 EN
 *
 * 缺陷本体
 * --------
 * `ZQ_CNN_Forward_SSEUtils::_concat_NCHW`（ZQCNN/ZQ_CNN_Forward_SSEUtils.cpp:4951）
 * 的多输入分支里，`output` 的形状是在**读完 `valid_inputs` 的形状之后**才改的，
 * 而 `valid_inputs[i]` 是**指针**。当 `output` 恰好就是某个 `valid_inputs[i]`
 * （即 Concat 层的 top 名字与某个 bottom 名字相同、但下标不同）时：
 *
 *   1. `output.ChangeSize(out_N, out_H, out_W, out_C, 0, 0)` 把那个输入**就地扩容**，
 *      它的 C 变成了 out_C（= 前面几个输入的 C 之和），内容被 Reset 清零；
 *   2. 随后的拷贝循环用 `in_C = valid_inputs[i]->GetC()` 取**扩容后**的 C，
 *      于是每个像素写 out_C 个 float，而这一路输入其实只占 out_C - 前缀C 个；
 *   3. 最后一个像素写完，**越出整块分配 A.C 个 float**（A.C 是排在它前面的那个输入的 C）。
 *
 * 为什么没被挡住
 * --------------
 * `ZQ_CNN_Net::_check_connect` 里有一道就地守卫，但它只比**同一下标**
 * （`tops[i][j] == bottoms[i][j]`，j < min(两者的长度)）。
 * `bottoms=[A,B] top=B` 时 j=0 比的是 B 与 A，放行。
 *
 * 本文件做什么
 * ------------
 * 不想为了复现去编 `ZQ_CNN_Forward_SSEUtils.cpp`（那个 TU 极重）。
 * 所以这里**逐字照抄** `_concat_NCHW` 多输入分支那个拷贝循环，
 * 配真实的 `ZQ_CNN_Tensor4D` 对象，在 ASan 下看它到底写不写越界。
 * 循环体是自包含的（只用到 Get* 与 memcpy），照抄保真。
 *
 * 判据
 * ----
 * 两个子进程（对齐种类交叉），每个跑一批用例：
 *   - top 别名 **第一个** 输入
 *   - top 别名 **第二个** 输入
 * ASan 下任何一个越界都会让子进程非零退出。
 * 另有一组"良性"用例（top 是独立张量）作对照，必须**不**报错。
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include "zq_check_child.h"
#include "ZQCNN/ZQ_CNN_Tensor4D.h"

using namespace ZQ;

#define RES_FILE "/tmp/zq_concat_alias_res.txt"

enum { K_A0 = 0, K_A128 = 1, K_A256 = 2 };
static const char* g_kind_name[] = { "align0", "align128bit", "align256bit" };

static ZQ_CNN_Tensor4D* make(int kind)
{
    if (kind == K_A0) return new ZQ_CNN_Tensor4D_NHW_C_Align0();
    if (kind == K_A128) return new ZQ_CNN_Tensor4D_NHW_C_Align128bit();
    return new ZQ_CNN_Tensor4D_NHW_C_Align256bit();
}

static void fill(ZQ_CNN_Tensor4D* t, float base)
{
    const int ps = t->GetPixelStep();
    float* p = t->GetFirstPixelPtr();
    for (int n = 0; n < t->GetN(); n++)
        for (int h = 0; h < t->GetH(); h++)
            for (int w = 0; w < t->GetW(); w++)
                for (int c = 0; c < ps; c++)
                    p[(int64_t)n * t->GetSliceStep() + (int64_t)h * t->GetWidthStep()
                      + (int64_t)w * t->GetPixelStep() + c] = base + c;
}

/* ---- 以下是 ZQ_CNN_Forward_SSEUtils.cpp:4990-5027 的逐字照抄 ----
 * 唯一改动：valid_inputs / output 由调用方传入（原文是函数参数）。
 */
static bool concat_copy_loop(const std::vector<ZQ_CNN_Tensor4D*>& valid_inputs, int axis,
                             ZQ_CNN_Tensor4D& output)
{
    int out_pixStep = output.GetPixelStep();
    int out_widthStep = output.GetWidthStep();
    int out_sliceStep = output.GetSliceStep();

    float* out_ptr = output.GetFirstPixelPtr();
    for (int i = 0; i < valid_inputs.size(); i++)
    {
        int in_N = valid_inputs[i]->GetN();
        int in_C = valid_inputs[i]->GetC();
        int in_H = valid_inputs[i]->GetH();
        int in_W = valid_inputs[i]->GetW();
        int in_pixStep = valid_inputs[i]->GetPixelStep();
        int in_widthStep = valid_inputs[i]->GetWidthStep();
        int in_sliceStep = valid_inputs[i]->GetSliceStep();
        const float* in_ptr = valid_inputs[i]->GetFirstPixelPtr();
        const float* in_slice_ptr = in_ptr;
        float* out_slice_ptr = out_ptr;
        for (int n = 0; n < in_N; n++)
        {
            const float* in_row_ptr = in_slice_ptr;
            float* out_row_ptr = out_slice_ptr;
            for (int h = 0; h < in_H; h++)
            {
                const float* in_pix_ptr = in_row_ptr;
                float* out_pix_ptr = out_row_ptr;
                for (int w = 0; w < in_W; w++)
                {
                    memcpy(out_pix_ptr, in_pix_ptr, sizeof(float)*in_C);
                    in_pix_ptr += in_pixStep;
                    out_pix_ptr += out_pixStep;
                }
                in_row_ptr += in_widthStep;
                out_row_ptr += out_widthStep;
            }
            in_slice_ptr += in_sliceStep;
            out_slice_ptr += out_sliceStep;
        }
        if (axis == 0)
            out_ptr += in_N*out_sliceStep;
        else if (axis == 1)
            out_ptr += in_C;
        else if (axis == 2)
            out_ptr += in_H*out_widthStep;
        else if (axis == 3)
            out_ptr += in_W*out_pixStep;
    }
    return true;
}
/* ---- 照抄结束 ---- */

// axis=1，只在 C 上拼。alias_slot: -1 = top 是独立张量（良性对照）
//                    0/1 = top 就是第 0/1 个输入
static int run_case(int kind, int N, int C0, int C1, int H, int W, int alias_slot)
{
    ZQ_CNN_Tensor4D* a = make(kind);
    ZQ_CNN_Tensor4D* b = make(kind);
    ZQ_CNN_Tensor4D* out = (alias_slot < 0) ? make(kind) : ((alias_slot == 0) ? a : b);
    int rc = 0;
    if (a->ChangeSize(N, H, W, C0, 0, 0) && b->ChangeSize(N, H, W, C1, 0, 0)) {
        fill(a, 1.f); fill(b, 1000.f);
        // _concat_NCHW 的多输入分支：先按目标形状重排 output
        if (out->ChangeSize(N, H, W, C0 + C1, 0, 0)) {
            std::vector<ZQ_CNN_Tensor4D*> vi;
            vi.push_back(a); vi.push_back(b);
            concat_copy_loop(vi, 1, *out);
        } else {
            rc = 2;
        }
    } else {
        rc = 1;
    }
    delete a; delete b;
    if (alias_slot < 0) delete out;
    return rc;
}

int main(int argc, char** argv)
{
    int kind = (argc > 1) ? atoi(argv[1]) : 0;
    int alias_slot = (argc > 2) ? atoi(argv[2]) : -1;
    int idx = (argc > 3) ? atoi(argv[3]) : 0;

    struct C { int N, C0, C1, H, W; };
    static const C cases[] = {
        {1,  3,  3, 1, 1},
        {1, 64, 64, 1, 1},
        {1, 64, 64, 3, 3},
        {1, 32, 96, 5, 7},
        {2, 128, 128, 2, 2},
        {1, 256, 512, 4, 4},
        {1, 1, 1, 1, 1},
        {3, 17, 23, 3, 5},
    };
    const int ncase = (int)(sizeof(cases) / sizeof(cases[0]));
    if (idx < 0 || idx >= ncase) return 0;

    int rc = run_case(kind, cases[idx].N, cases[idx].C0, cases[idx].C1,
                      cases[idx].H, cases[idx].W, alias_slot);
    // 结果文件是"子进程活着跑完了"的信号，harness 靠它判别（附录 CJ.4）
    FILE* fp = fopen(RES_FILE, "a");
    if (fp) { fprintf(fp, "%d %d %d %d %d %d %d %d %d\n", kind, alias_slot, idx,
                      cases[idx].N, cases[idx].C0, cases[idx].C1, cases[idx].H, cases[idx].W, rc);
              fclose(fp); }
    return 0;
}
