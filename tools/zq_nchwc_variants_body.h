// `ZQ_CNN_Net_NCHWC<Tensor4D>` 的**三个**张量变体是不是都编得过（附录 HD）。
//
// 背景：HC 那道门禁只实例化了 `NCHWC4`，因为那是生产里唯一被用的
// （`ZQ_CNN_MTCNN_NCHWC.h:54` 的 `std::vector<ZQ_CNN_Net_NCHWC<ZQ_CNN_Tensor4D_NCHWC4>>`）。
// 另外两个变体 `NCHWC1` / `NCHWC8` 在**仓库里一次都没被实例化过**
// —— `grep -rn "ZQ_CNN_Net_NCHWC<" ZQCNN/ SamplesZQCNN/` 只有 NCHWC4 一处。
//
// 而它们的**前向实现**（`ZQ_CNN_Forward_SSEUtils_NCHWC.cpp` 里那一大堆
// `InnerProduct(ZQ_CNN_Tensor4D_NCHWC1&, …)` / `…NCHWC8&`）是**存在的**，
// 也就是说：**有人写了运行时代码，却没有任何调用方**。
//
// 所以要量两件事：
//   (1) 三个变体**是不是都编得过**。模板宣称支持三种布局，
//       若有两个编不过，那"支持三种"这句话本身就是错的。
//   (2) 编得过的那几个，**能不能走完一次 LoadFrom + SaveModel 往返**
//       —— 跟 HC 同一套判据。
//
// (1) 是本门禁存在的理由；它用 `#if` 逐个变体分别编译，
// 失败的那个**只报不算**（否则这道门禁在编不过的变体上永远红），
// 但**必须报出来** —— "编不过"和"没人用"是两件事，混起来就成了"看起来没问题"。
#ifndef ZQ_NCHWC_VARIANTS_BODY_H
#define ZQ_NCHWC_VARIANTS_BODY_H
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <dirent.h>
#include "ZQCNN/ZQ_CNN_Tensor4D.h"
#include "ZQCNN/ZQ_CNN_Tensor4D_NCHWC.h"
#include "ZQCNN/ZQ_CNN_Net_NCHWC.h"
#include "zq_net_fwd_tripwires.h"
#include "zq_concat_getsize_real.h"

#define MODEL_DIR "/mnt/d/ZQCNN/model"

static long long file_size(const std::string& p)
{
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long long n = ftell(f);
    fclose(f);
    return n;
}

// **逐字节**比内容，差异只允许 fabs(原值) < 1e-12（`ignore_small_value` 的清零）。
//
// 为什么要逐字节而不只比长度：第一版这里只比长度，于是阳性对照
// （在 `ZQ_CNN_Tensor4D_NCHWC::ConvertToCompactNCHW` 写回那一行乘 1.0000001f）
// 让 `zq_nchwc_roundtrip` 红了 34 条，而**三个变体门禁全绿**。
// 长度对得上、内容已经错了 —— 这正是附录 HB 开头写的那句话：
// "只比长度"照不到 `SaveBinary_NCHW` / `LoadBinary_NCHW` 之间的不对称。
// 同一课我当天在 HB 补过一次，在这个新门禁里又犯了一次。
// **阳性对照不是走过场，它就是用来抓"我新写的判据太弱"的。**
//
// 返回 0 = 内容对（或差异全部可由清零解释）；非 0 = 有对不上的。
static int cmp_content(const std::string& a, const std::string& b, double thresh,
                       long* ndiff, long* nbad, long* first_bad)
{
    *ndiff = *nbad = 0; *first_bad = -1;
    FILE* fa = fopen(a.c_str(), "rb");
    FILE* fb = fopen(b.c_str(), "rb");
    if (!fa || !fb) {
        if (fa) fclose(fa);
        if (fb) fclose(fb);
        *nbad = 1; *first_bad = -2;
        return 1;
    }
    std::vector<char> ba, bb;
    fseek(fa, 0, SEEK_END); long na = ftell(fa);
    fseek(fb, 0, SEEK_END); long nb = ftell(fb);
    if (na != nb) {
        fclose(fa); fclose(fb);
        *nbad = 1; *first_bad = -2;
        return 1;
    }
    rewind(fa); rewind(fb);
    ba.resize((size_t)na); bb.resize((size_t)nb);
    if (na > 0) {
        if (fread(&ba[0], 1, (size_t)na, fa) != (size_t)na
            || fread(&bb[0], 1, (size_t)nb, fb) != (size_t)nb) {
            fclose(fa); fclose(fb);
            *nbad = 1; *first_bad = -2;
            return 1;
        }
    }
    fclose(fa); fclose(fb);
    long nflo = na / 4;
    int* pa = (int*)&ba[0];
    int* pb = (int*)&bb[0];
    float* va_ = (float*)&ba[0];
    for (long i = 0; i < nflo; i++) {
        if (pa[i] == pb[i]) continue;
        (*ndiff)++;
        if (fabs(va_[i]) >= thresh) {
            (*nbad)++;
            if (*first_bad < 0) *first_bad = i;
        }
    }
    return *nbad == 0 ? 0 : 1;
}

// 取第一个 NCHWC 能加载的模型（有配套权重的），够验往返用了
static bool pick_model(std::string& zp, std::string& mp)
{
    DIR* d = opendir(MODEL_DIR);
    if (!d) return false;
    std::vector<std::string> params;
    struct dirent* e;
    while ((e = readdir(d)) != NULL) {
        std::string n = e->d_name;
        if (n.size() > 9 && n.compare(n.size() - 9, 9, ".zqparams") == 0)
            params.push_back(std::string(MODEL_DIR) + "/" + n);
    }
    closedir(d);
    for (size_t i = 0; i < params.size(); i++) {
        std::string m = params[i].substr(0, params[i].size() - 9) + ".nchwbin";
        if (access(m.c_str(), R_OK) != 0) continue;
        zp = params[i]; mp = m;
        return true;
    }
    return false;
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("NCHWC 张量变体门禁（附录 HD）\n");
    printf("判据：模板宣称支持 NCHWC1 / NCHWC4 / NCHWC8 三种布局，\n");
    printf("      本门禁**逐个变体分别实例化**并走一次 LoadFrom + SaveModel 往返。\n");
    printf("      编不过的变体只报不算（这道门禁存在就是为了把它报出来）。\n\n");

    std::string zp, mp;
    if (!pick_model(zp, mp)) {
        printf("  model/ 下没有带配套 .nchwbin 的 .zqparams —— 门禁不能空跑\n");
        return 1;
    }
    {
        const char* b = strrchr(zp.c_str(), '/');
        printf("  取来试的模型：%s\n", b ? b + 1 : zp.c_str());
    }

    int ok = 0, bad = 0;
    std::string out;

#define TRY_VARIANT(TAG, T4D)                                                   \
    do {                                                                         \
        ZQ::ZQ_CNN_Net_NCHWC<T4D> net;                                           \
        bool loaded = net.LoadFrom(zp, mp);                                      \
        if (!loaded) {                                                           \
            printf("  %-10s **编过了**（实例化成功），但 LoadFrom 失败\n", TAG);  \
            bad++; break;                                                        \
        }                                                                        \
        out = std::string("/tmp/zq_var_") + TAG + ".nchwbin";                    \
        if (!net.SaveModel(out)) {                                               \
            printf("  %-10s **编过了**，但 SaveModel 失败\n", TAG);               \
            bad++; break;                                                        \
        }                                                                        \
        long long got = file_size(out), want = file_size(mp);                   \
        if (got != want) {                                                       \
            printf("  %-10s 往返长度对不上：回存 %lld / 原文件 %lld\n",          \
                   TAG, got, want);                                              \
            bad++; break;                                                        \
        }                                                                        \
        /* 长度对上**不等于**内容对上 —— 逐字节再比一遍。 */                     \
        long ndiff = 0, nbad = 0, first_bad = -1;                                \
        cmp_content(mp, out, 1e-12, &ndiff, &nbad, &first_bad);                 \
        if (nbad != 0) {                                                         \
            printf("  %-10s 往返**内容**对不上：%ld 个 float 无法用"           \
                   " ignore_small_value 解释，首个在 #%ld\n",                     \
                   TAG, nbad, first_bad);                                        \
            bad++; break;                                                        \
        }                                                                        \
        printf("  %-10s OK  实例化 + LoadFrom + SaveModel 往返 %lld 字节%s\n",  \
               TAG, got, ndiff == 0 ? "（逐字节相同）" : "（差异全部可由清零解释）"); \
        remove(out.c_str());                                                     \
        ok++;                                                                    \
    } while (0)

    // **每种变体都要能实例化**。这里用 #ifdef 而不是模板套模板 ——
    // 三种变体在同一个 TU 里同时实例化，链接期若有缺失符号会**一起**失败，
    // 那就分不清是"哪一个变体编不过"了。
#ifdef ZQ_NCHWC_VAR_1
    TRY_VARIANT("NCHWC1", ZQ::ZQ_CNN_Tensor4D_NCHWC1);
#endif
#ifdef ZQ_NCHWC_VAR_4
    TRY_VARIANT("NCHWC4", ZQ::ZQ_CNN_Tensor4D_NCHWC4);
#endif
#ifdef ZQ_NCHWC_VAR_8
    TRY_VARIANT("NCHWC8", ZQ::ZQ_CNN_Tensor4D_NCHWC8);
#endif

    if (ok == 0 && bad == 0) {
        printf("\n  没有任何变体被编译进来（ZQ_NCHWC_VAR_1/4/8 都没定义）"
               " —— 门禁没跑\n");
        return 1;
    }
    printf("\n  本次编译进来的变体：编过并往返正确 %d，有问题 %d\n", ok, bad);
    // 往返不对**判失败**。第一版这里写的是 `return 0`（理由是"NCHWC1/NCHWC8
    // 生产里没人用"）—— 但那样这道门禁就只剩"能编过"一条判据，
    // 而"能编过"正是本仓库反复栽跟头的那根轴（附录 GK：一个从没被编译过的
    // 文件带着一个 SyntaxError，回归全绿）。
    // 生产里没人用**不是**放过的理由 —— AGENTS.md「生产不可达所以不改」
    // 那条的补丁判据是"有没有可对照的正确实现"和"是不是内存安全问题"，
    // 不看可达性。所以这里是硬判据。
    return bad == 0 ? 0 : 1;
}

#endif // ZQ_NCHWC_VARIANTS_BODY_H
