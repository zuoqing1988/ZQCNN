/* `ZQlibFaceID` 的**文件读入**行为门禁 —— 附录 EI
 *
 * 为什么是这两个类
 * ----------------
 * 附录 EH 查完：`ZQlibFaceID` 29 个头里 26 个零门禁覆盖，且这个目录
 * **既不在两个构建里、也几乎没有任何行为门禁**。
 * 附录 EH 修完两个"Linux 上编不过"的头之后，22 个头能在 Linux 上编过 ——
 * 但**"能编过"不等于"能链接"**：多数头 include 了 OpenCV / ncnn / SeetaFace，
 * 本机没有它们的 Linux 库。
 *
 * `ZQ_FaceGroup.h` / `ZQ_FaceSearchTarget.h` 是**例外**：
 * 它们的 include 只有 `ZQ_FaceFeature.h` / `ZQ_CNN_BBox.h` / `<vector>` / `<stdio.h>`，
 * **不需要任何外部库**，所以这是 `ZQlibFaceID` 里**唯一能真正在 Linux 上跑行为门禁**的地方。
 *
 * 而它们恰好是"人脸库文件不可信"威胁模型下的**文件解析入口**
 * （报告开头的威胁模型把 `.imgfeat` 之类列为不可信输入）。
 *
 * 判据
 * ----
 * 1. **往返一致**：`WriteToFile` 写出的内容，`LoadFromFile` 读回来必须逐字段相同
 *    （feat_dim、特征向量、bbox 数组、with_box 标志）
 * 2. **恶意 `feat_dim` / `num` 必须被拒**：两个类的守卫分别是
 *    `0 <= feat_dim < 65535` 与 `0 <= num < 1000000`，用例要把**边界内外**都打一遍
 * 3. **截断文件必须被拒**（fread 返回值逐个比对）
 * 4. **失败时文件流位置必须回到调用前** —— 这是 `ZQ_FaceGroup::LoadFromFile` /
 *    `WriteToFile` 开头记 `pos = ftell(...)`、失败时 `fseek(pos)` 的**明确契约**，
 *    调用方靠它才能在一个流里连续解析多个 group。
 *    **这一条只有测试钉得住**：读代码看到 `fseek` 不等于它在所有失败路径上都执行了。
 * 5. `num == 0` 的空组必须**成功**（边界，不是越界）
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
#include "ZQlibFaceID/ZQ_FaceGroup.h"
#include "ZQlibFaceID/ZQ_FaceSearchTarget.h"

using namespace ZQ;

#define RES_FILE "/tmp/zq_facegroup_res.txt"
#define TMP_PATH "/tmp/zq_facegroup_case.bin"

// note: 0 无事 / 1 搭建失败 / 2 往返不一致 / 3 该拒却收下 / 4 该收却拒了
//       / 5 **失败后文件流位置没回退** / 6 数据错
static const char* g_note[] = {
    "", "搭建失败", "**往返不一致**", "**该拒却收下了**",
    "**该收却拒了**", "**失败后文件流位置没回退**", "**数据错**"
};

enum {
    OP_RT_BOX = 0, OP_RT_NOBOX, OP_RT_EMPTY,
    OP_BAD_FEATDIM_LO, OP_BAD_FEATDIM_HI, OP_BAD_FEATDIM_FAR,
    OP_BAD_NUM_LO, OP_BAD_NUM_HI, OP_BAD_NUM_FAR,
    OP_TRUNC_FEAT, OP_TRUNC_NUM,
    OP_POS_RESTORE,
    // ---- ZQ_FaceSearchTarget 那一层（附录 EI.6 点名的缺口）----
    OP_ST_RT, OP_ST_EMPTY, OP_ST_BADNUM_LO, OP_ST_BADNUM_HI, OP_ST_BADNUM_FAR,
    OP_ST_TRUNC, OP_ST_NOFILE
};

static void wr_int(FILE* f, int v) { fwrite(&v, sizeof(int), 1, f); }

// 造一个内容可预测的 group
static void fill_group(ZQ_FaceGroup& g, int nfeat, int dim, bool with_box, int seed)
{
    g.feat_dim = dim;
    g.face_feats.clear();
    g.face_boxes.clear();
    for (int i = 0; i < nfeat; i++)
    {
        ZQ_FaceFeature f;
        f.length = dim;
        f.pData = (float*)malloc(sizeof(float) * (dim > 0 ? dim : 1));
        for (int k = 0; k < dim; k++)
            f.pData[k] = (float)(seed * 1000 + i * 10 + k) * 0.5f + 1.0f;
        g.face_feats.push_back(f);
        // **只有 with_box 才填 bbox**。WithoutBox 的 WriteToFile/LoadFromFile
        // 根本不走 box 那一段，填了也不会被序列化 ——
        // 我第一版无条件填，往返比对时 face_boxes  sizes 不等，
        // 报成"往返不一致"，看起来像被测代码的缺陷。
        if (with_box)
        {
            ZQ_CNN_BBox b;
            memset(&b, 0, sizeof(b));
            b.score = (float)(seed + i) * 0.25f;
            b.row1 = i + seed; b.col1 = i + seed + 1;
            b.row2 = i + seed + 2; b.col2 = i + seed + 3;
            b.exist = true;
            g.face_boxes.push_back(b);
        }
    }
}

// **只 clear，不手动 free**。`ZQ_FaceFeature` 有析构函数（ZQ_FaceFeature.h:33-36）
// 会 `free(pData)`；我第一版在 free_group 里又 free 了一遍 —— **双 free**，
// 往返用例的子进程直接崩掉，父进程只看到"结果文件读不出来"。
// 又一次"门禁坏了长得像代码坏了"。
static void free_group(ZQ_FaceGroup& g)
{
    g.face_feats.clear();     // 让 ZQ_FaceFeature 的析构去 free
    g.face_boxes.clear();
}

static bool same_group(const ZQ_FaceGroup& a, const ZQ_FaceGroup& b)
{
    if (a.feat_dim != b.feat_dim) return false;
    if (a.face_feats.size() != b.face_feats.size()) return false;
    if (a.face_boxes.size() != b.face_boxes.size()) return false;
    for (size_t i = 0; i < a.face_feats.size(); i++)
    {
        if (a.face_feats[i].length != b.face_feats[i].length) return false;
        for (int k = 0; k < a.face_feats[i].length; k++)
            if (a.face_feats[i].pData[k] != b.face_feats[i].pData[k]) return false;
    }
    for (size_t i = 0; i < a.face_boxes.size(); i++)
    {
        const ZQ_CNN_BBox& x = a.face_boxes[i];
        const ZQ_CNN_BBox& y = b.face_boxes[i];
        if (x.score != y.score || x.row1 != y.row1 || x.col1 != y.col1
            || x.row2 != y.row2 || x.col2 != y.col2 || x.exist != y.exist) return false;
    }
    return true;
}

static void run_case(int op)
{
    long bad = 0; int note = 0;
    ZQ_FaceGroupWithBox gb;
    ZQ_FaceGroupWithoutBox gn;

    // ---- 判据 4 的载体：先在文件里塞一段"前置内容"，再让 group 解析失败，
    //      看 ftell 是否回到 group 起始位置 ----
    const int PAD = 17;

    FILE* f = fopen(TMP_PATH, "wb");
    if (!f) { note = 1; bad++; }
    else
    {
        if (op == OP_RT_BOX || op == OP_RT_NOBOX || op == OP_RT_EMPTY)
        {
            // 往返：写 -> 读 -> 逐字段比对
            const bool box = (op == OP_RT_BOX);
            const int nf = (op == OP_RT_EMPTY) ? 0 : 3;
            const int dim = (op == OP_RT_EMPTY) ? 0 : 5;
            ZQ_FaceGroup& src = box ? (ZQ_FaceGroup&)gb : (ZQ_FaceGroup&)gn;
            fill_group(src, nf, dim, box, 1);
            fclose(f);
            // ZQ_FaceGroup 的接口是 WriteToFile(FILE*) / LoadFromFile(FILE*)，
            // **没有**带路径的 SaveToFile/LoadFromFile（那是 ZQ_FaceSearchTarget 才有的）。
            f = fopen(TMP_PATH, "wb");
            if (!f) { note = 1; bad++; }
            else
            {
                const bool w = src.WriteToFile(f);
                fclose(f);
                if (!w) { note = 1; bad++; }
                else
                {
                    FILE* in = fopen(TMP_PATH, "rb");
                    if (!in) { note = 1; bad++; }
                    else
                    {
                        ZQ_FaceGroupWithBox gb2; ZQ_FaceGroupWithoutBox gn2;
                        const bool r = box ? gb2.LoadFromFile(in) : gn2.LoadFromFile(in);
                        fclose(in);
                        if (!r) { note = 4; bad++; }
                        else if (!same_group(src, box ? (ZQ_FaceGroup&)gb2 : (ZQ_FaceGroup&)gn2)) {
                            note = 2; bad++;
                        }
                        free_group((ZQ_FaceGroup&)gb2);
                        free_group((ZQ_FaceGroup&)gn2);
                    }
                }
            }
        }
        else if (op == OP_POS_RESTORE)
        {
            // 写 PAD 字节前置内容 + 一个**必然解析失败**的 group 头
            for (int i = 0; i < PAD; i++) fputc(0x5A, f);
            wr_int(f, -7);            // feat_dim = -7  -> 必然被拒
            fclose(f);

            f = fopen(TMP_PATH, "rb");
            if (!f) { note = 1; bad++; }
            else
            {
                for (int i = 0; i < PAD; i++) fgetc(f);   // 跳过前置内容
                const long pos_before = ftell(f);
                ZQ_FaceGroupWithoutBox g;
                const bool r = g.LoadFromFile(f);
                const long pos_after = ftell(f);
                if (r) { note = 3; bad++; }                  // 应当被拒却收下
                else if (pos_after != pos_before) { note = 5; bad++; }  // **位置没回退**
                free_group((ZQ_FaceGroup&)g);
                fclose(f);
            }
        }
        else if (op >= OP_ST_RT)
        {
            // ---- ZQ_FaceSearchTarget 那一层（附录 EI.6 点名的缺口）----
            // 它的守卫是 `num < 0 || num > 1000000`（**与 ZQ_FaceGroup 的
            // `num < 1000000` 差一**：这里是 `<=`，所以 1000000 在界内）。
            // 这条差异第一版我按 FaceGroup 的口径写期望，又是一次"把实现当规格"，
            // 这次是**反过来把其中一份的边界套到另一份上** ——
            // 两份实现的边界必须分别读。
            // **不要在这里 fclose(f)**：下面 OP_ST_BADNUM_* 还要往 f 里写 num。
            // 第一版在分支开头就 fclose(f)，于是 wr_int(f, num) 往**已关闭的 FILE***
            // 写 -> 子进程崩 -> 父进程只看到"结果文件读不出来"（又一次门禁自己的错）。
            ZQ_FaceSearchTarget st;
            if (op == OP_ST_NOFILE)
            {
                fclose(f);
                if (st.LoadFromFile("/tmp/zq_definitely_no_such_file.bin")) { note = 3; bad++; }
            }
            else if (op == OP_ST_RT || op == OP_ST_EMPTY)
            {
                fclose(f);
                const int n = (op == OP_ST_EMPTY) ? 0 : 3;
                for (int i = 0; i < n; i++)
                {
                    ZQ_FaceGroupWithoutBox g;
                    fill_group((ZQ_FaceGroup&)g, 2, 4, false, i + 1);
                    st.targets.push_back(g);
                }
                if (!st.SaveToFile(TMP_PATH)) { note = 1; bad++; }
                else
                {
                    ZQ_FaceSearchTarget st2;
                    if (!st2.LoadFromFile(TMP_PATH)) { note = 4; bad++; }
                    else if (st2.targets.size() != st.targets.size()) { note = 2; bad++; }
                    else
                        for (size_t i = 0; i < st.targets.size(); i++)
                            if (!same_group((ZQ_FaceGroup&)st.targets[i],
                                            (ZQ_FaceGroup&)st2.targets[i])) { note = 2; bad++; break; }
                    for (size_t i = 0; i < st2.targets.size(); i++)
                        free_group((ZQ_FaceGroup&)st2.targets[i]);
                }
                for (size_t i = 0; i < st.targets.size(); i++)
                    free_group((ZQ_FaceGroup&)st.targets[i]);
            }
            else
            {
                // 手写 num（后接内容或不接），验证 num 守卫
                int num = 2;
                if (op == OP_ST_BADNUM_LO)  num = -1;
                else if (op == OP_ST_BADNUM_HI) num = 1000001;    // 守卫是 > 1000000 才拒
                else if (op == OP_ST_BADNUM_FAR) num = 0x7FFFFFFF;
                wr_int(f, num);
                if (op == OP_ST_TRUNC)
                {
                    // 先用真实的 SaveToFile 写出一个 num=1 的文件，
                    // 再把开头的 num 改成 2 —— 模拟"头说 2、实际只有 1 个"。
                    // （ZQ_FaceSearchTarget::SaveToFile 收的是**路径**不是 FILE*，
                    //   第一版传了 FILE*，编译直接报错。）
                    fclose(f);
                    ZQ_FaceGroupWithoutBox g;
                    fill_group((ZQ_FaceGroup&)g, 1, 3, false, 1);
                    ZQ_FaceSearchTarget w;
                    w.targets.push_back(g);
                    if (!w.SaveToFile(TMP_PATH)) { note = 1; bad++; }
                    else
                    {
                        FILE* rp = fopen(TMP_PATH, "r+b");
                        if (rp) { fseek(rp, 0, SEEK_SET); wr_int(rp, 2); fclose(rp); }
                        ZQ_FaceSearchTarget rd;
                        if (rd.LoadFromFile(TMP_PATH)) { note = 3; bad++; }
                        else if (!rd.targets.empty()) { note = 6; bad++; }   // 失败后必须清空
                        for (size_t i = 0; i < rd.targets.size(); i++)
                            free_group((ZQ_FaceGroup&)rd.targets[i]);
                    }
                    for (size_t i = 0; i < w.targets.size(); i++)
                        free_group((ZQ_FaceGroup&)w.targets[i]);
                    free_group((ZQ_FaceGroup&)g);
                    remove(TMP_PATH);
                }
                else
                {
                    fclose(f);
                    ZQ_FaceSearchTarget rd;
                    if (rd.LoadFromFile(TMP_PATH)) { note = 3; bad++; }
                    for (size_t i = 0; i < rd.targets.size(); i++)
                        free_group((ZQ_FaceGroup&)rd.targets[i]);
                    remove(TMP_PATH);
                }
            }
        }
        else
        {
            // 恶意参数 / 截断：直接手写文件内容
            int feat_dim = 5, num = 2, with_box = 1;
            int handled = 0;   // 截断分支自己已经收尾，别再落进下面的通用块
            switch (op)
            {
                case OP_BAD_FEATDIM_LO:  feat_dim = -1;      break;
                case OP_BAD_FEATDIM_HI:  feat_dim = 65535;   break;   // 守卫是 < 65535
                case OP_BAD_FEATDIM_FAR: feat_dim = 100000;  break;
                case OP_BAD_NUM_LO:      num = -1;           break;
                case OP_BAD_NUM_HI:      num = 1000000;     break;   // 守卫是 > 1000000 才拒
                case OP_BAD_NUM_FAR:     num = 0x7FFFFFFF;  break;
                case OP_TRUNC_FEAT:      num = 2; feat_dim = 5;
                                          for (int i = 0; i < num; i++)
                                              for (int k = 0; k < feat_dim; k++) { float v = 1.f; fwrite(&v, 4, 1, f); }
                                          fclose(f);        // **只写特征，不写 num/with_box/box**
                                          f = fopen(TMP_PATH, "rb");
                                          if (!f) { note = 1; bad++; }
                                          else {
                                              ZQ_FaceGroupWithBox g;
                                              const bool r = g.LoadFromFile(f);
                                              if (r) { note = 3; bad++; }
                                              free_group((ZQ_FaceGroup&)g);
                                              fclose(f);
                                          }
                                          handled = 1; break;
                case OP_TRUNC_NUM:      num = 2; feat_dim = 5; with_box = 0;
                                          wr_int(f, feat_dim);
                                          // **必须是 sizeof(bool)（1 字节）** ——
                                          // ZQ_FaceGroup 的 WriteToFile/LoadFromFile
                                          // 都是按 sizeof(bool) 写/读 with_box 的。
                                          // 我第一版写成 sizeof(int)，文件头错位，
                                          // 后面读出来的 feat_dim 全是垃圾
                                          //（日志里 feat_dim = 318899360 就是这个）。
                                          fwrite(&with_box, sizeof(bool), 1, f);
                                          for (int i = 0; i < num - 1; i++)      // **少写一个**
                                              for (int k = 0; k < feat_dim; k++) { float v = 1.f; fwrite(&v, 4, 1, f); }
                                          fclose(f);
                                          f = fopen(TMP_PATH, "rb");
                                          if (!f) { note = 1; bad++; }
                                          else {
                                              ZQ_FaceGroupWithoutBox g;
                                              const bool r = g.LoadFromFile(f);
                                              if (r) { note = 3; bad++; }
                                              free_group((ZQ_FaceGroup&)g);
                                              fclose(f);
                                          }
                                          handled = 1; break;
                default: break;
            }
            if (note != 1 && bad == 0 && handled == 0)
            {
                // 上面没提前 fclose 的分支：写完头部后关闭
                fclose(f);
                f = fopen(TMP_PATH, "rb");
                if (!f) { note = 1; bad++; }
                else
                {
                    ZQ_FaceGroupWithBox g;
                    const bool r = g.LoadFromFile(f);
                    // 守卫是 `num >= 0 && num < 1000000`（ZQ_FaceGroup.h:45），
                    // 所以 **num = 1000000 应当被拒**。
                    // 我第一版把它标成"恰好在界内、应当被收下" —— **期望反了**，
                    // 差点把一处正确的守卫报成缺陷。
                    const bool should_reject = true;
                    if (should_reject && r) { note = 3; bad++; }
                    if (!should_reject && !r) { note = 4; bad++; }
                    free_group((ZQ_FaceGroup&)g);
                    fclose(f);
                }
            }
        }
    }
    free_group((ZQ_FaceGroup&)gb);
    free_group((ZQ_FaceGroup&)gn);
    remove(TMP_PATH);
    FILE* o = fopen(RES_FILE, "w");
    if (o) { fprintf(o, "%ld %ld %d\n", 1L - bad, bad, note); fclose(o); }
}

static int g_case = 0, g_ok = 0, g_bad = 0, g_crash = 0;

static void one(int op)
{
    g_case++;
    remove(RES_FILE);
    pid_t pid = fork();
    if (pid == 0) { zq_child_silence_stderr(); run_case(op); _exit(0); }
    int st = 0; waitpid(pid, &st, 0);
    long ok = 0, bad = 0; int note = 0, have = 0;
    FILE* f = fopen(RES_FILE, "r");
    if (f) { have = (fscanf(f, "%ld %ld %d", &ok, &bad, &note) == 3); fclose(f); }
    static const char* nm[] = {
        "往返(WithBox)", "往返(WithoutBox)", "往返(num=0 空组)",
        "feat_dim=-1", "feat_dim=65535(界外)", "feat_dim=100000",
        "num=-1", "num=1000000(界外)", "num=0x7FFFFFFF",
        "截断: 特征齐但缺 num", "截断: 少一个特征",
        "失败后位置回退",                                   // OP_POS_RESTORE
        "ST往返(3 个 target)", "ST往返(num=0 空表)",        // SearchTarget
        "ST num=-1", "ST num=1000001(界外)", "ST num=0x7FFFFFFF",
        "ST 截断: 头说2实际1", "ST 文件不存在"
    };
    const char* tag = (op >= 0 && op < (int)(sizeof(nm)/sizeof(nm[0]))) ? nm[op] : "?";
    if (!have || WIFSIGNALED(st)) {
        g_crash++;
        printf("  %-28s  没跑完%s\n", tag, WIFSIGNALED(st) ? "（信号）" : "（结果文件读不出来）");
        return;
    }
    if (bad > 0) { g_bad++; printf("  %-28s  %s\n", tag, g_note[note]); }
    else g_ok++;
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("ZQlibFaceID 文件读入门禁（附录 EI）\n");
    printf("为什么是这两个类：EH 修完之后 22 个头能在 Linux 上编过，但多数 include 了\n");
    printf("  OpenCV / ncnn / SeetaFace（本机没有 Linux 库）**链不过**；\n");
    printf("  ZQ_FaceGroup / ZQ_FaceSearchTarget 只依赖 ZQ_FaceFeature / ZQ_CNN_BBox / <vector>，\n");
    printf("  **不需要任何外部库** —— 是 ZQlibFaceID 里唯一能真正跑行为门禁的地方，\n");
    printf("  而它们正是\"人脸库文件不可信\"威胁模型下的解析入口。\n");
    printf("判据：① 往返一致 ② 恶意 feat_dim/num 被拒（边界内外都打）③ 截断被拒\n");
    printf("      ④ **失败时文件流位置必须回退**（只有测试钉得住这一条）⑤ num=0 空组必须成功\n\n");
    for (int op = 0; op <= OP_ST_NOFILE; op++) one(op);
    printf("\n共 %d 个用例：全对 %d，有错 %d，崩溃/搭建失败 %d\n", g_case, g_ok, g_bad, g_crash);
    if (g_bad || g_crash)
        printf("**每一项在下结论之前都要先用独立复现对一遍**（附录 CA.3）。\n");
    return (g_bad || g_crash) ? 1 : 0;
}
