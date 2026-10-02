// zq_facedb2_check.cpp —— ZQ_FaceDatabase / ZQ_FaceDatabaseCompact 的**分析路径**回归
//
// 起因（audit_k3_20261001.md 附录 BL）
// ----------------------------------------
// 附录 BK 覆盖的是 ZQ_FaceDatabaseCompact 的**解析**路径（LoadFromFile）。本文件
// 覆盖另一条同样吃"解析出来的库"、但此前零覆盖的路径：
//
//   解析出来的库 -> Search / SelectSubset / SelectSubsetDesiredNum
//                / DetectLowestPair / DetectRepeatPerson / ExportSimilarityForAllPairs
//
// 输入的**上界**已经被 BK 校验过（dim / person_num / feat_num 都有上界），但那个
// 上界本身撑不撑得住后面的算式，是另一回事。写这个测试时撞出四条真缺陷：
//   BL.1  ZQ_FaceDatabase::ExportSimilarityForAllPairs 是 public 且不校验库是否
//         为空，直接 persons[0].features[0].length -> 空 vector 上取下标。
//         同一文件里 _select_subset / _detect_repeat_person / _detect_lowest_pair
//         三兄弟开头都有 person_num == 0 的检查，只有它没有。
//   BL.2  四处 std::vector<float> scores(cur_num*cur_num) 是 **int** 相乘。
//         加载器允许单人最多 1e7 个特征，cur_num=65536 时 65536*65536 回绕成 0
//         -> vector<float>(0) 分配"成功" -> scores[0]=1 立刻 4 字节堆越界写。
//         ZQ_FaceDatabaseCompact 里同样的式子 cur_num 是 __int64，不回绕，但会
//         变成 17 GB 的分配请求，std::length_error / bad_alloc 直接逃到调用者。
//   BL.3  并行分支用 `#pragma omp critical` 往共享 vector push_back / 往共享 FILE*
//         fwrite，于是同样的输入 + 同样的线程数，**输出文件的内容顺序每次不同**。
//         单线程分支是确定的，所以现状是"并行版比单线程版更不可信"。
//         顺带：ExportSimilarityForAllPairs 的 fwrite 全在 critical 里，
//         所谓并行版实际是"串行 + 全局锁竞争"。
//   BL.4  Search 的四路输出 out_ids / out_scores / out_names / out_filenames
//         长度可以不一致：维度对不上时 ids/names/scores 照推，filenames 被跳过。
//         调用方按 ids 的下标去取 filenames[i] 就是越界。
//         （上一轮修 _find_the_best_matches 的"维度不匹配就跳过"只修了一半。）
//
// 本测试覆盖
//   用例 1 正常库：5 个分析入口 + save/load 往返后 Search 结果不变 + Search 的
//                  四路输出等长
//   用例 2 畸形文件：解析阶段就该干净失败（BL 系列的前置条件）
//   用例 3 空库：BL.1
//   用例 4 cur_num*cur_num 回绕：BL.2（两个库各来一次）
//   用例 5 8 线程 x 8 次：5 份产物逐字节相同，BL.3
//
// 全部在 ASan + LeakSanitizer 下跑。
// 取证用：-DZQ_PROBE_CASE=<n> 只跑第 n 个用例 —— 一次越界就会 abort 掉整个
// 进程，所以"修之前"必须一个用例一个进程地抓。
//   0=全部 1=正常 2=畸形 3=空库 4=回绕 5=确定性

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

#if !defined(_WIN32)
#include <unistd.h>
#endif

#include "zqlib_msvc_shim.h"
#include "ZQlibFaceID/ZQ_FaceDatabase.h"
#include "ZQlibFaceID/ZQ_FaceDatabaseCompact.h"

#ifndef ZQ_PROBE_CASE
#define ZQ_PROBE_CASE 0
#endif

static int g_fail = 0;
static const char* FEATS = "/tmp/zq_facedb2.feat";
static const char* NAMES = "/tmp/zq_facedb2.names";
static const char* OUT = "/tmp/zq_facedb2.out";

#define CASE_ALL          0
#define CASE_NORMAL       1
#define CASE_MALFORMED    2
#define CASE_EMPTY        3
#define CASE_SQUARE_OVF   4
#define CASE_DETERMINISM  5
#define CASE_COMPACT      6

static void check(bool cond, const char* what)
{
    if (!cond) { g_fail++; printf("  FAIL  %s\n", what); }
    else       { printf("  ok    %s\n", what); }
}

// 定义在下面（确定性那一段要用），这里先声明，用例 1 也要用
static std::vector<char> slurp(const char* path);

static void write_file(const char* path, const std::vector<char>& data)
{
    FILE* f = fopen(path, "wb");
    if (!f) { printf("cannot write %s\n", path); exit(2); }
    if (!data.empty())
        fwrite(&data[0], 1, data.size(), f);
    fclose(f);
}

static void put_i32(std::vector<char>& v, int x)
{
    char b[4];
    memcpy(b, &x, 4);
    v.insert(v.end(), b, b + 4);
}

static void put_f32(std::vector<char>& v, float x)
{
    char b[4];
    memcpy(b, &x, 4);
    v.insert(v.end(), b, b + 4);
}

static void put_names(int n)
{
    std::vector<char> v;
    char line[64];
    for (int i = 0; i < n; i++) {
        int len = sprintf(line, "person_%d\n", i);
        v.insert(v.end(), line, line + len);
    }
    write_file(NAMES, v);
}

// ---- 非 compact 的文件格式（ZQ_FaceDatabase::_load_feats_binary） ----
//   int feat_dim / int person_num
//   per person: int feat_num
//   per feat:   int len; len 字节（末字节必须 '\0'）; feat_dim 个 float
static std::vector<char> make_feats_plain(int dim, const std::vector<int>& per,
                                          int name_cap)
{
    std::vector<char> v;
    put_i32(v, dim);
    put_i32(v, (int)per.size());
    for (size_t i = 0; i < per.size(); i++) {
        // 非 compact 的布局是"每个人的 feat_num 紧跟在他自己的特征前面"，
        // 不是先把所有 count 写完再写所有特征（那是 compact 的布局）。
        put_i32(v, per[i]);
        for (int j = 0; j < per[i]; j++) {
            char nm[64];
            sprintf(nm, "img_%d_%d", (int)i, j);
            int len = (int)strlen(nm) + 1;
            if (name_cap > 0 && len > name_cap) len = name_cap;
            put_i32(v, len);
            if (len > 1) v.insert(v.end(), nm, nm + len - 1);
            v.push_back('\0');
            for (int k = 0; k < dim; k++)
                put_f32(v, (float)((i * 31 + j * 7 + k) % 13) * 0.05f);
        }
    }
    return v;
}

// ---- compact 的文件格式（ZQ_FaceDatabaseCompact::LoadFromFile，附录 BK 已验证） ----
//   int feat_dim / int person_num / per person: int face_num
//   然后**所有特征连续**排列，person 0 的在前
static std::vector<char> make_feats_compact(int dim, const std::vector<int>& per)
{
    std::vector<char> v;
    put_i32(v, dim);
    put_i32(v, (int)per.size());
    long long total = 0;
    for (size_t i = 0; i < per.size(); i++) total += per[i];
    for (size_t i = 0; i < per.size(); i++) put_i32(v, per[i]);
    for (long long i = 0; i < total * dim; i++)
        put_f32(v, (float)(i % 13) * 0.05f);
    return v;
}

// ---------------------------------------------------------------- 用例 1：正常路径
static void case_normal()
{
    printf("[case 1] 正常库：5 个分析入口 + save/load 往返 + Search 四路等长\n");
    std::vector<int> per;
    per.push_back(4); per.push_back(3); per.push_back(5);
    per.push_back(2); per.push_back(6);
    write_file(FEATS, make_feats_plain(8, per, 64));
    put_names((int)per.size());

    ZQ::ZQ_FaceDatabase db;
    check(db.LoadFromFileBinay(FEATS, NAMES), "LoadFromFileBinay 成功");

    // --- Search：维度对得上 ---
    ZQ::ZQ_FaceFeature q;
    q.ChangeSize(8);
    for (int k = 0; k < 8; k++) q.pData[k] = (float)(k % 5) * 0.1f;
    std::vector<ZQ::ZQ_FaceFeature> qv(1, q);
    std::vector<int> ids; std::vector<float> scores;
    std::vector<std::string> nms, fns;
    check(db.Search(qv, ids, scores, nms, fns, 3, 1), "Search(dim=8) 返回 true");
    check(!ids.empty() && (int)ids.size() <= 3, "Search 返回 1..3 个 id");
    check(ids.size() == nms.size() && ids.size() == fns.size()
          && ids.size() == scores.size(), "Search 四路输出等长（维度匹配时）");
    for (size_t i = 0; i < ids.size(); i++)
        printf("        #%zu id=%d name=%s file=%s score=%.4f\n", i, ids[i],
               nms[i].c_str(), fns[i].c_str(), scores[i]);

    // --- Search：维度对不上（BL.4） ---
    ZQ::ZQ_FaceFeature q2;
    q2.ChangeSize(4);
    for (int k = 0; k < 4; k++) q2.pData[k] = 0.2f;
    std::vector<int> ids2; std::vector<float> sc2;
    std::vector<std::string> nm2, fn2;
    db.Search(std::vector<ZQ::ZQ_FaceFeature>(1, q2), ids2, sc2, nm2, fn2, 3, 1);
    check(ids2.size() == sc2.size() && ids2.size() == nm2.size()
          && ids2.size() == fn2.size(), "Search 四路输出等长（维度不匹配时，BL.4）");
    check(fns.size() == fn2.size() || ids2.empty(), "维度不匹配不产生半截结果");

    // --- 5 个分析入口 ---
    // BL.5: 单线程分支**根本不碰** same_pair_num / notsame_pair_num,
    // 只有并行分支里有 tmp_same_pair_num。所以 max_thread_num=1 时调用方拿到的是
    // 自己传进去的初值（库里根本没写）。这里故意用 -1/-2 这样的哨兵值，
    // 免得"恰好是 0"看起来像对。
    __int64 all = -1, same = -1, notsame = -1;
    check(db.ExportSimilarityForAllPairs(OUT, "/tmp/zq_facedb2.flag", all, same, notsame, 1, false),
          "ExportSimilarityForAllPairs(单线程) 成功");
    // 4+3+5+2+6 = 20 张脸 -> C(20,2) = 190;
    // 同人对 = C(4,2)+C(3,2)+C(5,2)+C(2,2)+C(6,2) = 6+3+10+1+15 = 35
    check(all == 190, "all_pair_num = C(20,2) = 190");
    check(same == 35, "same_pair_num = 35（单线程分支也要写，BL.5）");
    check(notsame == 190 - 35, "notsame_pair_num = 155（BL.5）");
    check(same + notsame == all, "same + notsame == all");
    printf("        all=%lld same=%lld notsame=%lld\n",
           (long long)all, (long long)same, (long long)notsame);
    // 同样的库, 走并行分支, 三个计数必须一致
    std::vector<char> serial_score = slurp("/tmp/zq_facedb2.flag");
    {
        __int64 a2 = -1, s2 = -1, n2 = -1;
        check(db.ExportSimilarityForAllPairs(OUT, "/tmp/zq_facedb2.pf", a2, s2, n2, 4, false),
              "ExportSimilarityForAllPairs(4 线程) 成功");
        check(a2 == all && s2 == same && n2 == notsame,
              "4 线程与单线程的三个计数一致");
        // 审计修复 2026-10-02（附录 BL.3）的核心断言：并行分支的字节布局必须
        // 与单线程分支**完全一样**。flag 文件一个字节对一个记录，最容易验。
        std::vector<char> par_flag = slurp("/tmp/zq_facedb2.pf");
        check(par_flag == serial_score, "4 线程的 flag 文件与单线程逐字节相同");
    }
    check(db.SelectSubset(OUT, 1, 2, 0.1f), "SelectSubset(单线程) 成功");
    check(db.DetectLowestPair(OUT, 1, 0.9f), "DetectLowestPair(单线程) 成功");
    check(db.DetectRepeatPerson(OUT, 1, 0.1f), "DetectRepeatPerson(单线程) 成功");
    check(db.SelectSubsetDesiredNum(OUT, 2, 2, 3, 1, 0.1f), "SelectSubsetDesiredNum 成功");

    // --- 往返：save -> load -> Search 必须逐项相同 ---
    check(db.SaveToFileBinary(OUT, "/tmp/zq_facedb2.rt.names"), "SaveToFileBinary 成功");
    ZQ::ZQ_FaceDatabase db2;
    check(db2.LoadFromFileBinay(OUT, "/tmp/zq_facedb2.rt.names"), "自己的产物能读回来");
    std::vector<int> ids3; std::vector<float> sc3;
    std::vector<std::string> nm3, fn3;
    db2.Search(qv, ids3, sc3, nm3, fn3, 3, 1);
    check(ids3 == ids && nm3 == nms && fn3 == fns && sc3 == scores,
          "往返后 Search 结果逐项相同");
}

// ---------------------------------------------------------------- 用例 2：畸形输入
static void expect_load_fail(const char* tag, const std::vector<char>& feats, int names_n)
{
    write_file(FEATS, feats);
    put_names(names_n);
    ZQ::ZQ_FaceDatabase db;
    bool r = db.LoadFromFileBinay(FEATS, NAMES);
    if (r) { g_fail++; printf("  FAIL  %-32s 期望加载失败却成功了\n", tag); }
    else    printf("  ok    %-32s 干净失败\n", tag);
}

static void case_malformed()
{
    printf("[case 2] 畸形文件：解析阶段就该干净失败（BL 的前置条件）\n");
    std::vector<int> per;
    per.push_back(2); per.push_back(2);

    // 先确认**同一个生成器**造出来的"干净"文件是能加载的。
    // 不这么做的话，生成器本身写错布局时，下面每一条都会"干净失败"，
    // 看着全绿，其实验的是"随便什么文件都被拒绝"（2026-10-02 实踩过一次：
    // 把 compact 的布局写进了非 compact 的生成器，case 1/5 全挂而 case 2 全过）。
    {
        write_file(FEATS, make_feats_plain(8, per, 64));
        put_names((int)per.size());
        ZQ::ZQ_FaceDatabase db;
        check(db.LoadFromFileBinay(FEATS, NAMES), "基线：同一生成器的干净文件必须能加载");
    }

    { std::vector<char> v; put_i32(v, 0); put_i32(v, 1);
      expect_load_fail("feat_dim=0", v, 1); }
    { std::vector<char> v; put_i32(v, 5000); put_i32(v, 1);
      expect_load_fail("feat_dim=5000(>4096)", v, 1); }
    { std::vector<char> v; put_i32(v, 8); put_i32(v, 0);
      expect_load_fail("person_num=0", v, 0); }
    { std::vector<char> v; put_i32(v, 8); put_i32(v, 20000000);
      expect_load_fail("person_num=2e7", v, 1); }
    { std::vector<char> v = make_feats_plain(8, per, 64); v.resize(v.size() - 4);
      expect_load_fail("特征区被截断", v, 2); }
    { std::vector<char> v; put_i32(v, 8); put_i32(v, 1); put_i32(v, 1);
      put_i32(v, 4);
      v.push_back('a'); v.push_back('b'); v.push_back('c'); v.push_back('d');
      put_f32(v, 0.f);
      expect_load_fail("文件名未以 '\\0' 结尾", v, 1); }
    { std::vector<char> v; put_i32(v, 8); put_i32(v, 1); put_i32(v, 1);
      put_i32(v, 70000);
      expect_load_fail("len=70000(>65536)", v, 1); }
    { std::vector<char> v; put_i32(v, 8); put_i32(v, 1); put_i32(v, 1);
      put_i32(v, -5);
      expect_load_fail("len=-5", v, 1); }
    expect_load_fail("names 人数与 feats 不符", make_feats_plain(8, per, 64), 5);
}

// ---------------------------------------------------------------- 用例 3：空库
static void case_empty()
{
    printf("[case 3] 空库：4 个分析入口必须干净返回 false（BL.1）\n");
    ZQ::ZQ_FaceDatabase db;          // 一个人都没有
    __int64 all = -1, same = -1, notsame = -1;
    check(!db.ExportSimilarityForAllPairs(OUT, "/tmp/zq_facedb2.flag", all, same, notsame, 1, false),
          "空库 ExportSimilarityForAllPairs -> false");
    check(!db.SelectSubset(OUT, 1, 2, 0.1f), "空库 SelectSubset -> false");
    check(!db.DetectLowestPair(OUT, 1, 0.1f), "空库 DetectLowestPair -> false");
    check(!db.DetectRepeatPerson(OUT, 1, 0.1f), "空库 DetectRepeatPerson -> false");

    // Clear() 之后同上（Clear 是 public；加载失败后库里就是空的）
    std::vector<int> per;
    per.push_back(3);
    write_file(FEATS, make_feats_plain(8, per, 64));
    put_names(1);
    ZQ::ZQ_FaceDatabase db3;
    check(db3.LoadFromFileBinay(FEATS, NAMES), "先加载成功");
    db3.Clear();
    all = -1; same = -1; notsame = -1;
    check(!db3.ExportSimilarityForAllPairs(OUT, "/tmp/zq_facedb2.flag", all, same, notsame, 1, false),
          "Clear 之后 ExportSimilarityForAllPairs -> false");
    check(!db3.SelectSubset(OUT, 1, 2, 0.1f), "Clear 之后 SelectSubset -> false");
    check(!db3.DetectLowestPair(OUT, 1, 0.1f), "Clear 之后 DetectLowestPair -> false");
    check(!db3.DetectRepeatPerson(OUT, 1, 0.1f), "Clear 之后 DetectRepeatPerson -> false");
}

// ---------------------------------------------------------------- 用例 4：cur_num*cur_num 回绕
static void case_square_ovf()
{
    printf("[case 4] 单人 65536 个特征：cur_num*cur_num 回绕（BL.2）\n");
    const int N = 65536;            // 65536*65536 在 int 里正好回绕成 0
    std::vector<int> per;
    per.push_back(N);

    // --- 非 compact：int cur_num -> 回绕 -> 堆越界写 ---
    write_file(FEATS, make_feats_plain(1, per, 1));
    put_names(1);
    ZQ::ZQ_FaceDatabase db;
    check(db.LoadFromFileBinay(FEATS, NAMES), "65536 x 1 维的库加载成功");
    // 修之前: std::vector<float> scores(65536*65536) -> 0 个元素
    //          -> scores[0] = 1 是 4 字节堆越界写，ASan 立刻 abort
    check(!db.SelectSubset(OUT, 1, 1, 0.1f), "SelectSubset 对回绕规模干净返回 false");
    check(!db.DetectRepeatPerson(OUT, 1, 0.1f), "DetectRepeatPerson 对回绕规模干净返回 false");
    // 这里**故意不去**验 ExportSimilarityForAllPairs：它不建 cur_num^2 矩阵，
    // 所以确实不受 BL.2 影响，但它是 O(N^2) —— 65536 张脸就是 2.1e9 次点积，
    // ASan 下要跑好几分钟，放在门禁里不合适。它在用例 1 里已经验过了。

    // --- compact：__int64 cur_num -> 不回绕，但 65536^2*4B = 17 GB 分配请求 ---
    write_file(FEATS, make_feats_compact(1, per));
    put_names(1);
    ZQ::ZQ_FaceDatabaseCompact cdb;
    check(cdb.LoadFromFile(FEATS, NAMES), "compact 版 65536 x 1 维的库加载成功");
    // 修之前: std::vector<float> scores((__int64)65536*65536) -> 17 GB
    //          -> length_error / bad_alloc 逃到调用者
    check(!cdb.DetectRepeatPerson(OUT, 8, 0.1f), "compact DetectRepeatPerson 对回绕规模干净返回 false");
}

// ---------------------------------------------------------------- 用例 5：确定性
static std::vector<char> slurp(const char* path)
{
    std::vector<char> v;
    FILE* f = fopen(path, "rb");
    if (!f) return v;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n > 0) {
        v.resize(n);
        if ((long)fread(&v[0], 1, n, f) != n) v.clear();
    }
    fclose(f);
    return v;
}

static void case_determinism()
{
    printf("[case 5] 8 线程 x 8 次：5 份产物必须逐字节相同（BL.3）\n");
    // 人数必须**超过** DetectLowestPair / ExportSimilarityForAllPairs 用的
    // chunk_size(=100)，否则 parallel for 只有一个 chunk、全落到同一个线程上，
    // 于是"碰巧是确定的"——第一版就是 40 个人，四个入口里两个假绿。
    std::vector<int> per;
    for (int i = 0; i < 300; i++) per.push_back(3);
    write_file(FEATS, make_feats_plain(8, per, 64));
    put_names((int)per.size());
    ZQ::ZQ_FaceDatabase db;
    if (!db.LoadFromFileBinay(FEATS, NAMES)) { g_fail++; printf("  FAIL  加载失败\n"); return; }

    const int ROUND = 8;
    std::vector<char> b_sub, b_low, b_rep, b_score, b_flag;
    __int64 all = 0, same = 0, notsame = 0;
    bool score_same = true, low_same = true, rep_same = true, sub_same = true;

    // 库里的 ExportSimilarityForAllPairs / DetectLowestPair 每处理一个人就
    // printf 一行进度，300 人 x 8 轮 = 几千行噪声，会把门禁的日志淹掉。
    // 把 stdout 临时接到 /dev/null（ASan 的报告走 stderr，不受影响），
    // 跑完再接回来打印结论。
#if !defined(_WIN32)
    fflush(stdout);
    int saved_fd = dup(fileno(stdout));
    FILE* devnull = fopen("/dev/null", "w");
    if (devnull) dup2(fileno(devnull), fileno(stdout));
#endif
    for (int r = 0; r < ROUND; r++) {
        char fs[160], fc[160], fl[160], fp[160], fq[160];
        sprintf(fs, "%s.SCORE.%d", OUT, r);
        sprintf(fc, "%s.FLAG.%d",  OUT, r);
        sprintf(fl, "%s.LOW.%d",   OUT, r);
        sprintf(fp, "%s.REP.%d",   OUT, r);
        sprintf(fq, "%s.SUB.%d",   OUT, r);
        db.ExportSimilarityForAllPairs(fs, fc, all, same, notsame, 8, false);
        db.DetectLowestPair(fl, 8, 0.9f);
        db.DetectRepeatPerson(fp, 8, 0.1f);
        db.SelectSubset(fq, 8, 1, 0.1f);
        if (r == 0) {
            b_score = slurp(fs); b_flag = slurp(fc);
            b_low = slurp(fl); b_rep = slurp(fp); b_sub = slurp(fq);
        } else {
            if (slurp(fs) != b_score || slurp(fc) != b_flag) score_same = false;
            if (slurp(fl) != b_low)  low_same = false;
            if (slurp(fp) != b_rep)  rep_same = false;
            if (slurp(fq) != b_sub)  sub_same = false;
        }
    }
    // 再加一条最强的：8 线程的产物必须与**单线程**的产物逐字节相同。
    // 只比"8 轮之间一致"是不够的 —— 那只能说明它稳定，不能说明它对。
    std::vector<char> s_score, s_flag, s_low, s_rep, s_sub;
    {
        __int64 a1 = -1, m1 = -1, n1 = -1;
        char ts[160], tf[160], tl[160], tp[160], tq[160];
        sprintf(ts, "%s.S1SCORE", OUT);
        sprintf(tf, "%s.S1FLAG",  OUT);
        sprintf(tl, "%s.S1LOW",   OUT);
        sprintf(tp, "%s.S1REP",   OUT);
        sprintf(tq, "%s.S1SUB",   OUT);
        db.ExportSimilarityForAllPairs(ts, tf, a1, m1, n1, 1, false);
        db.DetectLowestPair(tl, 1, 0.9f);
        db.DetectRepeatPerson(tp, 1, 0.1f);
        db.SelectSubset(tq, 1, 1, 0.1f);
        s_score = slurp(ts); s_flag = slurp(tf);
        s_low = slurp(tl); s_rep = slurp(tp); s_sub = slurp(tq);
    }
#if !defined(_WIN32)
    fflush(stdout);
    if (devnull) { fclose(devnull); dup2(saved_fd, fileno(stdout)); }
    close(saved_fd);
#endif
    printf("        all=%lld same=%lld notsame=%lld; 产物字节数 sub=%d low=%d rep=%d score=%d flag=%d\n",
           (long long)all, (long long)same, (long long)notsame,
           (int)b_sub.size(), (int)b_low.size(), (int)b_rep.size(),
           (int)b_score.size(), (int)b_flag.size());
    check(!b_score.empty(), "ExportSimilarityForAllPairs 产出了非空文件");
    check(score_same, "ExportSimilarityForAllPairs 8 线程 x 8 次字节相同");
    check(low_same,   "DetectLowestPair        8 线程 x 8 次字节相同");
    check(rep_same,   "DetectRepeatPerson     8 线程 x 8 次字节相同");
    check(sub_same,   "SelectSubset           8 线程 x 8 次字节相同");
    check(b_score == s_score, "ExportSimilarityForAllPairs 的 score 文件：8 线程 == 单线程");
    check(b_flag  == s_flag,  "ExportSimilarityForAllPairs 的 flag  文件：8 线程 == 单线程");
    check(b_low   == s_low,   "DetectLowestPair    8 线程 == 单线程");
    check(b_rep   == s_rep,   "DetectRepeatPerson  8 线程 == 单线程");
    check(b_sub   == s_sub,   "SelectSubset        8 线程 == 单线程");
}

// ---------------------------------------------------------------- 用例 6：compact 库
// 附录 BL 的修法在 ZQ_FaceDatabaseCompact.h 上是**同一处改动**，所以也必须测。
// "同一个修法只改了一半"就是附录 BG 栽过的跟头（NCHW 修了 NCHWC 忘了）。
static void case_compact()
{
    printf("[case 6] ZQ_FaceDatabaseCompact：同样两条 BL.3 + BL.2\n");
    const int P = 300;
    std::vector<int> per;
    for (int i = 0; i < P; i++) per.push_back(3);
    write_file(FEATS, make_feats_compact(8, per));
    put_names(P);
    ZQ::ZQ_FaceDatabaseCompact cdb;
    if (!cdb.LoadFromFile(FEATS, NAMES)) { g_fail++; printf("  FAIL  compact 加载失败\n"); return; }
    check(true, "compact 库加载成功（300 人 x 3 脸 x 8 维）");

    // ---- BL.2：回绕规模干净返回 false 而不是 17 GB 的 bad_alloc ----
    {
        std::vector<int> big;
        big.push_back(65536);
        write_file(FEATS, make_feats_compact(1, big));
        put_names(1);
        ZQ::ZQ_FaceDatabaseCompact c2;
        check(c2.LoadFromFile(FEATS, NAMES), "compact 65536 x 1 维的库加载成功");
        check(!c2.DetectRepeatPerson(OUT, 8, 0.1f), "compact DetectRepeatPerson 对回绕规模干净返回 false");
    }

    // ---- BL.3：8 线程的产物必须与单线程逐字节相同 ----
    __int64 a1 = -1, s1 = -1, n1 = -1, a8 = -1, s8 = -1, n8 = -1;
    char q1s[160], q1f[160], q8s[160], q8f[160], qr1[160], qr8[160];
    sprintf(q1s, "%s.C1SCORE", OUT);
    sprintf(q1f, "%s.C1FLAG",  OUT);
    sprintf(q8s, "%s.C8SCORE", OUT);
    sprintf(q8f, "%s.C8FLAG",  OUT);
    sprintf(qr1, "%s.C1REP",   OUT);
    sprintf(qr8, "%s.C8REP",   OUT);
#if !defined(_WIN32)
    fflush(stdout);
    int sv = dup(fileno(stdout));
    FILE* dn = fopen("/dev/null", "w");
    if (dn) dup2(fileno(dn), fileno(stdout));
#endif
    cdb.ExportSimilarityForAllPairs(q1s, q1f, a1, s1, n1, 1, false);
    cdb.DetectRepeatPerson(qr1, 1, 0.1f, true);
    cdb.ExportSimilarityForAllPairs(q8s, q8f, a8, s8, n8, 8, false);
    cdb.DetectRepeatPerson(qr8, 8, 0.1f, true);
#if !defined(_WIN32)
    fflush(stdout);
    if (dn) { fclose(dn); dup2(sv, fileno(stdout)); }
    close(sv);
#endif
    // 900 张脸 -> C(900,2) = 404550; 同人对 = 300 * C(3,2) = 900
    check(a1 == 404550 && a8 == 404550, "compact all_pair_num = C(900,2) = 404550（单/多线程一致）");
    check(s1 == 900 && s8 == 900, "compact same_pair_num = 900（单/多线程一致）");
    check(n1 == 404550 - 900 && n8 == 404550 - 900, "compact notsame_pair_num 一致");
    check(slurp(q1s) == slurp(q8s), "compact score 文件：8 线程 == 单线程");
    check(slurp(q1f) == slurp(q8f), "compact flag  文件：8 线程 == 单线程");
    check(slurp(qr1) == slurp(qr8), "compact DetectRepeatPerson：8 线程 == 单线程");

    // 空库：compact 版的 _clear() 之后 person_num=0，_detect_repeat_person
    // 不像 _find_the_best_matches 那样查 person_num<=0（见附录 BL.7 的加固）
    ZQ::ZQ_FaceDatabaseCompact c3;
    check(!c3.ExportSimilarityForAllPairs(q1s, q1f, a1, s1, n1, 1, false),
          "compact 空库 ExportSimilarityForAllPairs -> false");
    check(!c3.DetectRepeatPerson(qr1, 1, 0.1f, true), "compact 空库 DetectRepeatPerson -> false");
}

int main()
{
    // 越界时 ASan 直接 abort，stdout 是块缓冲的 -> 崩溃前的输出全丢。
    // 上面那些 check 的结果正是崩溃时要看的证据，所以必须不缓冲。
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("ZQ_FaceDatabase 家族 分析路径 回归（附录 BL）\n");    printf("ASan 会在越界时 abort；畸形/极端输入必须干净返回 false\n");
    if (ZQ_PROBE_CASE != CASE_ALL)
        printf("（只跑用例 %d）\n", ZQ_PROBE_CASE);
    printf("\n");

    if (ZQ_PROBE_CASE == CASE_ALL || ZQ_PROBE_CASE == CASE_NORMAL)      case_normal();
    if (ZQ_PROBE_CASE == CASE_ALL || ZQ_PROBE_CASE == CASE_MALFORMED)   case_malformed();
    if (ZQ_PROBE_CASE == CASE_ALL || ZQ_PROBE_CASE == CASE_EMPTY)       case_empty();
    if (ZQ_PROBE_CASE == CASE_ALL || ZQ_PROBE_CASE == CASE_SQUARE_OVF)  case_square_ovf();
    if (ZQ_PROBE_CASE == CASE_ALL || ZQ_PROBE_CASE == CASE_DETERMINISM) case_determinism();
    if (ZQ_PROBE_CASE == CASE_ALL || ZQ_PROBE_CASE == CASE_COMPACT)     case_compact();

    printf("\n%s (g_fail = %d)\n", g_fail ? "FAILED" : "PASSED", g_fail);
    return g_fail ? 1 : 0;
}
