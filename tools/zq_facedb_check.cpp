// zq_facedb_check.cpp —— ZQ_FaceDatabaseCompact 解析**不可信人脸库文件**的回归测试
//
// 起因（audit_k3_20261001.md 附录 BK）
// -------------------------------------
// 本报告的威胁模型把**人脸库文件**（.feat/.names）明确列为不可信输入，
// 而 ZQ_FaceDatabaseCompact::LoadFromFile 是它的解析入口 —— **此前零测试覆盖**。
//
// 为了能给它写测试，先得让这个头能独立编译，这一撞就撞出三条真缺陷：
//   ① ZQ_FaceRecognizerUtils.h 用了 std::cout 却没 include <iostream>
//   ② GenerateRandomDatabase 里两个 malloc 不判 NULL
//   ③ `__int64 num_all_feats = num_person * num_feat_per_person;`
//      —— 两个 int 相乘，__int64 只是装饰
// 另加三处 printf/sprintf 的格式符与实参宽度不匹配。
//
// 本测试覆盖
//   正常：一个格式正确的库应当加载成功，dim/person_num/total_face_num 正确
//   畸形（每一条都必须**干净地返回 false**，而不是崩 / 半加载）：
//     - 文件不存在 / 空文件
//     - dim <= 0、person_num <= 0
//     - person_face_num 里出现 0 或负数
//     - 特征区被截断（声明的个数比文件里能读出来的多）
//     - names 的人数与 feats 的人数不一致
//     - person_face_num 累加到 int 回绕（person_num=2, 每人 0x40000000）
//
// 全部在 ASan + LeakSanitizer 下跑。ZQ_FaceDatabaseCompact 的拷贝/赋值
// 已经被禁掉（早前一轮：补了析构就会 double free），所以本测试只用栈对象。

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#include "zqlib_msvc_shim.h"
#include "ZQlibFaceID/ZQ_FaceDatabaseCompact.h"

static int g_fail = 0;
static const char* FEATS = "/tmp/zq_facedb.feat";
static const char* NAMES = "/tmp/zq_facedb.names";

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

// 拼一个"格式正确"的 feats 文件
static std::vector<char> make_feats(int dim, const std::vector<int>& per_person,
                                   bool truncate_feats)
{
    std::vector<char> v;
    put_i32(v, dim);
    put_i32(v, (int)per_person.size());
    long long total = 0;
    for (size_t i = 0; i < per_person.size(); i++) total += per_person[i];
    for (size_t i = 0; i < per_person.size(); i++) put_i32(v, per_person[i]);
    long long want = truncate_feats ? total - 1 : total;   // 少一个 -> fread 应当短读
    for (long long i = 0; i < want * dim; i++) {
        float f = (float)(i % 17) * 0.1f;
        char b[4];
        memcpy(b, &f, 4);
        v.insert(v.end(), b, b + 4);
    }
    return v;
}

// dim / person_num / total_face_num 都是 private，没有公开的取值接口，
// 所以这里只能断言 LoadFromFile 的**返回值** —— 那也正是对外的契约。
// （想更严就得往类里加 getter，本轮不加：那是 API 变更，不是审计。）
// 只写头部 + 一点点特征体：用来构造"人脸数很大、特征区却几乎没有"这种
// 回绕场景。第一版直接复用 make_feats，于是它自己要去 vector 里塞
// 8 维 x 0x7FFFFFFE 个 float（约 17 GB）—— 挂的是**测试**，不是库。
static std::vector<char> make_feats_header_only(int dim,
                                                const std::vector<int>& per_person)
{
    std::vector<char> v;
    put_i32(v, dim);
    put_i32(v, (int)per_person.size());
    for (size_t i = 0; i < per_person.size(); i++) put_i32(v, per_person[i]);
    for (int i = 0; i < 16; i++) { float f = 0.f; char b[4]; memcpy(b, &f, 4);
                                   v.insert(v.end(), b, b + 4); }
    return v;
}

static void expect_ok(const char* tag, int dim, const std::vector<int>& per)
{
    write_file(FEATS, make_feats(dim, per, false));
    put_names((int)per.size());
    ZQ::ZQ_FaceDatabaseCompact db;
    bool r = db.LoadFromFile(FEATS, NAMES);
    if (!r) g_fail++;
    printf("  %-26s 期望加载成功 -> %s (返回 %d)\n", tag, r ? "ok" : "FAIL", (int)r);
}

static void expect_fail(const char* tag, const std::vector<char>& feats, int names_n)
{
    write_file(FEATS, feats);
    put_names(names_n);
    ZQ::ZQ_FaceDatabaseCompact db;
    bool r = db.LoadFromFile(FEATS, NAMES);
    if (r) g_fail++;
    printf("  %-26s 期望干净失败 -> %s (返回 %d)\n", tag, r ? "FAIL" : "ok", (int)r);
}

int main()
{
    printf("ZQ_FaceDatabaseCompact 不可信文件解析 回归（附录 BK）\n");
    printf("ASan 会在越界时 abort；畸形输入必须干净返回 false\n\n");

    // ---- 正常 ----
    {
        std::vector<int> per;
        per.push_back(2); per.push_back(3);
        expect_ok("正常 2 人 / dim=8", 8, per);
    }
    {
        std::vector<int> per;
        per.push_back(1);
        expect_ok("正常 1 人 / dim=128", 128, per);
    }

    // ---- 畸形 ----
    {
        std::vector<char> empty;
        expect_fail("空文件", empty, 1);
    }
    {
        FILE* f = fopen(FEATS, "wb");
        if (f) fclose(f);
        remove(FEATS);
        put_names(1);
        ZQ::ZQ_FaceDatabaseCompact db;
        bool r = db.LoadFromFile(FEATS, NAMES);
        if (r) g_fail++;
        printf("  %-26s 期望干净失败 -> %s\n", "文件不存在", r ? "FAIL" : "ok");
    }
    {
        std::vector<char> v; put_i32(v, 0); put_i32(v, 1);
        expect_fail("dim = 0", v, 1);
    }
    {
        std::vector<char> v; put_i32(v, -5); put_i32(v, 1);
        expect_fail("dim = -5", v, 1);
    }
    {
        std::vector<char> v; put_i32(v, 8); put_i32(v, 0);
        expect_fail("person_num = 0", v, 1);
    }
    {
        std::vector<char> v; put_i32(v, 8); put_i32(v, -1);
        expect_fail("person_num = -1", v, 1);
    }
    {
        std::vector<int> per;
        per.push_back(0); per.push_back(1);
        expect_fail("某人脸数为 0", make_feats(8, per, false), 2);
    }
    {
        std::vector<int> per;
        per.push_back(-3); per.push_back(1);
        expect_fail("某人脸数为负", make_feats(8, per, false), 2);
    }
    {
        std::vector<int> per;
        per.push_back(2); per.push_back(3);
        expect_fail("特征区被截断", make_feats(8, per, true), 2);
    }
    {
        std::vector<int> per;
        per.push_back(2); per.push_back(3);
        write_file(FEATS, make_feats(8, per, false));
        put_names(3);                      // 名字比人脸多
        ZQ::ZQ_FaceDatabaseCompact db;
        bool r = db.LoadFromFile(FEATS, NAMES);
        if (r) g_fail++;
        printf("  %-26s 期望干净失败 -> %s (返回 %d)\n",
               "names 人数不一致", r ? "FAIL" : "ok", (int)r);
    }
    {
        // person_face_num 累加到 int 回绕：2 人各 0x40000000
        std::vector<int> per;
        per.push_back(0x40000000); per.push_back(0x40000000);
        expect_fail("人脸数累加 int 回绕", make_feats_header_only(8, per), 2);
    }

    remove(FEATS);
    remove(NAMES);

    if (g_fail) { printf("\n%d 条断言失败\n", g_fail); return 1; }
    printf("\n全部通过（无越界、畸形输入全部干净拒绝）\n");
    return 0;
}
