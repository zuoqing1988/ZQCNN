/* NCHWC 卷积 / 深度卷积的 **padding 键拒载** 门禁 —— 附录 DH
 *
 * 缺陷：NCHWC 这一族只支持**对称**的 `pad` / `pad_H` / `pad_W`，
 * 而 NCHW 那一族还认 `pad_type`（SAME/VALID）与非对称的
 * `pad_H_top` / `pad_H_bottom` / `pad_W_left` / `pad_W_right`。
 * 模型文件里写这些键时，NCHWC 的 `ReadParam` 原来只打一行
 * `warning: unknown para`，然后**按 pad=0 继续算** ——
 * 而 `ReadParam` 的返回条件（`has_num_output && has_kernelH && has_kernelW &&
 * has_bottom && has_top && has_name`）**不含任何 pad 标志**，
 * 所以这是**静默算错**：`in % stride != 0` 时整层错一格。
 *
 * 本门禁断言的是「**响亮地失败**」，不是「算对」——
 * 真正的支持是另一个独立决策（21 个前向函数的签名都要改）。
 * 判据：**加载必须失败，且失败信息里要指名那个键**。
 *
 * 对照组：受支持的写法（`pad` / `pad_H` / `pad_W`）**必须仍然能加载**。
 * 没有对照组的门禁很容易把「全都拒载」也判成通过。
 *
 * 走的是**完整 ZQ_CNN_Net_NCHWC 的 LoadFrom**，
 * 不是直接实例化层 —— 因为「静默」这个缺陷只有在**没人报错**时才是缺陷。
 */
#include "zq_check_child.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include "ZQCNN/ZQ_CNN_Tensor4D_NCHWC.h"
#include "ZQCNN/ZQ_CNN_Net_NCHWC.h"

#define RES_FILE "/tmp/zq_padreject_res.txt"
// 临时模型写在**当前工作目录**（不是 `model/`）。
// 门禁由 run_zqlib_checks 起进程时，cwd 是它自己那一轮的 WDIR（/tmp/zqchecks_<pid>_<ts>），
// 里面**没有 model/** —— 第一版写 `model/...`，15 个用例全部"没跑完"，
// 而症状看起来像是"库拒载了"。（本文件 HV.5：相对路径必须连 cwd 一起说清。）
// `LoadFrom` 接受任意路径，不要求文件在 model/ 下，所以直接写 cwd 即可。
#define MODEL_DIR "."

struct Case {
    const char* tag;      // 用例名
    const char* para;     // 附加到 Convolution 行上的参数
    int must_reject;      // 1 = 必须拒载并指名该键；0 = 必须能加载
};

static const Case g_cases[] = {
    { "conv pad_type=SAME",        " pad_type=SAME",        1 },
    { "conv pad_type=VALID",       " pad_type=VALID",       1 },
    { "conv pad_H_top",            " pad_H_top=1",          1 },
    { "conv pad_H_bottom",         " pad_H_bottom=1",       1 },
    { "conv pad_W_left",           " pad_W_left=1",         1 },
    { "conv pad_W_right",          " pad_W_right=1",        1 },
    { "conv same=1",               " same=1",               1 },
    { "conv valid=1",              " valid=1",              1 },
    { "dwconv pad_type=SAME",      " pad_type=SAME",        1 },
    { "dwconv pad_H_top",          " pad_H_top=1",          1 },
    { "dwconv pad_W_right",        " pad_W_right=1",        1 },
    // ---- 对照组：这些是 NCHWC 支持的写法，必须仍然能加载 ----
    { "conv pad=1 (支持)",         " pad=1",                0 },
    { "conv pad=1 pad_H=1 (支持)", " pad=1 pad_H=1",        0 },
    { "conv pad_W=1 (支持)",       " pad_W=1",              0 },
    { "dwconv pad=1 (支持)",       " pad=1",                0 },
};
static const int N_CASE = (int)(sizeof(g_cases) / sizeof(g_cases[0]));

static void write_model(const std::string& path, const Case& c)
{
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { printf("  写不出 %s\n", path.c_str()); exit(2); }
    const char* head = "Input name=data C=3 H=8 W=8\n";
    fwrite(head, 1, strlen(head), f);
    const char* kind = (strstr(c.tag, "dwconv") == 0) ? "Convolution" : "DepthwiseConvolution";
    char line[512];
    // DepthwiseConvolution 要求 num_output == bottom_C（=3），写 4 会因为
    // **另一个**原因加载失败，而症状（"受支持的写法被拒了"）看起来像是
    // 我们的拒载逻辑在误伤 —— 两个原因长得一模一样。
    const int nout = (strcmp(kind, "Convolution") == 0) ? 4 : 3;
    if (strcmp(kind, "Convolution") == 0)
        snprintf(line, sizeof(line),
                 "Convolution name=c1 bottom=data top=c1 num_output=%d kernel_H=3 kernel_W=3 "
                 "stride_H=1 stride_W=1%s\n", nout, c.para);
    else
        snprintf(line, sizeof(line),
                 "DepthwiseConvolution name=w1 bottom=data top=w1 num_output=%d kernel_H=3 kernel_W=3 "
                 "stride_H=1 stride_W=1%s\n", nout, c.para);
    fwrite(line, 1, strlen(line), f);
    fclose(f);
}

static void run_one(const Case& c)
{
    ZQ::ZQ_CNN_Net_NCHWC<ZQ::ZQ_CNN_Tensor4D_NCHWC4> net;
    std::string zqp = std::string(MODEL_DIR) + "/_padreject_tmp.zqparams";
    std::string nchw = std::string(MODEL_DIR) + "/_padreject_tmp.nchwbin";
    write_model(zqp, c);
    // 权重文件只需要存在且够长：拒载发生在**读权重之前**
    //（建层时 ReadParam 就返回 false 了）。
    FILE* w = fopen(nchw.c_str(), "wb");
    if (w) {
        static const float junk[4096] = { 0.f };
        fwrite(junk, sizeof(float), 4096, w);
        fclose(w);
    }

    bool ok = false;
    std::string msg;
    {
        // 捕获 stdout：拒载的说明是打在 stdout 上的，父进程要看到它
        char tmpl[] = "/tmp/zq_padreject_out_XXXXXX";
        int fd = mkstemp(tmpl);
        int saved = dup(1);
        dup2(fd, 1);
        ok = net.LoadFrom(zqp.c_str(), nchw.c_str(), true, 1e-9, true);
        fflush(stdout);
        dup2(saved, 1);
        close(saved);
        close(fd);
        FILE* rf = fopen(tmpl, "r");
        if (rf) {
            char buf[4096];
            size_t n = fread(buf, 1, sizeof(buf) - 1, rf);
            buf[n] = 0;
            msg = buf;
            fclose(rf);
        }
        remove(tmpl);
    }
    remove(zqp.c_str());
    remove(nchw.c_str());

    // 判据：拒载 => LoadFrom 返回 false **且** 输出里指名了那个键。
    // 注意只取**键名**（`=` 之前），不是整串 `key=value` ——
    // 消息里写的是 `does not support para 'pad_type'`，拿 `pad_type=SAME` 去 find
    // 必然找不到，于是 12 个用例误报成"没说清是哪个键"。
    const char* key = strchr(c.para, ' ');
    std::string full = (key != 0) ? std::string(key + 1) : std::string();
    const size_t eq = full.find('=');
    std::string want = (eq == std::string::npos) ? full : full.substr(0, eq);
    int named = (msg.find(want) != std::string::npos) ? 1 : 0;
    FILE* f = fopen(RES_FILE, "w");
    if (f) { fprintf(f, "%d %d\n", ok ? 1 : 0, named); fclose(f); }
}

static int g_case = 0, g_ok = 0, g_bad = 0, g_crash = 0;

static void one(const Case& c)
{
    g_case++;
    remove(RES_FILE);
    pid_t pid = fork();
    if (pid == 0) { zq_child_silence_stderr(); run_one(c); _exit(0); }
    int st = 0; waitpid(pid, &st, 0);
    int loaded = -1, named = 0, have = 0;
    FILE* f = fopen(RES_FILE, "r");
    if (f) { have = (fscanf(f, "%d %d", &loaded, &named) == 2); fclose(f); }
    if (!have) { g_crash++; printf("  %-28s 没跑完\n", c.tag); return; }
    if (WIFSIGNALED(st)) { g_crash++; printf("  %-28s CRASH(%d)\n", c.tag, WTERMSIG(st)); return; }
    int bad = 0;
    const char* why = "";
    if (c.must_reject) {
        if (loaded) { bad = 1; why = "**加载成功了**（应当拒载）"; }
        else if (!named) { bad = 1; why = "拒载了但**没说清是哪个键**"; }
    } else {
        if (!loaded) { bad = 1; why = "**受支持的写法被拒了**"; }
    }
    if (bad) { g_bad++; printf("  %-28s FAIL  %s\n", c.tag, why); }
    else { g_ok++; }
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("NCHWC 卷积/深度卷积：表达不了的 padding 键必须**拒载并指名**\n");
    printf("对照组：受支持的对称写法必须仍然能加载\n\n");
    for (int i = 0; i < N_CASE; i++)
        one(g_cases[i]);
    printf("\n共 %d 个用例：全对 %d，有错 %d，崩/未跑 %d\n", g_case, g_ok, g_bad, g_crash);
    return (g_bad || g_crash) ? 1 : 0;
}