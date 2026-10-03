/* 每个随仓库 .zqparams 都必须**通过参数解析与连通性检查** —— 附录 FD
 *
 * 为什么只查参数、不查权重
 * -----------------------
 * 权重是 66 MB / 27 个 `.nchwbin`，进不了快速通道。
 * 而本门禁要守的正是附录 EN 改的那一段：
 *
 *     ZQ_CNN_Net::LoadFrom
 *       -> _load_param_file()   每个层的 ReadParam（EM/EY/EZ 那些守卫）
 *       -> _check_connect()     blob 连通性 + 就地守卫（EN）
 *       -> _load_model_file()  <-- 66 MB，只是不查它
 *
 * 做法：给一个**故意不存在**的权重路径。`LoadFrom` 会先把前两步跑完，
 * 只在第三步失败并打印 `failed to open <file>`。
 * 于是判据变成：
 *
 *   输出里有 "failed to open"            => 参数与连通性**全过了** -> 通过
 *   输出里有 "unknown blob" / "changes shape but declares top ==" /
 *              "missing"                 => **被守卫拒了** -> 失败
 *
 * 这比"LoadFrom 返回 true/false"精确得多：两种失败都返回 false，
 * 只有区分消息才能说明"是权重没找到"而不是"模型被拒"。
 *
 * 一个子进程一个模型：某个模型让库段错误时不会连累其余 26 个。
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <dirent.h>
#include <unistd.h>
#include <sys/wait.h>
#include "zq_check_child.h"
#include "ZQCNN/ZQ_CNN_Tensor4D.h"
#include "ZQCNN/ZQ_CNN_Net.h"
// 链接期需要的那批 ZQ_CNN_Forward_SSEUtils 符号，用现成的**绊线桩**顶上。
// 本门禁只走 _load_param_file / _check_connect，**根本不会调到任何 Forward** ——
// 所以这些桩永远不该响；真响了说明"参数阶段"偷偷跑到了前向，
// 那本身就是一条要查的结论。
#include "zq_net_fwd_tripwires.h"
// 44+60 个绊线桩之外**唯一**的排除项：`_concat_NCHW_get_size`。
// 它是 Concat 的 LayerSetup 在**加载期**就合法调到的（与附录 EN 的记录一致），
// 而本门禁正好会走到那里 —— 所以必须给它真实现。
#include "zq_concat_getsize_real.h"

using namespace ZQ;

#define RES_FILE "/tmp/zq_modelparams_res.txt"
#define OUT_FILE "/tmp/zq_modelparams_out.txt"
#define MODEL_DIR "/mnt/d/ZQCNN/model"

// 走到这一步说明**参数解析 + 连通性都过了**，只是权重找不到
static const char* EXPECT_OK_MARK = "failed to open";
// 这些消息一旦出现，就是模型**被守卫拒了**
static const char* REJECT_MARKS[] = {
    "unknown blob",
    "changes shape but declares top ==",
    "missing ",
    "invalid conv params",
    "conv kernel/dilate overflow",
    "invalid pooling params",
    "must be specified for InnerProduct",
};

static int child(const char* zqparams)
{
    // 重定向自己的 stdout 到 OUT_FILE，父进程读它
    if (!freopen(OUT_FILE, "w", stdout)) _exit(4);
    ZQ_CNN_Net net;
    bool loaded = net.LoadFrom(zqparams, "/tmp/zq_definitely_missing.nchwbin");
    (void)loaded;               // 一定是 false；判据在消息里
    // **必须显式刷新**：父进程用 `_exit()` 收子进程，而 `_exit` 不跑 atexit、
    // 不刷缓冲。stdout 重定向到**文件**时是全缓冲的，那句
    // "failed to open ..." 就卡在缓冲里随进程一起消失 ——
    // 于是 27 个模型全部"没走到权重那步"，而其实每一条都打印过。
    fflush(NULL);
    std::cout.flush();
    return 0;
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("随仓库模型参数门禁（附录 FD）\n");
    printf("判据：每个 model/*.zqparams 都必须走完 ReadParam 与 _check_connect，\n");
    printf("      只允许停在 \"failed to open <权重>\" 这一步。\n\n");

    DIR* d = opendir(MODEL_DIR);
    if (!d) {
        printf("  打开 %s 失败 —— 门禁不能在没有模型的情况下报通过\n", MODEL_DIR);
        return 1;
    }
    std::vector<std::string> params;
    struct dirent* e;
    while ((e = readdir(d)) != NULL) {
        std::string n = e->d_name;
        if (n.size() > 9 && n.compare(n.size() - 9, 9, ".zqparams") == 0)
            params.push_back(std::string(MODEL_DIR) + "/" + n);
    }
    closedir(d);
    if (params.empty()) {
        printf("  没有找到 .zqparams —— 同样是\"没检查\"，不是通过\n");
        return 1;
    }

    int ok = 0, bad = 0, crash = 0;
    for (size_t i = 0; i < params.size(); i++) {
        remove(RES_FILE);
        remove(OUT_FILE);
        pid_t pid = fork();
        if (pid == 0) { zq_child_silence_stderr(); _exit(child(params[i].c_str())); }
        int st = 0; waitpid(pid, &st, 0);

        std::string out;
        FILE* f = fopen(OUT_FILE, "r");
        if (f) {
            char buf[4096];
            size_t n;
            while ((n = fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, n);
            fclose(f);
        }

        const char* base = strrchr(params[i].c_str(), '/');
        base = base ? base + 1 : params[i].c_str();

        if (WIFSIGNALED(st)) {
            crash++;
            printf("  %-34s 子进程被信号 %d 杀掉\n", base, WTERMSIG(st));
            continue;
        }
        // ASan 撞上致命错误时走 Die() -> _exit(1)，**不发信号**，
        // 所以 WIFSIGNALED 为假、退出码也不是约定的值。必须显式判"退出码非 0"，
        // 否则一个 sanitizer 崩溃会被记成"通过"（AGENTS.md「子进程写文件 + 父进程读
        // 文件」那条：默认写法会把崩溃算成通过）。
        if (WIFEXITED(st) && WEXITSTATUS(st) != 0) {
            crash++;
            printf("  %-34s 子进程非 0 退出（%d）—— 常见于 ASan 的 Die()\n",
                   base, WEXITSTATUS(st));
            continue;
        }
        if (out.find(EXPECT_OK_MARK) != std::string::npos) {
            ok++;
            printf("  %-34s OK（参数与连通性全过，只差权重文件）\n", base);
            continue;
        }
        // 找出第一个"被拒"的标记
        const char* why = NULL;
        for (size_t k = 0; k < sizeof(REJECT_MARKS) / sizeof(REJECT_MARKS[0]); k++) {
            if (out.find(REJECT_MARKS[k]) != std::string::npos) { why = REJECT_MARKS[k]; break; }
        }
        bad++;
        printf("  %-34s **FAIL**%s%s\n", base,
               why ? " 被守卫拒绝: " : " 没走到权重那步", why ? why : "");
        // 打印前几行，便于定位
        size_t pos = 0;
        for (int line = 0; line < 3 && pos < out.size(); line++) {
            size_t nl = out.find('\n', pos);
            if (nl == std::string::npos) nl = out.size();
            printf("        %s\n", out.substr(pos, nl - pos).c_str());
            pos = nl + 1;
        }
    }
    // **汇总行里不许出现字面的 "FAIL"。** harness 的 ASan 模式判据是
    // `grep -cE 'FAIL' <tag>.out`（tools/run_zqlib_checks.py:866）——
    // 它把输出里含 "FAIL" 的**行数**当作"断言失败数"。其他门禁都遵守
    // 「FAIL 只在真失败时出现」这个约定（见 zq_roi / zq_tile 的打印），
    // 只有我的汇总行**总是**打印 `OK 27，FAIL 0`，于是 27 通过 / 0 拒绝
    // 也被判成"1 条断言失败"。
    // 与 DY.5「消息为空把排查方向带偏」同族：**门禁的输出格式本身参与判定**。
    printf("\n共 %d 个模型：通过 %d，被守卫拒绝 %d，子进程异常 %d\n",
           (int)params.size(), ok, bad, crash);
    return (bad || crash) ? 1 : 0;
}
