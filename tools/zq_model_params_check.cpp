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

// 权重文件与同名 .zqparams 一一对应（附录 FD 量过：27 个里 23 个有配套权重，
// 缺的 4 个要么属 Model Zoo，要么配的是 `_bgr` 变体）。
// 取不到时**返回一个不存在的路径**，而不是 0。
// 第一版返回 0，而 `LoadFrom(const std::string&, ...)` 会用 `std::string(nullptr)`
// 构造形参 —— 那是 UB，表现为 **SIGABRT**。
// 于是"4 个没有配套权重的模型"在实测里全变成了"子进程被信号 6 杀掉"，
// 一度看着像库在加载缺权重的模型时会崩 —— **那是我造的，不是库的**。
static const char* weight_path_for(const char* zqparams)
{
    static char buf[1024];
    std::string z(zqparams);
    size_t dot = z.rfind('.');
    if (dot != std::string::npos) z = z.substr(0, dot);
    z += ".nchwbin";
    if (access(z.c_str(), R_OK) != 0)
        snprintf(buf, sizeof(buf), "/tmp/zq_no_such_weights.nchwbin");
    else
        snprintf(buf, sizeof(buf), "%s", z.c_str());
    return buf;
}

// 走到这一步说明**参数解析 + 连通性都过了**，只是权重找不到
static const char* EXPECT_OK_MARK = "failed to open";
// 解析阶段的**警告**：不是拒绝，但意味着某个参数名没被识别 ——
// .zqparams 里把 `kernel_size` 拼成 `kenerl_size` 就会静默走默认值，
// 而模型照样"加载成功"。所以这里把它们单列成"必须为零"。
// 2026-10-03 实测：27 个随仓库模型的解析阶段**零警告**，所以这条判据
// 现在就能立住，将来任何一个模型打错参数名都会被抓住。
static const char* WARN_MARKS[] = {
    "warning: unknown para",
    "warning:",                       // 兜底：任何 warning 行
};

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
    // **默认**给一个故意不存在的权重路径 —— 判据是"参数解析 + 连通性"，
    // 见文件头的说明。`ZQ_MODEL_FULL_LOAD=1` 时改用**真实**权重文件，
    // 于是连 `LoadBinary` 的尺寸对不对也一起验了（附录 GH）。
    const char* wpath = getenv("ZQ_MODEL_FULL_LOAD")
                        ? weight_path_for(zqparams)
                        : "/tmp/zq_definitely_missing.nchwbin";
    bool loaded = net.LoadFrom(zqparams, wpath);
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

    int ok = 0, bad = 0, crash = 0, skipped = 0;
    int full = (getenv("ZQ_MODEL_FULL_LOAD") != 0);
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
        // 全量加载模式下，成功时**不会**打印 "failed to open"，
        // 所以判据必须跟着模式走 —— 否则 23 个真加载成功的模型会被判成
        // "既没到权重那步、又没有拒绝标记"，也就是"没验到"。
        // （第一版忘了改这一处，23 个全红的现象就是这么来的。）
        if (full && out.find(EXPECT_OK_MARK) != std::string::npos
            && access(weight_path_for(params[i].c_str()), R_OK) != 0) {
            // 仓库里没有配套权重的模型（README:24 写明 SphereFace 等示例
            // 需要从 Model Zoo 另下）——**这不是缺陷**，记成"跳过"而不是失败。
            skipped++;
            printf("  %-34s 跳过（仓库里没有配套 .nchwbin）\n", base);
            continue;
        }
        bool reached_weights = full ? out.empty()
                                   : (out.find(EXPECT_OK_MARK) != std::string::npos);
        if (reached_weights) {
            // 解析阶段的**警告**也要判：模型照样"加载成功"，但某个参数名
            // 没被识别、静默走了默认值（`kernel_size` 拼成 `kenerl_size` 就是
            // 这个后果）。这是"能加载"这道判据**看不到**的一类问题。
            const char* warn = NULL;
            for (size_t k = 0; k < sizeof(WARN_MARKS) / sizeof(WARN_MARKS[0]); k++) {
                if (out.find(WARN_MARKS[k]) != std::string::npos) { warn = WARN_MARKS[k]; break; }
            }
            if (warn) {
                bad++;
                printf("  %-34s 解析阶段有警告（仍会加载成功，但参数可能没生效）\n", base);
                size_t pos = 0;
                for (int line = 0; line < 3 && pos < out.size(); line++) {
                    size_t nl = out.find('\n', pos);
                    if (nl == std::string::npos) nl = out.size();
                    printf("        %s\n", out.substr(pos, nl - pos).c_str());
                    pos = nl + 1;
                }
                continue;
            }
            ok++;
            printf("  %-34s OK（%s）\n", base,
                   full ? "参数 + 连通性 + 权重全部加载通过"
                        : "参数与连通性全过，只差权重文件");
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
    printf("\n共 %d 个模型：通过 %d（解析零警告%s），被守卫拒绝 %d，跳过 %d，子进程异常 %d\n",
           (int)params.size(), ok, full ? "，权重全部加载" : "，只差权重文件",
           bad, skipped, crash);
    return (bad || crash) ? 1 : 0;
}
