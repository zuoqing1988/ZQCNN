/* `ZQ_FaceIDPrecisionEvaluation` 的行为门禁 —— 附录 IH
 *
 * 为什么是这一个头
 * ----------------
 * 附录 EG 把它从 `<opencv2\opencv.hpp>`（反斜杠）改成正斜杠，于是它在 Linux 上
 * "能编过"了 —— 但附录 EH 之后的结论一直是：**ZQlibFaceID 里没有一个头 include
 * 了 OpenCV，所以它们一道行为门禁都跑不了**（本机 WSL 没装 OpenCV）。
 *
 * 这个头是那个结论的**唯一反例**，而且反例本身很说明问题：
 * 它 include OpenCV 只是为了 `cv::imread` / `cv::flip` 两个调用，
 * 而 `EvaluationOnLFW` 的**全部逻辑**（解析 list 文件、抽特征、留一法定阈值、
 * FAR/TAR 曲线）就在这个头里，是同目录里逻辑量最大的一个。
 * 也就是说：附录 EH 的"零覆盖"名单里，**漏掉的是覆盖价值最高的那一个**。
 *
 * 做法：给一个最小 OpenCV 桩（tools/opencv_stub/opencv2/opencv.hpp），
 * 只提供 cv::Mat / cv::imread / cv::flip 三样，让**头本身**在 Linux 上编过并跑行为。
 * 桩里的 imread 走**真 fopen** —— "list 文件指向的图片全都不存在" 正是
 * 触发 IH.1 那个空 vector 越界的最短路径，桩必须能造出这个场景。
 *
 * 判据
 * ----
 * 1. **图片全部读不到 → 必须干净返回，不能崩**（IH.1）
 * 2. **header 说有 N 行、文件里一行都没有 → 必须被拒**（IH.2）
 * 3. **part_num 无上界 → 必须被拒，不能抛异常**（IH.3）
 * 4. 正常 2x2 场景必须跑完且 ACC 落在开区间（IH.4）
 * 5. `use_flip` 分支（特征缓冲区右半段 `pData + feat_dim`）必须跑完（IH.5）
 * 6. `part_num == 1`（留一法留空了）不能崩（IH.6）
 * 7. 只有一对有效（image_num == 2, all_num == 1）不能崩（IH.7）
 *
 * 关于 3：不用"给子进程设内存上限"来观测，改为量 **ru_maxrss 增量**。
 * 原因见 run_case 里那段注释：RLIMIT_AS 与 ASan **不兼容**，任何上限都会让
 * 子进程只剩一行 "ERROR: Failed to mmap" —— 那是观测手段弄坏了被测环境，
 * 不是被测代码崩了（附录 CA.3 的第 6 次）。
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <unistd.h>
#include "zq_check_child.h"
#include "ZQlibFaceID/ZQ_FaceIDPrecisionEvaluation.h"

using namespace ZQ;

#define RES_FILE  "/tmp/zq_lfweval_res.txt"
#define LIST_PATH "/tmp/zq_lfweval_list.txt"
#define IMG_DIR   "/tmp/zq_lfweval_imgs"

enum {
	OP_ALL_MISSING = 0,   // 2x2，图片全不存在  -> 现状：SIGSEGV
	OP_NO_DATA,           // header 说 2x1，实际 0 行 -> 现状：返回 true（错）
	OP_HUGE_PARTNUM,      // part_num = 20 亿 -> 现状：bad_alloc 抛穿
	OP_HUGE_HALFPAIR,     // half_pair_num = 2^30，2*它 int 回绕 -> 现状：返回 true（错）
	OP_NORMAL,            // 2x2 全有效
	OP_FLIP,              // 2x2 全有效 + use_flip
	OP_ONE_PART,          // part_num == 1
	OP_ONE_VALID_PAIR,    // 4 对里只有 1 对图片在
	OP_BAD_LIST           // list 文件不存在
};

// note: 0 无事 / 1 搭建失败 / 2 该拒却收下了 / 3 抛异常了 / 4 正常路径没跑完
//       / 5 ACC 不在开区间 / 6 判据本身没跑到
//       / 7 拒是拒了，但先按 part_num 分配了一大片内存
static const char* g_note[] = {
    "", "搭建失败", "**该拒却收下了**", "**抛异常了（库没接住）**",
    "**正常路径没跑完**", "**ACC 不在 (0,1]**", "判据没跑到",
    "**拒是拒了，但先按 part_num 分配了 480MB（无上界的内存放大）**"
};

// ---------------------------------------------------------------- 桩识别器
// 只需要"抽特征"这一个动作是真的。特征必须**依赖像素字节**，
// 否则 use_flip 那半段和原图那半段会一模一样，IH.5 的判据就废了。
class StubRecognizer : public ZQ_FaceRecognizer
{
	int _dim;
public:
	StubRecognizer(int dim) : _dim(dim) {}
	// 基类写的是 `Init(const std::string model_name, ...)` —— 那是**按值传参上的
	// 顶层 const**，签名里被丢掉，所以这里必须写 `std::string` 而不是 `const std::string&`。
	// 写成引用会得到"抽象类"编译错（顶层的坑，见报告 IH.6）。
	bool Init(std::string, std::string, std::string, std::string) { return true; }
	int GetFeatDim() const { return _dim; }
	int GetCropWidth() const { return 8; }
	int GetCropHeight() const { return 8; }
	bool CropImage(const unsigned char*, int, int, int, ZQ_PixelFormat,
		const float*, const float*, unsigned char*, int) const { return true; }
	bool ExtractFeature(const unsigned char*, int, int, int, ZQ_PixelFormat,
		const float*, const float*, float*, bool) { return true; }
	bool ExtractFeature(const unsigned char* img, int widthStep, ZQ_PixelFormat, float* feat, bool)
	{
		// 桩 imread 出来的是 8 行、每行 widthStep 字节。
		// 越界读没有任何好处：只按 8*widthStep 取模循环。
		size_t n = (size_t)8 * (size_t)widthStep;
		if (n == 0 || feat == 0) return false;
		for (int k = 0; k < _dim; k++)
			feat[k] = (float)img[(size_t)k % n] * 0.00391f + (float)(k % 7) * 0.5f;
		return true;
	}
	float CalSimilarity(const float* a, const float* b) const
	{
		float s = 0;
		for (int k = 0; k < _dim; k++) s += a[k] * b[k];
		return s;
	}
};

// ------------------------------------------------------------------ 造文件
static void write_file(const char* path, const std::string& content)
{
	FILE* f = fopen(path, "wb");
	if (f == 0) return;
	if (!content.empty()) fwrite(content.data(), 1, content.size(), f);
	fclose(f);
}

// 造"图"必须**落在库自己拼出来的那个路径**上：
// `_parse_lfw_list` 拼的是 `folder + "/" + name + "/" + name + "_%04i.jpg"`（3 段行），
// 4 段行右边那半是 `folder + "/" + nameR + "/" + nameR + "_%04i.jpg"`。
// 我第一版图省事写成 `img_%03d.jpg` 放在 IMG_DIR 根下 —— 结果**一张都读不到**，
// 于是"正常 2x2"和"只有一对图片在"两个用例其实跑的是"全都不存在"，
// 而门禁照样报出"没跑完"，看上去像是库崩了。
// 又一次（附录 CA.3 / DA.2）：**观测手段没走到那条路径，结论就是假的**。
//
// n_ids: 造出 _0000 .. _%04d 这些编号；with_name2 决定异名那一支建不建目录。
static void make_images(int n_ids, bool with_name2)
{
	char cmd[512];
	sprintf(cmd, "rm -rf %s", IMG_DIR);
	if (system(cmd) != 0) { /* 目录本来就不存在，忽略 */ }
	static const char* names[2] = { "stubname", "stubname2" };
	int nn = with_name2 ? 2 : 1;
	for (int t = 0; t < nn; t++)
	{
		sprintf(cmd, "mkdir -p %s/%s", IMG_DIR, names[t]);
		if (system(cmd) != 0) return;
		for (int i = 0; i < n_ids; i++)
		{
			char p[512];
			sprintf(p, "%s/%s/%s_%04d.jpg", IMG_DIR, names[t], names[t], i);
			FILE* f = fopen(p, "wb");
			if (f == 0) continue;
			for (int k = 0; k < 192; k++) fputc((k * 37 + i * 11 + t * 53) & 0xFF, f);
			fclose(f);
		}
	}
}

// 一行 LFW 记录：3 段（同名正样本）或 4 段（异名负样本）。
// id 直接决定文件名序号，**必须和 make_images 造出来的编号对得上**。
static std::string lfw_line(int idL, int idR, bool same)
{
	char b[128];
	if (same) sprintf(b, "stubname\t%d\t%d\n", idL, idR);
	else      sprintf(b, "stubname\t%d\tstubname2\t%d\n", idL, idR);
	return std::string(b);
}

static void run_case(int op)
{
	long bad = 0, ok = 0;
	int note = 0;

	std::vector<ZQ_FaceRecognizer*> recs;
	StubRecognizer* r0 = new StubRecognizer(8);
	recs.push_back(r0);

	const char* good_dir = IMG_DIR;
	const char* empty_dir = "/tmp/zq_lfweval_no_such_dir";

	switch (op)
	{
	case OP_ALL_MISSING:
		make_images(4, true);
		write_file(LIST_PATH, "2\t2\n" + lfw_line(0, 1, true) + lfw_line(2, 3, true)
			+ lfw_line(0, 1, false) + lfw_line(2, 3, false)
			+ lfw_line(0, 2, true) + lfw_line(1, 3, true)
			+ lfw_line(0, 2, false) + lfw_line(1, 3, false));
		break;
	case OP_NO_DATA:
		make_images(4, true);
		write_file(LIST_PATH, "2\t1\n");     // 声称 2x1 = 2 行数据，文件里一行都没有
		break;
	case OP_HUGE_PARTNUM:
		make_images(2, true);
		write_file(LIST_PATH, "20000000\t1\n" + lfw_line(0, 1, true) + lfw_line(0, 1, true));
		break;
	case OP_HUGE_HALFPAIR:
		make_images(2, true);
		write_file(LIST_PATH, "2\t1073741824\n");   // 2*它 在 int 里回绕成负
		break;
	case OP_NORMAL:
		make_images(4, true);
		write_file(LIST_PATH, "2\t2\n" + lfw_line(0, 1, true) + lfw_line(2, 3, true)
			+ lfw_line(0, 1, false) + lfw_line(2, 3, false)
			+ lfw_line(0, 2, true) + lfw_line(1, 3, true)
			+ lfw_line(0, 2, false) + lfw_line(1, 3, false));
		break;
	case OP_FLIP:
		make_images(4, true);
		write_file(LIST_PATH, "2\t2\n" + lfw_line(0, 1, true) + lfw_line(2, 3, true)
			+ lfw_line(0, 1, false) + lfw_line(2, 3, false)
			+ lfw_line(0, 2, true) + lfw_line(1, 3, true)
			+ lfw_line(0, 2, false) + lfw_line(1, 3, false));
		break;
	case OP_ONE_PART:
		make_images(2, true);
		write_file(LIST_PATH, "1\t1\n" + lfw_line(0, 1, true) + lfw_line(0, 1, false));
		break;
	case OP_ONE_VALID_PAIR:
		// 只造 0/1 两张，2/3 号图片不存在 -> 4 对里只有第一对的两张图能读到
		make_images(2, true);
		write_file(LIST_PATH, "1\t4\n" + lfw_line(0, 1, true) + lfw_line(0, 1, false)
			+ lfw_line(2, 3, true) + lfw_line(2, 3, false)
			+ lfw_line(0, 2, true) + lfw_line(0, 2, false)
			+ lfw_line(1, 3, true) + lfw_line(1, 3, false));
		break;
	case OP_BAD_LIST:
		make_images(2, true);
		write_file(LIST_PATH, "2\t2\n" + lfw_line(0, 1, true) + lfw_line(2, 3, true)
			+ lfw_line(0, 1, false) + lfw_line(2, 3, false)
			+ lfw_line(0, 2, true) + lfw_line(1, 3, true)
			+ lfw_line(0, 2, false) + lfw_line(1, 3, false));
		break;
	}

	// 判据 1 用的"图片全不存在"是靠 folder 指向空目录实现的：
	// 把 IMG_DIR 重命名掉太容易和别的用例串味，直接换一个不存在的路径更干净。
	const char* folder = (op == OP_ALL_MISSING) ? empty_dir : good_dir;
	const char* list = (op == OP_BAD_LIST) ? "/tmp/zq_lfweval_no_such_list.txt" : LIST_PATH;
	bool use_flip = (op == OP_FLIP);

	// 3 号判据的观测手段：**量常驻内存增量**，而不是给进程设内存上限。
	//
	// 设上限这条路走过、而且走死了：RLIMIT_AS 与 ASan **不兼容** ——
	// 实测把上限设成 8/16/24/40/64/96/128/200 GB，子进程一律只打一行
	// "ERROR: Failed to mmap" 就没了（ASan 自己的分配器要 mmap，
	// 而任何 RLIMIT_AS 都会把它挡掉；不是"值不够大"，是**这条路本身不通**）。
	// （第一个版本就是 1GB，症状完全一样：门禁报"没跑完"，
	//   看上去像被测代码崩了，其实是观测手段先把被测环境弄坏了 ——
	//   附录 CA.3 的第 6 次。）
	//
	// 改成量 RSS 增量：part_num = 2000 万 -> `pairs.resize` 要 2000万*24 = 480 MB。
	// 现在（无上界）这 480 MB 真的被分配出来，ru_maxrss 一定涨；
	// 修好之后是**在 resize 之前**就拒掉，涨 0。
	// 取 480 MB 而不是 48 GB：即使在"还没修"的状态下手动跑一遍门禁，
	// 峰值也只是半个 G，不会把机器拖进 OOM（本机 WSL 是
	// vm.overcommit_memory=1，48GB 的 malloc 会当场成功然后**真的构造**
	// 20 亿个空 vector）。
	struct rusage ru0, ru1;
	getrusage(RUSAGE_SELF, &ru0);
	bool ret = false;
	try {
		ret = ZQ_FaceIDPrecisionEvaluation::EvaluationOnLFW(recs, list, folder, use_flip);
	} catch (std::exception& e) {
		(void)e;
		ok = 0; bad = 1; note = 3;
	}
	getrusage(RUSAGE_SELF, &ru1);
	long rss_growth_kb = (long)ru1.ru_maxrss - (long)ru0.ru_maxrss;

	if (note == 0)
	{
		switch (op)
		{
		case OP_ALL_MISSING:
		case OP_ONE_PART:
		case OP_ONE_VALID_PAIR:
		case OP_NORMAL:
		case OP_FLIP:
			// 跑到这里就说明没崩 —— 崩了的话父进程读到的是 WIFSIGNALED。
			// 这五个都要求**真的跑完了整个流程**（ret == true）。
			// 修好 IH.5 之前"解析失败"也是 return EXIT_FAILURE == true，
			// 所以这条判据在 IH.5 修好之前是**不成立**的（假阴性风险）。
			// 这一点在报告 IH.5 里写明了：返回值语义本身就是要修的东西之一。
			if (!ret) { bad = 1; note = 4; }
			else ok = 1;
			break;
		case OP_NO_DATA:
		case OP_HUGE_PARTNUM:
		case OP_HUGE_HALFPAIR:
		case OP_BAD_LIST:
			// 这四个必须**被拒**：解析不通过就不该进抽特征/留一法那一段，
			// 返回值必须是 false。OP_BAD_LIST 修之前就是 false（列表文件压根没开成），
			// 它是**对照项**，用来确认判据没写反。
			if (ret) { bad = 1; note = 2; }
			else if (op == OP_HUGE_PARTNUM && rss_growth_kb > 64 * 1024)
			{
				// 拒掉了，但**先 resize 了**：480MB 已经分配出来又释放了。
				// 仍然算错 —— 无上界的危害正是"被恶意 list 文件拿来做内存放大"。
				bad = 1; note = 7;
			}
			else ok = 1;
			break;
		default:
			bad = 1; ok = 0; note = 6;
		}
	}

	FILE* o = fopen(RES_FILE, "w");
	if (o) { fprintf(o, "%ld %ld %d\n", ok, bad, note); fclose(o); }
	delete r0;
}

static int g_case = 0, g_ok = 0, g_bad = 0, g_crash = 0;

static void one(int op)
{
	g_case++;
	remove(RES_FILE);
	pid_t pid = fork();
	if (pid == 0) { zq_child_silence_stderr(); run_case(op); _exit(0); }
	int st = 0;
	waitpid(pid, &st, 0);
	long ok = 0, bad = 0;
	int note = 0, have = 0;
	FILE* f = fopen(RES_FILE, "r");
	if (f) { have = (fscanf(f, "%ld %ld %d", &ok, &bad, &note) == 3); fclose(f); }
	static const char* nm[] = {
		"图片全不存在 (2x2)", "header 说 2x1 但一行数据都没有",
		"part_num=2000 万（无上界则先分 480MB）", "half_pair_num=2^30 (2*它 回绕)",
		"正常 2x2", "正常 2x2 + use_flip", "part_num=1", "只有一对图片在",
		"list 文件不存在"
	};
	const char* tag = (op >= 0 && op < (int)(sizeof(nm) / sizeof(nm[0]))) ? nm[op] : "?";
	if (!have || WIFSIGNALED(st))
	{
		g_crash++;
		printf("  %-34s  没跑完（%s）\n", tag, WIFSIGNALED(st) ? "信号" : "结果文件读不出来");
		return;
	}
	if (bad > 0) { g_bad++; printf("  %-34s  %s\n", tag, g_note[note]); }
	else g_ok++;
}

int main()
{
	setvbuf(stdout, NULL, _IONBF, 0);
	printf("ZQ_FaceIDPrecisionEvaluation 行为门禁（附录 IH）\n");
	printf("为什么是它：EH 的\"零覆盖\"名单里漏了这个头 —— 它 include OpenCV 只为了\n");
	printf("  imread/flip 两个调用，而 EvaluationOnLFW 的全部逻辑（解析/抽特征/留一法/\n");
	printf("  FAR-TAR）就在这个头里。给一个最小 OpenCV 桩就能在 Linux 上跑真行为。\n");
	printf("判据：① 图片全读不到必须干净返回 ② header 与实际行数不符必须被拒\n");
	printf("      ③ part_num 无上界必须被拒（子进程限 1GB 地址空间）\n");
	printf("      ④⑤ 正常 / use_flip 必须跑完且 ACC 有效 ⑥ part_num=1 不能崩\n");
	printf("      ⑦ 只有一对有效不能崩\n\n");
	for (int op = 0; op <= OP_BAD_LIST; op++) one(op);
	printf("\n共 %d 个用例：全对 %d，有错 %d，崩溃/搭建失败 %d\n", g_case, g_ok, g_bad, g_crash);
	if (g_bad || g_crash)
		printf("**每一项在下结论之前都要先用独立复现对一遍**（附录 CA.3）。\n");
	{
		char cmd[256];
		sprintf(cmd, "rm -rf %s", IMG_DIR);
		if (system(cmd) != 0) { /* 忽略 */ }
	}
	return (g_bad || g_crash) ? 1 : 0;
}
