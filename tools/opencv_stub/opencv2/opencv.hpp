/* 审计用 OpenCV 最小桩 —— 附录 IH
 *
 * 为什么要桩
 * ----------
 * `ZQlibFaceID/ZQ_FaceIDPrecisionEvaluation.h` include 了 `<opencv2/opencv.hpp>`，
 * 是 ZQlibFaceID 里**唯一一个 include 了 OpenCV 却被当成"评测代码"**的头：
 * 它其实是 `EvaluationOnLFW` 的**全部实现**（解析 list 文件、抽特征、
 * 留一法算阈值、算 FAR/TAR 曲线），逻辑量比同目录多数头都大。
 *
 * 而本机 WSL 里没有 OpenCV（`/usr/include/opencv4` 不存在，`3rdparty/opencv` 只有
 * Windows 的 build），于是这个头**一道行为门禁都跑不了** ——
 * 附录 EH~IJ 这一整轮里它是"已能编译但零覆盖"的那一类。
 *
 * 桩里只放这个头真正用到的东西（**别多写**，多写就变成第二份 OpenCV）：
 *   `cv::Mat`（.data / .step[0] / .empty()）、`cv::imread`、`cv::flip`
 * grep 确认过 ZQ_FaceIDPrecisionEvaluation.h 里没有别的 cv:: 用法。
 *
 * 语义说明
 * --------
 * `imread` 走**真 fopen**：文件不存在就返回空 Mat。
 * 这一条是判据本身的一部分 —— "list 文件里指向的图片全都不存在"
 * 正是触发附录 IH.1 那个空 vector 越界的最短路径，桩必须能造出这个场景。
 * 读到的字节按 8x8x3 填满（192 字节），足够让桩识别器算出可区分的特征。
 *
 * 桩**只在 tools/ 的门禁里通过 -I 生效**，主工程两个构建都看不到它。
 */
#ifndef ZQ_AUDIT_OPENCV_STUB_H_
#define ZQ_AUDIT_OPENCV_STUB_H_
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace cv
{
	class Mat
	{
	public:
		unsigned char* data;     // 被测头按 `imgL.data` 取裸指针
		size_t step[2];          // 被测头按 `imgL.step[0]` 取行跨度（真 OpenCV 里是 Size 的 step[]）
		int _rows, _cols;

		Mat() : data(0), _rows(0), _cols(0) { step[0] = 0; step[1] = 0; }
		~Mat() { release(); }
		// 拷贝构造/赋值：ZQ_FaceIDPrecisionEvaluation.h 里全是
		// `cv::Mat imgL = cv::imread(...)`，走的是值语义。
		Mat(const Mat& o) : data(0), _rows(0), _cols(0) { step[0] = 0; step[1] = 0; *this = o; }
		Mat& operator=(const Mat& o)
		{
			if (this == &o) return *this;
			release();
			if (o.data != 0)
			{
				_rows = o._rows; _cols = o._cols;
				step[0] = o.step[0]; step[1] = o.step[1];
				size_t n = (size_t)_rows * step[0];
				data = (unsigned char*)malloc(n > 0 ? n : 1);
				if (data != 0) memcpy(data, o.data, n);
				else { _rows = 0; _cols = 0; step[0] = 0; }
			}
			return *this;
		}
		bool empty() const { return data == 0 || _rows == 0 || _cols == 0; }
		void release() { if (data) { free(data); data = 0; } _rows = 0; _cols = 0; step[0] = 0; }
	};

	// 8x8x3 的确定性假图。内容来自文件字节，所以 flip 之后特征会变，
	// `pData + feat_dim` 那半段才有内容可比。
	inline Mat imread(const std::string& path, int flags = -1)
	{
		(void)flags;
		Mat m;
		FILE* f = fopen(path.c_str(), "rb");
		if (f == 0) return m;          // **真的去开文件**：路径错就返回空 Mat
		const int R = 8, C = 8, CH = 3;
		size_t n = (size_t)R * C * CH;
		unsigned char* buf = (unsigned char*)malloc(n);
		if (buf == 0) { fclose(f); return m; }
		size_t got = fread(buf, 1, n, f);
		fclose(f);
		if (got != n) { free(buf); return m; }   // 短文件也算读失败
		m.data = buf;
		m._rows = R; m._cols = C;
		m.step[0] = (size_t)C * CH; m.step[1] = CH;
		return m;
	}

	// 只实现 code==1（水平翻转）。真 OpenCV 的 flip 对非连续 Mat 也有定义，
	// 但这个头里所有 Mat 都是 imread 出来、连续的。
	inline void flip(const Mat& src, Mat& dst, int code)
	{
		if (code != 1) return;
		if (src.empty()) { dst.release(); return; }
		Mat out;
		out.data = (unsigned char*)malloc((size_t)src._rows * src.step[0]);
		if (out.data == 0) { dst.release(); return; }
		out._rows = src._rows; out._cols = src._cols;
		out.step[0] = src.step[0]; out.step[1] = src.step[1];
		for (int r = 0; r < src._rows; r++)
		{
			const unsigned char* s = src.data + (size_t)r * src.step[0];
			unsigned char* d = out.data + (size_t)r * out.step[0];
			for (int x = 0; x < src._cols; x++)
				for (int c = 0; c < src.step[1]; c++)
					d[(size_t)x * src.step[1] + c] = s[(size_t)(src._cols - 1 - x) * src.step[1] + c];
		}
		dst = out;   // 允许别名（imgL 同时是 src 和 dst）
	}
}

#endif
