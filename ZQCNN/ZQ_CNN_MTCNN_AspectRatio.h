#ifndef _ZQ_CNN_MTCNN_ASPECT_RATIO_H_
#define _ZQ_CNN_MTCNN_ASPECT_RATIO_H_
#pragma once
#include "ZQ_CNN_Net.h"
#include "ZQ_CNN_BBoxUtils.h"
#include <omp.h>
namespace ZQ
{
	class ZQ_CNN_MTCNN_AspectRatio
	{
	public:
		using string = std::string;
		ZQ_CNN_MTCNN_AspectRatio()
		{
			min_size = 60;
			thresh[0] = 0.6;
			thresh[1] = 0.7;
			thresh[2] = 0.8;
			nms_thresh[0] = 0.4;
			nms_thresh[1] = 0.5;
			nms_thresh[2] = 0.5;
			width = 0;
			height = 0;
			width_half = 0;
			height_half = 0;
			factor = 0.709;
			pnet_overlap_thresh_count = 3;
			pnet_size = 20;
			pnet_stride = 4;
			special_handle_very_big_face = false;
			force_run_pnet_multithread = false;
			show_debug_info = false;
			limit_r_num = 0;
			limit_o_num = 0;
			thread_num = 0;
			rnet_size = 0;
			onet_size = 0;
			lnet_size = 0;
			nms_thresh_per_scale = 0.495f;
		}
		~ZQ_CNN_MTCNN_AspectRatio()
		{

		}

	private:
#if __ARM_NEON
		const int BATCH_SIZE = 16;
#else
		const int BATCH_SIZE = 64;
#endif
		std::vector<ZQ_CNN_Net> pnet, rnet, onet;
		int thread_num;
		float thresh[3], nms_thresh[3];
		int min_size;
		int width, height;
		int width_half, height_half;
		float factor;
		int pnet_overlap_thresh_count;
		int pnet_size;
		int pnet_stride;
		int rnet_size;
		int onet_size;
		int lnet_size;
		bool special_handle_very_big_face;
		float nms_thresh_per_scale;
		bool force_run_pnet_multithread;
		std::vector<float> scales, scales_xhalf, scales_yhalf;
		std::vector<ZQ_CNN_Tensor4D_NHW_C_Align128bit> pnet_images, pnet_images_xhalf, pnet_images_yhalf;
		ZQ_CNN_Tensor4D_NHW_C_Align128bit input, input_xhalf, input_yhalf;
		ZQ_CNN_Tensor4D_NHW_C_Align128bit rnet_image, onet_image;
		bool show_debug_info;
		int limit_r_num;
		int limit_o_num;
	public:
		// 审计修复 2026-10-06（附录 II.21）：顺带把开关**同步给各个 net**。
		// 原来只改本类自己的 flag，而 net 是独立的（`pnet[i].show_debug_info`），
		// 于是「TurnOnShowDebugInfo() 之后 pnet 该不该打」这件事在两处各有一份状态、
		// 且从不互相通知。下面 _Pnet_stage 里那个「临时关掉、跑完恢复」就靠它
		// 才能知道该恢复成什么样。Init 之后才调 TurnOn 也没问题（net 全部建好）。
		void TurnOnShowDebugInfo()
		{
			show_debug_info = true;
			for (size_t i = 0; i < pnet.size(); i++) pnet[i].TurnOnShowDebugInfo();
			for (size_t i = 0; i < rnet.size(); i++) rnet[i].TurnOnShowDebugInfo();
			for (size_t i = 0; i < onet.size(); i++) onet[i].TurnOnShowDebugInfo();
			// 本变体没有 lnet 成员（它走 xhalf/yhalf 三族分派，不是 lnet 那一族）
		}
		void TurnOffShowDebugInfo()
		{
			show_debug_info = false;
			for (size_t i = 0; i < pnet.size(); i++) pnet[i].TurnOffShowDebugInfo();
			for (size_t i = 0; i < rnet.size(); i++) rnet[i].TurnOffShowDebugInfo();
			for (size_t i = 0; i < onet.size(); i++) onet[i].TurnOffShowDebugInfo();
			// 同上：没有 lnet 成员
		}		void SetLimit(int limit_r = 0, int limit_o = 0)
		{
			limit_r_num = limit_r;
			limit_o_num = limit_o;
		}

		bool Init(const string& pnet_param, const string& pnet_model, const string& rnet_param, const string& rnet_model,
			const string& onet_param, const string& onet_model, int thread_num = 1)
		{
			if (thread_num < 1)
				force_run_pnet_multithread = true;
			else
				force_run_pnet_multithread = false;
			thread_num = __max(1, thread_num);
			// 审计修复 2026-10-06（附录 II.13）：原来只夹**下界**。
			// 下面按 thread_num 份**逐份 LoadFrom**，每份都是一整套网络 + 权重 —— 
			// 传 100000 就是 30 万份模型常驻内存，直接 OOM / 换页失败。
			// Init 里除了 ret 之外没有任何资源预算，调用方给个离谱值就能把进程打死。
			// 上界取 128：超过 CPU 核数那么多份没有任何意义（每份独占一份 net 就是为了并行），
			// 而 128 份已经远超任何真实机器的核数，同时把最坏情况钉在一个可预期的量级。
			if (thread_num > 128)
			{
				printf("thread_num = %d is too large, clamp to 128\n", thread_num);
				thread_num = 128;
			}
			pnet.resize(thread_num);
			rnet.resize(thread_num);
			onet.resize(thread_num);
			bool ret = true;
			for (int i = 0; i < thread_num; i++)
			{
				ret = pnet[i].LoadFrom(pnet_param, pnet_model, true, 1e-9, true)
					&& rnet[i].LoadFrom(rnet_param, rnet_model, true, 1e-9, true)
					&& onet[i].LoadFrom(onet_param, onet_model, true, 1e-9, true);
				if (!ret)
					break;
			}
			if (!ret)
			{
				pnet.clear();
				rnet.clear();
				onet.clear();
				this->thread_num = 0;
			}
			else
				this->thread_num = thread_num;

			if (!ret)
				return false;
			if (show_debug_info)
			{
				printf("rnet = %.1f M, onet = %.1f M\n", rnet[0].GetNumOfMulAdd() / (1024.0*1024.0),
					onet[0].GetNumOfMulAdd() / (1024.0*1024.0));
			}
			int C, H, W;
			rnet[0].GetInputDim(C, H, W);
			rnet_size = H;
			onet[0].GetInputDim(C, H, W);
			onet_size = H;
			return ret;
		}

		bool InitFromBuffer(
			const char* pnet_param, __int64 pnet_param_len, const char* pnet_model, __int64 pnet_model_len,
			const char* rnet_param, __int64 rnet_param_len, const char* rnet_model, __int64 rnet_model_len,
			const char* onet_param, __int64 onet_param_len, const char* onet_model, __int64 onet_model_len,
			int thread_num = 1)
		{
			if (thread_num < 1)
				force_run_pnet_multithread = true;
			else
				force_run_pnet_multithread = false;
			thread_num = __max(1, thread_num);
			pnet.resize(thread_num);
			rnet.resize(thread_num);
			onet.resize(thread_num);
			
			bool ret = true;
			for (int i = 0; i < thread_num; i++)
			{
				ret = pnet[i].LoadFromBuffer(pnet_param, pnet_param_len, pnet_model, pnet_model_len, true, 1e-9, true)
					&& rnet[i].LoadFromBuffer(rnet_param, rnet_param_len, rnet_model, rnet_model_len, true, 1e-9, true)
					&& onet[i].LoadFromBuffer(onet_param, onet_param_len, onet_model, onet_model_len, true, 1e-9, true);
				if (!ret)
					break;
			}
			if (!ret)
			{
				pnet.clear();
				rnet.clear();
				onet.clear();
				this->thread_num = 0;
			}
			else
				this->thread_num = thread_num;

			if (!ret)
				return false;
			if (show_debug_info)
			{
				printf("rnet = %.1f M, onet = %.1f M\n", rnet[0].GetNumOfMulAdd() / (1024.0*1024.0),
					onet[0].GetNumOfMulAdd() / (1024.0*1024.0));
			}
			int C, H, W;
			rnet[0].GetInputDim(C, H, W);
			rnet_size = H;
			onet[0].GetInputDim(C, H, W);
			onet_size = H;
			return ret;
		}

		void SetPara(int w, int h, int min_face_size = 60, float pthresh = 0.6, float rthresh = 0.7, float othresh = 0.7,
			float nms_pthresh = 0.4, float nms_rthresh = 0.5, float nms_othresh = 0.5, float scale_factor = 0.709,
			int pnet_overlap_thresh_count = 3, int pnet_size = 20, int pnet_stride = 4, bool special_handle_very_big_face = false)
		{
			// 审计修复 2026-10-06（附录 II.14）：SetPara 公开、没有 w/h 的任何校验。
			// `float minside = __min(width, height);` 在 w 或 h 为 0 时是 0，
			// 于是 `scales.push_back((float)pnet_size / minside)` 得到 **+inf**，
			// 消费端 `(int)ceil(height * scales[i])` 是 float->int 的**未定义行为**，
			// 返回值（实现定义的垃圾）再进 `if (changedH < pnet_size) continue;` ——
			// 判据本身随之失效，后面所有几何全错。夹到 1 让金字塔退化成最小规模而不是 UB。
			if (w <= 0 || h <= 0)
			{
				printf("SetPara: invalid size %dx%d, clamp to 1x1\n", w, h);
				w = __max(1, w); h = __max(1, h);
			}
			min_size = __max(__max(1, pnet_size), min_face_size);
			thresh[0] = __max(0.1, pthresh); thresh[1] = __max(0.1, rthresh); thresh[2] = __max(0.1, othresh);
			nms_thresh[0] = __max(0.1, nms_pthresh); nms_thresh[1] = __max(0.1, nms_rthresh); nms_thresh[2] = __max(0.1, nms_othresh);
			// 审计修复 2026-10-06（附录 II.7）：原来这一行改的是**形参** `scale_factor`，
			// 成员 `factor` 从构造函数（factor = 0.709）起**从没被赋过值**。
			// 后果有两条，都是静默的：
			//   1. 调用方传的 `scale_factor` **完全无效** —— 传 0.5 也照样按 0.709 建金字塔；
			//   2. 下面的 `factor != scale_factor` 变成拿 0.709 跟**调用方传的原始值**比。
			//      只要调用方没恰好传 0.709（哪怕只是 0.7），条件**恒真**，于是每次 SetPara 
			//      都 `scales.clear()` + `pnet_images.clear()` 全量重建 —— 
			//      大图上这是「每次调用都把所有 scale 的图像重新分配一遍」。
			// 改成先把夹取结果落到局部量，成员在重建块**之前**赋新值（旧值先留副本给比较用）。
			const float new_factor = (float)__max(0.5f, __min(0.97f, scale_factor));
			const float old_factor = factor;
			this->factor = new_factor;
			this->pnet_overlap_thresh_count = __max(0, pnet_overlap_thresh_count);
			/* pnet_size/pnet_stride 分别是 while 的终止条件与整数除法的除数，
			   不校验的话 stride=0 直接整数除零 SIGFPE，size<0 时 minside 衰减到 0
			   仍满足 while 条件，scales 无限增长直到 OOM。夹到 1。 */
			// 审计修复 2026-10-06（附录 II.4）：先记下**旧值**。
			// 下面的失效判断要比较 pnet_size / min_size / special_handle_very_big_face
			// 有没有变，而紧接着的几行就把它们改掉了 —— 不留副本的话只能拿新值比新值，
			// 判断恒为「没变」。这正是原来漏掉这三个参数的原因。
			int old_pnet_size = this->pnet_size;
			int old_min_size = min_size;
			bool old_special_big = special_handle_very_big_face;
			this->pnet_size = __max(1, pnet_size);
			this->pnet_stride = __max(1, pnet_stride);
			this->special_handle_very_big_face = special_handle_very_big_face;
			if (pnet_size == 20 && pnet_stride == 4)
				nms_thresh_per_scale = 0.45;
			else
				nms_thresh_per_scale = 0.495;
			// 审计修复 2026-10-06（附录 II.4）：原来只比 width/height/scale_factor。
			// 但 scales / pnet_images 的**生成**还依赖 pnet_size、min_size 和
			// special_handle_very_big_face（`MIN_DET_SIZE = pnet_size`、`m = MIN_DET_SIZE/min_size`、
			// `ceil(scales[i]*minside) <= pnet_size` 那个过滤器、
			// `scales.push_back((float)pnet_size / minside)`）。
			// 同一对象二次 SetPara 只改 pnet_size 的话，缓存下来的 scale 全部按旧值生成：
			//   · 几何全错位；
			//   · 更糟的是它会打破「没有任何 scale 被 pnet_size 过滤」这个**没有写下来的不变量**，
			//     于是 scale_num < scales.size()，多线程路径的 `maps[scale_id][...]` 越界**写**
			//     （附录 II.3：mapH/mapW/maps 按「通过过滤的个数」建紧凑下标，
			//     而 task_scale_id 存的是 scales 的全局下标）。
			// 五个 MTCNN 变体的这一处是同一个形状，一并改，避免变体行为分叉。
			if (width != w || height != h || old_factor != new_factor
				|| old_pnet_size != this->pnet_size || old_min_size != min_size
				|| old_special_big != special_handle_very_big_face)
			{
				scales.clear();
				pnet_images.clear();

				width = w; height = h;
				float minside = __min(width, height);
				int MIN_DET_SIZE = pnet_size;
				float m = (float)MIN_DET_SIZE / min_size;
				minside *= m;
				while (minside > MIN_DET_SIZE)
				{
					scales.push_back(m);
					minside *= factor;
					m *= factor;
				}
				minside = __min(width, height);
				int count = scales.size();
				for (int i = scales.size() - 1; i >= 0; i--)
				{
					if (ceil(scales[i] * minside) <= pnet_size)
					{
						count--;
					}
				}
				if (special_handle_very_big_face)
				{
					if (count > 2)
						count--;

					scales.resize(count);
					if (count > 0)
					{
						float last_size = ceil(scales[count - 1] * minside);
			// 审计修复 2026-10-06（附录 II.15）：这个循环的次数是 ~minside/2，没有上界。
			// 20000x20000 的图 -> 近 1 万个 scale -> pnet_images.resize(1万)，
			// 随后每个都被 ResizeBilinear 分配 3x120x120x4 字节，**GB 级内存**。
			// 另外 last_size > INT_MAX 时 `int tmp_size = last_size - 1` 本身就是 UB。
			// 上界取 2000：即便 pnet_size 最小（12）也远超实际需要，
			// 而 2000 个 scale 的分数表仍然在可接受量级内。
						for (int tmp_size = last_size - 1; tmp_size >= pnet_size + 1 && count < 2000; tmp_size -= 2)
						{
							scales.push_back((float)tmp_size / minside);
							count++;
						}
					}

					scales.push_back((float)pnet_size / minside);
					count++;
				}
				else
				{
					scales.push_back((float)pnet_size / minside);
					count++;
				}

				pnet_images.resize(count);
				// 审计修复 2026-10-06（附录 II.3）：把「没有任何 scale 会被
				// `changedH < pnet_size || changedW < pnet_size` 过滤掉」这个**没有写下来的不变量**
				// 变成显式保证。整条流水线把两个**不同的下标空间**混着用：
				//   · mapH/mapW/maps 只为「通过过滤的 scale」建 —— 紧凑下标 0..scale_num-1；
				//   · task_scale_id.push_back(i) 存的是 scales 里的**全局**下标 i；
				//   · 消费端 `for (i = 0; i < maps.size(); i++)` 用紧凑下标 i 去取 scales[i]。
				// 只要有一个 scale 被过滤，scale_num < scales.size()，三处**同时**错位，
				// 其中 maps[scale_id] 那处是越界**写**。
				// 与其把三处都改成双下标（五个变体十几处，风险大），不如在源头把不变量做实：
				// 直接剔掉不满足条件的 scale，于是紧凑下标 == 全局下标，两种用法都恒成立。
				// 注意这里必须**保持原有顺序**（scales 是升序的，后面的 NMS 依赖它）。
			{
				int kept = 0;
				for (int i = 0; i < (int)scales.size(); i++)
			{
				int changedH = (int)ceil(height * scales[i]);
				int changedW = (int)ceil(width * scales[i]);
				if (changedH < pnet_size || changedW < pnet_size)
					continue;
				scales[kept++] = scales[i];
			}
				if (kept != (int)scales.size())
			{
				scales.resize(kept);
				count = kept;
			}
			}
			}

			if (width_half != w/2)
			{
				scales_xhalf.clear();
				pnet_images_xhalf.clear();

				width_half = w / 2;
				float minside = __min(width_half, height);
				int MIN_DET_SIZE = pnet_size;
				float m = (float)MIN_DET_SIZE / min_size;
				minside *= m;
				while (minside > MIN_DET_SIZE)
				{
					scales_xhalf.push_back(m);
					minside *= factor;
					m *= factor;
				}
				minside = __min(width_half, height);
				int count = scales_xhalf.size();
				for (int i = scales_xhalf.size() - 1; i >= 0; i--)
				{
					if (ceil(scales_xhalf[i] * minside) <= pnet_size)
					{
						count--;
					}
				}
				if (special_handle_very_big_face)
				{
					if (count > 2)
						count--;

					scales_xhalf.resize(count);
					if (count > 0)
					{
						float last_size = ceil(scales_xhalf[count - 1] * minside);
			// 审计修复 2026-10-06（附录 II.15）：这个循环的次数是 ~minside/2，没有上界。
			// 20000x20000 的图 -> 近 1 万个 scale -> pnet_images.resize(1万)，
			// 随后每个都被 ResizeBilinear 分配 3x120x120x4 字节，**GB 级内存**。
			// 另外 last_size > INT_MAX 时 `int tmp_size = last_size - 1` 本身就是 UB。
			// 上界取 2000：即便 pnet_size 最小（12）也远超实际需要，
			// 而 2000 个 scale 的分数表仍然在可接受量级内。
						for (int tmp_size = last_size - 1; tmp_size >= pnet_size + 1 && count < 2000; tmp_size -= 2)
						{
							scales_xhalf.push_back((float)tmp_size / minside);
							count++;
						}
					}

					scales_xhalf.push_back((float)pnet_size / minside);
					count++;
				}
				else
				{
					scales_xhalf.push_back((float)pnet_size / minside);
					count++;
				}

				pnet_images_xhalf.resize(count);
			}

			if (height_half != h / 2)
			{
				scales_yhalf.clear();
				pnet_images_yhalf.clear();

				height_half = h / 2;
				float minside = __min(width, height_half);
				int MIN_DET_SIZE = pnet_size;
				float m = (float)MIN_DET_SIZE / min_size;
				minside *= m;
				while (minside > MIN_DET_SIZE)
				{
					scales_yhalf.push_back(m);
					minside *= factor;
					m *= factor;
				}
				minside = __min(width, height_half);
				int count = scales_yhalf.size();
				for (int i = scales_yhalf.size() - 1; i >= 0; i--)
				{
					if (ceil(scales_yhalf[i] * minside) <= pnet_size)
					{
						count--;
					}
				}
				if (special_handle_very_big_face)
				{
					if (count > 2)
						count--;

					scales_yhalf.resize(count);
					if (count > 0)
					{
						float last_size = ceil(scales_yhalf[count - 1] * minside);
			// 审计修复 2026-10-06（附录 II.15）：这个循环的次数是 ~minside/2，没有上界。
			// 20000x20000 的图 -> 近 1 万个 scale -> pnet_images.resize(1万)，
			// 随后每个都被 ResizeBilinear 分配 3x120x120x4 字节，**GB 级内存**。
			// 另外 last_size > INT_MAX 时 `int tmp_size = last_size - 1` 本身就是 UB。
			// 上界取 2000：即便 pnet_size 最小（12）也远超实际需要，
			// 而 2000 个 scale 的分数表仍然在可接受量级内。
						for (int tmp_size = last_size - 1; tmp_size >= pnet_size + 1 && count < 2000; tmp_size -= 2)
						{
							scales_yhalf.push_back((float)tmp_size / minside);
							count++;
						}
					}

					scales_yhalf.push_back((float)pnet_size / minside);
					count++;
				}
				else
				{
					scales_yhalf.push_back((float)pnet_size / minside);
					count++;
				}

				pnet_images_yhalf.resize(count);
			}
		}

		bool Find(const unsigned char* bgr_img, int _width, int _height, int _widthStep, std::vector<ZQ_CNN_BBox>& results)
		{
			// 审计修复 2026-10-06（附录 II.17）：公开入口**自己**校验像素缓冲。
			// `_Pnet_stage` 里也有一道（同一批加的），但那是**下游**：
			// 公开 API 的参数契约应该在入口就成立，不能依赖「我恰好调的那个函数会检查」。
			// 三种笔误的后果：`bgr_img == nullptr` 空指针解引用；
			// `_widthStep <= 0` 时 `BGR_img + h*_widthStep` 越过缓冲**前端**；
			// `_widthStep < _width*3` 时本行最后一个像素读到下一行，
			// 到了**最后一行**就越过整个缓冲末尾 —— 堆越界**读**。
			if (bgr_img == 0 || _width <= 0 || _height <= 0 || _widthStep < _width * 3)
			{
				printf("Find: bad bgr buffer (img=%p, %dx%d, step=%d)\n",
					(void*)bgr_img, _width, _height, _widthStep);
				return false;
			}
			double t1 = omp_get_wtime();
			std::vector<ZQ_CNN_BBox> firstBbox, secondBbox, thirdBbox;
			if (!_Pnet_stage(bgr_img, _width, _height, _widthStep, firstBbox))
				return false;
			//results = firstBbox;
			//return true;
			if (limit_r_num > 0)
			{
				_select(firstBbox, limit_r_num, _width, _height);
			}

			double t2 = omp_get_wtime();
			if (!_Rnet_stage(firstBbox, secondBbox))
				return false;

			//results = secondBbox;
			//return true;

			if (limit_o_num > 0)
			{
				_select(secondBbox, limit_o_num, _width, _height);
			}

			double t3 = omp_get_wtime();
			if (!_Onet_stage(secondBbox, results))
				return false;

			double t4 = omp_get_wtime();
			if (show_debug_info)
			{
				printf("final found num: %d\n", (int)results.size());
				printf("total cost: %.3f ms (P: %.3f ms, R: %.3f ms, O: %.3f ms)\n",
					1000 * (t4 - t1), 1000 * (t2 - t1), 1000 * (t3 - t2), 1000 * (t4 - t3));
			}
			return true;
		}

	private:
		void _compute_Pnet_single_thread(std::vector<std::vector<float> >& maps,
			std::vector<int>& mapH, std::vector<int>& mapW, int& ori_num, int& xhalf_num, int& yhalf_num)
		{
			int scale_num = 0;
			for (int i = 0; i < scales.size(); i++)
			{
				int changedH = (int)ceil(height*scales[i]);
				int changedW = (int)ceil(width*scales[i]);
				if (changedH < pnet_size || changedW < pnet_size)
					continue;
				scale_num++;
				mapH.push_back((changedH - pnet_size) / pnet_stride + 1);
				mapW.push_back((changedW - pnet_size) / pnet_stride + 1);
			}
			ori_num = scale_num;
			scale_num = 0;
			for (int i = 0; i < scales_xhalf.size(); i++)
			{
				int changedH = (int)ceil(height*scales_xhalf[i]);
				int changedW = (int)ceil(width_half*scales_xhalf[i]);
				if (changedH < pnet_size || changedW < pnet_size)
					continue;
				scale_num++;
				mapH.push_back((changedH - pnet_size) / pnet_stride + 1);
				mapW.push_back((changedW - pnet_size) / pnet_stride + 1);
			}
			xhalf_num = scale_num;
			scale_num = 0;
			for (int i = 0; i < scales_yhalf.size(); i++)
			{
				int changedH = (int)ceil(height_half*scales_yhalf[i]);
				int changedW = (int)ceil(width*scales_yhalf[i]);
				if (changedH < pnet_size || changedW < pnet_size)
					continue;
				scale_num++;
				mapH.push_back((changedH - pnet_size) / pnet_stride + 1);
				mapW.push_back((changedW - pnet_size) / pnet_stride + 1);
			}
			yhalf_num = scale_num;
			int total_scale_num = ori_num + xhalf_num + yhalf_num;
			maps.resize(total_scale_num);
			for (int i = 0; i < total_scale_num; i++)
			{
				maps[i].resize(mapH[i] * mapW[i]);
			}

			for (int i = 0; i < total_scale_num; i++)
			{
				if (i < ori_num)
				{
					int changedH = (int)ceil(height*scales[i]);
					int changedW = (int)ceil(width*scales[i]);
					float cur_scale_x = (float)width / changedW;
					float cur_scale_y = (float)height / changedH;
					double t10 = omp_get_wtime();
					if (scales[i] != 1)
					{
						input.ResizeBilinear(pnet_images[i], changedW, changedH, 0, 0);
					}

					double t11 = omp_get_wtime();
					if (scales[i] != 1)
						pnet[0].Forward(pnet_images[i]);
					else
						pnet[0].Forward(input);
					double t12 = omp_get_wtime();
					if (show_debug_info)
						printf("Pnet [%d]: resolution [%dx%d], resize:%.3f ms, cost:%.3f ms\n",
							i, changedW, changedH, 1000 * (t11 - t10), 1000 * (t12 - t11));
					const ZQ_CNN_Tensor4D* score = pnet[0].GetBlobByName("prob1");
					// 审计修复 2026-10-06（附录 II.12）：GetBlobByName 找不到就返回 **0**
					// （ZQ_CNN_Net.h:295-300 / ZQ_CNN_Net_Interface），而 Init 对 blob 名**零校验**、
					// SetPara / Find 也不校验 —— 传一个概率层不叫 prob1 的模型进来，
					// 下面 score->GetH() / score->GetFirstPixelPtr() 就是空指针解引用。
					// 耐人寻味的是同一个函数里 keyPoint **有**判空（`if (keyPoint != 0)`），
					// score / location 没有 —— 判据不一致本身就是信号。
					if (score == 0) { printf("[MTCNN] blob not found in pnet\n"); continue; }
					//score p
					int scoreH = score->GetH();
					int scoreW = score->GetW();
					int scorePixStep = score->GetPixelStep();
					const float *p = score->GetFirstPixelPtr() + 1;
					for (int row = 0; row < scoreH; row++)
					{
						for (int col = 0; col < scoreW; col++)
						{
							if (row < mapH[i] && col < mapW[i])
								maps[i][row*mapW[i] + col] = *p;
							p += scorePixStep;
						}
					}
				}
				else if (i < ori_num + xhalf_num)
				{
					int j = i - ori_num;
					int changedH = (int)ceil(height*scales_xhalf[j]);
					int changedW = (int)ceil(width_half*scales_xhalf[j]);
					float cur_scale_x = (float)width_half / changedW;
					float cur_scale_y = (float)height / changedH;
					double t10 = omp_get_wtime();
					if (scales_xhalf[j] != 1)
					{
						input_xhalf.ResizeBilinear(pnet_images_xhalf[j], changedW, changedH, 0, 0);
					}

					double t11 = omp_get_wtime();
					if (scales_xhalf[j] != 1)
						pnet[0].Forward(pnet_images_xhalf[j]);
					else
						pnet[0].Forward(input_xhalf);
					double t12 = omp_get_wtime();
					if (show_debug_info)
						printf("Pnet [%d]: resolution [%dx%d], resize:%.3f ms, cost:%.3f ms\n",
							i, changedW, changedH, 1000 * (t11 - t10), 1000 * (t12 - t11));
					const ZQ_CNN_Tensor4D* score = pnet[0].GetBlobByName("prob1");
					// 审计修复 2026-10-06（附录 II.12）：GetBlobByName 找不到就返回 **0**
					// （ZQ_CNN_Net.h:295-300 / ZQ_CNN_Net_Interface），而 Init 对 blob 名**零校验**、
					// SetPara / Find 也不校验 —— 传一个概率层不叫 prob1 的模型进来，
					// 下面 score->GetH() / score->GetFirstPixelPtr() 就是空指针解引用。
					// 耐人寻味的是同一个函数里 keyPoint **有**判空（`if (keyPoint != 0)`），
					// score / location 没有 —— 判据不一致本身就是信号。
					if (score == 0) { printf("[MTCNN] blob not found in pnet\n"); continue; }
					//score p
					int scoreH = score->GetH();
					int scoreW = score->GetW();
					int scorePixStep = score->GetPixelStep();
					const float *p = score->GetFirstPixelPtr() + 1;
					for (int row = 0; row < scoreH; row++)
					{
						for (int col = 0; col < scoreW; col++)
						{
							if (row < mapH[i] && col < mapW[i])
								maps[i][row*mapW[i] + col] = *p;
							p += scorePixStep;
						}
					}
				}
				else
				{
					int k = i - ori_num - xhalf_num;
					int changedH = (int)ceil(height*scales_xhalf[k]);
					int changedW = (int)ceil(width_half*scales_xhalf[k]);
					float cur_scale_x = (float)width_half / changedW;
					float cur_scale_y = (float)height / changedH;
					double t10 = omp_get_wtime();
					if (scales_yhalf[k] != 1)
					{
						input_yhalf.ResizeBilinear(pnet_images_yhalf[k], changedW, changedH, 0, 0);
					}

					double t11 = omp_get_wtime();
					if (scales_yhalf[k] != 1)
						pnet[0].Forward(pnet_images_yhalf[k]);
					else
						pnet[0].Forward(input_yhalf);
					double t12 = omp_get_wtime();
					if (show_debug_info)
						printf("Pnet [%d]: resolution [%dx%d], resize:%.3f ms, cost:%.3f ms\n",
							i, changedW, changedH, 1000 * (t11 - t10), 1000 * (t12 - t11));
					const ZQ_CNN_Tensor4D* score = pnet[0].GetBlobByName("prob1");
					// 审计修复 2026-10-06（附录 II.12）：GetBlobByName 找不到就返回 **0**
					// （ZQ_CNN_Net.h:295-300 / ZQ_CNN_Net_Interface），而 Init 对 blob 名**零校验**、
					// SetPara / Find 也不校验 —— 传一个概率层不叫 prob1 的模型进来，
					// 下面 score->GetH() / score->GetFirstPixelPtr() 就是空指针解引用。
					// 耐人寻味的是同一个函数里 keyPoint **有**判空（`if (keyPoint != 0)`），
					// score / location 没有 —— 判据不一致本身就是信号。
					if (score == 0) { printf("[MTCNN] blob not found in pnet\n"); continue; }
					//score p
					int scoreH = score->GetH();
					int scoreW = score->GetW();
					int scorePixStep = score->GetPixelStep();
					const float *p = score->GetFirstPixelPtr() + 1;
					for (int row = 0; row < scoreH; row++)
					{
						for (int col = 0; col < scoreW; col++)
						{
							if (row < mapH[i] && col < mapW[i])
								maps[i][row*mapW[i] + col] = *p;
							p += scorePixStep;
						}
					}
				}	
			}
		}

		void _compute_Pnet_multi_thread(std::vector<std::vector<float> >& maps,
			std::vector<int>& mapH, std::vector<int>& mapW, int& ori_num, int& xhalf_num, int& yhalf_num)
		{
			if (thread_num <= 1)
			{
				for (int i = 0; i < scales.size(); i++)
				{
					int changedH = (int)ceil(height*scales[i]);
					int changedW = (int)ceil(width*scales[i]);
					if (changedH < pnet_size || changedW < pnet_size)
						continue;
					if (scales[i] != 1)
					{
						input.ResizeBilinear(pnet_images[i], changedW, changedH, 0, 0);
					}
				}
				for (int i = 0; i < scales_xhalf.size(); i++)
				{
					int changedH = (int)ceil(height*scales_xhalf[i]);
					int changedW = (int)ceil(width_half*scales_xhalf[i]);
					if (changedH < pnet_size || changedW < pnet_size)
						continue;
					if (scales_xhalf[i] != 1)
					{
						input_xhalf.ResizeBilinear(pnet_images_xhalf[i], changedW, changedH, 0, 0);
					}
				}
				for (int i = 0; i < scales_yhalf.size(); i++)
				{
					int changedH = (int)ceil(height_half*scales_yhalf[i]);
					int changedW = (int)ceil(width*scales_yhalf[i]);
					if (changedH < pnet_size || changedW < pnet_size)
						continue;
					if (scales_yhalf[i] != 1)
					{
						input_yhalf.ResizeBilinear(pnet_images_yhalf[i], changedW, changedH, 0, 0);
					}
				}
			}
			else
			{
				ori_num = scales.size();
				xhalf_num = scales_xhalf.size();
				yhalf_num = scales_yhalf.size();
				int total_scale_num = ori_num + xhalf_num + yhalf_num;
#pragma omp parallel for num_threads(thread_num) schedule(dynamic, 1)
				for (int i = 0; i < total_scale_num; i++)
				{
					if (i < ori_num)
					{
						int changedH = (int)ceil(height*scales[i]);
						int changedW = (int)ceil(width*scales[i]);
						if (changedH < pnet_size || changedW < pnet_size)
							continue;
						if (scales[i] != 1)
						{
							input.ResizeBilinear(pnet_images[i], changedW, changedH, 0, 0);
						}
					}
					else if (i < ori_num + xhalf_num)
					{
						int j = i - ori_num;
						int changedH = (int)ceil(height*scales_xhalf[j]);
						int changedW = (int)ceil(width_half*scales_xhalf[j]);
						if (changedH < pnet_size || changedW < pnet_size)
							continue;
						if (scales_xhalf[j] != 1)
						{
							input_xhalf.ResizeBilinear(pnet_images_xhalf[j], changedW, changedH, 0, 0);
						}
					}
					else
					{
						int k = i - ori_num - xhalf_num;
						int changedH = (int)ceil(height_half*scales_yhalf[k]);
						int changedW = (int)ceil(width*scales_yhalf[k]);
						if (changedH < pnet_size || changedW < pnet_size)
							continue;
						if (scales_yhalf[k] != 1)
						{
							input_yhalf.ResizeBilinear(pnet_images_yhalf[k], changedW, changedH, 0, 0);
						}
					}
				}
			}
			int scale_num = 0;
			for (int i = 0; i < scales.size(); i++)
			{
				int changedH = (int)ceil(height*scales[i]);
				int changedW = (int)ceil(width*scales[i]);
				if (changedH < pnet_size || changedW < pnet_size)
					continue;
				scale_num++;
				mapH.push_back((changedH - pnet_size) / pnet_stride + 1);
				mapW.push_back((changedW - pnet_size) / pnet_stride + 1);
			}
			ori_num = scale_num;
			scale_num = 0;
			for (int i = 0; i < scales_xhalf.size(); i++)
			{
				int changedH = (int)ceil(height*scales_xhalf[i]);
				int changedW = (int)ceil(width_half*scales_xhalf[i]);
				if (changedH < pnet_size || changedW < pnet_size)
					continue;
				scale_num++;
				mapH.push_back((changedH - pnet_size) / pnet_stride + 1);
				mapW.push_back((changedW - pnet_size) / pnet_stride + 1);
			}
			xhalf_num = scale_num;
			scale_num = 0;
			for (int i = 0; i < scales_yhalf.size(); i++)
			{
				int changedH = (int)ceil(height_half*scales_yhalf[i]);
				int changedW = (int)ceil(width*scales_yhalf[i]);
				if (changedH < pnet_size || changedW < pnet_size)
					continue;
				scale_num++;
				mapH.push_back((changedH - pnet_size) / pnet_stride + 1);
				mapW.push_back((changedW - pnet_size) / pnet_stride + 1);
			}
			yhalf_num = scale_num;
			int total_scale_num = ori_num + xhalf_num + yhalf_num;
			maps.resize(total_scale_num);
			for (int i = 0; i < total_scale_num; i++)
			{
				maps[i].resize(mapH[i] * mapW[i]);
			}

			std::vector<int> task_rect_off_x;
			std::vector<int> task_rect_off_y;
			std::vector<int> task_rect_width;
			std::vector<int> task_rect_height;
			std::vector<float> task_scale;
			std::vector<int> task_scale_id;

			int stride = pnet_stride;
			const int block_size = 64 * stride;
			int cellsize = pnet_size;
			int border_size = cellsize - stride;
			int overlap_border_size = cellsize / stride;
			int jump_size = block_size - border_size;
			for (int i = 0; i < total_scale_num; i++)
			{
				if (i < ori_num)
				{
					int changeH = (int)ceil(height*scales[i]);
					int changeW = (int)ceil(width*scales[i]);
					if (changeH < pnet_size || changeW < pnet_size)
						continue;
					int block_H_num = 0;
					int block_W_num = 0;
					int start = 0;
					while (start < changeH)
					{
						block_H_num++;
						if (start + block_size >= changeH)
							break;
						start += jump_size;
					}
					start = 0;
					while (start < changeW)
					{
						block_W_num++;
						if (start + block_size >= changeW)
							break;
						start += jump_size;
					}
					for (int s = 0; s < block_H_num; s++)
					{
						for (int t = 0; t < block_W_num; t++)
						{
							int rect_off_x = t * jump_size;
							int rect_off_y = s * jump_size;
							int rect_width = __min(changeW, rect_off_x + block_size) - rect_off_x;
							int rect_height = __min(changeH, rect_off_y + block_size) - rect_off_y;
							if (rect_width >= cellsize && rect_height >= cellsize)
							{
								task_rect_off_x.push_back(rect_off_x);
								task_rect_off_y.push_back(rect_off_y);
								task_rect_width.push_back(rect_width);
								task_rect_height.push_back(rect_height);
								task_scale.push_back(scales[i]);
								task_scale_id.push_back(i);
							}
						}
					}
				}
				else if (i < ori_num + xhalf_num)
				{
					int j = i - ori_num;
					int changeH = (int)ceil(height*scales_xhalf[j]);
					int changeW = (int)ceil(width_half*scales_xhalf[j]);
					if (changeH < pnet_size || changeW < pnet_size)
						continue;
					int block_H_num = 0;
					int block_W_num = 0;
					int start = 0;
					while (start < changeH)
					{
						block_H_num++;
						if (start + block_size >= changeH)
							break;
						start += jump_size;
					}
					start = 0;
					while (start < changeW)
					{
						block_W_num++;
						if (start + block_size >= changeW)
							break;
						start += jump_size;
					}
					for (int s = 0; s < block_H_num; s++)
					{
						for (int t = 0; t < block_W_num; t++)
						{
							int rect_off_x = t * jump_size;
							int rect_off_y = s * jump_size;
							int rect_width = __min(changeW, rect_off_x + block_size) - rect_off_x;
							int rect_height = __min(changeH, rect_off_y + block_size) - rect_off_y;
							if (rect_width >= cellsize && rect_height >= cellsize)
							{
								task_rect_off_x.push_back(rect_off_x);
								task_rect_off_y.push_back(rect_off_y);
								task_rect_width.push_back(rect_width);
								task_rect_height.push_back(rect_height);
								task_scale.push_back(scales_xhalf[j]);
								task_scale_id.push_back(i);
							}
						}
					}
				}
				else
				{
					int k = i - ori_num - xhalf_num;
					int changeH = (int)ceil(height_half*scales_yhalf[k]);
					int changeW = (int)ceil(width*scales_yhalf[k]);
					if (changeH < pnet_size || changeW < pnet_size)
						continue;
					int block_H_num = 0;
					int block_W_num = 0;
					int start = 0;
					while (start < changeH)
					{
						block_H_num++;
						if (start + block_size >= changeH)
							break;
						start += jump_size;
					}
					start = 0;
					while (start < changeW)
					{
						block_W_num++;
						if (start + block_size >= changeW)
							break;
						start += jump_size;
					}
					for (int s = 0; s < block_H_num; s++)
					{
						for (int t = 0; t < block_W_num; t++)
						{
							int rect_off_x = t * jump_size;
							int rect_off_y = s * jump_size;
							int rect_width = __min(changeW, rect_off_x + block_size) - rect_off_x;
							int rect_height = __min(changeH, rect_off_y + block_size) - rect_off_y;
							if (rect_width >= cellsize && rect_height >= cellsize)
							{
								task_rect_off_x.push_back(rect_off_x);
								task_rect_off_y.push_back(rect_off_y);
								task_rect_width.push_back(rect_width);
								task_rect_height.push_back(rect_height);
								task_scale.push_back(scales_yhalf[k]);
								task_scale_id.push_back(i);
							}
						}
					}
				}
			}

			//
			int task_num = task_scale.size();
			std::vector<ZQ_CNN_Tensor4D_NHW_C_Align128bit> task_pnet_images(thread_num);

			if (thread_num <= 1)
			{
				for (int i = 0; i < task_num; i++)
				{
					int thread_id = omp_get_thread_num();
					int scale_id = task_scale_id[i];
					float cur_scale = task_scale[i];
					int i_rect_off_x = task_rect_off_x[i];
					int i_rect_off_y = task_rect_off_y[i];
					int i_rect_width = task_rect_width[i];
					int i_rect_height = task_rect_height[i];
					if (scale_id < ori_num)
					{
						if (scale_id == 0 && scales[0] == 1)
						{
							if (!input.ROI(task_pnet_images[thread_id],
								i_rect_off_x, i_rect_off_y, i_rect_width, i_rect_height, 0, 0))
								continue;
						}
						else
						{
							if (!pnet_images[scale_id].ROI(task_pnet_images[thread_id],
								i_rect_off_x, i_rect_off_y, i_rect_width, i_rect_height, 0, 0))
								continue;
						}
					}
					else if (scale_id < ori_num + xhalf_num)
					{
						int j = scale_id - ori_num;
						if (j == 0 && scales_xhalf[0] == 1)
						{
							if (!input_xhalf.ROI(task_pnet_images[thread_id],
								i_rect_off_x, i_rect_off_y, i_rect_width, i_rect_height, 0, 0))
								continue;
						}
						else
						{
							if (!pnet_images_xhalf[j].ROI(task_pnet_images[thread_id],
								i_rect_off_x, i_rect_off_y, i_rect_width, i_rect_height, 0, 0))
								continue;
						}
					}
					else
					{
						int k = scale_id - ori_num - xhalf_num;
						if (k == 0 && scales_yhalf[0] == 1)
						{
							if (!input_yhalf.ROI(task_pnet_images[thread_id],
								i_rect_off_x, i_rect_off_y, i_rect_width, i_rect_height, 0, 0))
								continue;
						}
						else
						{
							if (!pnet_images_yhalf[k].ROI(task_pnet_images[thread_id],
								i_rect_off_x, i_rect_off_y, i_rect_width, i_rect_height, 0, 0))
								continue;
						}
					}

					if (!pnet[thread_id].Forward(task_pnet_images[thread_id]))
						continue;
					const ZQ_CNN_Tensor4D* score = pnet[thread_id].GetBlobByName("prob1");
					// 审计修复 2026-10-06（附录 II.12）：GetBlobByName 找不到就返回 **0**
					// （ZQ_CNN_Net.h:295-300 / ZQ_CNN_Net_Interface），而 Init 对 blob 名**零校验**、
					// SetPara / Find 也不校验 —— 传一个概率层不叫 prob1 的模型进来，
					// 下面 score->GetH() / score->GetFirstPixelPtr() 就是空指针解引用。
					// 耐人寻味的是同一个函数里 keyPoint **有**判空（`if (keyPoint != 0)`），
					// score / location 没有 —— 判据不一致本身就是信号。
					if (score == 0) { printf("[MTCNN] blob not found in pnet\n"); continue; }

					int task_count = 0;
					//score p
					int scoreH = score->GetH();
					int scoreW = score->GetW();
					int scorePixStep = score->GetPixelStep();
					const float *p = score->GetFirstPixelPtr() + 1;
					ZQ_CNN_BBox bbox;
					ZQ_CNN_OrderScore order;
					for (int row = 0; row < scoreH; row++)
					{
						for (int col = 0; col < scoreW; col++)
						{
							int real_row = row + i_rect_off_y / stride;
							int real_col = col + i_rect_off_x / stride;
							if (real_row < mapH[scale_id] && real_col < mapW[scale_id])
								maps[scale_id][real_row*mapW[scale_id] + real_col] = *p;

							p += scorePixStep;
						}
					}
				}
			}
			else
			{
#pragma omp parallel for num_threads(thread_num)
				for (int i = 0; i < task_num; i++)
				{
					int thread_id = omp_get_thread_num();
					int scale_id = task_scale_id[i];
					float cur_scale = task_scale[i];
					int i_rect_off_x = task_rect_off_x[i];
					int i_rect_off_y = task_rect_off_y[i];
					int i_rect_width = task_rect_width[i];
					int i_rect_height = task_rect_height[i];
// 审计修复 2026-10-06（附录 II.8）：这一支原来用 **`i`（任务下标）** 判族、
// 算 j/k，而下面取张量时又用 `scale_id`。单线程那一份（:905 起）三处
// 全部用 `scale_id`，是正确样板；`if (i < ori_num)` 这种写法在 4 个兄弟头里
// **0 次命中** —— 这份 xhalf/yhalf 分派是本文件独有的。
// 后果：任务表按 scale 顺序追加（外层 i 是 scale、内层是块），所以
// `scale_id <= i`，两者在「一个 scale 产出多个块」时分离。i 越过
// ori_num / ori_num+xhalf_num 后进入错误的族，而 `j = i - ori_num` /
// `k = i - ori_num - xhalf_num` **可以是负数，也可以 >= 容器大小** =>
// `std::vector::operator[]` 越界 -> 在野的 ZQ_CNN_Tensor4D_NHW_C_Align128bit
// 上调 .ROI() -> 野指针解引用。
// 触发条件（thread_num >= 2 且 scales[0] == 1，即 min_face_size <= pnet_size）：
//   1920x1080, min_face=20 -> tasks(scale0)=40 > ori+xhalf=26，k = -26 起
//   3840x2160, min_face=20 -> tasks(scale0)=144 > 30，k = -114 起
// 中间那段（i 落在 xhalf 区间但 scale_id 还是 0）不崩，但会索引到
// **错误的金字塔层** —— 结果全错且完全静默。
					if (scale_id < ori_num)
					{
						if (scale_id == 0 && scales[0] == 1)
						{
							if (!input.ROI(task_pnet_images[thread_id],
								i_rect_off_x, i_rect_off_y, i_rect_width, i_rect_height, 0, 0))
								continue;
						}
						else
						{
							if (!pnet_images[scale_id].ROI(task_pnet_images[thread_id],
								i_rect_off_x, i_rect_off_y, i_rect_width, i_rect_height, 0, 0))
								continue;
						}
					}
					else if (scale_id < ori_num + xhalf_num)
					{
						int j = scale_id - ori_num;
						if (j == 0 && scales_xhalf[0] == 1)
						{
							if (!input_xhalf.ROI(task_pnet_images[thread_id],
								i_rect_off_x, i_rect_off_y, i_rect_width, i_rect_height, 0, 0))
								continue;
						}
						else
						{
							if (!pnet_images_xhalf[j].ROI(task_pnet_images[thread_id],
								i_rect_off_x, i_rect_off_y, i_rect_width, i_rect_height, 0, 0))
								continue;
						}
					}
					else
					{
						int k = scale_id - ori_num - xhalf_num;
						if (k == 0 && scales_yhalf[0] == 1)
						{
							if (!input_yhalf.ROI(task_pnet_images[thread_id],
								i_rect_off_x, i_rect_off_y, i_rect_width, i_rect_height, 0, 0))
								continue;
						}
						else
						{
							if (!pnet_images_yhalf[k].ROI(task_pnet_images[thread_id],
								i_rect_off_x, i_rect_off_y, i_rect_width, i_rect_height, 0, 0))
								continue;
						}
					}

					if (!pnet[thread_id].Forward(task_pnet_images[thread_id]))
						continue;
					const ZQ_CNN_Tensor4D* score = pnet[thread_id].GetBlobByName("prob1");
					// 审计修复 2026-10-06（附录 II.12）：GetBlobByName 找不到就返回 **0**
					// （ZQ_CNN_Net.h:295-300 / ZQ_CNN_Net_Interface），而 Init 对 blob 名**零校验**、
					// SetPara / Find 也不校验 —— 传一个概率层不叫 prob1 的模型进来，
					// 下面 score->GetH() / score->GetFirstPixelPtr() 就是空指针解引用。
					// 耐人寻味的是同一个函数里 keyPoint **有**判空（`if (keyPoint != 0)`），
					// score / location 没有 —— 判据不一致本身就是信号。
					if (score == 0) { printf("[MTCNN] blob not found in pnet\n"); continue; }

					int task_count = 0;
					//score p
					int scoreH = score->GetH();
					int scoreW = score->GetW();
					int scorePixStep = score->GetPixelStep();
					const float *p = score->GetFirstPixelPtr() + 1;
					ZQ_CNN_BBox bbox;
					ZQ_CNN_OrderScore order;
					for (int row = 0; row < scoreH; row++)
					{
						for (int col = 0; col < scoreW; col++)
						{
							int real_row = row + i_rect_off_y / stride;
							int real_col = col + i_rect_off_x / stride;
							if (real_row < mapH[scale_id] && real_col < mapW[scale_id])
								maps[scale_id][real_row*mapW[scale_id] + real_col] = *p;

							p += scorePixStep;
						}
					}
				}
			}
		}

		bool _Pnet_stage(const unsigned char* bgr_img, int _width, int _height, int _widthStep, std::vector<ZQ_CNN_BBox>& firstBbox)
		{
			if (thread_num <= 0)
				return false;

			// 审计修复 2026-10-06（附录 II.17）：bgr 这条入口原来只校验宽高，
			// **不校验像素缓冲本身**：`ConvertFromBGR` 里是
			//     bgr_row = BGR_img + h*_widthStep;  然后 bgr_pix = bgr_row; 逐像素 bgr_pix += 3
			// 于是 ① `bgr_img == nullptr` 立刻空指针解引用；
			// ② `_widthStep <= 0` 时 bgr_row 原地不动或**往回走**，越过缓冲区**前端**读；
			// ③ `_widthStep < _width*3` 时本行最后一个像素会读到下一行，
			//    到了**最后一行**就越过整个缓冲末尾 —— 堆越界**读**。
			// 三种都是调用方一个笔误就能踩到的（传 nullptr、传 0、忘了算对齐填充），
			// 而且症状是随机崩溃或花屏，没有任何提示。
			if (bgr_img == 0 || _width <= 0 || _height <= 0 || _widthStep < _width * 3)
			{
				printf("Find: bad bgr buffer (img=%p, %dx%d, step=%d)\n",
					(void*)bgr_img, _width, _height, _widthStep);
				return false;
			}

			double t1 = omp_get_wtime();
			firstBbox.clear();
			if (width != _width || height != _height)
				return false;
			if (!input.ConvertFromBGR(bgr_img, width, height, _widthStep))
				return false;
			if (!input_yhalf.ConvertFromBGR(bgr_img, width, height / 2, _widthStep * 2))
				return false;
			const __int64 xhalf_size = (__int64)width_half * _height * 3;
			if (xhalf_size <= 0 || xhalf_size > 0x7FFFFFFF)
				return false;
			std::vector<unsigned char> bgr_img_xhalf((size_t)xhalf_size);
			int widthStep_half = width_half * 3;
			for (int i = 0; i < _height; i++)
			{
				for (int j = 0; j < width_half; j++)
				{
					bgr_img_xhalf[i*widthStep_half + j * 3 + 0] = bgr_img[i*_widthStep + j * 6 + 0];
					bgr_img_xhalf[i*widthStep_half + j * 3 + 1] = bgr_img[i*_widthStep + j * 6 + 1];
					bgr_img_xhalf[i*widthStep_half + j * 3 + 2] = bgr_img[i*_widthStep + j * 6 + 2];
				}
			}
			if (!input_xhalf.ConvertFromBGR(&bgr_img_xhalf[0], width_half, height, widthStep_half))
				return false;
			double t2 = omp_get_wtime();
			if (show_debug_info)
				printf("convert cost: %.3f ms\n", 1000 * (t2 - t1));

			std::vector<std::vector<float> > maps;
			std::vector<int> mapH;
			std::vector<int> mapW;
			int ori_num, xhalf_num, yhalf_num;
			if (thread_num == 1 && !force_run_pnet_multithread)
			{
				// 审计修复 2026-10-06（附录 II.21）：原来是 `TurnOffShowDebugInfo()`
				// **只关不恢复**（恢复那一行被注释掉了，就挂在下一行）。
				// 后果：用户 TurnOnShowDebugInfo() 之后，第一次 Find 把 pnet 的调试
				// 永久关掉，之后再也开不回来；而且**多线程分支根本没关**，
				// 于是 thread_num=1 和 thread_num>1 的调试输出**完全相反** ——
				// 这不是取舍，是漏了。改成按本类的 show_debug_info 记下原值再恢复。
				const bool pnet_debug_was = show_debug_info;
				pnet[0].TurnOffShowDebugInfo();
				_compute_Pnet_single_thread(maps, mapH, mapW, ori_num, xhalf_num, yhalf_num);
				if (pnet_debug_was)
					pnet[0].TurnOnShowDebugInfo();   // 恢复调用前的状态（附录 II.21）
			}
			else
			{
				_compute_Pnet_multi_thread(maps, mapH, mapW, ori_num, xhalf_num, yhalf_num);
			}
			int total_scale_num = ori_num + xhalf_num + yhalf_num;
			ZQ_CNN_OrderScore order;
			std::vector<std::vector<ZQ_CNN_BBox> > bounding_boxes(total_scale_num);
			std::vector<std::vector<ZQ_CNN_OrderScore> > bounding_scores(total_scale_num);
			const int block_size = 32;
			int stride = pnet_stride;
			int cellsize = pnet_size;
			int border_size = cellsize / stride;

			for (int i = 0; i < total_scale_num; i++)
			{
				double t13 = omp_get_wtime();
				int changedH, changedW;
				if (i < ori_num)
				{
					changedH = (int)ceil(height*scales[i]);
					changedW = (int)ceil(width*scales[i]);
				}
				else if (i < ori_num + xhalf_num)
				{
					int j = i - ori_num;
					changedH = (int)ceil(height*scales_xhalf[j]);
					changedW = (int)ceil(width_half*scales_xhalf[j]);
				}
				else
				{
					int k = i - ori_num - xhalf_num;
					changedH = (int)ceil(height_half*scales_yhalf[k]);
					changedW = (int)ceil(width*scales_yhalf[k]);
				}
				
				if (changedH < pnet_size || changedW < pnet_size)
					continue;
				float cur_scale_x = (float)width / changedW;
				float cur_scale_y = (float)height / changedH;

				int count = 0;
				//score p
				int scoreH = mapH[i];
				int scoreW = mapW[i];
				const float *p = &maps[i][0];
				if (scoreW <= block_size && scoreH < block_size)
				{

					ZQ_CNN_BBox bbox;
					ZQ_CNN_OrderScore order;
					for (int row = 0; row < scoreH; row++)
					{
						for (int col = 0; col < scoreW; col++)
						{
							if (*p > thresh[0])
							{
								bbox.score = *p;
								order.score = *p;
								order.oriOrder = count;
								bbox.row1 = stride*row;
								bbox.col1 = stride*col;
								bbox.row2 = stride*row + cellsize;
								bbox.col2 = stride*col + cellsize;
								bbox.exist = true;
								bbox.area = (bbox.row2 - bbox.row1)*(bbox.col2 - bbox.col1);
								bbox.need_check_overlap_count = (row >= border_size && row < scoreH - border_size)
									&& (col >= border_size && col < scoreW - border_size);
								bounding_boxes[i].push_back(bbox);
								bounding_scores[i].push_back(order);
								count++;
							}
							p++;
						}
					}
					int before_count = bounding_boxes[i].size();
					ZQ_CNN_BBoxUtils::_nms(bounding_boxes[i], bounding_scores[i], nms_thresh_per_scale, "Union", pnet_overlap_thresh_count);
					int after_count = bounding_boxes[i].size();
					for (int j = 0; j < after_count; j++)
					{
						ZQ_CNN_BBox& bbox = bounding_boxes[i][j];
						bbox.row1 = round(bbox.row1 *cur_scale_y);
						bbox.col1 = round(bbox.col1 *cur_scale_x);
						bbox.row2 = round(bbox.row2 *cur_scale_y);
						bbox.col2 = round(bbox.col2 *cur_scale_x);
						bbox.area = (bbox.row2 - bbox.row1)*(bbox.col2 - bbox.col1);
					}
					double t14 = omp_get_wtime();
					if (show_debug_info)
						printf("nms cost: %.3f ms, (%d-->%d)\n", 1000 * (t14 - t13), before_count, after_count);
				}
				else
				{
					int before_count = 0, after_count = 0;
					int block_H_num = __max(1, scoreH / block_size);
					int block_W_num = __max(1, scoreW / block_size);
					int block_num = block_H_num*block_W_num;
					int width_per_block = scoreW / block_W_num;
					int height_per_block = scoreH / block_H_num;
					std::vector<std::vector<ZQ_CNN_BBox> > tmp_bounding_boxes(block_num);
					std::vector<std::vector<ZQ_CNN_OrderScore> > tmp_bounding_scores(block_num);
					std::vector<int> block_start_w(block_num), block_end_w(block_num);
					std::vector<int> block_start_h(block_num), block_end_h(block_num);
					for (int bh = 0; bh < block_H_num; bh++)
					{
						for (int bw = 0; bw < block_W_num; bw++)
						{
							int bb = bh * block_W_num + bw;
							block_start_w[bb] = (bw == 0) ? 0 : (bw*width_per_block - border_size);
							// 审计修复 2026-10-06（附录 II.11）：原来判的是 `block_num - 1`（**块总数**减一），
							// 应该是 `block_W_num - 1`（**每行的块数**减一）。
							// bb = bh*block_W_num + bw，所以 `bb == block_num-1` 只在「最后一行块的最后一列块」
							// 成立 —— 于是**只有那一个块**延伸到 scoreW/scoreH，
							// 每一行最后 `scoreW - block_W_num*width_per_block` 列、以及前 block_H_num-1 个行块的
							// 最后一整行**从来没被扫过**。不越界（block_start/end 都还在 [0,scoreW] 内），
							// 是**召回率**缺陷：贴边的脸会漏。五个变体同款。
							block_end_w[bb] = (bw == block_W_num - 1) ? scoreW : ((bw + 1)*width_per_block);
							block_start_h[bb] = (bh == 0) ? 0 : (bh*height_per_block - border_size);
							block_end_h[bb] = (bh == block_H_num - 1) ? scoreH : ((bh + 1)*height_per_block);
						}
					}
					int chunk_size = 1;// ceil((float)block_num / thread_num);
					if (thread_num <= 1)
					{
						for (int bb = 0; bb < block_num; bb++)
						{
							ZQ_CNN_BBox bbox;
							ZQ_CNN_OrderScore order;
							int count = 0;
							for (int row = block_start_h[bb]; row < block_end_h[bb]; row++)
							{
								p = &maps[i][0] + row*scoreW + block_start_w[bb];
								for (int col = block_start_w[bb]; col < block_end_w[bb]; col++)
								{
									if (*p > thresh[0])
									{
										bbox.score = *p;
										order.score = *p;
										order.oriOrder = count;
										bbox.row1 = stride*row;
										bbox.col1 = stride*col;
										bbox.row2 = stride*row + cellsize;
										bbox.col2 = stride*col + cellsize;
										bbox.exist = true;
										bbox.need_check_overlap_count = (row >= border_size && row < scoreH - border_size)
											&& (col >= border_size && col < scoreW - border_size);
										bbox.area = (bbox.row2 - bbox.row1)*(bbox.col2 - bbox.col1);
										tmp_bounding_boxes[bb].push_back(bbox);
										tmp_bounding_scores[bb].push_back(order);
										count++;
									}
									p++;
								}
							}
							int tmp_before_count = tmp_bounding_boxes[bb].size();
							ZQ_CNN_BBoxUtils::_nms(tmp_bounding_boxes[bb], tmp_bounding_scores[bb], nms_thresh_per_scale, "Union", pnet_overlap_thresh_count);
							int tmp_after_count = tmp_bounding_boxes[bb].size();
							before_count += tmp_before_count;
							after_count += tmp_after_count;
						}
					}
					else
					{
// 审计修复 2026-10-06（附录 II.10）：这一行原来**缺 reduction 子句** —— 
// `before_count` / `after_count` 是上面声明的共享 int，在 parallel for 里
// 无锁 `+=` 是数据竞争，每次跑打印出来的数字都可能不一样。
// 主副本 `ZQ_CNN_MTCNN.h:932` 早就是带
// `reduction(+:before_count, after_count)` 的写法 —— 这四份是没同步过去的旧拷贝。
#pragma omp parallel for schedule(dynamic, chunk_size) num_threads(thread_num) reduction(+:before_count, after_count)
						for (int bb = 0; bb < block_num; bb++)
						{
							ZQ_CNN_BBox bbox;
							ZQ_CNN_OrderScore order;
							int count = 0;
							for (int row = block_start_h[bb]; row < block_end_h[bb]; row++)
							{
								const float* p = &maps[i][0] + row*scoreW + block_start_w[bb];
								for (int col = block_start_w[bb]; col < block_end_w[bb]; col++)
								{
									if (*p > thresh[0])
									{
										bbox.score = *p;
										order.score = *p;
										order.oriOrder = count;
										bbox.row1 = stride*row;
										bbox.col1 = stride*col;
										bbox.row2 = stride*row + cellsize;
										bbox.col2 = stride*col + cellsize;
										bbox.exist = true;
										bbox.need_check_overlap_count = (row >= border_size && row < scoreH - border_size)
											&& (col >= border_size && col < scoreW - border_size);
										bbox.area = (bbox.row2 - bbox.row1)*(bbox.col2 - bbox.col1);
										tmp_bounding_boxes[bb].push_back(bbox);
										tmp_bounding_scores[bb].push_back(order);
										count++;
									}
									p++;
								}
							}
							int tmp_before_count = tmp_bounding_boxes[bb].size();
							ZQ_CNN_BBoxUtils::_nms(tmp_bounding_boxes[bb], tmp_bounding_scores[bb], nms_thresh_per_scale, "Union", pnet_overlap_thresh_count);
							int tmp_after_count = tmp_bounding_boxes[bb].size();
							before_count += tmp_before_count;
							after_count += tmp_after_count;
						}
					}

					count = 0;
					for (int bb = 0; bb < block_num; bb++)
					{
						std::vector<ZQ_CNN_BBox>::iterator it = tmp_bounding_boxes[bb].begin();
						for (; it != tmp_bounding_boxes[bb].end(); it++)
						{
							if ((*it).exist)
							{
								bounding_boxes[i].push_back(*it);
								order.score = (*it).score;
								order.oriOrder = count;
								bounding_scores[i].push_back(order);
								count++;
							}
						}
					}

					//ZQ_CNN_BBoxUtils::_nms(bounding_boxes[i], bounding_scores[i], nms_thresh_per_scale, "Union", 0);
					after_count = bounding_boxes[i].size();
					for (int j = 0; j < after_count; j++)
					{
						ZQ_CNN_BBox& bbox = bounding_boxes[i][j];
						bbox.row1 = round(bbox.row1 *cur_scale_y);
						bbox.col1 = round(bbox.col1 *cur_scale_x);
						bbox.row2 = round(bbox.row2 *cur_scale_y);
						bbox.col2 = round(bbox.col2 *cur_scale_x);
						bbox.area = (bbox.row2 - bbox.row1)*(bbox.col2 - bbox.col1);
					}
					double t14 = omp_get_wtime();
					if (show_debug_info)
						printf("nms cost: %.3f ms, (%d-->%d)\n", 1000 * (t14 - t13), before_count, after_count);
				}

			}

			std::vector<ZQ_CNN_OrderScore> firstOrderScore;
			int count = 0;
			for (int i = 0; i < total_scale_num; i++)
			{
				std::vector<ZQ_CNN_BBox>::iterator it = bounding_boxes[i].begin();
				for (; it != bounding_boxes[i].end(); it++)
				{
					if ((*it).exist)
					{
						if (i < ori_num)
						{
							it->scale_x = 1;
							it->scale_y = 1;
						}
						else if(i < ori_num + xhalf_num)
						{
							it->scale_x = 0.5;
							it->scale_y = 1;
						}
						else
						{
							it->scale_x = 1;
							it->scale_y = 0.5;
						}
						firstBbox.push_back(*it);
						order.score = (*it).score;
						order.oriOrder = count;
						firstOrderScore.push_back(order);
						count++;
					}
				}
			}


			//the first stage's nms
			if (count < 1) return false;
			double t15 = omp_get_wtime();
			ZQ_CNN_BBoxUtils::_nms(firstBbox, firstOrderScore, nms_thresh[0], "Union", 0, 1);
			ZQ_CNN_BBoxUtils::_refine_and_square_bbox(firstBbox, width, height, true);
			double t16 = omp_get_wtime();
			if (show_debug_info)
				printf("nms cost: %.3f ms\n", 1000 * (t16 - t15));
			if (show_debug_info)
				printf("first stage candidate count: %d\n", count);
			double t3 = omp_get_wtime();
			if (show_debug_info)
				printf("stage 1: cost %.3f ms\n", 1000 * (t3 - t2));
			return true;
		}

		bool _Rnet_stage(std::vector<ZQ_CNN_BBox>& firstBbox, std::vector<ZQ_CNN_BBox>& secondBbox)
		{
			double t3 = omp_get_wtime();
			secondBbox.clear();
			std::vector<ZQ_CNN_BBox>::iterator it = firstBbox.begin();
			std::vector<ZQ_CNN_OrderScore> secondScore;
			std::vector<int> src_off_x, src_off_y, src_rect_w, src_rect_h;
			int r_count = 0;
			for (; it != firstBbox.end(); it++)
			{
				if ((*it).exist)
				{
					int off_x = it->col1;
					int off_y = it->row1;
					int rect_w = it->col2 - off_x;
					int rect_h = it->row2 - off_y;
					if (/*off_x < 0 || off_x + rect_w > width || off_y < 0 || off_y + rect_h > height ||*/ rect_w <= 0.5*min_size || rect_h <= 0.5*min_size)
					{
						(*it).exist = false;
						continue;
					}
					else
					{
						src_off_x.push_back(off_x);
						src_off_y.push_back(off_y);
						src_rect_w.push_back(rect_w);
						src_rect_h.push_back(rect_h);
						r_count++;
						secondBbox.push_back(*it);
					}
				}
			}

			int batch_size = BATCH_SIZE;
			int per_num = ceil((float)r_count / thread_num);
			int need_thread_num = thread_num;
			if (per_num > batch_size)
			{
				need_thread_num = ceil((float)r_count / batch_size);
				per_num = batch_size;
			}
			std::vector<ZQ_CNN_Tensor4D_NHW_C_Align128bit> task_rnet_images(need_thread_num);
			std::vector<std::vector<int> > task_src_off_x(need_thread_num);
			std::vector<std::vector<int> > task_src_off_y(need_thread_num);
			std::vector<std::vector<int> > task_src_rect_w(need_thread_num);
			std::vector<std::vector<int> > task_src_rect_h(need_thread_num);
			std::vector<std::vector<ZQ_CNN_BBox> > task_secondBbox(need_thread_num);


			for (int i = 0; i < need_thread_num; i++)
			{
				int st_id = per_num*i;
				int end_id = __min(r_count, per_num*(i + 1));
				int cur_num = end_id - st_id;
				if (cur_num > 0)
				{
					task_src_off_x[i].resize(cur_num);
					task_src_off_y[i].resize(cur_num);
					task_src_rect_w[i].resize(cur_num);
					task_src_rect_h[i].resize(cur_num);
					task_secondBbox[i].resize(cur_num);
					for (int j = 0; j < cur_num; j++)
					{
						task_src_off_x[i][j] = src_off_x[st_id + j];
						task_src_off_y[i][j] = src_off_y[st_id + j];
						task_src_rect_w[i][j] = src_rect_w[st_id + j];
						task_src_rect_h[i][j] = src_rect_h[st_id + j];
						task_secondBbox[i][j] = secondBbox[st_id + j];
					}
				}
			}

			if (thread_num <= 1)
			{
				for (int pp = 0; pp < need_thread_num; pp++)
				{
					// 审计修复 2026-10-06（附录 II.9）：原来只判**外层** vector 的 size。
					// 外层 `task_src_off_x` 的大小是 need_thread_num，**恒 >= 1**，
					// 判它永远不成立；真正会为空的是**当前槽位** `task_src_off_x[pp]`
					// （r_count == 0 时 per_num = ceil(0/thread_num) = 0，cur_num 也为 0）。
					// 本文件 Rnet 那一族早就是 `size() == 0 || task_src_off_x[pp].size() == 0`，
					// Pnet / Onet / lnet 这几处是**没同步过去的旧拷贝**。
					// 当前不崩只是因为 `ResizeBilinearRect` 在 rect_num == 0 时返回 false
					// 被下一个 if 接住了 —— 守卫没起作用，靠下游兜着。两种写法一起留着，判据才准。
					if (task_src_off_x.size() == 0 || task_src_off_x[pp].size() == 0)
						continue;
					if (!input.ResizeBilinearRect(task_rnet_images[pp], rnet_size, rnet_size, 0, 0,
						task_src_off_x[pp], task_src_off_y[pp], task_src_rect_w[pp], task_src_rect_h[pp]))
					{
					// 审计修复 2026-10-06（附录 II.18）：原来这里是裸的 `continue`，
					// 于是**这一槽的框一个都没被评过**，却仍然 exist=true、
					// score 还是 **Pnet 的旧分数**（task_secondBbox[pp] 是从上一层整体拷来的），
					// 随后的汇总把它们全部并进下一阶段的 NMS。这些框带着**偏高的 Pnet 分数**
					// 参加该阶段的 NMS，会把真正的框当 hero 抑制掉。
					// 正确写法是**清空这一槽**：没被评过的框不该带着别人的分数进下一阶段。
						task_secondBbox[pp].clear();
						continue;
					}
					rnet[0].Forward(task_rnet_images[pp]);
					const ZQ_CNN_Tensor4D* score = rnet[0].GetBlobByName("prob1");
					const ZQ_CNN_Tensor4D* location = rnet[0].GetBlobByName("conv5-2");
					// 审计修复 2026-10-06（附录 II.12）：GetBlobByName 找不到就返回 **0**
					// （ZQ_CNN_Net.h:295-300 / ZQ_CNN_Net_Interface），而 Init 对 blob 名**零校验**、
					// SetPara / Find 也不校验 —— 传一个概率层不叫 prob1 的模型进来，
					// 下面 score->GetH() / score->GetFirstPixelPtr() 就是空指针解引用。
					// 耐人寻味的是同一个函数里 keyPoint **有**判空（`if (keyPoint != 0)`），
					// score / location 没有 —— 判据不一致本身就是信号。
					if (score == 0 || location == 0) { printf("[MTCNN] blob not found in rnet\n"); continue; }
					const float* score_ptr = score->GetFirstPixelPtr();
					const float* location_ptr = location->GetFirstPixelPtr();
					int score_sliceStep = score->GetSliceStep();
					int location_sliceStep = location->GetSliceStep();
					int task_count = 0;
					for (int i = 0; i < task_secondBbox[pp].size(); i++)
					{
						if (score_ptr[i*score_sliceStep + 1] > thresh[1])
						{
							for (int j = 0; j < 4; j++)
								task_secondBbox[pp][i].regreCoord[j] = location_ptr[i*location_sliceStep + j];
							task_secondBbox[pp][i].area = task_src_rect_w[pp][i] * task_src_rect_h[pp][i];
							task_secondBbox[pp][i].score = score_ptr[i*score_sliceStep + 1];
							task_count++;
						}
						else
						{
							task_secondBbox[pp][i].exist = false;
						}
					}
					if (task_count < 1)
					{
						task_secondBbox[pp].clear();
						continue;
					}
					for (int i = task_secondBbox[pp].size() - 1; i >= 0; i--)
					{
						if (!task_secondBbox[pp][i].exist)
							task_secondBbox[pp].erase(task_secondBbox[pp].begin() + i);
					}
				}
			}
			else
			{
#pragma omp parallel for num_threads(thread_num) schedule(dynamic,1)
				for (int pp = 0; pp < need_thread_num; pp++)
				{
					int thread_id = omp_get_thread_num();
					// 审计修复 2026-10-06（附录 II.9）：原来只判**外层** vector 的 size。
					// 外层 `task_src_off_x` 的大小是 need_thread_num，**恒 >= 1**，
					// 判它永远不成立；真正会为空的是**当前槽位** `task_src_off_x[pp]`
					// （r_count == 0 时 per_num = ceil(0/thread_num) = 0，cur_num 也为 0）。
					// 本文件 Rnet 那一族早就是 `size() == 0 || task_src_off_x[pp].size() == 0`，
					// Pnet / Onet / lnet 这几处是**没同步过去的旧拷贝**。
					// 当前不崩只是因为 `ResizeBilinearRect` 在 rect_num == 0 时返回 false
					// 被下一个 if 接住了 —— 守卫没起作用，靠下游兜着。两种写法一起留着，判据才准。
					if (task_src_off_x.size() == 0 || task_src_off_x[pp].size() == 0)
						continue;
					if (!input.ResizeBilinearRect(task_rnet_images[pp], rnet_size, rnet_size, 0, 0,
						task_src_off_x[pp], task_src_off_y[pp], task_src_rect_w[pp], task_src_rect_h[pp]))
					{
					// 审计修复 2026-10-06（附录 II.18）：原来这里是裸的 `continue`，
					// 于是**这一槽的框一个都没被评过**，却仍然 exist=true、
					// score 还是 **Pnet 的旧分数**（task_secondBbox[pp] 是从上一层整体拷来的），
					// 随后的汇总把它们全部并进下一阶段的 NMS。这些框带着**偏高的 Pnet 分数**
					// 参加该阶段的 NMS，会把真正的框当 hero 抑制掉。
					// 正确写法是**清空这一槽**：没被评过的框不该带着别人的分数进下一阶段。
						task_secondBbox[pp].clear();
						continue;
					}
					rnet[thread_id].Forward(task_rnet_images[pp]);
					const ZQ_CNN_Tensor4D* score = rnet[thread_id].GetBlobByName("prob1");
					const ZQ_CNN_Tensor4D* location = rnet[thread_id].GetBlobByName("conv5-2");
					// 审计修复 2026-10-06（附录 II.12）：GetBlobByName 找不到就返回 **0**
					// （ZQ_CNN_Net.h:295-300 / ZQ_CNN_Net_Interface），而 Init 对 blob 名**零校验**、
					// SetPara / Find 也不校验 —— 传一个概率层不叫 prob1 的模型进来，
					// 下面 score->GetH() / score->GetFirstPixelPtr() 就是空指针解引用。
					// 耐人寻味的是同一个函数里 keyPoint **有**判空（`if (keyPoint != 0)`），
					// score / location 没有 —— 判据不一致本身就是信号。
					if (score == 0 || location == 0) { printf("[MTCNN] blob not found in rnet\n"); continue; }
					const float* score_ptr = score->GetFirstPixelPtr();
					const float* location_ptr = location->GetFirstPixelPtr();
					int score_sliceStep = score->GetSliceStep();
					int location_sliceStep = location->GetSliceStep();
					int task_count = 0;
					for (int i = 0; i < task_secondBbox[pp].size(); i++)
					{
						if (score_ptr[i*score_sliceStep + 1] > thresh[1])
						{
							for (int j = 0; j < 4; j++)
								task_secondBbox[pp][i].regreCoord[j] = location_ptr[i*location_sliceStep + j];
							task_secondBbox[pp][i].area = task_src_rect_w[pp][i] * task_src_rect_h[pp][i];
							task_secondBbox[pp][i].score = score_ptr[i*score_sliceStep + 1];
							task_count++;
						}
						else
						{
							task_secondBbox[pp][i].exist = false;
						}
					}
					if (task_count < 1)
					{
						task_secondBbox[pp].clear();
						continue;
					}
					for (int i = task_secondBbox[pp].size() - 1; i >= 0; i--)
					{
						if (!task_secondBbox[pp][i].exist)
							task_secondBbox[pp].erase(task_secondBbox[pp].begin() + i);
					}
				}
			}

			int count = 0;
			for (int i = 0; i < need_thread_num; i++)
			{
				count += task_secondBbox[i].size();
			}
			secondBbox.resize(count);
			secondScore.resize(count);
			int id = 0;
			for (int i = 0; i < need_thread_num; i++)
			{
				for (int j = 0; j < task_secondBbox[i].size(); j++)
				{
					secondBbox[id] = task_secondBbox[i][j];
					secondScore[id].score = secondBbox[id].score;
					secondScore[id].oriOrder = id;
					id++;
				}
			}

			ZQ_CNN_BBoxUtils::_nms(secondBbox, secondScore, nms_thresh[1], "Union");
			//ZQ_CNN_BBoxUtils::_nms(secondBbox, secondScore, nms_thresh[1], "Min");
			for (int i = 0; i < secondBbox.size(); i++)
			{
				float h = secondBbox[i].row2 - secondBbox[i].row1 + 1;
				float w = secondBbox[i].col2 - secondBbox[i].col1 + 1;
				float ratio = h / w;
				if (ratio > 1.5)
				{
					secondBbox[i].scale_x = 1;
					secondBbox[i].scale_y = 0.5;
				}
				else if (ratio < 1.0 / 1.5)
				{
					secondBbox[i].scale_x = 0.5;
					secondBbox[i].scale_y = 1;
				}
				else
				{
					secondBbox[i].scale_x = 1;
					secondBbox[i].scale_y = 1;
				}
			}
			ZQ_CNN_BBoxUtils::_refine_and_square_bbox(secondBbox, width, height, true);
			count = secondBbox.size();

			double t4 = omp_get_wtime();
			if (show_debug_info)
				printf("run Rnet [%d] times, candidate after nms: %d \n", r_count, count);
			if (show_debug_info)
				printf("stage 2: cost %.3f ms\n", 1000 * (t4 - t3));

			return true;
		}

		bool _Onet_stage(std::vector<ZQ_CNN_BBox>& secondBbox, std::vector<ZQ_CNN_BBox>& thirdBbox)
		{
			double t4 = omp_get_wtime();
			thirdBbox.clear();
			std::vector<ZQ_CNN_BBox>::iterator it = secondBbox.begin();
			std::vector<ZQ_CNN_OrderScore> thirdScore;
			std::vector<ZQ_CNN_BBox> early_accept_thirdBbox;
			std::vector<int> src_off_x, src_off_y, src_rect_w, src_rect_h;
			int o_count = 0;
			for (; it != secondBbox.end(); it++)
			{
				if ((*it).exist)
				{
					int off_x = it->col1;
					int off_y = it->row1;
					int rect_w = it->col2 - off_x;
					int rect_h = it->row2 - off_y;
					if (/*off_x < 0 || off_x + rect_w > width || off_y < 0 || off_y + rect_h > height ||*/ rect_w <= 0.5*min_size || rect_h <= 0.5*min_size)
					{
						(*it).exist = false;
						continue;
					}
					else
					{
						src_off_x.push_back(off_x);
						src_off_y.push_back(off_y);
						src_rect_w.push_back(rect_w);
						src_rect_h.push_back(rect_h);
						o_count++;
						thirdBbox.push_back(*it);
					}
				}
			}

			int batch_size = BATCH_SIZE;
			int per_num = ceil((float)o_count / thread_num);
			int need_thread_num = thread_num;
			if (per_num > batch_size)
			{
				need_thread_num = ceil((float)o_count / batch_size);
				per_num = batch_size;
			}

			std::vector<ZQ_CNN_Tensor4D_NHW_C_Align128bit> task_onet_images(need_thread_num);
			std::vector<std::vector<int> > task_src_off_x(need_thread_num);
			std::vector<std::vector<int> > task_src_off_y(need_thread_num);
			std::vector<std::vector<int> > task_src_rect_w(need_thread_num);
			std::vector<std::vector<int> > task_src_rect_h(need_thread_num);
			std::vector<std::vector<ZQ_CNN_BBox> > task_thirdBbox(need_thread_num);

			for (int i = 0; i < need_thread_num; i++)
			{
				int st_id = per_num*i;
				int end_id = __min(o_count, per_num*(i + 1));
				int cur_num = end_id - st_id;
				if (cur_num > 0)
				{
					task_src_off_x[i].resize(cur_num);
					task_src_off_y[i].resize(cur_num);
					task_src_rect_w[i].resize(cur_num);
					task_src_rect_h[i].resize(cur_num);
					task_thirdBbox[i].resize(cur_num);
					for (int j = 0; j < cur_num; j++)
					{
						task_src_off_x[i][j] = src_off_x[st_id + j];
						task_src_off_y[i][j] = src_off_y[st_id + j];
						task_src_rect_w[i][j] = src_rect_w[st_id + j];
						task_src_rect_h[i][j] = src_rect_h[st_id + j];
						task_thirdBbox[i][j] = thirdBbox[st_id + j];
					}
				}
			}

			if (thread_num <= 1)
			{
				for (int pp = 0; pp < need_thread_num; pp++)
				{
					// 审计修复 2026-10-06（附录 II.9）：原来只判**外层** vector 的 size。
					// 外层 `task_src_off_x` 的大小是 need_thread_num，**恒 >= 1**，
					// 判它永远不成立；真正会为空的是**当前槽位** `task_src_off_x[pp]`
					// （r_count == 0 时 per_num = ceil(0/thread_num) = 0，cur_num 也为 0）。
					// 本文件 Rnet 那一族早就是 `size() == 0 || task_src_off_x[pp].size() == 0`，
					// Pnet / Onet / lnet 这几处是**没同步过去的旧拷贝**。
					// 当前不崩只是因为 `ResizeBilinearRect` 在 rect_num == 0 时返回 false
					// 被下一个 if 接住了 —— 守卫没起作用，靠下游兜着。两种写法一起留着，判据才准。
					if (task_src_off_x.size() == 0 || task_src_off_x[pp].size() == 0)
						continue;
					if (!input.ResizeBilinearRect(task_onet_images[pp], onet_size, onet_size, 0, 0,
						task_src_off_x[pp], task_src_off_y[pp], task_src_rect_w[pp], task_src_rect_h[pp]))
					{
					// 审计修复 2026-10-06（附录 II.18）：原来这里是裸的 `continue`，
					// 于是**这一槽的框一个都没被评过**，却仍然 exist=true、
					// score 还是 **Pnet 的旧分数**（task_thirdBbox[pp] 是从上一层整体拷来的），
					// 随后的汇总把它们全部并进下一阶段的 NMS。这些框带着**偏高的 Pnet 分数**
					// 参加该阶段的 NMS，会把真正的框当 hero 抑制掉。
					// 正确写法是**清空这一槽**：没被评过的框不该带着别人的分数进下一阶段。
						task_thirdBbox[pp].clear();
						continue;
					}
					double t31 = omp_get_wtime();
					onet[0].Forward(task_onet_images[pp]);
					double t32 = omp_get_wtime();
					const ZQ_CNN_Tensor4D* score = onet[0].GetBlobByName("prob1");
					const ZQ_CNN_Tensor4D* location = onet[0].GetBlobByName("conv6-2");
					// 审计修复 2026-10-06（附录 II.12）：GetBlobByName 找不到就返回 **0**
					// （ZQ_CNN_Net.h:295-300 / ZQ_CNN_Net_Interface），而 Init 对 blob 名**零校验**、
					// SetPara / Find 也不校验 —— 传一个概率层不叫 prob1 的模型进来，
					// 下面 score->GetH() / score->GetFirstPixelPtr() 就是空指针解引用。
					// 耐人寻味的是同一个函数里 keyPoint **有**判空（`if (keyPoint != 0)`），
					// score / location 没有 —— 判据不一致本身就是信号。
					if (score == 0 || location == 0) { printf("[MTCNN] blob not found in onet\n"); continue; }
					const ZQ_CNN_Tensor4D* keyPoint = onet[0].GetBlobByName("conv6-3");
					const float* score_ptr = score->GetFirstPixelPtr();
					const float* location_ptr = location->GetFirstPixelPtr();
					const float* keyPoint_ptr = 0;
					if (keyPoint != 0)
						keyPoint_ptr = keyPoint->GetFirstPixelPtr();
					int score_sliceStep = score->GetSliceStep();
					int location_sliceStep = location->GetSliceStep();
					int keyPoint_sliceStep = 0;
					if (keyPoint != 0)
						keyPoint_sliceStep = keyPoint->GetSliceStep();
					int task_count = 0;
					ZQ_CNN_OrderScore order;
					for (int i = 0; i < task_thirdBbox[pp].size(); i++)
					{
						if (score_ptr[i*score_sliceStep + 1] > thresh[2])
						{
							for (int j = 0; j < 4; j++)
								task_thirdBbox[pp][i].regreCoord[j] = location_ptr[i*location_sliceStep + j];
							if (keyPoint != 0)
							{
								// 审计修复 2026-10-06（附录 II.19）：原来写的是 `__min(5, C / 2)`，
								// 但下面**两半**都读：`keyPoint_ptr[i*sliceStep + num]` 和 `[... + num + 5]`。
								// 本行读到的最大下标是 `kp_num - 1 + 5`，要不越出行必须 `kp_num <= C - 5`：
								//   C=4  -> kp_num=2，最大读 6 >= 4  => 读到**下一行**；
								//   C=8  -> kp_num=4，最大读 8 >= 8  => 正好越过本行；
								//   C=10 -> kp_num=5，最大读 9 <  10 => 才安全。
								// 越出行本身还在缓冲里（不算越界），但**最后一个样本**再读就越过缓冲末尾，
								// 而且即使不崩，取到的也是**下一个样本的坐标** —— 结果静默错乱。
								// 正确上限是 `C - 5`（右半要从下标 5 开始），不是 `C / 2`。
								int kp_num = keyPoint->GetC() - 5;
								if (kp_num > 5) kp_num = 5;
								if (kp_num < 0) kp_num = 0;
								for (int num = 0; num < kp_num; num++)
								{
									task_thirdBbox[pp][i].ppoint[num] = task_thirdBbox[pp][i].col1 +
										(task_thirdBbox[pp][i].col2 - task_thirdBbox[pp][i].col1)*keyPoint_ptr[i*keyPoint_sliceStep + num];
									task_thirdBbox[pp][i].ppoint[num + 5] = task_thirdBbox[pp][i].row1 +
										(task_thirdBbox[pp][i].row2 - task_thirdBbox[pp][i].row1)*keyPoint_ptr[i*keyPoint_sliceStep + num + 5];
								}
							}
							task_thirdBbox[pp][i].area = task_src_rect_w[pp][i] * task_src_rect_h[pp][i];
							task_thirdBbox[pp][i].score = score_ptr[i*score_sliceStep + 1];
							task_count++;
						}
						else
						{
							task_thirdBbox[pp][i].exist = false;
						}
					}

					if (task_count < 1)
					{
						task_thirdBbox[pp].clear();
						continue;
					}
					for (int i = task_thirdBbox[pp].size() - 1; i >= 0; i--)
					{
						if (!task_thirdBbox[pp][i].exist)
							task_thirdBbox[pp].erase(task_thirdBbox[pp].begin() + i);
					}
				}
			}
			else
			{
#pragma omp parallel for num_threads(thread_num) schedule(dynamic,1)
				for (int pp = 0; pp < need_thread_num; pp++)
				{
					int thread_id = omp_get_thread_num();
					// 审计修复 2026-10-06（附录 II.9）：原来只判**外层** vector 的 size。
					// 外层 `task_src_off_x` 的大小是 need_thread_num，**恒 >= 1**，
					// 判它永远不成立；真正会为空的是**当前槽位** `task_src_off_x[pp]`
					// （r_count == 0 时 per_num = ceil(0/thread_num) = 0，cur_num 也为 0）。
					// 本文件 Rnet 那一族早就是 `size() == 0 || task_src_off_x[pp].size() == 0`，
					// Pnet / Onet / lnet 这几处是**没同步过去的旧拷贝**。
					// 当前不崩只是因为 `ResizeBilinearRect` 在 rect_num == 0 时返回 false
					// 被下一个 if 接住了 —— 守卫没起作用，靠下游兜着。两种写法一起留着，判据才准。
					if (task_src_off_x.size() == 0 || task_src_off_x[pp].size() == 0)
						continue;
					if (!input.ResizeBilinearRect(task_onet_images[pp], onet_size, onet_size, 0, 0,
						task_src_off_x[pp], task_src_off_y[pp], task_src_rect_w[pp], task_src_rect_h[pp]))
					{
					// 审计修复 2026-10-06（附录 II.18）：原来这里是裸的 `continue`，
					// 于是**这一槽的框一个都没被评过**，却仍然 exist=true、
					// score 还是 **Pnet 的旧分数**（task_thirdBbox[pp] 是从上一层整体拷来的），
					// 随后的汇总把它们全部并进下一阶段的 NMS。这些框带着**偏高的 Pnet 分数**
					// 参加该阶段的 NMS，会把真正的框当 hero 抑制掉。
					// 正确写法是**清空这一槽**：没被评过的框不该带着别人的分数进下一阶段。
						task_thirdBbox[pp].clear();
						continue;
					}
					double t31 = omp_get_wtime();
					onet[thread_id].Forward(task_onet_images[pp]);
					double t32 = omp_get_wtime();
					const ZQ_CNN_Tensor4D* score = onet[thread_id].GetBlobByName("prob1");
					const ZQ_CNN_Tensor4D* location = onet[thread_id].GetBlobByName("conv6-2");
					// 审计修复 2026-10-06（附录 II.12）：GetBlobByName 找不到就返回 **0**
					// （ZQ_CNN_Net.h:295-300 / ZQ_CNN_Net_Interface），而 Init 对 blob 名**零校验**、
					// SetPara / Find 也不校验 —— 传一个概率层不叫 prob1 的模型进来，
					// 下面 score->GetH() / score->GetFirstPixelPtr() 就是空指针解引用。
					// 耐人寻味的是同一个函数里 keyPoint **有**判空（`if (keyPoint != 0)`），
					// score / location 没有 —— 判据不一致本身就是信号。
					if (score == 0 || location == 0) { printf("[MTCNN] blob not found in onet\n"); continue; }
					const ZQ_CNN_Tensor4D* keyPoint = onet[thread_id].GetBlobByName("conv6-3");
					const float* score_ptr = score->GetFirstPixelPtr();
					const float* location_ptr = location->GetFirstPixelPtr();
					const float* keyPoint_ptr = 0;
					if (keyPoint != 0)
						keyPoint_ptr = keyPoint->GetFirstPixelPtr();
					int score_sliceStep = score->GetSliceStep();
					int location_sliceStep = location->GetSliceStep();
					int keyPoint_sliceStep = 0;
					if (keyPoint != 0)
						keyPoint_sliceStep = keyPoint->GetSliceStep();
					int task_count = 0;
					ZQ_CNN_OrderScore order;
					for (int i = 0; i < task_thirdBbox[pp].size(); i++)
					{
						if (score_ptr[i*score_sliceStep + 1] > thresh[2])
						{
							for (int j = 0; j < 4; j++)
								task_thirdBbox[pp][i].regreCoord[j] = location_ptr[i*location_sliceStep + j];
							if (keyPoint != 0)
							{
								// 审计修复 2026-10-06（附录 II.19）：原来写的是 `__min(5, C / 2)`，
								// 但下面**两半**都读：`keyPoint_ptr[i*sliceStep + num]` 和 `[... + num + 5]`。
								// 本行读到的最大下标是 `kp_num - 1 + 5`，要不越出行必须 `kp_num <= C - 5`：
								//   C=4  -> kp_num=2，最大读 6 >= 4  => 读到**下一行**；
								//   C=8  -> kp_num=4，最大读 8 >= 8  => 正好越过本行；
								//   C=10 -> kp_num=5，最大读 9 <  10 => 才安全。
								// 越出行本身还在缓冲里（不算越界），但**最后一个样本**再读就越过缓冲末尾，
								// 而且即使不崩，取到的也是**下一个样本的坐标** —— 结果静默错乱。
								// 正确上限是 `C - 5`（右半要从下标 5 开始），不是 `C / 2`。
								int kp_num = keyPoint->GetC() - 5;
								if (kp_num > 5) kp_num = 5;
								if (kp_num < 0) kp_num = 0;
								for (int num = 0; num < kp_num; num++)
								{
									task_thirdBbox[pp][i].ppoint[num] = task_thirdBbox[pp][i].col1 +
										(task_thirdBbox[pp][i].col2 - task_thirdBbox[pp][i].col1)*keyPoint_ptr[i*keyPoint_sliceStep + num];
									task_thirdBbox[pp][i].ppoint[num + 5] = task_thirdBbox[pp][i].row1 +
										(task_thirdBbox[pp][i].row2 - task_thirdBbox[pp][i].row1)*keyPoint_ptr[i*keyPoint_sliceStep + num + 5];
								}
							}
							task_thirdBbox[pp][i].area = task_src_rect_w[pp][i] * task_src_rect_h[pp][i];
							task_thirdBbox[pp][i].score = score_ptr[i*score_sliceStep + 1];
							task_count++;
						}
						else
						{
							task_thirdBbox[pp][i].exist = false;
						}
					}

					if (task_count < 1)
					{
						task_thirdBbox[pp].clear();
						continue;
					}
					for (int i = task_thirdBbox[pp].size() - 1; i >= 0; i--)
					{
						if (!task_thirdBbox[pp][i].exist)
							task_thirdBbox[pp].erase(task_thirdBbox[pp].begin() + i);
					}
				}
			}

			int count = 0;
			for (int i = 0; i < need_thread_num; i++)
			{
				count += task_thirdBbox[i].size();
			}
			thirdBbox.resize(count);
			thirdScore.resize(count);
			int id = 0;
			for (int i = 0; i < need_thread_num; i++)
			{
				for (int j = 0; j < task_thirdBbox[i].size(); j++)
				{
					thirdBbox[id] = task_thirdBbox[i][j];
					thirdScore[id].score = task_thirdBbox[i][j].score;
					thirdScore[id].oriOrder = id;
					id++;
				}
			}
			ZQ_CNN_OrderScore order;
			for (int i = 0; i < early_accept_thirdBbox.size(); i++)
			{
				order.score = early_accept_thirdBbox[i].score;
				order.oriOrder = count++;
				thirdScore.push_back(order);
				thirdBbox.push_back(early_accept_thirdBbox[i]);
			}
			for (int i = 0; i < secondBbox.size(); i++)
			{
				float h = secondBbox[i].row2 - secondBbox[i].row1 + 1;
				float w = secondBbox[i].col2 - secondBbox[i].col1 + 1;
				float ratio = h / w;
				if (ratio > 1.5)
				{
					secondBbox[i].scale_x = 1;
					secondBbox[i].scale_y = 0.5;
				}
				else if (ratio < 1.0 / 1.5)
				{
					secondBbox[i].scale_x = 0.5;
					secondBbox[i].scale_y = 1;
				}
				else
				{
					secondBbox[i].scale_x = 1;
					secondBbox[i].scale_y = 1;
				}
			}
			ZQ_CNN_BBoxUtils::_refine_and_square_bbox(thirdBbox, width, height, false);
			ZQ_CNN_BBoxUtils::_nms(thirdBbox, thirdScore, nms_thresh[2], "Min");
			double t5 = omp_get_wtime();
			if (show_debug_info)
				printf("run Onet [%d] times, candidate before nms: %d \n", o_count, count);
			if (show_debug_info)
				printf("stage 3: cost %.3f ms\n", 1000 * (t5 - t4));

			return true;
		}


		void _select(std::vector<ZQ_CNN_BBox>& bbox, int limit_num, int width, int height)
		{
			// 审计修复 2026-10-06（附录 II.20）：原来这里是 `bbox.resize(limit_num)` ——
			// 直接按**插入顺序**截断。`firstBbox` / `secondBbox` 的插入顺序是
			// 「先按 scale、再按 scale 内顺序」，而 scale 是从小到大走的，
			// 所以保留下来的恰好是**最小尺度**的那一批，而不是分数最高的。
			// `SetLimit(r, o)` 的用途是给 Rnet/Onet 的计算量封顶，
			// 封顶当然应该留最有希望的框；现在这样等于**随机丢掉高分框、留下低分框**。
			//
			// `width` / `height` 两个形参从头到尾就没被用过（调用点传的是
			// input.GetW()/GetH()），一并留着以免动签名。
			int in_num = (int)bbox.size();
			if (limit_num <= 0)
			{
				bbox.clear();
				return;
			}
			if (limit_num >= in_num)
				return;
			// 稳定选择：分数相同时保持原插入顺序 —— 否则同分框之间的相对次序
			// 取决于 std::sort 的实现，输出就不再可复现（附录 CA 那类问题）。
			std::vector<ZQ_CNN_BBox> keep;
			keep.resize(limit_num);
			// 用「插入序」当次级键做一次稳定的部分选择：
			// 先按分数降序排下标，同分按下标升序 —— 等价于稳定取前 limit_num 个。
			std::vector<int> idx(in_num);
			for (int i = 0; i < in_num; i++)
				idx[i] = i;
			const std::vector<ZQ_CNN_BBox>& ref = bbox;
			std::stable_sort(idx.begin(), idx.end(),
				[&ref](int a, int b) { return ref[a].score > ref[b].score; });
			for (int i = 0; i < limit_num; i++)
				keep[i] = bbox[idx[i]];
			bbox.swap(keep);
		}
	};
}

#endif
