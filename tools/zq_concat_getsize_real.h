/* `_concat_NCHW_get_size` 的**逐字照抄**实现 —— 附录 EN
 *
 * 为什么它不能是绊线
 * ------------------
 * `ZQ_CNN_Layer_Concat::LayerSetup` 在**加载期**就要算输出形状，走的正是它。
 * tools/zq_net_fwd_tripwires.h 里其余 44 个都是绊线（被调到就 `_exit(3)`），
 * 只有这一个必须给真实现 —— 做成绊线的话，任何"Concat 合法"的用例都会红，
 * 而 `zq_concat_alias_check.cpp` 的两个良性对照恰恰必须是合法 Concat。
 * 第一版就是这么写的，红了还一度以为是守卫没修好。
 *
 * 谁在用它
 * --------
 *   - tools/zq_concat_alias_check.cpp（6 例的门禁）
 *   - tools/zq_model_load_probe.cpp（28 个真实模型的加载探针）
 *
 * 同步义务（**这是一个真实的、有代价的排除项**）
 * ------------------------------------------------
 * 下面这份是 2026-10-03 从 `ZQCNN/ZQ_CNN_Forward_SSEUtils.cpp:4894` 抄的。
 * 真实现改了而这里没跟上，两处都会**静默地**按旧逻辑算形状。
 * 缓解：① 函数只读形状、不碰数据，抄错的后果是形状算错而不是内存问题，
 * 而形状算错会在真正跑该模型的 sample 回归里立刻暴露；
 * ② 它的返回值在这两个消费者里都不是判据（门禁只看 LoadFrom 成功与否）。
 *
 * 真要消除这个风险，只能编 ZQ_CNN_Forward_SSEUtils.cpp —— 那个 TU 一个文件引用
 * 半个库，会拖进单编 5 分钟以上的 conv GEMM（附录 EC.1 的取舍）。
 */
#ifndef ZQ_CONCAT_GETSIZE_REAL_H_
#define ZQ_CONCAT_GETSIZE_REAL_H_

#include <vector>
#include "ZQCNN/ZQ_CNN_Tensor4D.h"
// 必须显式 include：**这个类只在这里被"定义成员函数"**，
// 而 forward declaration 不足以定义成员。
// 少了这一行时，单独编译本头的调用方会报
//   invalid use of incomplete type 'class ZQ::ZQ_CNN_Forward_SSEUtils'
// （2026-10-03 在 `zq_model_params_check.cpp` 上撞到；此前两个消费者
//  都碰巧先 include 了 `ZQ_CNN_Net.h`，把它间接带进来了。）
#include "ZQCNN/ZQ_CNN_Forward_SSEUtils.h"

namespace ZQ
{
class ZQ_CNN_Forward_SSEUtils;

// 逐字照抄 ZQCNN/ZQ_CNN_Forward_SSEUtils.cpp:4894 的 _concat_NCHW_get_size
bool ZQ_CNN_Forward_SSEUtils::_concat_NCHW_get_size(const std::vector<ZQ_CNN_Tensor4D*>& inputs, int axis,
	int& out_N, int& out_C, int& out_H, int& out_W)
{
	if (axis < 0 || axis >= 4)
		return false;
	int in_num = (int)inputs.size();
	std::vector<ZQ_CNN_Tensor4D*> valid_inputs;
	for (int i = 0; i < inputs.size(); i++)
	{
		if (inputs[i] == 0)
			continue;
		inputs[i]->GetShape(out_N, out_C, out_H, out_W);
		if (out_N > 0 && out_C > 0 && out_H > 0 && out_W > 0)
			valid_inputs.push_back(inputs[i]);
	}

	if (valid_inputs.size() == 0)
	{
		out_N = out_H = out_W = out_C = 0;
		return true;
	}
	else if (valid_inputs.size() == 1)
	{
		valid_inputs[0]->GetShape(out_N, out_C, out_H, out_W);
		return true;
	}
	else
	{
		int standard_dim[4];
		valid_inputs[0]->GetShape(standard_dim[0], standard_dim[1], standard_dim[2], standard_dim[3]);
		int sum_out = standard_dim[axis];
		for (int i = 1; i < valid_inputs.size(); i++)
		{
			if (valid_inputs[i] == 0)
				return false;
			int cur_dim[4];
			valid_inputs[i]->GetShape(cur_dim[0], cur_dim[1], cur_dim[2], cur_dim[3]);
			for (int j = 0; j < 4; j++)
			{
				if (axis == j)
				{
					sum_out += cur_dim[j];
				}
				else if (cur_dim[j] != standard_dim[j])
				{
					return false;
				}
			}
		}
		standard_dim[axis] = sum_out;
		out_N = standard_dim[0];
		out_C = standard_dim[1];
		out_H = standard_dim[2];
		out_W = standard_dim[3];
		return true;
	}
}

}  // namespace ZQ

#endif  // ZQ_CONCAT_GETSIZE_REAL_H_
