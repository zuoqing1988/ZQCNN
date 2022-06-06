#include "ZQ_CNN_CompileConfig.h"
#include "ZQ_CNN_SSDDetectorPytorch.h"
#include "ZQ_CNN_LoadConfigUtils.h"
#include <algorithm>


using namespace ZQ;

ZQ_CNN_SSDDetectorPytorch::ZQ_CNN_SSDDetectorPytorch()
{
	prob_thresh = 0.3;
	iou_thresh = 0.5;
	top_k = -1;
	image_std = 1.0;
	image_size_x = image_size_y = 300;
	use_gray = false;
	show_debug_info = false;
	center_variance = 0.1;
	size_variance = 0.2;
}

ZQ_CNN_SSDDetectorPytorch::~ZQ_CNN_SSDDetectorPytorch()
{

}

void ZQ_CNN_SSDDetectorPytorch::SetParam(float prob_thresh, float iou_thresh, int top_k)
{
	this->prob_thresh = prob_thresh;
	this->iou_thresh = iou_thresh;
	this->top_k = top_k;
}

bool ZQ_CNN_SSDDetectorPytorch::UseGray() const
{
	return use_gray;
}

void ZQ_CNN_SSDDetectorPytorch::SetShowDebugInfo(bool b)
{
	this->show_debug_info = b;
}

void ZQ_CNN_SSDDetectorPytorch::GetInputSize(int &h, int& w, int& c)const
{
	w = this->image_size_x;
	h = this->image_size_y;
	c = this->use_gray ? 1 : 3;
}

float ZQ_CNN_SSDDetectorPytorch::GetMinSize() const
{
	if (specs.size() > 0)
		return specs[0].box_min;
	else
		return 0;
}

bool ZQ_CNN_SSDDetectorPytorch::LoadModel(const std::string& param_file, const std::string& bin_file, const std::string& cfg_file)
{
	if (!runner.LoadFrom(param_file, bin_file))
	{
		if (show_debug_info)
			printf("failed to load ssd net\n");
		return false;
	}
	if (show_debug_info)
	{
		printf("num_MADD = %.1f M\n", runner.GetNumOfMulAdd() / 1024.0 / 1024.0);
	}
	if (!_load_cfg(cfg_file.c_str()))
	{
		if (show_debug_info)
			printf("failed to load ssd cfg\n");
		return false;
	}

	_generate_ssd_priors(specs, image_size_x, image_size_y, true, prior_boxes);

	return true;
}

bool ZQ_CNN_SSDDetectorPytorch::Detect(const unsigned char* im_data, int im_width, int im_height, int widthStep,
	std::vector<ZQ_CNN_SSDDetectorUtils::BBox>& output)
{
	ZQ_CNN_Tensor4D_NHW_C_Align128bit input0, input1;
	if (use_gray)
	{
		if (!input0.ConvertFromGray(im_data, im_width, im_height, widthStep, 0, 1))
			return false;
	}
	else
	{
		if (!input0.ConvertFromBGR(im_data, im_width, im_height, widthStep, 0, 1))
			return false;
	}

	int in_c, in_h, in_w;
	GetInputSize(in_h, in_w, in_c);
	if (im_width != in_w || im_height != in_h)
	{
		if (!input0.ResizeBilinear(input1, in_w, in_h, 0, 0))
		{
			return false;
		}
		if (!_detect(input1, output))
			return false;
	}
	else
	{
		if (!_detect(input0, output))
			return false;
	}

	for (int i = 0; i < output.size(); i++)
	{
		output[i].xmin *= im_width;
		output[i].ymin *= im_height;
		output[i].xmax *= im_width;
		output[i].ymax *= im_height;
	}
	return true;
}

bool ZQ_CNN_SSDDetectorPytorch::DetectMultiScale(const unsigned char* im_data, int im_width, int im_height, int widthStep, int min_size,
	std::vector<ZQ_CNN_SSDDetectorUtils::BBox>& output)
{
	int min_ssd_det_size = GetMinSize();
	if (min_ssd_det_size <= 0)
		return false;
	int in_c, in_w, in_h;
	GetInputSize(in_h, in_w, in_c);

	float scale = (float)min_ssd_det_size / min_size;
	int scaled_width = im_width*scale;
	int scaled_height = im_height*scale;
	if (scaled_width <= in_w || scaled_height <= in_h)
		return Detect(im_data, im_width, im_height, widthStep, output);

	ZQ_CNN_Tensor4D_NHW_C_Align128bit input0, input1;
	if (use_gray)
	{
		if (!input0.ConvertFromGray(im_data, im_width, im_height, widthStep, 0, 1))
			return false;
	}
	else
	{
		if (!input0.ConvertFromBGR(im_data, im_width, im_height, widthStep, 0, 1))
			return false;
	}

	int last_w = scaled_width;
	int last_h = scaled_height;
	int count = 1;
	std::vector<int> scaled_widths = { last_w };
	std::vector<int> scaled_heights = { last_h };
	while (last_w > in_w && last_h > in_h)
	{
		int need_w = last_w / 2;
		int need_h = last_h / 2;
		if (need_w < in_w || need_h < in_h)
		{
			count++;
			scaled_widths.push_back(in_w);
			scaled_heights.push_back(in_h);
			break;
		}
		else
		{
			count++;
			last_w = need_w;
			last_h = need_h;
			scaled_widths.push_back(need_w);
			scaled_heights.push_back(need_h);
		}
	}
	std::vector<ZQ_CNN_Tensor4D_NHW_C_Align128bit> scaled_imgs(count);
	for (int i = 0; i < count; i++)
	{
		if (i == 0)
		{
			input0.ResizeBilinear(scaled_imgs[i], scaled_widths[i], scaled_heights[i], 0, 0);
		}
		else
		{
			scaled_imgs[i-1].ResizeBilinear(scaled_imgs[i], scaled_widths[i], scaled_heights[i], 0, 0);
		}
	}
	
	int overlap_size = __min(in_w*0.5, min_ssd_det_size * 1.5);
	std::vector<ZQ_CNN_SSDDetectorUtils::BBox> all_bboxes;
	
	for (int i = 0; i < count; i++)
	{
		int cur_w = scaled_widths[i];
		int cur_h = scaled_heights[i];
		//printf("cur_size(WXH) = %d x %d\n", cur_w, cur_h);
		float scale_x = (float)im_width / cur_w;
		float scale_y = (float)im_height / cur_h;
		int block_w = ceil((float)(cur_w - in_w) / (in_w - overlap_size) + 1);
		int block_h = ceil((float)(cur_h - in_h) / (in_h - overlap_size) + 1);
		for (int bh = 0; bh < block_h; bh++)
		{
			for (int bw = 0; bw < block_w; bw++)
			{
				int rect_x = 0, rect_y = 0, rect_w = 0, rect_h = 0;
				if (bw == block_w - 1)
					rect_x = cur_w - in_w;
				else
					rect_x = bw * (in_w - overlap_size);
				if (bh == block_h - 1)
					rect_y = cur_h - in_h;
				else
					rect_y = bh * (in_h - overlap_size);
				rect_w = in_w;
				rect_h = in_h;
				scaled_imgs[i].ROI(input1, rect_x, rect_y, rect_w, rect_h, 0, 0);
				std::vector<ZQ_CNN_SSDDetectorUtils::BBox> cur_bboxes;
				if (!_detect(input1, cur_bboxes))
				{
					printf("failed\n");
					continue;
				}
				for (int k = 0; k < cur_bboxes.size(); k++)
				{
					cur_bboxes[k].xmin = (cur_bboxes[k].xmin*in_w + rect_x) * scale_x;
					cur_bboxes[k].xmax = (cur_bboxes[k].xmax*in_w + rect_x) * scale_x;
					cur_bboxes[k].ymin = (cur_bboxes[k].ymin*in_h + rect_y) * scale_y;
					cur_bboxes[k].ymax = (cur_bboxes[k].ymax*in_h + rect_y) * scale_y;
				}
				all_bboxes.insert(all_bboxes.end(), cur_bboxes.begin(), cur_bboxes.end());
			}
		}
	}
	//printf("num_all_bboxes = %d\n", all_bboxes.size());
	_hard_nms(all_bboxes, output, 0.5, -1, 10000);
	return true;
}

bool ZQ_CNN_SSDDetectorPytorch::_detect(ZQ_CNN_Tensor4D& input, std::vector<ZQ_CNN_SSDDetectorUtils::BBox>& output)
{
	if (!runner.Forward(input))
	{
		return false;
	}

	const ZQ_CNN_Tensor4D* cls = runner.GetBlobByName("cls");
	const ZQ_CNN_Tensor4D* loc = runner.GetBlobByName("loc");
	if (cls == 0 || loc == 0)
		return false;

	int cls_H = cls->GetH();
	int cls_W = cls->GetW();
	int cls_C = cls->GetC();
	int loc_H = loc->GetH();
	int loc_W = loc->GetW();
	int loc_C = loc->GetC();
	std::vector<float> cls_data(cls_H*cls_W*cls_C), loc_data(loc_H*loc_W*loc_C);
	cls->ConvertToCompactNCHW(cls_data.data());
	loc->ConvertToCompactNCHW(loc_data.data());


	if (show_debug_info)
	{
		printf("cls HxWxC= %d x %d x %d\n", cls_H, cls_W, cls_C);
		printf("loc HxWxC= %d x %d x %d\n", cls_H, cls_W, cls_C);
	}
	int N = cls_C;
	int num_classes = cls_H;
	if (N * 4 != prior_boxes.size())
	{
		//show info
		if (show_debug_info)
		{
			printf("prior boxes don't match locations\n");
		}
		return false;
	}


	ZQ_CNN_SSDDetectorPytorch::_post_process(loc_data.data(), cls_data.data(), prior_boxes.data(),
		N, num_classes, center_variance, size_variance);
	ZQ_CNN_SSDDetectorPytorch::_detection(loc_data.data(), cls_data.data(), N, num_classes, output, prob_thresh,
		0.5f, iou_thresh, top_k, 0.5f, 200);
	return true;
}

bool ZQ_CNN_SSDDetectorPytorch::_load_cfg(const char* file)
{
	std::fstream fin(file, std::ios::in);
	if (!fin.is_open())
	{
		if (show_debug_info)
		{
			printf("failed to open file %s\n", file);
		}
		return false;
	}
	return _load_cfg_from_file_or_buffer(fin, NULL, 0);
}

bool ZQ_CNN_SSDDetectorPytorch::_load_cfg_from_buffer(const char* buffer, __int64 buffer_len)
{
	std::fstream fin;
	return _load_cfg_from_file_or_buffer(fin, buffer, buffer_len);
}

bool ZQ_CNN_SSDDetectorPytorch::_load_cfg_from_file_or_buffer(std::fstream& fin, const char* buffer, __int64 buffer_len)
{
	bool has_image_mean = false;
	bool has_image_std = false;
	bool has_center_variance = false;
	bool has_size_variance = false;
	bool has_aspect_ratios = false;
	bool has_use_gray = false;
	bool has_image_size_x = false;
	bool has_image_size_y = false;
	std::vector<float> aspect_ratios;
	std::vector<std::string> spec_names;
	const int buf_len = 512;
	char buf[buf_len];
	const int spec_max_num = 20;
	for (int i = 0; i < spec_max_num; i++)
	{
		snprintf(buf, buf_len, "spec%d", i + 1);
		spec_names.push_back(std::string(buf));
	}
	std::vector<ZQ_CNN_SSDDetectorUtils::SSDSpec> spec_vals(spec_max_num);
	std::vector<bool> has_spec_vals(spec_max_num, false);

	std::string line;
	while (ZQ_CNN_LoadConfigUtils::Get_line(fin, buffer, buffer_len, line))
	{
		std::vector<std::vector<std::string> > cur_splits = ZQ_CNN_LoadConfigUtils::Split_line(line);
		if (show_debug_info)
		{
			for (int i = 0; i < cur_splits.size(); i++)
			{
				printf("[----%d]\n", i);
				for (int j = 0; j < cur_splits[i].size(); j++)
					printf("%s\n", cur_splits[i][j].c_str());
			}
		}
		if (cur_splits.size() != 2)
			continue;
		if (cur_splits[0].size() != 1)
			continue;
		std::string param_name = ZQ_CNN_LoadConfigUtils::Remove_blank(cur_splits[0][0]);
		const char* param_name_ptr = param_name.c_str();
		if (ZQ_CNN_LoadConfigUtils::My_strcmpi(param_name_ptr, "image_mean") == 0)
		{
			image_mean.clear();
			for (int j = 0; j < cur_splits[1].size(); j++)
			{
				image_mean.push_back(atof(cur_splits[1][j].c_str()));
			}
			has_image_mean = true;
		}
		else if (ZQ_CNN_LoadConfigUtils::My_strcmpi(param_name_ptr, "image_std") == 0)
		{
			image_std = atof(cur_splits[1][0].c_str());
			has_image_std = true;
		}
		else if (ZQ_CNN_LoadConfigUtils::My_strcmpi(param_name_ptr, "center_variance") == 0)
		{
			center_variance = atof(cur_splits[1][0].c_str());
			has_center_variance = true;
		}
		else if (ZQ_CNN_LoadConfigUtils::My_strcmpi(param_name_ptr, "size_variance") == 0)
		{
			size_variance = atof(cur_splits[1][0].c_str());
			has_size_variance = true;
		}
		else if (ZQ_CNN_LoadConfigUtils::My_strcmpi(param_name_ptr, "aspect_ratios") == 0)
		{
			aspect_ratios.clear();
			for (int j = 0; j < cur_splits[1].size(); j++)
			{
				std::string val_str = ZQ_CNN_LoadConfigUtils::Remove_blank(cur_splits[1][j]);
				const char* val_str_ptr = val_str.c_str();
				if (ZQ_CNN_LoadConfigUtils::My_strcmpi(val_str_ptr, "None") != 0)
					aspect_ratios.push_back(atof(val_str_ptr));
			}
			has_aspect_ratios = true;
		}
		else if (ZQ_CNN_LoadConfigUtils::My_strcmpi(param_name_ptr, "use_gray") == 0)
		{
			std::string val_str = ZQ_CNN_LoadConfigUtils::Remove_blank(cur_splits[1][0]);
			const char* val_str_ptr = val_str.c_str();
			if (ZQ_CNN_LoadConfigUtils::My_strcmpi(val_str_ptr, "true") == 0)
			{
				use_gray = true;
				has_use_gray = true;
			}
			else if (ZQ_CNN_LoadConfigUtils::My_strcmpi(val_str_ptr, "false") == 0)
			{
				use_gray = false;
				has_use_gray = true;
			}
			else
			{
				use_gray = atoi(val_str_ptr);
				has_use_gray = true;
			}
		}
		else if (ZQ_CNN_LoadConfigUtils::My_strcmpi(param_name_ptr, "image_size_x") == 0)
		{
			image_size_x = atoi(cur_splits[1][0].c_str());
			has_image_size_x = true;
		}
		else if (ZQ_CNN_LoadConfigUtils::My_strcmpi(param_name_ptr, "image_size_y") == 0)
		{
			image_size_y = atoi(cur_splits[1][0].c_str());
			has_image_size_y = true;
		}
		else
		{
			for (int n = 0; n < spec_names.size(); n++)
			{
				if (ZQ_CNN_LoadConfigUtils::My_strcmpi(param_name_ptr, spec_names[n].c_str()) == 0)
				{
					if (cur_splits[1].size() == 6)
					{
						spec_vals[n].feature_map_size_x = atoi(cur_splits[1][0].c_str());
						spec_vals[n].feature_map_size_y = atoi(cur_splits[1][1].c_str());
						spec_vals[n].shrinkage_x = atoi(cur_splits[1][2].c_str());
						spec_vals[n].shrinkage_y = atoi(cur_splits[1][3].c_str());
						spec_vals[n].box_min = atoi(cur_splits[1][4].c_str());
						spec_vals[n].box_max = atoi(cur_splits[1][5].c_str());
						has_spec_vals[n] = true;
					}
				}
			}
		}
	}

	if (!has_image_mean)
	{
		//show info
		if (show_debug_info)
		{
			printf("image_mean is missing\n");
		}
		return false;
	}
	if (!has_image_std)
	{
		//show info
		if (show_debug_info)
		{
			printf("image_std is missing\n");
		}
		return false;
	}
	if (!has_center_variance)
	{
		//show info
		if (show_debug_info)
		{
			printf("center_variance is missing\n");
		}
		return false;
	}
	if (!has_size_variance)
	{
		//show info
		if (show_debug_info)
		{
			printf("size_variance is missing\n");
		}
		return false;
	}
	if (!has_aspect_ratios)
	{
		//show info
		if (show_debug_info)
		{
			printf("aspect_ratios is missing\n");
		}
		return false;
	}
	if (!has_use_gray)
	{
		//show info
		if (show_debug_info)
		{
			printf("use_gray is missing\n");
		}
		return false;
	}
	if (!has_image_size_x)
	{
		//show info
		if (show_debug_info)
		{
			printf("image_size_x is missing\n");
		}
		return false;
	}
	if (!has_image_size_y)
	{	//show info
		if (show_debug_info)
		{
			printf("image_size_y is missing\n");
		}
		return false;
	}

	int valid_spec_num = 0;
	for (int j = 0; j < spec_vals.size(); j++)
	{
		if (!has_spec_vals[j])
		{
			break;
		}
		else
		{
			valid_spec_num++;
		}
	}
	if (valid_spec_num == 0)
	{
		//show info
		printf("no valid spec\n");
		return false;
	}

	if (use_gray)
	{
		if (image_mean.size() != 1)
		{
			image_mean.resize(1);
			if (show_debug_info)
			{
				printf("image_mean.reisze(1)\n");
			}
		}
	}
	else
	{
		if (image_mean.size() > 3)
		{
			image_mean.resize(3);
			if (show_debug_info)
			{
				printf("image_mean.reisze(3)\n");
			}
		}
		else if (image_mean.size() < 3)
		{
			float last_val = image_mean.back();
			for (int j = image_mean.size(); j < 3; j++)
				image_mean.push_back(last_val);
			if (show_debug_info)
			{
				printf("image_mean padded to 3 elements with last value\n");
			}
		}
	}


	specs.resize(valid_spec_num);
	int aspect_ratio_num = aspect_ratios.size();
	for (int i = 0; i < valid_spec_num; i++)
	{
		spec_vals[i].aspect_ratio_num = aspect_ratio_num;
		for (int j = 0; j < aspect_ratio_num; j++)
			spec_vals[i].aspect_ratios[j] = aspect_ratios[j];
		specs[i] = spec_vals[i];
	}


	return true;
}


bool ZQ_CNN_SSDDetectorPytorch::LoadLabel(const char* file, std::vector<std::string>& names, bool show_debug_info)
{
	std::fstream fin(file, std::ios::in);
	if (!fin.is_open())
	{
		if (show_debug_info)
		{
			printf("failed to open file %s\n", file);
		}
		return false;
	}
	return _load_label_from_file_or_buffer(fin, NULL, 0, names, show_debug_info);
}

bool ZQ_CNN_SSDDetectorPytorch::LoadLabelFromBuffer(const char* buffer, __int64 buffer_len, std::vector<std::string>& names,
	bool show_debug_info)
{
	std::fstream fin;
	return _load_label_from_file_or_buffer(fin, buffer, buffer_len, names, show_debug_info);
}

bool ZQ_CNN_SSDDetectorPytorch::_load_label_from_file_or_buffer(std::fstream& fin, const char* buffer, __int64 buffer_len,
	std::vector<std::string>& names, bool show_debug_info)
{
	std::string line;
	names.clear();
	while (ZQ_CNN_LoadConfigUtils::Get_line(fin, buffer, buffer_len, line))
	{
		std::vector<std::string> cur_splits = ZQ_CNN_LoadConfigUtils::Split_c(line.c_str(), ',');
		for (int j = 0; j < cur_splits.size(); j++)
		{
			std::string label_name = ZQ_CNN_LoadConfigUtils::Remove_blank(cur_splits[j]);
			if (show_debug_info)
			{
				printf("%d:%s\n", j, label_name.c_str());
			}
			if (label_name != "")
			{
				names.push_back(label_name);
			}
		}
	}
	return true;
}


/*
locations: N x 4
probs: N x num_classes
*/
void ZQ_CNN_SSDDetectorPytorch::_post_process(float* locations, float* probs, const float* priors, int N, int num_classes, float center_variance, float size_variance)
{
	_softmax(probs, N, num_classes);
	std::vector<float> boxes(N * 4);
	_convert_locations_to_boxes(locations, priors, N, boxes.data(), center_variance, size_variance);
	_center_form_to_corner_form(boxes.data(), N);
	memcpy(locations, boxes.data(), sizeof(float)*N * 4);
}

/*
locations: N x 4
probs: N x num_classes
*/
void ZQ_CNN_SSDDetectorPytorch::_detection(const float* locations, const float* probs, int N, int num_classes,
	std::vector<ZQ_CNN_SSDDetectorUtils::BBox>& output, float prob_thresh, float nms_thresh, float iou_thresh,
	int top_k, float sigma, int candidate_size)
{
	output.clear();

	for (int class_id = 1; class_id < num_classes; class_id++)
	{
		std::vector<ZQ_CNN_SSDDetectorUtils::BBox> input_bboxes, output_bboxes;
		for (int i = 0; i < N; i++)
		{
			float cur_prob = probs[i*num_classes + class_id];
			if (cur_prob >= prob_thresh)
			{
				ZQ_CNN_SSDDetectorUtils::BBox bbox;
				bbox.prob = cur_prob;
				bbox.class_id = class_id;
				bbox.xmin = locations[i * 4 + 0];
				bbox.ymin = locations[i * 4 + 1];
				bbox.xmax = locations[i * 4 + 2];
				bbox.ymax = locations[i * 4 + 3];
				input_bboxes.push_back(bbox);
			}
		}

		_hard_nms(input_bboxes, output_bboxes, iou_thresh, top_k, candidate_size);
		output.insert(output.end(), output_bboxes.begin(), output_bboxes.end());
	}
}

void ZQ_CNN_SSDDetectorPytorch::_generate_ssd_priors(const std::vector<ZQ_CNN_SSDDetectorUtils::SSDSpec>& specs, int image_size_x, int image_size_y,
	bool clamp, std::vector<float>& prior_box)
{
	prior_box.clear();
	for (int s = 0; s < specs.size(); s++)
	{
		int feat_map_x = specs[s].feature_map_size_x;
		int feat_map_y = specs[s].feature_map_size_y;
		float scale_x = image_size_x / specs[s].shrinkage_x;
		float scale_y = image_size_y / specs[s].shrinkage_y;
		for (int j = 0; j < feat_map_y; j++)
		{
			for (int i = 0; i < feat_map_x; i++)
			{
				float x_center = (i + 0.5f) / scale_x;
				float y_center = (j + 0.5f) / scale_y;

				//small sized square box
				float size = specs[s].box_min;
				float h = size / image_size_y;
				float w = size / image_size_x;
				prior_box.push_back(x_center);
				prior_box.push_back(y_center);
				prior_box.push_back(w);
				prior_box.push_back(h);

				//big sized square box
				size = sqrt(specs[s].box_min*specs[s].box_max);
				float h_1 = size / image_size_y;
				float w_1 = size / image_size_x;
				prior_box.push_back(x_center);
				prior_box.push_back(y_center);
				prior_box.push_back(w_1);
				prior_box.push_back(h_1);

				// change h/w ratio of the small sized box
				for (int k = 0; k < specs[s].aspect_ratio_num; k++)
				{
					float ratio = sqrt(specs[s].aspect_ratios[k]);
					prior_box.push_back(x_center);
					prior_box.push_back(y_center);
					prior_box.push_back(w * ratio);
					prior_box.push_back(h / ratio);

					prior_box.push_back(x_center);
					prior_box.push_back(y_center);
					prior_box.push_back(w / ratio);
					prior_box.push_back(h * ratio);
				}
			}
		}
	}

	if (clamp)
	{
		for (int i = 0; i < prior_box.size(); i++)
		{
			prior_box[i] = std::max<float>(0.0f, std::min<float>(1.0f, prior_box[i]));
		}
	}
}

/*
locations: N x 4
priors: N x 4
boxes: N x 4
*/
void ZQ_CNN_SSDDetectorPytorch::_convert_locations_to_boxes(const float* locations, const float* priors, int N,
	float* boxes, float center_variance, float size_variance)
{
	for (int i = 0; i < N; i++)
	{
		float cx = priors[i * 4 + 0];
		float cy = priors[i * 4 + 1];
		float w = priors[i * 4 + 2];
		float h = priors[i * 4 + 3];
		boxes[i * 4 + 0] = locations[i * 4 + 0] * center_variance*w + cx;
		boxes[i * 4 + 1] = locations[i * 4 + 1] * center_variance*h + cy;
		boxes[i * 4 + 2] = exp(locations[i * 4 + 2] * size_variance)*w;
		boxes[i * 4 + 3] = exp(locations[i * 4 + 3] * size_variance)*h;
	}
}

void ZQ_CNN_SSDDetectorPytorch::_center_form_to_corner_form(float* boxes, int N)
{
	for (int i = 0; i < N; i++)
	{
		float cx = boxes[i * 4 + 0];
		float cy = boxes[i * 4 + 1];
		float w = boxes[i * 4 + 2];
		float h = boxes[i * 4 + 3];
		boxes[i * 4 + 0] = cx - 0.5f*w;
		boxes[i * 4 + 1] = cy - 0.5f*h;
		boxes[i * 4 + 2] = cx + 0.5f*w;
		boxes[i * 4 + 3] = cy + 0.5f*h;
	}
}

void ZQ_CNN_SSDDetectorPytorch::_softmax(float* probs, int N, int num_classes)
{
	int C = num_classes;
	for (int i = 0; i < N; i++)
	{
		float max_val = -FLT_MAX;
		for (int j = 0; j < C; j++)
			max_val = __max(max_val, probs[i*C + j]);
		float sum_val = 0;
		for (int j = 0; j < C; j++)
		{
			float tmp_val = exp(probs[i*C + j] - max_val);
			sum_val += tmp_val;
			probs[i*C + j] = tmp_val;
		}
		sum_val = 1.0f / sum_val;
		for (int j = 0; j < C; j++)
			probs[i*C + j] *= sum_val;
	}
}

float ZQ_CNN_SSDDetectorPytorch::_iou_of(const ZQ_CNN_SSDDetectorUtils::BBox& bbox0, const ZQ_CNN_SSDDetectorUtils::BBox& bbox1, float eps)
{
	float max_of_xmin = __max(bbox0.xmin, bbox1.xmin);
	float max_of_ymin = __max(bbox0.ymin, bbox1.ymin);
	float min_of_xmax = __min(bbox0.xmax, bbox1.xmax);
	float min_of_ymax = __min(bbox0.ymax, bbox1.ymax);

	float overlap_area = 0;
	if (min_of_xmax > max_of_xmin && min_of_ymax - max_of_ymin)
		overlap_area = (min_of_xmax - max_of_xmin)*(min_of_ymax - max_of_ymin);
	float area0 = (bbox0.xmax - bbox0.xmin)*(bbox0.ymax - bbox0.ymin);
	float area1 = (bbox1.xmax - bbox1.xmin)*(bbox1.ymax - bbox1.ymin);
	return overlap_area / (area0 + area1 - overlap_area + eps);
}

void ZQ_CNN_SSDDetectorPytorch::_hard_nms(const std::vector<ZQ_CNN_SSDDetectorUtils::BBox>& input, std::vector<ZQ_CNN_SSDDetectorUtils::BBox>& output,
	float iou_thresh, int top_k, int candidate_size)
{
	int box_num = input.size();
	std::vector<ZQ_CNN_SSDDetectorUtils::OrderScore> order_scores(box_num);
	std::vector<bool> exist_flags(box_num, false);
	for (int i = 0; i < box_num; i++)
	{
		order_scores[i].ori_order = i;
		order_scores[i].score = input[i].prob;
	}
	std::sort(order_scores.begin(), order_scores.end(), ZQ_CNN_SSDDetectorUtils::_cmp_score);
	if (candidate_size > 0)
	{
		if (box_num > candidate_size)
		{
			order_scores.erase(order_scores.begin(), order_scores.begin() + box_num - candidate_size);
		}
	}
	for (int i = 0; i < order_scores.size(); i++)
	{
		int i_order = order_scores[i].ori_order;
		exist_flags[i_order] = true;
	}
	std::vector<int> heros;
	while (order_scores.size() > 0)
	{
		int i_order = order_scores.back().ori_order;
		order_scores.pop_back();
		if (!exist_flags[i_order])
			continue;
		heros.push_back(i_order);
		if (top_k > 0 && heros.size() >= top_k)
			break;
		exist_flags[i_order] = false;//delete it
		for (int j = 0; j < order_scores.size(); j++)
		{
			int j_order = order_scores[j].ori_order;
			if (!exist_flags[j_order])
				continue;

			float cur_iou = _iou_of(input[i_order], input[j_order]);
			if (cur_iou > iou_thresh)
			{
				exist_flags[j_order] = false;//delete it
			}
		}
	}

	output.resize(heros.size());
	for (int i = 0; i < heros.size(); i++)
	{
		output[i] = input[heros[i]];
	}
}