#ifndef _ZQ_CNN_SSD_DETECTOR_PYTORCH_H_
#define _ZQ_CNN_SSD_DETECTOR_PYTORCH_H_
#pragma once

#include "ZQ_CNN_SSDDetectorUtils.h"
#include "ZQ_CNN_Net.h"
#include <fstream>

namespace ZQ
{
	class ZQ_CNN_SSDDetectorPytorch
	{
	public:
		ZQ_CNN_SSDDetectorPytorch();
		~ZQ_CNN_SSDDetectorPytorch();

		bool LoadModel(const std::string& param_file, const std::string& bin_file, const std::string& cfg_file);

		bool Detect(const unsigned char* im_data, int im_width, int im_height, int widthStep,
			std::vector<ZQ_CNN_SSDDetectorUtils::BBox>& output);

		void SetParam(float prob_thresh = 0.3f, float iou_thresh = 0.5f, int top_k = -1);

		bool UseGray() const;

		void SetShowDebugInfo(bool b);

		void GetInputSize(int &h, int& w, int& c)const;

		float GetMinSize() const;

	private:
		ZQ_CNN_Net runner;
		bool show_debug_info;
		std::vector<float> prior_boxes;
		// set param
		float prob_thresh;
		float iou_thresh;
		int top_k;

		// load form cfg
		std::vector<ZQ_CNN_SSDDetectorUtils::SSDSpec> specs;
		std::vector<float> image_mean;
		float image_std;
		int image_size_x, image_size_y;
		bool use_gray;
		float center_variance;
		float size_variance;

	private:

		bool _load_cfg(const char* cfg_file);

		bool _load_cfg_from_buffer(const char* buffer, __int64 buffer_len);

		bool _load_cfg_from_file_or_buffer(std::fstream& fin, const char* buffer, __int64 buffer_len);

	public:
		static bool LoadLabel(const char* label_file, std::vector<std::string>& names, bool show_debug_info = false);

		static bool LoadLabelFromBuffer(const char* buffer, __int64 buffer_len, std::vector<std::string>& names, bool show_debug_info = false);

	private:
		static bool _load_label_from_file_or_buffer(std::fstream& fin, const char* buffer, __int64 buffer_len, std::vector<std::string>& names, bool show_debug_info);

	private:

		/*
		locations: N x 4
		probs: N x num_classes
		*/
		static void _post_process(float* locations, float* probs, const float* priors, int N,
			int num_classes, float center_variance, float size_variance);

		/*
		locations: N x 4
		probs: N x num_classes
		*/
		static void _detection(const float* locations, const float* probs, int N, int num_classes,
			std::vector<ZQ_CNN_SSDDetectorUtils::BBox>& output, float prob_thresh = 0.5, float nms_thresh = 0.5f,
			float iou_thresh = 0.5, int top_k = 200, float sigma = 0.5, int candidate_size = 200);

	private:

		static void _generate_ssd_priors(const std::vector<ZQ_CNN_SSDDetectorUtils::SSDSpec>& specs, int image_size_x, int image_size_y,
			bool clamp, std::vector<float>& prior_box);

		/*
		locations: N x 4
		priors: N x 4
		boxes: N x 4
		*/
		static void _convert_locations_to_boxes(const float* locations, const float* priors, int N,
			float* boxes, float center_variance, float size_variance);

		static void _center_form_to_corner_form(float* boxes, int N);

		static void _softmax(float* probs, int N, int num_classes);

		static float _iou_of(const ZQ_CNN_SSDDetectorUtils::BBox& bbox0, const ZQ_CNN_SSDDetectorUtils::BBox& bbox1, float eps = 1e-5);

	public:
		static void _hard_nms(const std::vector<ZQ_CNN_SSDDetectorUtils::BBox>& input, std::vector<ZQ_CNN_SSDDetectorUtils::BBox>& output,
			float iou_thresh, int top_k = -1, int candidate_size = 200);
	};
}

#endif
