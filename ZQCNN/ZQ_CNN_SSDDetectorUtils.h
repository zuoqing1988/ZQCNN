#ifndef _ZQ_CNN_SSD_DETECTOR_UTILS_H_
#define _ZQ_CNN_SSD_DETECTOR_UTILS_H_
#pragma once

#include <string.h>

namespace ZQ
{
	class ZQ_CNN_SSDDetectorUtils
	{
	public:
		class OrderScore
		{
		public:
			float score;
			int ori_order;

			OrderScore()
			{
				score = 0;
				ori_order = 0;
			}
		};
		static bool _cmp_score(const OrderScore& lsh, const OrderScore& rsh)
		{
			return lsh.score < rsh.score;
		}

		class BBox
		{
		public:
			BBox() {
				xmin = xmax = ymin = ymax = 0;
				class_id = 0;
				prob = 0;
			}
			float xmin, xmax, ymin, ymax;
			int class_id;
			float prob;
		};

		class SSDSpec
		{
		public:
			SSDSpec()
			{
				feature_map_size_x = feature_map_size_y = 0;
				shrinkage_x = shrinkage_y = 0;
				box_min = box_max = 0;
				aspect_ratio_num = 0;
				memset(aspect_ratios, 0, sizeof(float) * 16);
			}
			int feature_map_size_x, feature_map_size_y;
			float shrinkage_x, shrinkage_y;
			float box_min, box_max;
			float aspect_ratios[16];
			int aspect_ratio_num;

		};
	};
}
#endif
