#ifndef _ZQ_FACE_EXTRACTOR_H_
#define _ZQ_FACE_EXTRACTOR_H_
#pragma once
#include "ZQ_FaceGroup.h"
#include "ZQ_FaceSearchTarget.h"
#include "ZQ_FaceDetector.h"
#include "ZQ_FaceRecognizer.h"
namespace ZQ
{
	class ZQ_FaceExtractor
	{
	public:
		static void SortByAreaDecend(std::vector<ZQ_CNN_BBox>& boxes)
		{
			ZQ_CNN_BBox tmp;
			int num = boxes.size();
			for (int pass = 1; pass < num; pass++)
			{
				for (int i = 0; i < num - 1; i++)
				{
					if (boxes[i].area < boxes[i + 1].area)
					{
						tmp = boxes[i];
						boxes[i] = boxes[i + 1];
						boxes[i + 1] = tmp;
					}
				}
			}
		}


		/*max_face_num = 0 means not limitation*/
		static bool ExtractFacesFromOneImage(const unsigned char* in_img, int in_width, int in_height, int in_widthStep,
			ZQ_PixelFormat pixFmt, ZQ_FaceDetector& detector, ZQ_FaceRecognizer& recognizer, int min_face_size,
			ZQ_FaceGroup& group, int max_face_num)
		{
			return ExtractFacesFromOneImageROI(in_img, in_width, in_height, in_widthStep, pixFmt, 0, 0, 1, 1,
				detector, recognizer, min_face_size, group, max_face_num);
		}

		/*max_face_num = 0 means not limitation*/
		static bool ExtractFacesFromOneImageROI(const unsigned char* in_img, int in_width, int in_height, int in_widthStep,
			ZQ_PixelFormat pixFmt, float roi_min_x, float roi_min_y, float roi_max_x, float roi_max_y,
			ZQ_FaceDetector& detector, ZQ_FaceRecognizer& recognizer, int min_face_size,
			ZQ_FaceGroup& group, int max_face_num)
		{
			std::vector<ZQ_CNN_BBox> bbox;
			if (!detector.FindFaceROI(in_img, in_width, in_height, in_widthStep, pixFmt,
				roi_min_x, roi_min_y, roi_max_x, roi_max_y, min_face_size,1.2, bbox))
				return false;

			int num = bbox.size();
			SortByAreaDecend(bbox);
			if (max_face_num > 0 && num > max_face_num)
				num = __min(num, max_face_num);

			group.feat_dim = recognizer.GetFeatDim();
			group.face_feats.resize(num);
			int feat_dim = recognizer.GetFeatDim();
			for (int i = 0; i < num; i++)
			{
				group.face_feats[i].ChangeSize(feat_dim);

				if (!recognizer.ExtractFeature(in_img, in_width, in_height, in_widthStep, pixFmt,
					bbox[i].ppoint, bbox[i].ppoint + 5, group.face_feats[i].pData, true))
				{
					return false;
				}
			}

			if (group.HasBox())
			{
				group.face_boxes = bbox;
			}

			return true;
		}

		static bool GenerateTargetFromOneImage(
			const unsigned char* in_img, int in_width, int in_height, int in_widthStep, ZQ_PixelFormat pixFmt,
			ZQ_FaceDetector& detector, ZQ_FaceRecognizer& recognizer, int min_face_size, ZQ_FaceSearchTarget& target)
		{
			ZQ_FaceGroupWithBox group;
			if (!ExtractFacesFromOneImage(in_img, in_width, in_height, in_widthStep, pixFmt, detector, recognizer, min_face_size, group, 0))
				return false;

			int num = group.face_feats.size();
			target.targets.resize(num);
			for (int i = 0; i < num; i++)
			{
				if (target.targets[i].HasBox())
					target.targets[i].face_boxes.push_back(group.face_boxes[i]);
				target.targets[i].face_feats.push_back(group.face_feats[i]);
				target.targets[i].feat_dim = group.feat_dim;
			}
			return true;
		}
	};
	
}
#endif
