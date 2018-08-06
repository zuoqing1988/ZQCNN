#ifndef _ZQ_CPU_RENDERER_2D_WORKSPACE_H_
#define _ZQ_CPU_RENDERER_2D_WORKSPACE_H_
#pragma once

#include "ZQ_DoubleImage.h"
#include "ZQ_TextureSampler.h"
#include "ZQ_CPURenderer3DWorkSpace.h"
#include "ZQ_Vec2D.h"

namespace ZQ
{
	class ZQ_CPURenderer2DWorkspace
	{
	public:
		ZQ_CPURenderer2DWorkspace(unsigned int w, unsigned int h)
		{
			min_clip_near_depth = 1e-3;
			opt.width = w;
			opt.height = h;
			color_buffer.allocate(w, h, 4);
			depth_buffer.allocate(w, h, 1);
			opt.enable_depth_test = true;
			opt.enable_texture = false;
			opt.default_color[0] = opt.default_color[1] = opt.default_color[2] = opt.default_color[3] = 0;
			opt.enable_alpha_blend = false;
			opt.depth_clip_near = 1;
			opt.depth_clip_far = 1000;
			opt.sampler = 0;
		}

		~ZQ_CPURenderer2DWorkspace(){}


	private:
		ZQ_DImage<float> color_buffer;
		ZQ_DImage<float> depth_buffer;
		float min_clip_near_depth;
		ZQ_CPURenderer3DWorkspace::RenderOptions opt;

	public:
		void EnableDepthTest(){opt.enable_depth_test = true;}
		void DisableDepthTest(){opt.enable_depth_test = false;}

		void EnableTexture(){opt.enable_texture = true;}
		void DisableTexture(){opt.enable_texture = false;}

		void EnableTextureSampleCubic(){opt.enable_texture_sample_cubic = true;}
		void DisableTextureSampleCubic(){opt.enable_texture_sample_cubic = false;}

		void EnableAlphaBlend(){opt.enable_alpha_blend = true;}
		void DisableAlphaBlend(){opt.enable_alpha_blend = false;}

		void SetAlphaBlendMode(const ZQ_CPURenderer3DWorkspace::ALPHA_BLEND_MODE mode){opt.alpha_blend_mode = mode;}

		void SetDefaultColor(const float r, const float g, const float b, const float a)
		{
			opt.default_color[0] = __min(1, __max(0, r));
			opt.default_color[1] = __min(1, __max(0, g));
			opt.default_color[2] = __min(1, __max(0, b));
			opt.default_color[3] = __min(1, __max(0, a));
		}

		bool SetBackground(const ZQ_DImage<float>& image)
		{
			int im_width = image.width();
			int im_height = image.height();
			int im_nChannels = image.nchannels();
			if (opt.width != im_width || opt.height != im_height || im_nChannels > 4)
				return false;
			float*& color_buffer_data = color_buffer.data();
			const float*& im_data = image.data();
			for (int h = 0; h < im_height; h++)
			{
				for (int w = 0; w < im_width; w++)
				{
					memcpy(color_buffer_data + (h*im_width + w) * 4, im_data + (h*im_width + w)*im_nChannels, sizeof(float)*im_nChannels);
				}
			}
			return true;
		}

		bool SetClip(const float _near, const float _far)
		{
			if (_near > _far)
				return false;
			if (_near < min_clip_near_depth)
				return false;

			opt.depth_clip_near = _near;
			opt.depth_clip_far = _far;
			return true;
		}

		void GetClip(float& _near, float& _far) const
		{
			_near = opt.depth_clip_near;
			_far = opt.depth_clip_far;
		}

		void ClearColorBuffer(const float r, const float g, const float b, const float a)
		{
			float*& color_data = color_buffer.data();
			float _r = __min(1, __max(0, r));
			float _g = __min(1, __max(0, g));
			float _b = __min(1, __max(0, b));
			float _a = __min(1, __max(0, a));
			for (int i = 0; i < opt.width*opt.height; i++)
			{
				color_data[i * 4 + 0] = _r;
				color_data[i * 4 + 1] = _g;
				color_data[i * 4 + 2] = _b;
				color_data[i * 4 + 3] = _a;
			}
		}

		void ClearDepthBuffer(const float d)
		{
			float*& depth_data = depth_buffer.data();
			for (int i = 0; i < opt.width*opt.height; i++)
				depth_data[i] = d;
		}

		unsigned int GetBufferWidth() const {return opt.width;}
		unsigned int GetBufferHeight() const {return opt.height;}
		const float*& GetColorBufferPtr() const {return color_buffer.data();}

		void BindSampler(const ZQ_TextureSampler<float>* s){opt.sampler = s;}
		void RenderIndexedTriangles(const float* vertices, const int* indices, const int vertex_num, const int triangle_num, const ZQ_CPURenderer3DWorkspace::VERTEX_FORMAT format)
		{
			int nChannels = 0;
			switch (format)
			{
			case ZQ_CPURenderer3DWorkspace::VERTEX_POSITION3:
				nChannels = 3;
				break;
			case ZQ_CPURenderer3DWorkspace::VERTEX_POSITION3_COLOR4:
				nChannels = 7;
				break;
			case ZQ_CPURenderer3DWorkspace::VERTEX_POSITION3_TEXCOORD2:
				nChannels = 5;
				break;
			}

			for (int i = 0; i < triangle_num; i++)
			{
				const float* vertex1 = vertices + indices[i * 3 + 0] * nChannels;
				const float* vertex2 = vertices + indices[i * 3 + 1] * nChannels;
				const float* vertex3 = vertices + indices[i * 3 + 2] * nChannels;
				ZQ_Vec2D pt1(vertex1[0], vertex1[1]);
				ZQ_Vec2D pt2(vertex2[0], vertex2[1]);
				ZQ_Vec2D pt3(vertex3[0], vertex3[1]);
				ZQ_CPURenderer3DWorkspace::_split_triangle_and_render_in_window(color_buffer, depth_buffer, opt, pt1, vertex1, pt2, vertex2, pt3, vertex3, format);
			}
		}
	};
}

#endif