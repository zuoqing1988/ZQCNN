#ifndef _ZQ_CPU_RENDERER_3D_WORKSPACE_H_
#define _ZQ_CPU_RENDERER_3D_WORKSPACE_H_
#pragma once

#include "ZQ_DoubleImage.h"
#include "ZQ_TextureSampler.h"
#include "ZQ_ScanLinePolygonFill.h"
#include "ZQ_Vec2D.h"
#include "ZQ_Vec3D.h"
#include "ZQ_MathBase.h"

namespace ZQ
{
	class ZQ_CPURenderer3DWorkspace
	{
		friend class ZQ_CPURenderer2DWorkspace;
	
	public:
		ZQ_CPURenderer3DWorkspace(unsigned int w, unsigned int h, bool zAxis_in)
		{
			min_clip_near_depth = 1e-3;
			this->zAxis_in = zAxis_in;
			opt.width = w;
			opt.height = h;
			opt.cx = w*0.5;
			opt.cy = h*0.5;
			opt.focal_len = h;
			memset(view_matrix, 0, sizeof(float)* 16);
			view_matrix[0] = view_matrix[5] = view_matrix[10] = view_matrix[15] = 1;
			memset(world_matrix, 0, sizeof(float)* 16);
			world_matrix[0] = world_matrix[5] = world_matrix[10] = world_matrix[15] = 1;
			color_buffer.allocate(w, h, 4);
			depth_buffer.allocate(w, h, 1);
			opt.enable_depth_test = true;
			opt.enable_texture = false;
			opt.depth_clip_far = 100;
			opt.depth_clip_near = min_clip_near_depth;
			opt.default_color[0] = opt.default_color[1] = opt.default_color[2] = opt.default_color[3] = 0;
			opt.enable_alpha_blend = false;
			opt.sampler = 0;
		}

		~ZQ_CPURenderer3DWorkspace(){}

		enum ALPHA_BLEND_MODE{
			ALPHABLEND_SRC_PLUS_DST,
			ALPHABLEND_SRC_ALPHA_DST_ONE_MINUS_SRC,
			ALPHABLEND_SRC_ONE_DST_ONE_MINUS_SRC,
			ALPHABLEND_SRC_ONE_DST_ONE_MINUS_SRC_EACHCHANNEL
		};
		enum VERTEX_FORMAT
		{
			VERTEX_POSITION3,
			VERTEX_POSITION3_COLOR4,
			VERTEX_POSITION3_TEXCOORD2
		};

	protected:
		class RenderOptions
		{
		public:
			bool enable_depth_test;
			bool enable_texture;
			bool enable_texture_sample_cubic;
			bool enable_alpha_blend;
			ALPHA_BLEND_MODE alpha_blend_mode;
			float default_color[4];
			float depth_clip_near;
			float depth_clip_far;
			int width, height;
			float cx, cy;
			float focal_len;
			const ZQ_TextureSampler<float>* sampler;
		};
		enum ClipMode{
			CLIPMODE_NEAR, CLIPMODE_FAR, CLIPMODE_LEFT, CLIPMODE_RIGHT, CLIPMODE_BOTTOM, CLIPMODE_TOP
		};

	private:
		float min_clip_near_depth;
		bool zAxis_in;
		float view_matrix[16];
		float world_matrix[16];
		ZQ_DImage<float> color_buffer;
		ZQ_DImage<float> depth_buffer;

		RenderOptions opt;

	public:
		void EnableDepthTest(){ opt.enable_depth_test = true; }
		void DisableDepthTest(){ opt.enable_depth_test = false; }

		void EnableTexture(){ opt.enable_texture = true; }
		void DisableTexture(){ opt.enable_texture = false; }

		void EnableTextureSampleCubic(){ opt.enable_texture_sample_cubic = true; }
		void DisableTextureSampleCubic(){ opt.enable_texture_sample_cubic = false; }

		void EnableAlphaBlend(){ opt.enable_alpha_blend = true; }
		void DisableAlphaBlend(){ opt.enable_alpha_blend = false; }

		void SetAlphaBlendMode(const ALPHA_BLEND_MODE mode){ opt.alpha_blend_mode = mode; }

		void SetDefaultColor(const float r, const float g, const float b, const float a)
		{
			opt.default_color[0] = __min(1, __max(0, r));
			opt.default_color[1] = __min(1, __max(0, g));
			opt.default_color[2] = __min(1, __max(0, b));
			opt.default_color[3] = __min(1, __max(0, a));
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

		void SetIntrinsicPara(const float _cx, const float _cy, const float _focal_len)
		{
			opt.cx = _cx;
			opt.cy = _cy;
			opt.focal_len = _focal_len;
		}

		void SetWorldMatrix(const float* world_mat)
		{
			memcpy(world_matrix, world_mat, sizeof(float)* 16);
		}

		const float* GetWorldMatrix() const
		{
			return world_matrix;
		}

		void SetViewMatrix(const float* view_mat)
		{
			memcpy(view_matrix, view_mat, sizeof(float)* 16);
		}

		const float* GetViewMatrix() const
		{
			return view_matrix;
		}

		bool LookAt(ZQ_Vec3D eyepos, ZQ_Vec3D target, ZQ_Vec3D updir)
		{
			ZQ_Vec3D z_dir;
			if (zAxis_in)
				z_dir = target - eyepos;
			else
				z_dir = eyepos - target;
				
			z_dir.Normalized();
			ZQ_Vec3D x_dir = updir.CrossProduct(z_dir);
			if (x_dir.Length() == 0)
				return false;

			x_dir.Normalized();
			ZQ_Vec3D y_dir = z_dir.CrossProduct(x_dir);

			if (y_dir.Length() == 0)
				return false;
			y_dir.Normalized();

			float inv_view[16] =
			{
				x_dir.x, y_dir.x, z_dir.x, eyepos.x,
				x_dir.y, y_dir.y, z_dir.y, eyepos.y,
				x_dir.z, y_dir.z, z_dir.z, eyepos.z,
				0, 0, 0, 1
			};

			ZQ_MathBase::MatrixInverse(inv_view, 4, view_matrix);
			return true;
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

		unsigned int GetBufferWidth() const { return opt.width; }
		unsigned int GetBufferHeight() const { return opt.height; }
		const float*& GetColorBufferPtr() const { return color_buffer.data(); }
		const float*& GetDepthBufferPtr() const { return depth_buffer.data(); }

		void BindSampler(const ZQ_TextureSampler<float>* s){ opt.sampler = s; }
		bool RenderIndexedTriangles(const float* vertices, const int* indices, const int vertex_num, const int triangle_num, const VERTEX_FORMAT format)
		{
			int nChannels = 0;
			switch (format)
			{
			case VERTEX_POSITION3:
				nChannels = 3;
				break;
			case VERTEX_POSITION3_COLOR4:
				nChannels = 7;
				break;
			case VERTEX_POSITION3_TEXCOORD2:
				nChannels = 5;
				break;
			default:
				return false;
			}

			float model_view[16];
			ZQ_MathBase::MatrixMul(view_matrix, world_matrix, 4, 4, 4, model_view);

			float* new_vertices = new float[vertex_num*nChannels];
			memcpy(new_vertices, vertices, sizeof(float)*nChannels*vertex_num);
			for (int i = 0; i < vertex_num; i++)
			{
				new_vertices[i*nChannels + 0] = model_view[0] * vertices[i*nChannels + 0] + model_view[1] * vertices[i*nChannels + 1] + model_view[2] * vertices[i*nChannels + 2] + model_view[3];
				new_vertices[i*nChannels + 1] = model_view[4] * vertices[i*nChannels + 0] + model_view[5] * vertices[i*nChannels + 1] + model_view[6] * vertices[i*nChannels + 2] + model_view[7];
				new_vertices[i*nChannels + 2] = model_view[8] * vertices[i*nChannels + 0] + model_view[9] * vertices[i*nChannels + 1] + model_view[10] * vertices[i*nChannels + 2] + model_view[11];
				if (!zAxis_in)
					new_vertices[i*nChannels + 2] *= -1;
			}

			for (int i = 0; i < triangle_num; i++)
			{
				//printf("i=%d\n", i);
				if (!_render_triangle(color_buffer, depth_buffer, opt, new_vertices + indices[i * 3 + 0] * nChannels, new_vertices + indices[i * 3 + 1] * nChannels, new_vertices + indices[i * 3 + 2] * nChannels, format))
				{
					delete[]new_vertices;
					return false;
				}
			}

			delete[]new_vertices;
			return true;
		}


	protected:
		static float _area_of_triangle(const ZQ_Vec2D& pt1, const ZQ_Vec2D& pt2, const ZQ_Vec2D& pt3)
		{
			float x1 = pt2.x - pt1.x;
			float y1 = pt2.y - pt1.y;
			float x2 = pt3.x - pt1.x;
			float y2 = pt3.y - pt1.y;
			return 0.5*fabs(x1*y2 - x2*y1);
		}

		static float _area_of_triangle(const ZQ_Vec3D& pt1, const ZQ_Vec3D& pt2, const ZQ_Vec3D& pt3)
		{
			ZQ_Vec3D dir12 = pt2 - pt1;
			ZQ_Vec3D dir13 = pt3 - pt1;
			ZQ_Vec3D nv = dir12.CrossProduct(dir13);
			return 0.5*nv.Length();
		}

		static bool _is_in_range(const ZQ_Vec2D pt, int width, int height)
		{
			return pt.x >= 0 && pt.x <= width - 1 && pt.y >= 0 && pt.y <= height - 1;
		}

		static bool _render_triangle(ZQ_DImage<float>& color_buffer, ZQ_DImage<float>& depth_buffer, const RenderOptions& opt, const float* vertex1, const float* vertex2, const float* vertex3, const VERTEX_FORMAT format)
		{
			const int max_tri_num_after_clip_depth = 5;
			ZQ_Vec3D tmp_pts1[max_tri_num_after_clip_depth], tmp_pts2[max_tri_num_after_clip_depth];
			tmp_pts1[0].x = vertex1[0];
			tmp_pts1[0].y = vertex1[1];
			tmp_pts1[0].z = vertex1[2];
			tmp_pts1[1].x = vertex2[0];
			tmp_pts1[1].y = vertex2[1];
			tmp_pts1[1].z = vertex2[2];
			tmp_pts1[2].x = vertex3[0];
			tmp_pts1[2].y = vertex3[1];
			tmp_pts1[2].z = vertex3[2];
			int len1 = 3, len2;

			_Sutherland_Hodgeman_clip_depth(tmp_pts1, tmp_pts2, len1, len2, CLIPMODE_NEAR, opt.depth_clip_near, opt.depth_clip_far);
			_Sutherland_Hodgeman_clip_depth(tmp_pts2, tmp_pts1, len2, len1, CLIPMODE_FAR, opt.depth_clip_near, opt.depth_clip_far);
			if (len1 < 3)
				return true;

			float** tmp_vertices1 = 0;
			if (!_generate_interpolate_vertices_after_clip_depth(vertex1, vertex2, vertex3, len1, tmp_pts1, tmp_vertices1, format))
			{
				return false;
			}

			if (tmp_vertices1 == 0)
				return true;
			
			ZQ_Vec2D tmp_2d[max_tri_num_after_clip_depth];
			for (int i = 0; i < len1; i++)
			{
				tmp_2d[i].x = tmp_vertices1[i][0] / tmp_vertices1[i][2] * opt.focal_len + opt.cx;
				tmp_2d[i].y = tmp_vertices1[i][1] / tmp_vertices1[i][2] * opt.focal_len + opt.cy;
			}

			for (int i = 0; i < len1 - 2; i++)
			{
				if (!_split_triangle_and_render_in_window(color_buffer, depth_buffer, opt, tmp_2d[0], tmp_vertices1[0], tmp_2d[i + 1], tmp_vertices1[i + 1], tmp_2d[i + 2], tmp_vertices1[i + 2], format))
				{
					_release_interpolate_vertices(len1, tmp_vertices1);
					return false;
				}
			}

			_release_interpolate_vertices(len1, tmp_vertices1);
			return true;
		}

		static bool _render_triangle_whole_in_window(ZQ_DImage<float>& color_buffer, ZQ_DImage<float>& depth_buffer, const RenderOptions& opt, const ZQ_Vec2D& pt1, const float* vertex1, const ZQ_Vec2D& pt2, const float* vertex2, const ZQ_Vec2D& pt3, const float* vertex3, const VERTEX_FORMAT format)
		{
			if (vertex1 == 0 || vertex1 == 0 || vertex2 == 0)
				return false;

			float d1 = vertex1[2];
			float d2 = vertex2[2];
			float d3 = vertex3[2];

			std::vector<ZQ::ZQ_Vec2D> poly, pixels;
			poly.push_back(pt1);
			poly.push_back(pt2);
			poly.push_back(pt3);
			if (!ZQ::ZQ_ScanLinePolygonFill::ScanLinePolygonFill(poly, pixels))
				return false;
			if (pixels.size() == 0)
				return true;

			float*& depth_buffer_data = depth_buffer.data();
			float*& color_buffer_data = color_buffer.data();
			int width = color_buffer.width();
			int height = color_buffer.height();
			for (int i = 0; i < pixels.size(); i++)
			{
				ZQ::ZQ_Vec2D cur_pt = pixels[i];
				float area3 = _area_of_triangle(pt1, pt2, cur_pt);
				float area1 = _area_of_triangle(pt2, pt3, cur_pt);
				float area2 = _area_of_triangle(pt3, pt1, cur_pt);
				int cur_x = cur_pt.x;
				int cur_y = cur_pt.y;
				if (cur_x < 0 || cur_x >= width || cur_y < 0 || cur_y >= height)
					continue;

				float sum_area = area1 + area2 + area3;
				if (sum_area == 0)
					continue;

				float w1 = area1 / sum_area;
				float w2 = area2 / sum_area;
				float w3 = area3 / sum_area;

				float depth = 1.0 / (w1 / d1 + w2 / d2 + w3 / d3);
				if (opt.enable_depth_test)
				{
					if (depth < opt.depth_clip_near || depth > opt.depth_clip_far || (depth_buffer_data[cur_y*width + cur_x] > 0 && depth >= depth_buffer_data[cur_y*width + cur_x]))
						continue;
				}
				depth_buffer_data[cur_y*width + cur_x] = depth;

				float src_color[4] = { 0 };

				int nChannels_Pos = 3;
				int nChannels_Color = 4;
				int nChannels_Tex = 2;
				switch (format)
				{
				case VERTEX_POSITION3:

					memcpy(src_color, opt.default_color, sizeof(float)* 4);

					break;
				case VERTEX_POSITION3_COLOR4:
					nChannels_Pos = 3;
					nChannels_Color = 4;
					for (int cc = 0; cc < nChannels_Color; cc++)
					{
						src_color[cc] = depth*(vertex1[nChannels_Pos + cc] / d1 * w1 + vertex2[nChannels_Pos + cc] / d2 * w2 + vertex3[nChannels_Pos + cc] / d3 * w3);
					}

					break;
				case VERTEX_POSITION3_TEXCOORD2:
					nChannels_Pos = 3;
					nChannels_Tex = 2;
					float tex_coord[2];
					for (int cc = 0; cc < nChannels_Tex; cc++)
					{
						tex_coord[cc] = depth*(vertex1[nChannels_Pos + cc] / d1 * w1 + vertex2[nChannels_Pos + cc] / d2 * w2 + vertex3[nChannels_Pos + cc] / d3 * w3);
					}
					if (opt.sampler)
						opt.sampler->Sample_NormalizedCoord(tex_coord[0], tex_coord[1], src_color, opt.enable_texture_sample_cubic);

					break;
				}

				if (opt.enable_alpha_blend)
				{
					switch (opt.alpha_blend_mode)
					{
					case ALPHABLEND_SRC_PLUS_DST:
						for (int cc = 0; cc < 4; cc++)
						{
							color_buffer_data[(cur_y*width + cur_x) * 4 + cc] += src_color[cc];
							color_buffer_data[(cur_y*width + cur_x) * 4 + cc] = __min(1, __max(0, color_buffer_data[(cur_y*width + cur_x) * 4 + cc]));
						}
						break;
					case ALPHABLEND_SRC_ALPHA_DST_ONE_MINUS_SRC:
						for (int cc = 0; cc < 4; cc++)
						{
							float src_alpha = src_color[3];
							color_buffer_data[(cur_y*width + cur_x) * 4 + cc] *= (1 - src_alpha);
							color_buffer_data[(cur_y*width + cur_x) * 4 + cc] += src_color[cc]*src_alpha;
						}
						break;
					case ALPHABLEND_SRC_ONE_DST_ONE_MINUS_SRC:			
						for (int cc = 0; cc < 4; cc++)
						{
							float src_alpha = src_color[3];
							color_buffer_data[(cur_y*width + cur_x) * 4 + cc] *= (1 - src_alpha);
							color_buffer_data[(cur_y*width + cur_x) * 4 + cc] += src_color[cc];
						}
						break;
					case ALPHABLEND_SRC_ONE_DST_ONE_MINUS_SRC_EACHCHANNEL:
						for (int cc = 0; cc < 4; cc++)
						{
							color_buffer_data[(cur_y*width + cur_x) * 4 + cc] *= (1 - src_color[cc]);
							color_buffer_data[(cur_y*width + cur_x) * 4 + cc] += src_color[cc];
						}
						break;
					}
				}
				else
				{
					memcpy(color_buffer_data + (cur_y*width + cur_x) * 4, src_color, sizeof(float)* 4);
				}
			}

			return true;
		}

		static bool _generate_interpolate_vertices_after_clip_depth(const float* vertex1, const float* vertex2, const float* vertex3, int len, const ZQ_Vec3D* interpolate_pts, float**& interpolate_vertices, const VERTEX_FORMAT format)
		{
			ZQ_Vec3D pt1(vertex1[0], vertex1[1], vertex1[2]);
			ZQ_Vec3D pt2(vertex2[0], vertex2[1], vertex2[2]);
			ZQ_Vec3D pt3(vertex3[0], vertex3[1], vertex3[2]);

			if (1e-16 > _area_of_triangle(pt1, pt2, pt3))
				return true;

			interpolate_vertices = new float*[len];
			memset(interpolate_vertices, 0, sizeof(float*)*len);

			int nChannels_Pos = 3;
			int nChannels_Color = 4;
			int nChannels_Tex = 2;

			for (int i = 0; i < len; i++)
			{
				ZQ_Vec3D cur_pt = interpolate_pts[i];
				float area3 = _area_of_triangle(pt1, pt2, cur_pt);
				float area1 = _area_of_triangle(pt2, pt3, cur_pt);
				float area2 = _area_of_triangle(pt3, pt1, cur_pt);
				float sum_area = area1 + area2 + area3;
				if (sum_area == 0)
					continue;

				float w1 = area1 / sum_area;
				float w2 = area2 / sum_area;
				float w3 = area3 / sum_area;

				switch (format)
				{
				case VERTEX_POSITION3:

					interpolate_vertices[i] = new float[nChannels_Pos];

					interpolate_vertices[i][0] = cur_pt.x;
					interpolate_vertices[i][1] = cur_pt.y;
					interpolate_vertices[i][2] = cur_pt.z;

					break;
				case VERTEX_POSITION3_COLOR4:
					interpolate_vertices[i] = new float[nChannels_Pos + nChannels_Color];
					interpolate_vertices[i][0] = cur_pt.x;
					interpolate_vertices[i][1] = cur_pt.y;
					interpolate_vertices[i][2] = cur_pt.z;
					for (int cc = 0; cc < nChannels_Color; cc++)
					{
						interpolate_vertices[i][nChannels_Pos + cc] = vertex1[nChannels_Pos + cc] * w1 + vertex2[nChannels_Pos + cc] * w2 + vertex3[nChannels_Pos + cc] * w3;
					}

					break;
				case VERTEX_POSITION3_TEXCOORD2:
					interpolate_vertices[i] = new float[nChannels_Pos + nChannels_Tex];
					interpolate_vertices[i][0] = cur_pt.x;
					interpolate_vertices[i][1] = cur_pt.y;
					interpolate_vertices[i][2] = cur_pt.z;

					for (int cc = 0; cc < nChannels_Tex; cc++)
					{
						interpolate_vertices[i][nChannels_Pos + cc] = vertex1[nChannels_Pos + cc] * w1 + vertex2[nChannels_Pos + cc] * w2 + vertex3[nChannels_Pos + cc] * w3;
					}
					break;
				}
			}

			return true;
		}

		static bool _generate_interpolate_vertices_after_clip_boundary(const ZQ_Vec2D& pt1, const float* vertex1, const ZQ_Vec2D& pt2, const float* vertex2, const ZQ_Vec2D& pt3, const float* vertex3, int len, const ZQ_Vec2D* interpolate_pts, float**& interpolate_vertices, const VERTEX_FORMAT format)
		{
			float d1 = vertex1[2];
			float d2 = vertex2[2];
			float d3 = vertex3[2];
			if (d1 <= 0 || d2 <= 0 || d3 <= 0)
				return false;

			interpolate_vertices = new float*[len];
			memset(interpolate_vertices, 0, sizeof(float*)*len);

			int nChannels_Pos = 3;
			int nChannels_Color = 4;
			int nChannels_Tex = 2;

			for (int i = 0; i < len; i++)
			{
				ZQ_Vec2D cur_pt = interpolate_pts[i];
				float area3 = _area_of_triangle(pt1, pt2, cur_pt);
				float area1 = _area_of_triangle(pt2, pt3, cur_pt);
				float area2 = _area_of_triangle(pt3, pt1, cur_pt);
				float sum_area = area1 + area2 + area3;
				if (sum_area == 0)
				{
					sum_area = 1e-6;
				}
				

				float w1 = area1 / sum_area;
				float w2 = area2 / sum_area;
				float w3 = 1 - w1 - w2;

				float depth = 1.0 / (w1 / d1 + w2 / d2 + w3 / d3);

				switch (format)
				{
				case VERTEX_POSITION3:

					interpolate_vertices[i] = new float[nChannels_Pos];

					interpolate_vertices[i][0] = cur_pt.x;
					interpolate_vertices[i][1] = cur_pt.y;
					interpolate_vertices[i][2] = depth;

					break;
				case VERTEX_POSITION3_COLOR4:
					interpolate_vertices[i] = new float[nChannels_Pos + nChannels_Color];
					interpolate_vertices[i][0] = cur_pt.x;
					interpolate_vertices[i][1] = cur_pt.y;
					interpolate_vertices[i][2] = depth;
					for (int cc = 0; cc < nChannels_Color; cc++)
					{
						interpolate_vertices[i][nChannels_Pos + cc] = depth*(vertex1[nChannels_Pos + cc] / d1 * w1 + vertex2[nChannels_Pos + cc] / d2 * w2 + vertex3[nChannels_Pos + cc] / d3 * w3);
					}

					break;
				case VERTEX_POSITION3_TEXCOORD2:
					interpolate_vertices[i] = new float[nChannels_Pos + nChannels_Tex];
					interpolate_vertices[i][0] = cur_pt.x;
					interpolate_vertices[i][1] = cur_pt.y;
					interpolate_vertices[i][2] = depth;

					for (int cc = 0; cc < nChannels_Tex; cc++)
					{
						interpolate_vertices[i][nChannels_Pos + cc] = depth*(vertex1[nChannels_Pos + cc] / d1 * w1 + vertex2[nChannels_Pos + cc] / d2 * w2 + vertex3[nChannels_Pos + cc] / d3 * w3);
					}
					break;
				}
			}

			return true;
		}

		static void _release_interpolate_vertices(int len, float**& vertices)
		{
			if (vertices)
			{
				for (int i = 0; i < len; i++)
				{
					if (vertices[i])
						delete[]vertices[i];
				}
				delete[]vertices;
				vertices = 0;
			}
		}

		static bool _split_triangle_and_render_in_window(ZQ_DImage<float>& color_buffer, ZQ_DImage<float>& depth_buffer, const RenderOptions& opt, 
			const ZQ_Vec2D& pt1, const float* vertex1, const ZQ_Vec2D& pt2, const float* vertex2, const ZQ_Vec2D& pt3, const float* vertex3, const VERTEX_FORMAT format)
		{
			const int max_tri_num_after_clip_boundary = 8;
			ZQ_Vec2D tmp_pts1[max_tri_num_after_clip_boundary] = { pt1, pt2, pt3 };
			ZQ_Vec2D tmp_pts2[max_tri_num_after_clip_boundary];
			int len1 = 3, len2;
			_Sutherland_Hodgeman_clip_boundary(tmp_pts1, tmp_pts2, len1, len2, CLIPMODE_BOTTOM, opt.width, opt.height);
			_Sutherland_Hodgeman_clip_boundary(tmp_pts2, tmp_pts1, len2, len1, CLIPMODE_RIGHT, opt.width, opt.height);
			_Sutherland_Hodgeman_clip_boundary(tmp_pts1, tmp_pts2, len1, len2, CLIPMODE_TOP, opt.width, opt.height);
			_Sutherland_Hodgeman_clip_boundary(tmp_pts2, tmp_pts1, len2, len1, CLIPMODE_LEFT, opt.width, opt.height);
			
			if (len1 < 3)
				return true;

			float** tmp_vertices1 = 0;
			if (!_generate_interpolate_vertices_after_clip_boundary(pt1, vertex1, pt2, vertex2, pt3, vertex3, len1, tmp_pts1, tmp_vertices1, format))
			{
				return false;
			}

			for (int i = 0; i < len1 - 2; i++)
			{
				if (!_render_triangle_whole_in_window(color_buffer, depth_buffer, opt, tmp_pts1[0], tmp_vertices1[0], tmp_pts1[i + 1], tmp_vertices1[i + 1], tmp_pts1[i + 2], tmp_vertices1[i + 2], format))
				{
					_release_interpolate_vertices(len1, tmp_vertices1);
					return false;
				}
			}
			_release_interpolate_vertices(len1, tmp_vertices1);
			return true;
		}


		static void _Sutherland_Hodgeman_clip_depth(const ZQ_Vec3D* in_vertices, ZQ_Vec3D* out_vertices,  const int& in_len, int& out_len, ClipMode mode, float depth_clip_near, float depth_clip_far)
		{
			out_len = 0;
			switch (mode)
			{
			case CLIPMODE_NEAR: case CLIPMODE_FAR:
				ZQ_Vec3D s = in_vertices[in_len - 1];
				for (int j = 0; j < in_len; j++)
				{
					ZQ_Vec3D p = in_vertices[j];
					if (_is_inside_clip_depth(p,mode,depth_clip_near,depth_clip_far))
					{
						if (_is_inside_clip_depth(s, mode, depth_clip_near, depth_clip_far))
						{
							out_vertices[out_len++] = p;
						}
						else
						{
							ZQ_Vec3D cross_pt;
							_intersect_with_clip_depth(s, p, mode, cross_pt, depth_clip_near, depth_clip_far);
							out_vertices[out_len++] = cross_pt;
							out_vertices[out_len++] = p;
						}
					}
					else if (_is_inside_clip_depth(s, mode, depth_clip_near, depth_clip_far))
					{
						ZQ_Vec3D cross_pt;
						_intersect_with_clip_depth(s, p, mode, cross_pt, depth_clip_near, depth_clip_far);
						out_vertices[out_len++] = cross_pt;

					}
					s = p;
				}
				break;
			}
		}

		static void _Sutherland_Hodgeman_clip_boundary(const ZQ_Vec2D* in_vertices, ZQ_Vec2D* out_vertices, const int& in_len, int& out_len, ClipMode mode, int width, int height)
		{
			out_len = 0;
			switch (mode)
			{
			case CLIPMODE_LEFT: case CLIPMODE_RIGHT:case CLIPMODE_BOTTOM:case CLIPMODE_TOP:
				ZQ_Vec2D s = in_vertices[in_len - 1];
				for (int j = 0; j < in_len; j++)
				{
					ZQ_Vec2D p = in_vertices[j];
					if (_is_inside_clip_boundary(p, mode, width, height))
					{
						if (_is_inside_clip_boundary(s, mode, width, height))
						{
							out_vertices[out_len++] = p;
						}
						else
						{
							ZQ_Vec2D cross_pt;
							_intersect_with_clip_boundary(s, p, mode, cross_pt, width, height);
							out_vertices[out_len++] = cross_pt;
							out_vertices[out_len++] = p;
						}
					}
					else if (_is_inside_clip_boundary(s, mode, width, height))
					{
						ZQ_Vec2D cross_pt;
						_intersect_with_clip_boundary(s, p, mode, cross_pt, width, height);
						out_vertices[out_len++] = cross_pt;

					}
					s = p;
				}
				break;
			}
		}

		static bool _is_inside_clip_depth(const ZQ_Vec3D& p, ClipMode mode, float depth_clip_near, float depth_clip_far)
		{
			switch (mode)
			{
			case CLIPMODE_NEAR:
				return p.z >= depth_clip_near;
				break;
			case CLIPMODE_FAR:
				return p.z <= depth_clip_far;
				break;
			}
			return false;
		}

		static bool _is_inside_clip_boundary(const ZQ_Vec2D& p, ClipMode mode, int width, int height)
		{
			switch (mode)
			{
			case CLIPMODE_LEFT:
				return p.x >= 0;
				break;
			case CLIPMODE_RIGHT:
				return p.x <= width-1;
				break;
			case CLIPMODE_BOTTOM:
				return p.y >= 0;
				break;
			case CLIPMODE_TOP:
				return p.y <= height - 1;
				break;
			}
			return false;
		}
		
		static bool _intersect_with_clip_depth(const ZQ_Vec3D& s, const ZQ_Vec3D& p, ClipMode mode, ZQ_Vec3D& cross_pt, float depth_clip_near, float depth_clip_far)
		{
			switch (mode)
			{
			case CLIPMODE_NEAR:
				if (p.z == s.z)
					return false;
				cross_pt.z = depth_clip_near;
				cross_pt.x = s.x + (p.x - s.x)*(depth_clip_near - s.z) / (p.z - s.z);
				cross_pt.y = s.y + (p.y - s.y)*(depth_clip_near - s.z) / (p.z - s.z);
				return true;
				break;
			case CLIPMODE_FAR:
				if (p.z == s.z)
					return false;
				cross_pt.z = depth_clip_far;
				cross_pt.x = s.x + (p.x - s.x)*(depth_clip_near - s.z) / (p.z - s.z);
				cross_pt.y = s.y + (p.y - s.y)*(depth_clip_near - s.z) / (p.z - s.z);
				return true;
				break;
			}
			return false;
		}

		static bool _intersect_with_clip_boundary(const ZQ_Vec2D& s, const ZQ_Vec2D& p, ClipMode mode, ZQ_Vec2D& cross_pt, int width, int height)
		{
			switch (mode)
			{
			case CLIPMODE_LEFT:
				if (p.x == s.x)
					return false;
				cross_pt.x = 0;
				cross_pt.y = s.y + (p.y - s.y)*(0 - s.x) / (p.x - s.x);
				return true;
				break;
			case CLIPMODE_RIGHT:
				if (p.x == s.x)
					return false;
				cross_pt.x = width - 1;
				cross_pt.y = s.y + (p.y - s.y)*(width - 1 - s.x) / (p.x - s.x);
				return true;
				break;
			case CLIPMODE_BOTTOM:
				if (p.y == s.y)
					return false;
				cross_pt.y = 0;
				cross_pt.x = s.x + (p.x - s.x)*(0 - s.y) / (p.y - s.y);
				return true;
				break;
			case CLIPMODE_TOP:
				if (p.y == s.y)
					return false;
				cross_pt.y = height - 1;
				cross_pt.x = s.x + (p.x - s.x)*(height - 1 - s.y) / (p.y - s.y);
				return true;
				break;
			}
			return false;
		}
		
	};
}

#endif