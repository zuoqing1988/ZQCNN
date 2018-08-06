#ifndef _ZQ_MERGE_IMAGE_H_
#define _ZQ_MERGE_IMAGE_H_
#pragma once

#include "ZQ_MergeImageOptions.h"
#include "ZQ_DoubleImage.h"
#include "ZQ_ImageIO.h"
#include "ZQ_CPURenderer2DWorkspace.h"
#include "ZQ_MathBase.h"

namespace ZQ
{
	class ZQ_MergeImage
	{
		typedef ZQ_DImage<float> DImage;
		typedef ZQ_Vec2D Vec2;
	public:
		static bool Go(const ZQ_MergeImageOptions& opt)
		{
			switch (opt.methodType)
			{
			case ZQ_MergeImageOptions::METHOD_MERGE_DIRECTLY:
			{
				return Go_MergeDirectly(opt);
			}
			break;
			case ZQ_MergeImageOptions::METHOD_MERGE_DENSITY:
			{
				return Go_MergeDensity(opt);
			}
			break;

			case ZQ_MergeImageOptions::METHOD_MERGE_SOURCE_PATCH:
			{
				return Go_MergeSourcePatch(opt);
			}
			break;

			case ZQ_MergeImageOptions::METHOD_IMAGE_BLUR:
			{
				return Go_ImageBlur(opt);
			}
			break;

			case ZQ_MergeImageOptions::METHOD_MERGE_LOW_HIGH:
			{
				return Go_MergeLowHigh(opt);
			}
			break;
			default:
				break;
			}

			return true;
		}
	private:
		static bool Go_MergeDirectly(const ZQ_MergeImageOptions& opt)
		{
			int num = opt.directSources.size();
			if (num == 0)
			{
				if (opt.display_running_info)
					printf("error: no merge sources\n");
				return false;
			}

			if (!opt.has_output_file)
			{
				if (opt.display_running_info)
					printf("error: no output file\n");
				return false;
			}

			std::vector<DImage> inputs(num);

			for (int i = 0; i < num; i++)
			{
				if (!_load(inputs[i], opt.directSources[i].file, 1))
				{
					if (opt.display_running_info)
					{
						printf("error: failed to load %s\n", opt.directSources[i].file);
					}
					return false;
				}
			}

			DImage output;

			if (!MergeDirectly(output, inputs, opt.display_running_info))
			{
				if (opt.display_running_info)
					printf("error: failed to merge density sources\n");
				return false;
			}

			if (!_save(output, opt.outputFile))
			{
				if (opt.display_running_info)
					printf("error: failed to save %s\n", opt.outputFile);
				return false;
			}

			return true;
		}

		static bool Go_MergeDensity(const ZQ_MergeImageOptions& opt)
		{
			const float m_pi = atan(1.0)*4.0;
			int num = opt.mergeSources.size();
			if (num == 0)
			{
				if (opt.display_running_info)
					printf("error: no merge sources\n");
				return false;
			}

			if (!opt.has_background_size)
			{
				if (opt.display_running_info)
					printf("error: no background size for merge\n");
				return false;
			}
			if (!opt.has_output_file)
			{
				if (opt.display_running_info)
					printf("error: no output file\n");
				return false;
			}

			std::vector<DImage> inputs(num);
			std::vector<Vec2> trans(num);
			std::vector<float> rots(num);
			std::vector<Vec2> target_size(num);


			for (int i = 0; i < num; i++)
			{
				if (!_load(inputs[i], opt.mergeSources[i].file, 1))
				{
					if (opt.display_running_info)
					{
						printf("error: failed to load %s\n", opt.mergeSources[i].file);
					}
					return false;
				}
				trans[i] = opt.mergeSources[i].trans;
				rots[i] = opt.mergeSources[i].rot_angle / 180.0 * m_pi;
				target_size[i] = opt.mergeSources[i].target_size;
			}

			int nChannels = inputs[0].nchannels();
			int width = opt.backgroundWidth;
			int height = opt.backgroundHeight;
			DImage output(width, height, nChannels);

			if (!MergeDensity(output, inputs, target_size, rots, trans, opt.merge_mode_blend, opt.yAxisUp, opt.display_running_info))
			{
				if (opt.display_running_info)
					printf("error: failed to merge density sources\n");
				return false;
			}

			if (!_save(output, opt.outputFile))
			{
				if (opt.display_running_info)
					printf("error: failed to save %s\n", opt.outputFile);
				return false;
			}
			return true;
		}

		static bool Go_MergeSourcePatch(const ZQ_MergeImageOptions& opt)
		{
			if (!opt.has_source_file)
			{
				if (opt.display_running_info)
					printf("error: no source file\n");
				return false;
			}
			if (!opt.has_patch_file)
			{
				if (opt.display_running_info)
					printf("error: no patch file\n");
				return false;
			}
			if (!opt.has_mask_file)
			{
				if (opt.display_running_info)
					printf("error: no mask file\n");
				return false;
			}
			if (!opt.has_output_file)
			{
				if (opt.display_running_info)
					printf("error: no output file\n");
				return false;
			}

			DImage source, patch, mask, output;
			if (!_load(source, opt.sourceFile, 1))
			{
				if (opt.display_running_info)
					printf("error: failed to load %s\n", opt.sourceFile);
				return false;
			}
			if (!_load(patch, opt.patchFile, 1))
			{
				if (opt.display_running_info)
					printf("error: failed to load %s\n", opt.patchFile);
				return false;
			}
			if (!_load(mask, opt.maskFile, 0))
			{
				if (opt.display_running_info)
					printf("error: failed to load %s\n", opt.maskFile);
				return false;
			}
			if (!MergeSourcePatch(output, source, patch, mask))
			{
				if (opt.display_running_info)
					printf("error: failed to merge source and patch\n");
				return false;
			}

			if (!_save(output, opt.outputFile))
			{
				if (opt.display_running_info)
					printf("error: failed to save %s\n", opt.outputFile);
				return false;
			}
			return true;
		}

		static bool Go_ImageBlur(const ZQ_MergeImageOptions& opt)
		{
			if (!opt.has_source_file)
			{
				if (opt.display_running_info)
					printf("error: no source file\n");
				return false;
			}
			if (!opt.has_output_file && !opt.has_high_part_file)
			{
				if (opt.display_running_info)
					printf("error: no output file and no high part file (need at least one of them)\n");
				return false;
			}

			DImage input, low, high;
			if (!_load(input, opt.sourceFile, 1))
			{
				if (opt.display_running_info)
					printf("error: failed to load %s\n", opt.sourceFile);
				return false;
			}
			DecomposeByBlur(low, high, input, opt.blur_sigma, opt.blur_fsize);

			if (opt.has_output_file)
			{
				if (!_save(low, opt.outputFile))
				{
					if (opt.display_running_info)
						printf("error: failed to save %s\n", opt.outputFile);
					return false;
				}
			}

			if (opt.has_high_part_file)
			{
				high.Addwith(0.5);
				if (!_save(high, opt.highPartFile))
				{
					if (opt.display_running_info)
						printf("error: failed to save %s\n", opt.highPartFile);
					return false;
				}
			}
			return true;
		}

		static bool Go_MergeLowHigh(const ZQ_MergeImageOptions& opt)
		{
			if (!opt.has_source_file)
			{
				if (opt.display_running_info)
					printf("error: no source file\n");
				return false;
			}
			if (!opt.has_high_part_file)
			{
				if (opt.display_running_info)
					printf("error: no high part file\n");
				return false;
			}
			if (!opt.has_output_file)
			{
				if (opt.display_running_info)
					printf("error: no output file\n");
				return false;
			}

			DImage low, high, output;
			if (!_load(low, opt.sourceFile, 1))
			{
				if (opt.display_running_info)
					printf("error: no source file\n");
				return false;
			}
			if (!_load(high, opt.highPartFile, 1))
			{
				if (opt.display_running_info)
					printf("error: no high part file\n");
				return false;
			}
			if (!low.matchDimension(high))
			{
				if (opt.display_running_info)
					printf("error: dimension don't match\n");
				return false;
			}
			high.Addwith(-0.5);
			output.Add(low, high);
			if (!_save(output, opt.outputFile))
			{
				if (opt.display_running_info)
					printf("failed to save %s\n", opt.outputFile);
				return false;
			}
			return true;
		}

	public:
		static void MakeMatrix(const Vec2& scale, const float rot_rad, const Vec2& trans, float output_mat[9])
		{
			float mat_scale[9] =
			{
				scale.x, 0, 0,
				0, scale.y, 0,
				0, 0, 1
			};

			float mat_rot[9] =
			{
				cos(rot_rad), -sin(rot_rad), 0,
				sin(rot_rad), cos(rot_rad), 0,
				0, 0, 1
			};

			float mat_trans[9] =
			{
				1, 0, trans.x,
				0, 1, trans.y,
				0, 0, 1
			};

			float mat_rot_scale[9];
			ZQ_MathBase::MatrixMul(mat_rot, mat_scale, 3, 3, 3, mat_rot_scale);
			ZQ_MathBase::MatrixMul(mat_trans, mat_rot_scale, 3, 3, 3, output_mat);
		}

	public:
		static bool MergeDirectly(DImage& output, const std::vector<DImage>& inputs,bool display)
		{
			int num = inputs.size();
			if (num == 0)
			{
				if (display)
					printf("no input\n");
				return false;
			}

			int width = inputs[0].width();
			int height = inputs[0].height();
			int nChannels = inputs[0].nchannels();

			for (int i = 1; i < inputs.size(); i++)
			{
				if (!inputs[i].matchDimension(width, height, nChannels))
				{
					if (display)
					{
						printf("dimensions don't match\n");
						return false;
					}
				}
			}

			output.allocate(width, height, nChannels);
			float*& output_ptr = output.data();
			for (int i = 0; i < inputs.size(); i++)
			{
				const float*& input_ptr = inputs[i].data();
				for (int pp = 0; pp < width*height*nChannels; pp++)
				{
					output_ptr[pp] = 1 - (1 - output_ptr[pp])*(1 - input_ptr[pp]);
				}
			}

			return true;
		}

		static bool MergeDensity(DImage& output, const std::vector<DImage>& inputs, const std::vector<Vec2>& target_size, const std::vector<float>& rots, const std::vector<Vec2>& trans, const bool blend_mode, const bool yAxisUp, const bool display)
		{
			int num = inputs.size();
			if (num != target_size.size() || num != rots.size() || num != trans.size())
				return false;

			output.reset();
			int back_width = output.width();
			int back_height = output.height();
			int nChannels = output.nchannels();
			ZQ_CPURenderer2DWorkspace renderer(back_width, back_height);
			renderer.SetClip(-100, 100);
			renderer.ClearDepthBuffer(100);
			renderer.ClearColorBuffer(0, 0, 0, 0);
			if (blend_mode)
			{
				renderer.DisableDepthTest();
				renderer.EnableAlphaBlend();
				renderer.SetAlphaBlendMode(ZQ_CPURenderer3DWorkspace::ALPHABLEND_SRC_ONE_DST_ONE_MINUS_SRC_EACHCHANNEL);
			}
			else
			{
				renderer.EnableDepthTest();
			}

			for (int i = 0; i < num; i++)
			{
				if (yAxisUp)
				{
					DImage tmp_img;
					inputs[i].FlipY(tmp_img);
					if (!_renderToBackground(renderer, tmp_img, target_size[i], rots[i], trans[i]))
						return false;
				}
				else
				{
					if (!_renderToBackground(renderer, inputs[i], target_size[i], rots[i], trans[i]))
						return false;
				}
			}

			const float*& buffer_data = renderer.GetColorBufferPtr();
			float*& output_data = output.data();
			for (int i = 0; i < back_width*back_height; i++)
			{
				for (int cc = 0; cc < nChannels && cc < 4; cc++)
					output_data[i*nChannels + cc] = buffer_data[i * 4 + cc];
			}

			if (yAxisUp)
				output.FlipY();

			return true;
		}

	private:
		static bool _renderToBackground(ZQ::ZQ_CPURenderer2DWorkspace& renderer, const DImage& source, const Vec2& target_size, const float rot_rad, const Vec2& trans)
		{
			int src_width = source.width();
			int src_height = source.height();
			int nChannels = source.nchannels();
			int back_width = renderer.GetBufferWidth();
			int back_height = renderer.GetBufferHeight();

			float half_src_width = target_size.x / 2.0;
			float half_src_height = target_size.y / 2.0;
			float half_back_width = back_width / 2.0;
			float half_back_height = back_height / 2.0;

			float corners[20] =
			{
				-half_src_width, -half_src_height, 0, 0, 0,
				half_src_width, -half_src_height, 0, 1, 0,
				half_src_width, half_src_height, 0, 1, 1,
				-half_src_width, half_src_height, 0, 0, 1
			};

			int indices[6] =
			{
				0, 1, 2,
				0, 2, 3
			};

			float mat_source_to_back[9];
			MakeMatrix(Vec2(1, 1), rot_rad, trans + Vec2(half_back_width, half_back_height), mat_source_to_back);


			for (int cor = 0; cor < 4; cor++)
			{
				float cur_coord[3] =
				{
					corners[cor * 5 + 0], corners[cor * 5 + 1], 1
				};
				float out_coord[3];
				ZQ_MathBase::MatrixMul(mat_source_to_back, cur_coord, 3, 3, 1, out_coord);
				corners[cor * 5 + 0] = out_coord[0];
				corners[cor * 5 + 1] = out_coord[1];
			}

			const float*& src_data = source.data();
			DImage tex_img(src_width, src_height, 4);
			float*& tex_data = tex_img.data();
			for (int i = 0; i < src_height; i++)
			{
				for (int j = 0; j < src_width; j++)
				{
					for (int cc = 0; cc < nChannels && cc < 4; cc++)
						tex_data[(i*src_width + j) * 4 + cc] = src_data[(i*src_width + j)*nChannels + cc];
				}
			}

			ZQ_TextureSampler<float> sampler;
			sampler.BindImage(source, false);
			renderer.BindSampler(&sampler);
			renderer.RenderIndexedTriangles(corners, indices, 4, 2, ZQ_CPURenderer3DWorkspace::VERTEX_POSITION3_TEXCOORD2);
			renderer.BindSampler(0);
			return true;
		}

	public:
		static bool MergeSourcePatch(DImage& output, const DImage& source, const DImage& patch, const DImage& mask)
		{
			int width = source.width();
			int height = source.height();
			int nChannels = source.nchannels();
			if (!patch.matchDimension(width, height, nChannels))
				return false;
			if (!mask.matchDimension(width, height, 1))
				return false;

			if (!output.matchDimension(width, height, nChannels))
				output.allocate(width, height, nChannels);

			const float*& mask_data = mask.data();
			const float*& source_data = source.data();
			const float*& patch_data = patch.data();
			float*& output_data = output.data();
			for (int i = 0; i < width*height; i++)
			{
				if (mask_data[i] < 0.5)
				{
					memcpy(output_data + i*nChannels, source_data + i*nChannels, sizeof(float)*nChannels);
				}
				else
				{
					memcpy(output_data + i*nChannels, patch_data + i*nChannels, sizeof(float)*nChannels);
				}
			}

			return true;
		}

		static void DecomposeByBlur(DImage& low, DImage& high, const DImage& input, const float sigma, const int fsize)
		{
			input.GaussianSmoothing(low, sigma, fsize);
			high.Subtract(input, low);
		}


	private: /* IO */
		static bool _load(DImage& img, const char* file, const int isColor)
		{
			const char* suffix_di2 = ".di2";
			int suffix_di2_len = strlen(suffix_di2);

			int filename_len = strlen(file);
			if (filename_len >= suffix_di2_len)
			{
				if (_strcmpi(file + filename_len - suffix_di2_len, suffix_di2) == 0)
					return img.loadImage(file);
			}

			return ZQ_ImageIO::loadImage(img, file, isColor);
		}

		static bool _save(const DImage&img, const char* file)
		{
			const char* suffix_di2 = ".di2";
			int suffix_di2_len = strlen(suffix_di2);

			int filename_len = strlen(file);
			if (filename_len >= suffix_di2_len)
			{
				if (_strcmpi(file + filename_len - suffix_di2_len, suffix_di2) == 0)
					return img.saveImage(file);
			}

			return ZQ_ImageIO::saveImage(img, file);
		}

	};
}

#endif
