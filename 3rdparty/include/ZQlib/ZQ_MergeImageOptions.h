#ifndef _ZQ_MERGE_IMAGE_OPTIONS_H_
#define _ZQ_MERGE_IMAGE_OPTIONS_H_
#pragma once

#include <vector>
#include <string.h>
#include "ZQ_Vec2D.h"


namespace ZQ
{
	class ZQ_MergeImageOptions
	{
		typedef ZQ_Vec2D Vec2;
		enum CONST_VAL{FILENAME_LEN=200};
	public:
		enum MethodType
		{
			METHOD_MERGE_DIRECTLY,
			METHOD_MERGE_DENSITY,
			METHOD_MERGE_SOURCE_PATCH,
			METHOD_IMAGE_BLUR,
			METHOD_MERGE_LOW_HIGH
		};
		struct MergeSource
		{
			
			char file[FILENAME_LEN];
			Vec2 trans;
			float rot_angle;
			Vec2 target_size;
		};
		struct DirectSource
		{
			char file[FILENAME_LEN];
		};

	public:
		ZQ_MergeImageOptions(){Reset();}
		~ZQ_MergeImageOptions(){}

	public:
		MethodType methodType;
		std::vector<MergeSource> mergeSources;
		std::vector<DirectSource> directSources;
		bool has_background_size;
		int backgroundWidth;
		int backgroundHeight;
		bool has_source_file;
		char sourceFile[FILENAME_LEN];
		bool has_patch_file;
		char patchFile[FILENAME_LEN];
		bool has_mask_file;
		char maskFile[FILENAME_LEN];
		bool has_output_file;
		char outputFile[FILENAME_LEN];
		bool has_high_part_file;
		char highPartFile[FILENAME_LEN];
		bool merge_mode_blend;
		bool display_running_info;
		float blur_sigma;
		int blur_fsize;
		bool yAxisUp;

	public:

		void Reset()
		{
			methodType = METHOD_MERGE_DENSITY;
			mergeSources.clear();
			has_background_size = false;
			backgroundWidth = 0;
			backgroundHeight = 0;
			has_source_file = false;
			sourceFile[0] = '\0';
			has_patch_file = false;
			patchFile[0] = '\0';
			has_mask_file = false;
			maskFile[0] = '\0';
			has_output_file = false;
			outputFile[0] = '\0';
			has_high_part_file = false;
			highPartFile[0] = '\0';
			merge_mode_blend = false;
			display_running_info = false;
			blur_sigma = 2;
			blur_fsize = 2;
			yAxisUp = false;
		}

		bool HandleParas(const int argc, const char** argv)
		{
			for(int i = 0;i < argc;i++)
			{
				if(_strcmpi(argv[i],"methodtype") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					if(_strcmpi(argv[i],"MERGE_DIRECTLY") == 0)
						methodType = METHOD_MERGE_DIRECTLY;
					else if(_strcmpi(argv[i],"MERGE_DENSITY") == 0)
						methodType = METHOD_MERGE_DENSITY;
					else if(_strcmpi(argv[i],"MERGE_SOURCE_PATCH") == 0)
						methodType = METHOD_MERGE_SOURCE_PATCH;
					else if(_strcmpi(argv[i],"IMAGE_BLUR") == 0)
						methodType = METHOD_IMAGE_BLUR;
					else if(_strcmpi(argv[i],"MERGE_LOW_HIGH") == 0)
						methodType = METHOD_MERGE_LOW_HIGH;
					else
					{
						printf("unknown methodType: %s\n",argv[i]);
						return false;
					}
				}
				else if(_strcmpi(argv[i],"MergeSource") == 0)
				{
					MergeSource src;

					i++;
					if(i >= argc)
					{
						printf("the value of MergeSource: file ?\n");
						return false;
					}
					strcpy(src.file,argv[i]);

					i++;
					if(i >= argc)
					{
						printf("the value of MergeSource: trans_x ?\n");
						return false;
					}
					src.trans.x = atof(argv[i]);

					i++;
					if(i >= argc)
					{
						printf("the value of MergeSource: trans_y ?\n");
						return false;
					}
					src.trans.y = atof(argv[i]);

					i++;
					if(i >= argc)
					{
						printf("the value of MergeSource: rot_angle ?\n");
						return false;
					}
					src.rot_angle = atof(argv[i]);

					i++;
					if(i >= argc)
					{
						printf("the value of MergeSource: target_x ?\n");
						return false;
					}
					src.target_size.x = atof(argv[i]);

					i++;
					if(i >= argc)
					{
						printf("the value of MergeSource: target_y ?\n");
						return false;
					}
					src.target_size.y = atof(argv[i]);

					mergeSources.push_back(src);
				}
				else if(_strcmpi(argv[i],"DirectSource") == 0)
				{
					DirectSource src;

					i++;
					if(i >= argc)
					{
						printf("the value of DirectSource: file ?\n");
						return false;
					}
					strcpy(src.file,argv[i]);
					directSources.push_back(src);
				}
				else if(_strcmpi(argv[i],"BackgroundSize") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of BackgroundSize:x ?\n");
						return false;
					}
					backgroundWidth = atoi(argv[i]);

					i++;
					if(i >= argc)
					{
						printf("the value of BackgroundSize:y ?\n");
						return false;
					}
					backgroundHeight = atoi(argv[i]);

					has_background_size = true;
				}
				else if(_strcmpi(argv[i],"SourceFile") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					strcpy(sourceFile,argv[i]);
					has_source_file = true;
				}
				else if(_strcmpi(argv[i],"PatchFile") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					strcpy(patchFile,argv[i]);
					has_patch_file = true;
				}
				else if(_strcmpi(argv[i],"MaskFile") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					strcpy(maskFile,argv[i]);
					has_mask_file = true;
				}
				else if(_strcmpi(argv[i],"OutputFile") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					strcpy(outputFile,argv[i]);
					has_output_file = true;
				}
				else if(_strcmpi(argv[i],"HighPartFile") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					strcpy(highPartFile,argv[i]);
					has_high_part_file = true;
				}
				else if(_strcmpi(argv[i],"MergeModeBlend") == 0)
					merge_mode_blend = true;
				else if(_strcmpi(argv[i],"Display") == 0)
					display_running_info = true;	
				else if(_strcmpi(argv[i],"sigma") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s\n",argv[i-1]);
						return false;
					}
					blur_sigma = atof(argv[i]);
				}
				else if(_strcmpi(argv[i],"fsize") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s\n",argv[i-1]);
						return false;
					}
					blur_fsize = atoi(argv[i]);
				}
				else if(_strcmpi(argv[i],"AxisUp") == 0)
				{
					yAxisUp = true;
				}
				else if(_strcmpi(argv[i],"AxisDown") == 0)
				{
					yAxisUp = false;
				}
				else
				{
					printf("unknown para: %s\n",argv[i]);
					return false;
				}
			}
			return true;
		}
	};
}

#endif