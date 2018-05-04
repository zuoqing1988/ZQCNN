#ifndef _ZQ_CNN_NET_H_
#define _ZQ_CNN_NET_H_
#pragma once
#include "ZQ_CNN_Defines.h"
#include "ZQ_CNN_Layer.h"
#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
namespace ZQ
{
	class ZQ_CNN_Net
	{
	public:
		ZQ_CNN_Net() :has_input_layer(false),show_debug_info(false), has_innerproduct_layer(false) {}
		~ZQ_CNN_Net() { _clear(); };

	private:
		std::vector<ZQ_CNN_Layer*> layers;	
		std::vector<ZQ_CNN_Tensor4D*> blobs; //blobs[0] stores a pointer to input blob
		std::map<std::string, int> map_name_to_layer_idx;
		std::map<std::string, int> map_name_to_blob_idx; 
		std::vector<std::vector<int>> bottoms;
		std::vector<std::vector<int>> tops;	//tops[0][0] stores input blob pointer
		std::string input_name;
		bool has_input_layer;
		bool show_debug_info;
		bool has_innerproduct_layer;
		int input_C, input_H, input_W;
	public:
		void TurnOnShowDebugInfo() { show_debug_info = true; }
		void TurnOffShowDebugInfo() { show_debug_info = false; }
		bool LoadFrom(const std::string& param_file, const std::string& model_file)
		{
			_clear();
			if (!_load_param_file(param_file))
			{
				_clear();
				return false;
			}
			if (!_check_connect())
			{
				_clear();
				return false;
			}
			if (!_load_model_file(model_file))
			{
				_clear();
				return false;
			}
			return true;
		}

		/*it may change input in case of padding, but the data will not be lost*/
		bool Forward(ZQ_CNN_Tensor4D& input)
		{
			if (map_name_to_blob_idx.size() == 0 || map_name_to_layer_idx.size() == 0 || tops.size() == 0)
				return false;
			if (has_innerproduct_layer)
			{
				if (input.GetH() != input_H || input.GetW() != input_W || input.GetC() != input_C)
				{
					std::cout << "The dimenson doesnot match with the needed\n";
					return false;
				}
			}
			blobs[0] = &input;
			
			for (int i = 0; i < layers.size(); i++)
			{
				std::vector<ZQ_CNN_Tensor4D*> bottom_ptrs, top_ptrs;
				for (int j = 0; j < bottoms[i].size(); j++)
					bottom_ptrs.push_back(blobs[bottoms[i][j]]);
				for (int j = 0; j < tops[i].size(); j++)
					top_ptrs.push_back(blobs[tops[i][j]]);
				layers[i]->show_debug_info = show_debug_info;
				if (!layers[i]->Forward(&bottom_ptrs, &top_ptrs))
				{
					blobs[0] = 0;
					tops[0][0] = 0;
					printf("failed to run layer: %s\n", layers[i]->name.c_str());
					return false;
				}
			}
			blobs[0] = 0;
			tops[0][0] = 0;
			return true;
		}

		const ZQ_CNN_Tensor4D* GetBlobByName(std::string name) 
		{
			std::map<std::string, int>::iterator it = map_name_to_blob_idx.find(name);
			if (it == map_name_to_blob_idx.end())
				return 0;
			else
			{
				return blobs[it->second];
			}
		}

	private:
		void _clear()
		{
			for (int i = 0; i < layers.size(); i++)
			{
				if (layers[i])
					delete layers[i];
			}
			layers.clear();
			map_name_to_layer_idx.clear();
			//blob[0] is a pointer to input blob, DONT FREE
			for (int i = 1; i < blobs.size(); i++)
			{
				if (blobs[i])
					delete blobs[i];
			}
			blobs.clear();
			map_name_to_layer_idx.clear();
			has_input_layer = false;
		}

		bool _load_param_file(const std::string& file)
		{
			std::fstream fin(file, std::ios::in);
			if (!fin.is_open())
			{
				std::cout << "failed to open file " << file << "\n";
				return false;
			}
			std::string line;
			int buf_len = 200;
			std::vector<char> buf(buf_len+1);
			while (std::getline(fin,line))
			{
				buf[0] = '\0';
				if (sscanf_s(line.c_str(), "%s", &buf[0], buf_len) == 0)
					continue;
				if (_strcmpi(&buf[0], "Convolution") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_Convolution();
					if (cur_layer == 0) {
						std::cout << "failed to create a Convolution layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line,false))
					{
						delete cur_layer;
						return false;
					}
				}
				else if (_strcmpi(&buf[0], "BatchNormScale") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_BatchNormScale();
					if (cur_layer == 0) {
						std::cout << "failed to create a BatchNorm layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line, false))
					{
						delete cur_layer;
						return false;
					}
				}
				else if (_strcmpi(&buf[0], "BatchNorm") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_BatchNorm();
					if (cur_layer == 0) {
						std::cout << "failed to create a BatchNorm layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line,false))
					{
						delete cur_layer;
						return false;
					}
				}
				else if (_strcmpi(&buf[0], "Scale") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_Scale();
					if (cur_layer == 0) {
						std::cout << "failed to create a Scale layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line,false))
					{
						delete cur_layer;
						return false;
					}
				}
				else if (_strcmpi(&buf[0], "PReLU") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_PReLU();
					if (cur_layer == 0) {
						std::cout << "failed to create a PReLU layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line,false))
					{
						delete cur_layer;
						return false;
					}
				}
				else if (_strcmpi(&buf[0], "ReLU") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_ReLU();
					if (cur_layer == 0) {
						std::cout << "failed to create a ReLU layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line, false))
					{
						delete cur_layer;
						return false;
					}
				}
				else if (_strcmpi(&buf[0], "Softmax") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_Softmax();
					if (cur_layer == 0) {
						std::cout << "failed to create a Softmax layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line,false))
					{
						delete cur_layer;
						return false;
					}
				}
				else if (_strcmpi(&buf[0], "Pooling") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_Pooling();
					if (cur_layer == 0) {
						std::cout << "failed to create a Pooling layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line,false))
					{
						delete cur_layer;
						return false;
					}
				}
				else if (_strcmpi(&buf[0], "Dropout") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_Dropout();
					if (cur_layer == 0) {
						std::cout << "failed to create a Dropout layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line,false))
					{
						delete cur_layer;
						return false;
					}
				}
				else if (_strcmpi(&buf[0], "InnerProduct") == 0)
				{
					has_innerproduct_layer = true;
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_InnerProduct();
					if (cur_layer == 0) {
						std::cout << "failed to create a InnerProduct layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line,false))
					{
						delete cur_layer;
						return false;
					}
				}
				else if (_strcmpi(&buf[0], "Eltwise") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_Eltwise();
					if (cur_layer == 0) {
						std::cout << "failed to create a Eltwise layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line, false))
					{
						delete cur_layer;
						return false;
					}
				}
				else if (_strcmpi(&buf[0], "LRN") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_LRN();
					if (cur_layer == 0) {
						std::cout << "failed to create a LRN layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line, false))
					{
						delete cur_layer;
						return false;
					}
				}
				else if (_strcmpi(&buf[0], "Input") == 0)
				{
					if (has_input_layer)
					{
						printf("Already have input layer\n");
						return false;
					}

					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_Input();
					if (cur_layer == 0) {
						std::cout << "failed to create a Input layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line,true))
					{
						delete cur_layer;
						return false;
					}
					cur_layer->GetTopDim(input_C, input_H, input_W);
				}
				line = "";
				
			}
			return true;
		}

		bool _load_model_file(const std::string& file)
		{
			int layer_num = layers.size();
			if (layers.size() == 0)
				return false;
			FILE* in = 0;
			fopen_s(&in, file.c_str(), "rb");
			if (in == 0)
			{
				std::cout << "failed to open " << file << "\n";
				return false;
			}
			for (int i = 0; i < layer_num; i++)
			{
				if (!layers[i]->LoadBinary_NCHW(in))
				{
					fclose(in);
					std::cout << "Failed to load Binary for layer " << layers[i]->name << "\n";
					return false;
				}
			}
			fclose(in);
			return true;
		}

		bool _add_layer_and_blobs(ZQ_CNN_Layer* cur_layer, const std::string& line, bool is_input_layer)
		{
			if (!cur_layer->ReadParam(line))
			{
				return false;
			}
			std::string layer_name = cur_layer->name;
			if (is_input_layer)
			{
				tops.resize(1);
				bottoms.resize(1);
				tops[0].push_back(0);
				map_name_to_blob_idx[cur_layer->top_names[0]] = 0;
				blobs.push_back(0);
				has_input_layer = true;
				layers.push_back(cur_layer);
				map_name_to_layer_idx[layer_name] = layers.size() - 1;
			}
			else
			{
				if (map_name_to_layer_idx.find(layer_name) == map_name_to_layer_idx.end())
				{
					std::vector<int> bottom_idx, top_idx;
					std::vector<std::string>& cur_bottom_names = cur_layer->bottom_names;
					for (int i = 0; i < cur_bottom_names.size(); i++)
					{
						std::map<std::string, int>::iterator name_it = map_name_to_blob_idx.find(cur_bottom_names[i]);
						if (name_it == map_name_to_blob_idx.end())
						{
							int idx = blobs.size();
							ZQ_CNN_Tensor4D* blob = new ZQ_CNN_Tensor4D_NHW_C_Align256bit();
							if (blob == 0)
							{
								std::cout << "failed to allocate a ZQ_CNN_Tensor4D\n";
								return false;
							}
							blobs.push_back(blob);
							bottom_idx.push_back(blobs.size()-1);
							map_name_to_blob_idx[cur_bottom_names[i]] = idx;
						}
						else
						{
							bottom_idx.push_back(name_it->second);
						}
					}
					std::vector<std::string>& cur_top_names = cur_layer->top_names;
					for (int i = 0; i < cur_top_names.size(); i++)
					{
						std::map<std::string, int>::iterator name_it = map_name_to_blob_idx.find(cur_top_names[i]);
						if (name_it == map_name_to_blob_idx.end())
						{
							int idx = blobs.size();
							ZQ_CNN_Tensor4D* blob = new ZQ_CNN_Tensor4D_NHW_C_Align256bit();
							if (blob == 0)
							{
								std::cout << "failed to allocate a ZQ_CNN_Tensor4D\n";
								return false;
							}
							blobs.push_back(blob);
							top_idx.push_back(blobs.size()-1);
							map_name_to_blob_idx[cur_top_names[i]] = idx;
						}
						else
						{
							top_idx.push_back(name_it->second);
						}
					}
					bottoms.push_back(bottom_idx);
					tops.push_back(top_idx);
					layers.push_back(cur_layer);
					map_name_to_layer_idx[layer_name] = layers.size() - 1;
				}
				else
				{
					std::cout << "There's already a layer named " << layer_name << "!\n";
					return false;
				}
			}
			return true;
		}

		bool _check_connect()
		{
			int blob_num = blobs.size();
			int layer_num = layers.size();
			if (layers.size() == 0 || blob_num == 0)
				return false;
			ZQ_CNN_Layer_Input* input_layer = (ZQ_CNN_Layer_Input*)(layers[0]);
			if (has_innerproduct_layer)
			{
				if (!input_layer->has_H_val || !input_layer->has_W_val)
				{
					std::cout << "Input dim must be specified for InnerProduct layer\n";
					return false;
				}
			}
			std::vector<bool> visited(blob_num);
			std::vector<int> blob_dim_C(blob_num);
			std::vector<int> blob_dim_H(blob_num);
			std::vector<int> blob_dim_W(blob_num);
			visited[0] = true;
			blob_dim_C[0] = input_layer->C;
			blob_dim_H[0] = input_layer->H;
			blob_dim_W[0] = input_layer->W;

			for (int i = 1; i < blob_num; i++)
				visited[i] = false;

			for (int i = 1; i < layer_num; i++)
			{
				std::vector<std::string>& bottom_names = layers[i]->bottom_names;
				int cur_bottom_c = 0;
				int cur_bottom_h = 0;
				int cur_bottom_w = 0;
				for (int j = 0; j < bottom_names.size(); j++)
				{
					std::map<std::string, int>::iterator name_it = map_name_to_blob_idx.find(bottom_names[j]);
					if (!visited[name_it->second])
					{
						std::cout << "unknown blob " << bottom_names[j] << " in Layer " << layers[i]->name << "\n";
						return false;
					}
					if (j == 0)
					{
						cur_bottom_c = blob_dim_C[name_it->second];
						cur_bottom_h = blob_dim_H[name_it->second];
						cur_bottom_w = blob_dim_W[name_it->second];
						layers[i]->SetBottomDim(cur_bottom_c,cur_bottom_h,cur_bottom_w);
					}
					else
					{
						if (blob_dim_C[name_it->second] != cur_bottom_c || blob_dim_H[name_it->second] != cur_bottom_h 
							|| blob_dim_W[name_it->second] != cur_bottom_w)
						{
							std::cout << "Dimension mismatch detected in layer " << layers[i]->name << "\n";
							return false;
						}
					}
				}
				std::vector<std::string>& top_names = layers[i]->top_names;
				for (int j = 0; j < top_names.size(); j++)
				{
					std::map<std::string, int>::iterator name_it = map_name_to_blob_idx.find(top_names[j]);
					visited[name_it->second] = true;
					layers[i]->GetTopDim(blob_dim_C[name_it->second], blob_dim_H[name_it->second], blob_dim_W[name_it->second]);
				}
			}
			return true;
		}
	};
}

#endif
