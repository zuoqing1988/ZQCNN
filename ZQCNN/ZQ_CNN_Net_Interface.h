#ifndef _ZQ_CNN_NET_INTERFACE_H_
#define _ZQ_CNN_NET_INTERFACE_H_
#pragma once
#include "ZQ_CNN_Tensor4D_Interface.h"
#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
namespace ZQ
{
	class ZQ_CNN_Net_Interface
	{
	public:
		ZQ_CNN_Net_Interface() {}
		virtual ~ZQ_CNN_Net_Interface() {}

	public:
		virtual void TurnOnShowDebugInfo() = 0;
		virtual void TurnOffShowDebugInfo() = 0;
		virtual void GetInputDim(int& in_C, int& in_H, int& in_W)const = 0;
		virtual bool LoadFrom(const std::string& param_file, const std::string& model_file, bool merge_bn = false, float ignore_small_value = 1e-12,
			bool merge_prelu = false) = 0;
		
		virtual bool LoadFromBuffer(const char*& param_buffer, __int64 param_buffer_len, const char*& model_buffer, __int64 model_buffer_len,
			bool merge_bn = false, float ignore_small_value = 1e-12, bool merge_prelu = false) = 0;
		
		virtual bool Forward(ZQ_CNN_Net_Interface* input) = 0;
		
		virtual const ZQ_CNN_Tensor4D* GetBlobByName(std::string name) = 0;

	};
}

#endif
