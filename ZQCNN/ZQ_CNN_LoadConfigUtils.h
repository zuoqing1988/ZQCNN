#ifndef _ZQ_CNN_LOAD_CONFIG_UTILS_H_
#define _ZQ_CNN_LOAD_CONFIG_UTILS_H_
#pragma once

#include <string>
#include <vector>
#include <fstream>
#include "ZQ_CNN_CompileConfig.h"

namespace ZQ
{
	class ZQ_CNN_LoadConfigUtils
	{
	public:
		static bool Get_line(std::fstream& fin, const char*& buffer, __int64& buffer_len, std::string& line);

		static std::vector<std::vector<std::string> > Split_line(const std::string& line);

		static std::string Remove_blank(const std::string& in);

		static std::vector<std::string>  Split_c(const char* str, char ch);

		static int My_strcmpi(const char* str1, const char* str2);
	};
}


#endif
