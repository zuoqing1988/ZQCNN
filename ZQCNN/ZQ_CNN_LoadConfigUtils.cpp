#include "ZQ_CNN_LoadConfigUtils.h"

using namespace ZQ;

bool ZQ_CNN_LoadConfigUtils::Get_line(std::fstream& fin, const char*& buffer, __int64& buffer_len, std::string& line)
{
	if (buffer == 0)
	{
		if (!fin.is_open() || fin.eof())
			return false;
		std::getline(fin, line);
		return true;
	}
	else
	{
		if (buffer_len <= 0)
			return false;
		__int64 i = 0;
		for (; i < buffer_len; i++)
		{
			if (buffer[i] != '\n' && buffer[i] != '\r')
				break;
		}
		if (i == buffer_len)
			return false;

		__int64 j = i;
		for (; j < buffer_len; j++)
		{
			if (buffer[j] == '\n' || buffer[j] == '\r')
				break;
		}

		if (j == i)
			return false;
		__int64 cur_len = j - i;
		line.clear();
		line.append(buffer + i, cur_len);
		buffer += cur_len;
		buffer_len -= cur_len;
		return true;
	}
}

std::vector<std::vector<std::string> > ZQ_CNN_LoadConfigUtils::Split_line(const std::string& line)
{
	std::vector<std::string> first_splits = Split_c(line.c_str(), ':');
	int num = (int)first_splits.size();
	std::vector<std::vector<std::string> > second_splits(num);
	for (int n = 0; n < num; n++)
	{
		second_splits[n] = Split_c(first_splits[n].c_str(), ',');
	}
	return second_splits;
}

std::string ZQ_CNN_LoadConfigUtils::Remove_blank(const std::string& in)
{
	const char* ptr = in.c_str();
	std::vector<char> buf;
	for (int i = 0; i < strlen(ptr); i++)
	{
		if (ptr[i] != ' ' && ptr[i] != '\t')
			buf.push_back(ptr[i]);
	}
	buf.push_back('\0');
	return std::string(buf.data());
}

std::vector<std::string>  ZQ_CNN_LoadConfigUtils::Split_c(const char* str, char ch)
{
	std::vector<std::string> out;
	int len = (int)strlen(str);
	std::vector<char> buf(len + 1);
	int i = 0, j = 0;

	while (1)
	{
		for (i = j; i < len && str[i] != ch; i++);
		if (i >= len)
		{
			break;
		}
		int tmp_len = i - j;
		if (tmp_len > 0)
			memcpy(buf.data(), str + j, tmp_len * sizeof(char));
		buf[tmp_len] = '\0';
		out.push_back(std::string(buf.data()));
		j = i + 1;
	}
	int tmp_len = i - j;
	if (tmp_len > 0)
		memcpy(buf.data(), str + j, tmp_len * sizeof(char));
	buf[tmp_len] = '\0';
	out.push_back(std::string(buf.data()));
	return out;
}

int ZQ_CNN_LoadConfigUtils::My_strcmpi(const char* str1, const char* str2)
{
	char c1, c2;
	while (true)
	{
		c1 = tolower(*str1);
		c2 = tolower(*str2);
		str1++;
		str2++;
		if (c1 < c2)
			return -1;
		else if (c1 > c2)
			return 1;

		if (c1 == '\0')
			break;
	}
	return 0;
}