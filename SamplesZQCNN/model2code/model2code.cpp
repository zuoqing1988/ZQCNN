#include <stdio.h>
#include <stdlib.h>
#ifndef __int64 
#define __int64 long long
#endif

int main(int argc, const char** argv)
{
	if (argc != 5)
	{
		printf("%s param_file model_file code_file prefix\n", argv[0]);
		return EXIT_FAILURE;
	}
	const char* param_file = argv[1];
	const char* model_file = argv[2];
	const char* code_file = argv[3];
	const char* prefix = argv[4];
	FILE* in1 = 0, *in2 = 0;
#if defined(_WIN32)
	if (0 != fopen_s(&in1, param_file, "r"))
#else
	if (0 ==(in1 = fopen(param_file, "r")))
#endif
	{
		printf("failed to open file %s\n", param_file);
		return EXIT_FAILURE;
	}
#if defined(_WIN32)
	if (0 != fopen_s(&in2, model_file, "rb"))
#else
	if (0 == (in2 = fopen(model_file, "rb")))
#endif
	{
		printf("failed to open file %s\n", model_file);
		fclose(in1);
		return EXIT_FAILURE;
	}

	FILE* out = 0;
#if defined(_WIN32)
	if (0 != fopen_s(&out, code_file, "w"))
#else
	if (0 == (out = fopen(code_file, "w")))
#endif
	{
		printf("failed to create file %s\n", code_file);
		fclose(in1);
		fclose(in2);
		return EXIT_FAILURE;
	}
#if defined(_WIN32)
	_fseeki64(in1, 0, SEEK_END);
	__int64 param_len = _ftelli64(in1);
	char* buffer = (char*)malloc(param_len);
	_fseeki64(in1, 0, SEEK_SET);
#else
	fseek(in1, 0, SEEK_END);
	__int64 param_len = ftell(in1);
	char* buffer = (char*)malloc(param_len);
	fseek(in1, 0, SEEK_SET);
#endif
	fread(buffer, 1, param_len, in1);
	fclose(in1);
	fprintf(out, "__int64 %s_param_len = %lld;\n", prefix, param_len);
	fprintf(out, "char %s_param[%lld] = {\n", prefix, param_len);
	for (__int64 i = 0; i < param_len; i++)
	{
		if (i == param_len-1)
		{
			fprintf(out, "%d};\n", buffer[i]);
		}
		else
		{
			fprintf(out, "%d,", buffer[i]);
		}
		
		if ((i+1) % 100 == 0)
			fprintf(out, "\n");
	}

#if defined(_WIN32)
	_fseeki64(in2, 0, SEEK_END);
	__int64 model_len = _ftelli64(in2);
	free(buffer);
	buffer = (char*)malloc(model_len);
	_fseeki64(in2, 0, SEEK_SET);
#else
	fseek(in2, 0, SEEK_END);
	__int64 model_len = ftell(in2);
	free(buffer);
	buffer = (char*)malloc(model_len);
	fseek(in2, 0, SEEK_SET);
#endif
	fread(buffer, 1, model_len, in2);
	fclose(in2);
	fprintf(out, "__int64 %s_model_len = %lld;\n", prefix, model_len);
	fprintf(out, "char %s_model[%lld] = {\n", prefix, model_len);
	for (__int64 i = 0; i < model_len; i++)
	{
		if (i == model_len - 1)
		{
			fprintf(out, "%d};\n", buffer[i]);
		}
		else
		{
			fprintf(out, "%d,", buffer[i]);
		}

		if ((i+1) % 100 == 0)
			fprintf(out, "\n");
	}
	fclose(out);

	return EXIT_SUCCESS;
}