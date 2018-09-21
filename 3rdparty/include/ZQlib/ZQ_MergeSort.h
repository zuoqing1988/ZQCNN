#ifndef _ZQ_MERGE_SORT_H_
#define _ZQ_MERGE_SORT_H_

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <iostream>

namespace ZQ
{
	class ZQ_MergeSort
	{
	public:
		template<class T>
		static void MergeSort(T* vals, __int64 num, bool ascending_dir);

		template<class T>
		static void MergeSort(T* vals, int* idx, __int64 num, bool ascending_dir);

		template<class T>
		static void MergeSortWithData(T* vals, void* data, int data_elt_size, __int64 num, bool ascending_dir);

		template<class T>
		static bool MergeSort_OOC(const char* src_val_file, const char* dst_val_file, bool ascending_dir,
			int max_mem_size_in_KB = 100);

		template<class T>
		static bool MergeSortWithData_OOC(const char* src_val_file, const char* dst_val_file,
			const char* src_data_file, const char* dst_data_file, int data_elt_size, bool ascending_dir,
			int max_mem_size_in_KB = 100);
		
	private:
		template<class T>
		static void _mergeSort(T* vals,__int64 start, __int64 end, bool ascending_dir);

		template<class T>
		static void _mergeSort(T* vals, int* idx, __int64 start, __int64 end, bool ascending_dir);

		template<class T>
		static void _mergeSortWithData(T* vals, void* data, int data_elt_size, __int64 start, __int64 end, bool ascending_dir);

		template<class T>
		static bool _mergeSort_OOC(FILE* in_val_file, FILE* tmp_val_files[2], FILE* in_data_file, FILE* tmp_data_files[2],
			__int64 num, int data_elt_size, __int64 block_size, int rest_iter, bool ascending_dir, bool has_data);

		class OOC_Buffer
		{
		public:
			char* val_buffer;
			int elt_size;
			__int64 buffer_size;
			__int64 offset;
			__int64 total_num;
			__int64 cur_idx;
			__int64 cur_total_off;
			__int64 cur_buffer_off;
			FILE* f_ptr;

		public:
			bool Bind(FILE* f_ptr, __int64 off, __int64 total_num, int elt_size, void* buffer, __int64 buffer_size)
			{
				this->f_ptr = f_ptr;
				this->offset = off;
				this->total_num = total_num;
				this->val_buffer = (char*)buffer;
				this->elt_size = elt_size;
				this->buffer_size = buffer_size / elt_size;
				cur_idx = 0;
				cur_buffer_off = 0;
				cur_total_off = 0;
				return true;
			}

			bool isEnd() const { return cur_total_off == total_num; }
			
			
			bool GetNextVal(void* val) 
			{
				if (!isEnd())
				{
					if (cur_total_off == 0)
					{
						_fseeki64(f_ptr, offset, SEEK_SET);
						__int64 need_read_num = __min(buffer_size, total_num - cur_idx);
						__int64 readed_num = fread(val_buffer,elt_size , need_read_num, f_ptr);
						if (need_read_num != readed_num)
						{
							return false;
						}
						cur_idx += readed_num;
					}
					

					memcpy(val, val_buffer + cur_buffer_off*elt_size, elt_size);
					cur_buffer_off++;
					cur_total_off++;

					if (cur_buffer_off == buffer_size)
					{
						if (cur_idx < total_num)
						{
							_fseeki64(f_ptr, offset + cur_idx * elt_size, SEEK_SET);
							__int64 need_read_num = __min(buffer_size, total_num - cur_idx);
							if (need_read_num > 0)
							{
								if (need_read_num != fread(val_buffer, elt_size, need_read_num, f_ptr))
									return false;
							}
							cur_buffer_off = 0;
							cur_idx += need_read_num;
						}
					}
					return true;
				}
				else
				{
					return false;
				}	
			}

			bool SetNextVal(void* val)
			{
				if (isEnd())
					return false;
				memcpy(val_buffer + cur_buffer_off*elt_size, val, elt_size);
				cur_buffer_off++;
				cur_total_off++;
				if (cur_idx+cur_buffer_off == total_num)
				{
					_fseeki64(f_ptr, offset + cur_idx * elt_size, SEEK_SET);
					if (cur_buffer_off != fwrite(val_buffer, elt_size, cur_buffer_off, f_ptr))
					{
						return false;
					}
					cur_idx = total_num;
				}
				else if(cur_buffer_off == buffer_size)
				{
					_fseeki64(f_ptr, offset + cur_idx * elt_size, SEEK_SET);
					if (buffer_size != fwrite(val_buffer, elt_size, buffer_size, f_ptr))
					{
						return false;
					}
					cur_idx += buffer_size;
					cur_buffer_off = 0;
				}
				return true;
			}
		};
		
	};
		
	/********************* definitions ***********************************/

	template<class T>
	void ZQ_MergeSort::MergeSort(T* vals, __int64 num, bool ascending_dir)
	{
		_mergeSort(vals,0,num-1,ascending_dir);
	}

	template<class T>
	void ZQ_MergeSort::MergeSort(T* vals, int* idx, __int64 num, bool ascending_dir)
	{
		_mergeSort(vals,idx,0,num-1,ascending_dir);
	}

	template<class T>
	void ZQ_MergeSort::MergeSortWithData(T* vals, void* data, int data_elt_size, __int64 num, bool ascending_dir)
	{
		_mergeSortWithData(vals, data, data_elt_size, 0, num - 1, ascending_dir);
	}

	template<class T>
	void ZQ_MergeSort::_mergeSort(T* vals,__int64 start, __int64 end, bool ascending_dir)
	{
		if(start >= end)
			return ;

		__int64 mid = (start+end)/2;
		__int64 left_len = mid-start+1;
		__int64 right_len = end-mid;
		T* tmp_left = new T[left_len];
		T* tmp_right = new T[right_len];
		for(__int64 i = 0;i < left_len;i++)
			tmp_left[i] = vals[start+i];
		for(__int64 i = 0;i < right_len;i++)
			tmp_right[i] = vals[mid+1+i];

		_mergeSort(tmp_left,0,left_len-1,ascending_dir);
		_mergeSort(tmp_right,0,right_len-1,ascending_dir);

		__int64 i_idx = 0;
		__int64 j_idx = 0;
		__int64 k_idx = 0;
		for(;i_idx < left_len && j_idx < right_len;)
		{
			if((tmp_left[i_idx] > tmp_right[j_idx]) != ascending_dir)
			{
				vals[start+k_idx] = tmp_left[i_idx];
				i_idx++;
				k_idx++;
			}
			else
			{
				vals[start+k_idx] = tmp_right[j_idx];
				j_idx++;
				k_idx++;
			}
		}
		if(i_idx == left_len)
		{
			for(;j_idx < right_len;j_idx++,k_idx++)
				vals[start+k_idx] = tmp_right[j_idx];
		}
		else
		{
			for(; i_idx < left_len;i_idx++,k_idx++)
				vals[start+k_idx] = tmp_left[i_idx];
		}
		delete []tmp_left;
		delete []tmp_right;
		return ;
	}

	template<class T>
	void ZQ_MergeSort::_mergeSort(T* vals, int* idx, __int64 start, __int64 end, bool ascending_dir)
	{
		if(start >= end)
			return ;

		__int64 mid = (start+end)/2;
		__int64 left_len = mid-start+1;
		__int64 right_len = end-mid;
		T* tmp_left = new T[left_len];
		T* tmp_right = new T[right_len];
		int* tmp_left_idx = new int[left_len];
		int* tmp_right_idx = new int[right_len];
		for(__int64 i = 0;i < left_len;i++)
		{
			tmp_left[i] = vals[start+i];
			tmp_left_idx[i] = idx[start+i];
		}
		for(__int64 i = 0;i < right_len;i++)
		{
			tmp_right[i] = vals[mid+1+i];
			tmp_right_idx[i] = idx[mid+1+i];
		}

		_mergeSort(tmp_left,tmp_left_idx,0,left_len-1,ascending_dir);
		_mergeSort(tmp_right,tmp_right_idx,0,right_len-1,ascending_dir);

		__int64 i_idx = 0;
		__int64 j_idx = 0;
		__int64 k_idx = 0;
		for(;i_idx < left_len && j_idx < right_len;)
		{
			if((tmp_left[i_idx] > tmp_right[j_idx]) != ascending_dir)
			{
				vals[start+k_idx] = tmp_left[i_idx];
				idx[start+k_idx] = tmp_left_idx[i_idx];
				i_idx++;
				k_idx++;
			}
			else
			{
				vals[start+k_idx] = tmp_right[j_idx];
				idx[start+k_idx] = tmp_right_idx[j_idx];
				j_idx++;
				k_idx++;
			}
		}
		if(i_idx == left_len)
		{
			for(;j_idx < right_len;j_idx++,k_idx++)
			{
				vals[start+k_idx] = tmp_right[j_idx];
				idx[start+k_idx] = tmp_right_idx[j_idx];
			}
		}
		else
		{
			for(; i_idx < left_len;i_idx++,k_idx++)
			{
				vals[start+k_idx] = tmp_left[i_idx];
				idx[start+k_idx] = tmp_left_idx[i_idx];
			}
		}
		delete []tmp_left;
		delete []tmp_right;
		delete []tmp_left_idx;
		delete []tmp_right_idx;
		return ;
	}

	template<class T>
	void ZQ_MergeSort::_mergeSortWithData(T* vals, void* data, int data_elt_size, __int64 start, __int64 end, bool ascending_dir)
	{
		if (start >= end)
			return;

		__int64 mid = (start + end) / 2;
		__int64 left_len = mid - start + 1;
		__int64 right_len = end - mid;
		T* tmp_left = new T[left_len];
		T* tmp_right = new T[right_len];
		void* tmp_data_left = malloc(data_elt_size*left_len);
		void* tmp_data_right = malloc(data_elt_size*right_len);
		memcpy(tmp_data_left, (char*)data + start*data_elt_size, data_elt_size*left_len);
		memcpy(tmp_data_right, (char*)data + (mid + 1)*data_elt_size, data_elt_size*right_len);
		for (__int64 i = 0; i < left_len; i++)
		{
			tmp_left[i] = vals[start + i];
		}
		for (__int64 i = 0; i < right_len; i++)
		{
			tmp_right[i] = vals[mid + 1 + i];
		}

		_mergeSortWithData(tmp_left, tmp_data_left, data_elt_size, 0, left_len - 1, ascending_dir);
		_mergeSortWithData(tmp_right, tmp_data_right, data_elt_size, 0, right_len - 1, ascending_dir);

		__int64 i_idx = 0;
		__int64 j_idx = 0;
		__int64 k_idx = 0;
		for (; i_idx < left_len && j_idx < right_len;)
		{
			if ((tmp_left[i_idx] > tmp_right[j_idx]) != ascending_dir)
			{
				vals[start + k_idx] = tmp_left[i_idx];
				memcpy((char*)data + (start + k_idx)*data_elt_size, (char*)tmp_data_left + i_idx*data_elt_size, data_elt_size);
				i_idx++;
				k_idx++;
			}
			else
			{
				vals[start + k_idx] = tmp_right[j_idx];
				memcpy((char*)data + (start + k_idx)*data_elt_size, (char*)tmp_data_right + j_idx*data_elt_size, data_elt_size);
				j_idx++;
				k_idx++;
			}
		}
		if (i_idx == left_len)
		{
			for (; j_idx < right_len; j_idx++, k_idx++)
			{
				vals[start + k_idx] = tmp_right[j_idx];
				memcpy((char*)data + (start + k_idx)*data_elt_size, (char*)tmp_data_right + j_idx*data_elt_size, data_elt_size);
			}
		}
		else
		{
			for (; i_idx < left_len; i_idx++, k_idx++)
			{
				vals[start + k_idx] = tmp_left[i_idx];
				memcpy((char*)data + (start + k_idx)*data_elt_size, (char*)tmp_data_left + i_idx*data_elt_size, data_elt_size);
			}
		}
		delete[]tmp_left;
		delete[]tmp_right;
		free(tmp_data_left);
		free(tmp_data_right);
		return;
	}

	template<class T>
	bool ZQ_MergeSort::MergeSort_OOC(const char* src_val_file, const char* dst_val_file, bool ascending_dir,
		int max_mem_size_in_KB)
	{
		FILE* in = 0;
		if (0 != fopen_s(&in, src_val_file, "rb"))
			return false;

		__int64 val_size = sizeof(T);
		_fseeki64(in, 0, SEEK_END);
		__int64 total_len = _ftelli64(in);
		if (total_len % val_size != 0 || total_len == 0)
		{
			fclose(in);
			return false;
		}

		__int64 num = total_len / val_size;
		_fseeki64(in, 0, SEEK_SET);


		max_mem_size_in_KB = __max(max_mem_size_in_KB, 1);
		__int64 max_block_size = (__int64)max_mem_size_in_KB * 1024 / (4 * val_size);
		__int64 block_size = 1;
		while (block_size * 2 <= max_block_size)
			block_size *= 2;

		int rest_iter = 0;
		__int64 tmp_block_size = block_size;
		while (tmp_block_size < num)
		{
			tmp_block_size *= 2;
			rest_iter++;
		}

		/* allocate file begin */
		std::string tmp_dst_file_name = std::string(dst_val_file) + ".tmp";
		const char* tmp_filename[2] = {
			dst_val_file,
			tmp_dst_file_name.c_str()
		};
		FILE* tmp_file[2] = { 0 };
		
		if (0 != fopen_s(&tmp_file[0], tmp_filename[0], "wb+"))
		{
			fclose(in);
			printf("failed to create file %s\n", tmp_filename[0]);
			return false;
		}
		if (0 != _fseeki64(tmp_file[0], total_len-1, SEEK_SET)
			|| EOF == fputc('\0', tmp_file[0]))
		{
			fclose(in);
			fclose(tmp_file[0]);
			printf("failed to allocate space for file %s\n", tmp_filename[0]);
			return false;
		}
		_fseeki64(tmp_file[0], 0, SEEK_SET);

		if (rest_iter > 0)
		{
			if (0 != fopen_s(&tmp_file[1], tmp_filename[1], "wb+"))
			{
				fclose(in);
				fclose(tmp_file[0]);
				printf("failed to create file %s\n", tmp_filename[1]);
				return false;
			}
			if (0 != _fseeki64(tmp_file[1], total_len - 1, SEEK_SET)
				|| EOF == fputc('\0', tmp_file[1]))
			{
				fclose(in);
				fclose(tmp_file[0]);
				fclose(tmp_file[1]);
				printf("failed to allocate space for file %s\n", tmp_filename[1]);
				return false;
			}
			_fseeki64(tmp_file[1], 0, SEEK_SET);
		}
		
		/* allocate file end  */

		FILE* in_data_file = 0;
		FILE* tmp_data_file[2] = { 0 };
		bool ret = _mergeSort_OOC<T>(in, tmp_file, in_data_file, tmp_data_file, num, val_size, block_size, rest_iter, 
			ascending_dir, false);

		fclose(in);
		if (tmp_file[0])
			fclose(tmp_file[0]);
		if (tmp_file[1])
			fclose(tmp_file[1]);
		if (tmp_data_file[0])
			fclose(tmp_data_file[0]);
		if (tmp_data_file[1])
			fclose(tmp_data_file[1]);
		return ret;
	}

	template<class T>
	bool ZQ_MergeSort::MergeSortWithData_OOC(const char* src_val_file, const char* dst_val_file,
		const char* src_data_file, const char* dst_data_file, int data_elt_size, bool ascending_dir,
		int max_mem_size_in_KB)
	{
		FILE* in = 0, *in_data = 0;
		if (0 != fopen_s(&in, src_val_file, "rb"))
			return false;

		__int64 val_size = sizeof(T);
		_fseeki64(in, 0, SEEK_END);
		__int64 total_len = _ftelli64(in);
		if (total_len % val_size != 0 || total_len == 0)
		{
			fclose(in);
			return false;
		}

		__int64 num = total_len / val_size;
		_fseeki64(in, 0, SEEK_SET);

		if (0 != fopen_s(&in_data, src_data_file, "rb"))
			return false;

		_fseeki64(in_data, 0, SEEK_END);
		if (num*data_elt_size != _ftelli64(in_data))
		{
			fclose(in);
			fclose(in_data);
			return false;
		}
		_fseeki64(in_data, 0, SEEK_SET);

		max_mem_size_in_KB = __max(max_mem_size_in_KB, 1);
		__int64 max_block_size = (__int64)max_mem_size_in_KB * 1024 / (4 * (val_size+data_elt_size));
		__int64 block_size = 1;
		while (block_size * 2 <= max_block_size)
			block_size *= 2;

		int rest_iter = 0;
		__int64 tmp_block_size = block_size;
		while (tmp_block_size < num)
		{
			tmp_block_size *= 2;
			rest_iter++;
		}

		/* allocate file begin */
		std::string tmp_dst_file_name = std::string(dst_val_file) + ".tmp";
		std::string tmp_dst_data_file_name = std::string(dst_data_file) + ".tmp";
		const char* tmp_filename[2] = {
			dst_val_file,
			tmp_dst_file_name.c_str()
		};
		const char* tmp_data_filename[2] = {
			dst_data_file,
			tmp_dst_data_file_name.c_str()
		};
		FILE* tmp_file[2] = { 0 };
		FILE* tmp_data_file[2] = { 0 };

		if (0 != fopen_s(&tmp_file[0], tmp_filename[0], "wb+"))
		{
			fclose(in);
			printf("failed to create file %s\n", tmp_filename[0]);
			return false;
		}
		if (0 != _fseeki64(tmp_file[0], total_len - 1, SEEK_SET)
			|| EOF == fputc('\0', tmp_file[0]))
		{
			fclose(in);
			fclose(tmp_file[0]);
			printf("failed to allocate space for file %s\n", tmp_filename[0]);
			return false;
		}
		_fseeki64(tmp_file[0], 0, SEEK_SET);

		if (0 != fopen_s(&tmp_data_file[0], tmp_data_filename[0], "wb+"))
		{
			fclose(in);
			fclose(tmp_file[0]);
			printf("failed to create file %s\n", tmp_data_filename[0]);
			return false;
		}
		if (0 != _fseeki64(tmp_data_file[0], num*data_elt_size - 1, SEEK_SET)
			|| EOF == fputc('\0', tmp_data_file[0]))
		{
			fclose(in);
			fclose(tmp_file[0]);
			fclose(tmp_data_file[0]);
			printf("failed to allocate space for file %s\n", tmp_data_filename[0]);
			return false;
		}
		_fseeki64(tmp_data_file[0], 0, SEEK_SET);

		if (rest_iter > 0)
		{
			if (0 != fopen_s(&tmp_file[1], tmp_filename[1], "wb+"))
			{
				fclose(in);
				fclose(tmp_file[0]);
				fclose(tmp_data_file[0]);
				printf("failed to create file %s\n", tmp_filename[1]);
				return false;
			}
			if (0 != _fseeki64(tmp_file[1], total_len - 1, SEEK_SET)
				|| EOF == fputc('\0', tmp_file[1]))
			{
				fclose(in);
				fclose(tmp_file[0]);
				fclose(tmp_data_file[0]);
				fclose(tmp_file[1]);
				printf("failed to allocate space for file %s\n", tmp_filename[1]);
				return false;
			}
			_fseeki64(tmp_file[1], 0, SEEK_SET);

			if (0 != fopen_s(&tmp_data_file[1], tmp_data_filename[1], "wb+"))
			{
				fclose(in);
				fclose(tmp_file[0]);
				fclose(tmp_data_file[0]);
				fclose(tmp_file[1]);
				printf("failed to create file %s\n", tmp_data_filename[1]);
				return false;
			}
			if (0 != _fseeki64(tmp_data_file[1], num*data_elt_size - 1, SEEK_SET)
				|| EOF == fputc('\0', tmp_data_file[1]))
			{
				fclose(in);
				fclose(tmp_file[0]);
				fclose(tmp_data_file[0]);
				fclose(tmp_file[1]);
				fclose(tmp_data_file[1]);
				printf("failed to allocate space for file %s\n", tmp_data_filename[1]);
				return false;
			}
			_fseeki64(tmp_data_file[1], 0, SEEK_SET);
		}

		/* allocate file end  */

		bool ret = _mergeSort_OOC<T>(in, tmp_file, in_data, tmp_data_file, num, data_elt_size, block_size, rest_iter,
			ascending_dir, true);

		fclose(in);
		fclose(in_data);
		if (tmp_file[0])
			fclose(tmp_file[0]);
		if (tmp_file[1])
			fclose(tmp_file[1]);
		if (tmp_data_file[0])
			fclose(tmp_data_file[0]);
		if (tmp_data_file[1])
			fclose(tmp_data_file[1]);
		return ret;
	}


	template<class T>
	bool ZQ_MergeSort::_mergeSort_OOC(FILE* in_val_file, FILE* tmp_val_files[2], FILE* in_data_file, FILE* tmp_data_files[2],
		__int64 num, int data_elt_size, __int64 block_size, int rest_iter, bool ascending_dir, bool has_data)
	{
		__int64 val_size = sizeof(T);
		__int64 tmp_block_size = block_size;

		int tmp_file_idx = rest_iter % 2 == 0 ? 0 : 1;

		/* in-core sort begin */
		_fseeki64(in_val_file, 0, SEEK_SET);
		__int64 rest = num % block_size;
		__int64 nBlock = num / block_size;
		T* val_block_buffer = (T*)malloc(val_size*block_size * 2);
		T* val_block_buffer1 = val_block_buffer;
		T* val_block_buffer2 = val_block_buffer + block_size;
		char* data_block_buffer = 0, *data_block_buffer1 = 0, *data_block_buffer2 = 0;
		if (has_data)
		{
			data_block_buffer = (char*)malloc(data_elt_size*block_size * 2);
			data_block_buffer1 = data_block_buffer;
			data_block_buffer2 = data_block_buffer + data_elt_size*block_size;
		}
		for (__int64 i = 0; i < nBlock; i++)
		{
			if (block_size != fread(val_block_buffer1, val_size, block_size, in_val_file))
			{
				free(val_block_buffer);
				if (has_data)free(data_block_buffer);
				return false;
			}
			if (has_data)
			{
				if (block_size != fread(data_block_buffer1, data_elt_size, block_size, in_data_file))
				{
					free(val_block_buffer);
					free(data_block_buffer);
					return false;
				}
			}
			if (has_data)
			{
				ZQ_MergeSort::MergeSortWithData(val_block_buffer1, data_block_buffer1, data_elt_size, block_size, ascending_dir);
			}
			else
			{
				ZQ_MergeSort::MergeSort(val_block_buffer1, block_size, ascending_dir);
			}
			if (block_size != fwrite(val_block_buffer1, val_size, block_size, tmp_val_files[tmp_file_idx]))
			{
				free(val_block_buffer);
				if (has_data)free(data_block_buffer);
				return false;
			}
			if (has_data)
			{
				if (block_size != fwrite(data_block_buffer1, data_elt_size, block_size, tmp_data_files[tmp_file_idx]))
				{
					free(val_block_buffer);
					free(data_block_buffer);
					return false;
				}
			}
		}

		if (rest != 0)
		{
			if (rest != fread(val_block_buffer1, val_size, rest, in_val_file))
			{
				free(val_block_buffer);
				if (has_data)free(data_block_buffer);
				return false;
			}
			if (has_data)
			{
				if (rest != fread(data_block_buffer1, data_elt_size, rest, in_data_file))
				{
					free(val_block_buffer);
					free(data_block_buffer);
					return false;
				}
			}
			if(has_data)
				ZQ_MergeSort::MergeSortWithData(val_block_buffer1, data_block_buffer1, data_elt_size, rest, ascending_dir);
			else
				ZQ_MergeSort::MergeSort(val_block_buffer1, rest, ascending_dir);

			if (rest != fwrite(val_block_buffer1, val_size, rest, tmp_val_files[tmp_file_idx]))
			{
				free(val_block_buffer);
				if(has_data)free(data_block_buffer);
				return false;
			}
			if (has_data)
			{
				if (rest != fwrite(data_block_buffer1, data_elt_size, rest, tmp_data_files[tmp_file_idx]))
				{
					free(val_block_buffer);
					free(data_block_buffer);
					return false;
				}
			}
		}
		fflush(tmp_val_files[tmp_file_idx]);
		_fseeki64(tmp_val_files[tmp_file_idx], 0, SEEK_SET);
		if (has_data)
		{
			fflush(tmp_data_files[tmp_file_idx]);
			_fseeki64(tmp_data_files[tmp_file_idx], 0, SEEK_SET);
		}
		
		/* in-core sort end */

		T* out_val_buffer = (T*)malloc(val_size*block_size * 2);
		char* out_data_buffer = 0;
		if (has_data)
		{
			out_data_buffer = (char*)malloc(data_elt_size*block_size * 2);
		}
		tmp_block_size = block_size;
		for(int rest_it = 0; rest_it < rest_iter; rest_it++)
		{
			printf("%d/%d %lld\n", rest_it+1, rest_iter, tmp_block_size);
			int other_file_idx = 1 - tmp_file_idx;
			//
			_fseeki64(tmp_val_files[tmp_file_idx], 0, SEEK_SET);
			_fseeki64(tmp_val_files[other_file_idx], 0, SEEK_SET);
			if (has_data)
			{
				_fseeki64(tmp_data_files[tmp_file_idx], 0, SEEK_SET);
				_fseeki64(tmp_data_files[other_file_idx], 0, SEEK_SET);
			}

			__int64 tmp_nBlock = (num + tmp_block_size - 1) / tmp_block_size;
			__int64 tmp_rest_nBlock = tmp_nBlock % 2;
			OOC_Buffer buffer1, buffer2, out_buffer;
			OOC_Buffer data_buffer1, data_buffer2, data_out_buffer;
			__int64 bb;
			T val1, val2;
			std::vector<char> data(data_elt_size);
			char* data_ptr = &data[0];
			for (bb = 0; bb < tmp_nBlock / 2; bb++)
			{
				__int64 num1 = tmp_block_size;
				__int64 num2 = __min(num - tmp_block_size*(bb * 2 + 1), tmp_block_size);
				__int64 out_num = num1 + num2;
				buffer1.Bind(tmp_val_files[tmp_file_idx], tmp_block_size*(bb * 2)*val_size, num1, val_size, val_block_buffer1, block_size*val_size);
				buffer2.Bind(tmp_val_files[tmp_file_idx], tmp_block_size*(bb * 2 + 1)*val_size, num2, val_size, val_block_buffer2, block_size*val_size);
				out_buffer.Bind(tmp_val_files[other_file_idx], tmp_block_size*bb * 2 * val_size, out_num, val_size, out_val_buffer, block_size * 2 * val_size);
				if (has_data)
				{
					data_buffer1.Bind(tmp_data_files[tmp_file_idx], tmp_block_size*(bb * 2)*data_elt_size, num1, data_elt_size, data_block_buffer1, block_size*data_elt_size);
					data_buffer2.Bind(tmp_data_files[tmp_file_idx], tmp_block_size*(bb * 2 + 1)*data_elt_size, num2, data_elt_size, data_block_buffer2, block_size*data_elt_size);
					data_out_buffer.Bind(tmp_data_files[other_file_idx], tmp_block_size*bb * 2 * data_elt_size, out_num, data_elt_size, out_data_buffer, block_size * 2 * data_elt_size);
				}
				if (!buffer1.GetNextVal(&val1))
				{
					return false;
				}
				if (!buffer2.GetNextVal(&val2))
				{
					return false;
				}
				
				__int64 count1 = 1;
				__int64 count2 = 1;
				while (true)
				{
					if (val1 > val2 != ascending_dir)
					{
						out_buffer.SetNextVal(&val1);
						if (has_data)
						{
							if (!data_buffer1.GetNextVal(data_ptr))
							{
								return false;
							}
							if (!data_out_buffer.SetNextVal(data_ptr))
							{
								return false;
							}
						}
						if (buffer1.isEnd())
						{
							if (has_data)
							{
								if (!data_buffer1.isEnd())
								{
									printf("error\n");
								}
							}
							if (buffer2.isEnd())
							{
								out_buffer.SetNextVal(&val2);
								if (has_data)
								{
									if (!data_buffer2.GetNextVal(data_ptr))
									{
										return false;
									}
									if (!data_out_buffer.SetNextVal(data_ptr))
									{
										return false;
									}
								}
							}
							break;
						}
						buffer1.GetNextVal(&val1);
						count1++;
					}
					else
					{
						out_buffer.SetNextVal(&val2);
						if (has_data)
						{
							if (!data_buffer2.GetNextVal(data_ptr))
							{
								return false;
							}
							if (!data_out_buffer.SetNextVal(data_ptr))
							{
								return false;
							}
						}
						if (buffer2.isEnd())
						{
							if (has_data)
							{
								if (!data_buffer2.isEnd())
								{
									printf("error\n");
								}
							}
							if (buffer1.isEnd())
							{
								out_buffer.SetNextVal(&val1);
								if (has_data)
								{
									if (!data_buffer1.GetNextVal(data_ptr))
									{
										return false;
									}
									if (!data_out_buffer.SetNextVal(data_ptr))
									{
										return false;
									}
								}
							}
							break;
						}
						buffer2.GetNextVal(&val2);
						count2++;
					}
					//printf("%lld %lld\n", count1, count2);
				}

				if (buffer1.isEnd())
				{
					if (!buffer2.isEnd())
					{
						out_buffer.SetNextVal(&val2);
						if (has_data)
						{
							if (!data_buffer2.GetNextVal(data_ptr))
							{
								return false;
							}
							if (!data_out_buffer.SetNextVal(data_ptr))
							{
								return false;
							}
						}
					}
					while (!buffer2.isEnd())
					{
						buffer2.GetNextVal(&val2);
						out_buffer.SetNextVal(&val2);
						if (has_data)
						{
							if (!data_buffer2.GetNextVal(data_ptr))
							{
								return false;
							}
							if (!data_out_buffer.SetNextVal(data_ptr))
							{
								return false;
							}
						}
					}
					//printf("%lld\n", out_buffer.cur_total_off);
				}
				else
				{
					if (!buffer1.isEnd())
					{
						out_buffer.SetNextVal(&val1);
						if (has_data)
						{
							if (!data_buffer1.GetNextVal(data_ptr))
							{
								return false;
							}
							if (!data_out_buffer.SetNextVal(data_ptr))
							{
								return false;
							}
						}
					}
					while (!buffer1.isEnd())
					{
						buffer1.GetNextVal(&val1);
						out_buffer.SetNextVal(&val1);
						if (has_data)
						{
							if (!data_buffer1.GetNextVal(data_ptr))
							{
								return false;
							}
							if (!data_out_buffer.SetNextVal(data_ptr))
							{
								return false;
							}
						}
					}
					//printf("%lld\n", out_buffer.cur_total_off);
				}
			}

			if (tmp_rest_nBlock == 1)
			{
				__int64 num1 = __min(num - tmp_block_size*bb * 2, tmp_block_size);
				__int64 out_num = num1;
				buffer1.Bind(tmp_val_files[tmp_file_idx], tmp_block_size*(bb * 2)*val_size, num1, val_size, val_block_buffer1, block_size*val_size);
				out_buffer.Bind(tmp_val_files[other_file_idx], tmp_block_size*bb * 2*val_size, out_num, val_size, out_val_buffer, block_size * 2 * val_size);
				if (has_data)
				{
					data_buffer1.Bind(tmp_data_files[tmp_file_idx], tmp_block_size*(bb * 2)*data_elt_size, num1, data_elt_size, data_block_buffer1, block_size*data_elt_size);
					data_out_buffer.Bind(tmp_data_files[other_file_idx], tmp_block_size*bb * 2*data_elt_size, out_num, data_elt_size, out_data_buffer, block_size * 2 * data_elt_size);
				}
				while (!buffer1.isEnd())
				{
					buffer1.GetNextVal(&val1);
					out_buffer.SetNextVal(&val1);
					if (has_data)
					{
						if (!data_buffer1.GetNextVal(data_ptr))
						{
							return false;
						}
						if (!data_out_buffer.SetNextVal(data_ptr))
						{
							return false;
						}
					}
				}
			}

			fflush(tmp_val_files[tmp_file_idx]);
			if (has_data)
				fflush(tmp_data_files[tmp_file_idx]);
			//
			tmp_file_idx = 1 - tmp_file_idx;
			tmp_block_size *= 2;
		}

		free(val_block_buffer);
		free(out_val_buffer);
		if (has_data)
		{
			free(data_block_buffer);
			free(out_data_buffer);
		}
		return true;
	}
}

#endif