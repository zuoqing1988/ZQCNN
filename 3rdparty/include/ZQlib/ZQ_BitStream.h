/********************************/
/*you can use this class to read and write a bitstream.
/*the cuurent version don't support random write or read.
/* you can only write at the tail and read from the head.
/* be careful when using this.
/*
/*Zuo Qing, 2010.11.22-2010.11.25
/* modified by Zuo Qing, 2015-02-09
/* 
/********************************/

#ifndef _ZQ_BITSTREAM_H_
#define _ZQ_BITSTREAM_H_

#pragma once

namespace ZQ
{
	class ZQ_BitStream
	{
	public:
		/*be sure bytelen >= 1*/
		ZQ_BitStream(unsigned long bytelen) :maxlen(bytelen)
		{
			this->data = (unsigned char*)malloc(sizeof(unsigned char) * bytelen);
			this->index = 0;
			this->offset = 0;
			this->realBitlen = 0;
		}
		~ZQ_BitStream()
		{
			free(this->data);
			this->data = 0;
		}

	private:
		/*max storage maxlen is initial length*/
		unsigned long maxlen;

		/*it stores the bits*/
		unsigned char * data;

		/*byte offset*/
		unsigned long index;
		/*bit offset of data[index]*/
		unsigned long offset;

		/*the useful bit length. bits after this length is random.*/
		unsigned long realBitlen;

	public:
		/*let the pointer back to start position*/
		void BackToStart()
		{
			this->index = 0;
			this->offset = 0;
		}

		/*delete all the bits, and let all pointers to start position*/
		void Reset()
		{
			this->index = 0;
			this->offset = 0;
			this->realBitlen = 0;
		}

		/*get one bit, if return true, the pointers will go to the next position, if return false, pointers will keep*/
		bool GetBit(bool *value)
		{
			if (this->index >= this->maxlen)
				return false;

			unsigned char m_byte = this->data[this->index];
			m_byte >>= (7 - this->offset);
			m_byte &= 1;
			*value = (m_byte == 1);
			this->offset++;
			if (this->offset == 8)
			{
				this->offset = 0;
				this->index++;
			}
			return true;
		}

		/*get one byte(8-bit), if true, the pointer will go to the next position, if false, pointers will keep*/
		bool GetByte(unsigned char *value)
		{
			unsigned long oldIndex = this->index;
			unsigned long oldoffset = this->offset;
			int i = 8;
			unsigned char byteValue = 0;
			while (i > 0)
			{
				bool b;
				byteValue <<= 1;
				if (!GetBit(&b))
				{
					this->index = oldIndex;
					this->offset = oldoffset;
					return false;
				}
				byteValue += b ? 1 : 0;
				i--;
			}
			*value = byteValue;
			return true;
		}

		/*add one bit to the tail, it's dangerous to AddBit after calling BackToStart*/
		/*if return true, the poointers will go to next position, if return false, keep*/
		bool AddBit(bool value)
		{
			if (this->index >= maxlen)
				return false;

			int shift = 7 - this->offset;
			unsigned char m_byte = 255;
			m_byte <<= shift + 1;
			this->data[this->index] &= m_byte;
			m_byte = value ? 1 : 0;
			m_byte <<= shift;
			this->data[this->index] |= m_byte;

			this->offset++;
			if (this->offset >= 8)
			{
				this->offset = 0;
				this->index++;
			}
			this->realBitlen++;
			return true;
		}

		/*add one byte to the tail, it's dangerous to AddByte after calling BackToStart*/
		/*if return true, the poointers will go to next position, if return false, keep*/
		bool AddByte(unsigned char value)
		{
			unsigned long oldIndex = this->index;
			unsigned long oldoffset = this->offset;

			int i = 8;
			while (i > 0)
			{
				int bitshift = i - 1;
				bool b = (value >> bitshift) & 1;
				if (!AddBit(b))
				{
					this->index = oldIndex;
					this->offset = oldoffset;
					return false;
				}
				i--;
			}
			return true;
		}

		/*set data to the bit-stream,the pointers will be at the tail,so you can add more bits or bytes*/
		/*if you want to getbits from the strem, you should calling BackToStart before you GetBit*/
		/*the len is the length of bytes you want to set, and reallen is the real length you have set*/
		/*if return false, it will change the original bits or pointers.*/
		bool SetData(const unsigned char * data,unsigned long len,unsigned long* reallen)
		{
			if (data == 0)
				return false;
			if (len <= this->maxlen)
			{
				memcpy(this->data, data, len);
				*reallen = len;

			}
			else
			{
				memcpy(this->data, data, this->maxlen);
				*reallen = this->maxlen;
			}
			this->index = *reallen;
			this->offset = 0;
			this->realBitlen = 8 * this->maxlen;
			return true;
		}

		/*export the bit stream, bytelen is the byte length you have export and bitlen is the useful bit length*/
		bool ExportData(unsigned char ** data, unsigned long* bytelen, unsigned long* bitlen)
		{
			*bitlen = this->realBitlen;
			if ((*bitlen) % 8 == 0)
				*bytelen = (*bitlen) >> 3;
			else
				*bytelen = ((*bitlen) >> 3) + 1;

			*data = (unsigned char*)malloc(sizeof(unsigned char) * (*bytelen));
			memcpy(*data, this->data, *bytelen);
			return true;
		}

	};

}


#endif