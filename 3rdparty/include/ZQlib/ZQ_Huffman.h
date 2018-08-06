/*********************************************************/
/*it's a powerful class to do huffman code.
/*but you can only encode bytes, if you want encode non-bytes 
/*you have to changge them into bytes. if you want to encode 
/*N bits when N is not multiple of eight, I suggest you cut 
/*the bits into bytes and a several-bits addition.
/*
/*Zuo Qing, 2010.11.22-2010.11.25
/*modified by Zuo Qing, 2015-02-09
/*
/*********************************************************/
#ifndef _ZQ_HUFFMAN_H_
#define _ZQ_HUFFMAN_H_

#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "ZQ_BitStream.h"

#ifndef MAX_HUFFMAN_BYTES_LENGTH
#define MAX_HUFFMAN_BYTES_LENGTH 1024 * 1024 * 500
#endif

namespace ZQ
{
	/*node of the huffman Tree*/
	class ZQ_HuffmanNode
	{
	public:
		ZQ_HuffmanNode() :index(-1), freq(0), leftChild(0), rightChild(0){}
		~ZQ_HuffmanNode(){}

	public:
		/*index is useful when 0-255, is useless when index == -1*/
		int index;
		/*the freqency of the index*/
		unsigned long freq;

		/*bigger freq child*/
		ZQ_HuffmanNode* leftChild; 
		/*smaller freq child*/
		ZQ_HuffmanNode* rightChild;
	public:
		static void ReleaseNode(ZQ_HuffmanNode** node)
		{
			if (*node == 0)
				return;
			ZQ_HuffmanNode::ReleaseNode(&((*node)->leftChild));
			ZQ_HuffmanNode::ReleaseNode(&((*node)->rightChild));
			free(*node);
		}
	};

	/*you can use this class to encode a HuffmanCode to a bit-stream or decode a HuffmanCode from a bit-stream*/
	class ZQ_HuffmanCode
	{	
		friend class ZQ_Huffman;
		friend class ZQ_HuffmanEncoder;
		friend class ZQ_HuffmanDecoder;
	public:
		ZQ_HuffmanCode(){ memset(code, 0, sizeof(char*) * 256); }

		~ZQ_HuffmanCode(){ _clear(); }

	protected:
		char* code[256];

	public:

		/*decode from a bit-stream*/
		bool ImportFromBitStream(const unsigned char * bitStream, unsigned long byteLen)
		{
			_clear();
			if (bitStream == 0 || byteLen < 1)
				return false;
			ZQ_BitStream* m_bitstream = new ZQ_BitStream(byteLen);
			unsigned long reallen;
			m_bitstream->SetData(bitStream, byteLen, &reallen);
			unsigned char n, index, huffcodelen;
			unsigned long N;

			m_bitstream->BackToStart();
			bool bv;
			if (!m_bitstream->GetBit(&bv))
			{
				delete m_bitstream;
				_clear();
				return false;
			}
			if (!m_bitstream->GetByte(&n))
			{
				delete m_bitstream;
				_clear();
				return false;
			}
			if (bv)
			{
				N = n + 256;
			}
			else
			{
				N = n;
			}
			if (N > 256)
			{
				delete m_bitstream;
				_clear();
				return false;
			}
			for (int i = 0; i < N; i++)
			{
				if (!m_bitstream->GetByte(&index))
				{
					delete m_bitstream;
					_clear();
					return false;
				}
				if (!m_bitstream->GetByte(&huffcodelen))
				{
					delete m_bitstream;
					_clear();
					return false;
				}
				this->code[index] = (char*)malloc(sizeof(char) * (huffcodelen + 1));
				memset(this->code[index], 0, huffcodelen + 1);
				for (int l = 0; l < huffcodelen; l++)
				{
					bool b;
					if (!m_bitstream->GetBit(&b))
					{
						delete m_bitstream;
						_clear();
						return false;
					}
					(this->code[index])[l] = b ? '1' : '0';
				}

			}
			return true;
		}

		/*encode to a bit-stream*/
		bool ExportToBitStream(unsigned char ** bitStream,unsigned long* byteLen)
		{
			unsigned long m_bitlen = 0, m_bytelen = 0;
			unsigned long N = 0;
			m_bitlen += 8;//for the number of grayscales coded

			for (int i = 0; i < 256; i++)
			{
				if (this->code[i] != 0)
				{
					N++;
					m_bitlen += 8;//for grayscale
					m_bitlen += 8;//for its huffmancode bitlen

					/*if there is only one value, this->code[i] will be ""*/
					if (strlen(this->code[i]) != 0)
						m_bitlen += strlen(this->code[i]);
					else
						m_bitlen += 1;
				}
			}
			m_bytelen = m_bitlen / 8 + 1;
			ZQ_BitStream* m_bitstream = new ZQ_BitStream(m_bytelen);
			m_bitstream->Reset();
			if (N == 256)
				m_bitstream->AddBit(true);
			else
				m_bitstream->AddBit(false);
			m_bitstream->AddByte((unsigned char)N);
			for (int i = 0; i < 256; i++)
			{
				if (this->code[i] != 0)
				{
					m_bitstream->AddByte((unsigned char)i);
					int len = strlen(this->code[i]);

					/*if there is only one value, this->code[i] will be ""*/
					if (len != 0)
						m_bitstream->AddByte((unsigned char)len);
					else
						m_bitstream->AddByte((unsigned char)1);

					/*if there is only one value, this->code[i] will be ""*/
					if (len == 0)
					{
						m_bitstream->AddBit(false);
					}
					else
					{
						for (int l = 0; l < len; l++)
						{
							m_bitstream->AddBit((this->code[i])[l] != '0');
						}
					}
				}
			}
			unsigned long tmp;
			m_bitstream->ExportData(bitStream, byteLen, &tmp);
			delete m_bitstream;

			return true;
		}

		/*disply the huffmancode*/
		void Print()
		{
			printf("start!\n");
			for (int i = 0; i < 256; i++)
			{
				if (this->code[i] != 0)
				{
					printf("%3d:%s\n", i, this->code[i]);
				}
			}
			printf("end!\n");

		}


	private:
		/*clear all data*/
		void _clear()
		{
			for (int i = 0; i < 256; i++)
			{
				if (code[i] != 0)
				{
					free(code[i]);
					code[i] = 0;
				}
			}
		}
	};

	/*you can use this class to analysis a byte-sream data, dan generate a HuffmanCode*/
	class ZQ_Huffman
	{
	public:
		ZQ_Huffman() :length(0), data(0), huffmanTree(0)
		{
			for (int i = 0; i < 256; i++)
				hist[i] = 0;
		}
		~ZQ_Huffman(){ _destroyHuffmanTree(); }

	private:
		/*data length*/
		unsigned long length;

		/*data*/
		const unsigned char * data;

		/*hist or say freq*/
		unsigned long hist[256];

		/*huffman tree*/
		ZQ_HuffmanNode* huffmanTree;

	public:

		/*set data length*/
		void SetDataLength(unsigned long length){ this->length = length; }

		/*set data,not a copy, just a pointer*/
		void SetData(const unsigned char * data){ this->data = data; }

		/*generate huffman tree*/
		bool HuffmanEncode()
		{
			if (!_calHist())
				return false;

			if (!_buildHuffmanTree())
				return false;
			return true;
		}

		/*export huffmancode*/
		ZQ_HuffmanCode* ExportHuffmanCode()
		{
			ZQ_HuffmanCode* m_HuffCode = new ZQ_HuffmanCode();
			ZQ_HuffmanNode* nodeStack[256];
			int stackFlag[256];
			char bitStack[256] = { 0 };
			int top = -1;

			if (this->huffmanTree == 0)
				return 0;

			//post-order bi-tree
			ZQ_HuffmanNode* root = huffmanTree;
			nodeStack[++top] = root;
			stackFlag[top] = 0;
			bitStack[top] = '\0';
			while (top != -1)
			{
				root = nodeStack[top];
				if (root != 0)
				{
					if (stackFlag[top] == 0)
					{
						stackFlag[top] ++;
						bitStack[top] = '0';
						bitStack[top + 1] = '\0';
						root = root->leftChild;
						nodeStack[++top] = root;
						stackFlag[top] = 0;
						bitStack[top] = '\0';
					}
					else if (stackFlag[top] == 1)
					{
						stackFlag[top] ++;
						bitStack[top] = '1';
						bitStack[top + 1] = '\0';
						root = root->rightChild;
						nodeStack[++top] = root;
						stackFlag[top] = 0;
						bitStack[top] = '\0';
					}
					else
					{
						//visit root
						if (root->index != -1)
						{
							m_HuffCode->code[root->index] = (char*)malloc(sizeof(char)*(top + 1));
							strcpy(m_HuffCode->code[root->index], bitStack);
						}
						top--;
						if (top != -1)
							bitStack[top] = '\0';
					}
				}
				else
				{
					top--;
					if (top != -1)
						bitStack[top] = '\0';
				}
			}

			return m_HuffCode;
		}

	private:

		/*calculate hist*/
		bool _calHist()
		{
			if (data == 0)
				return false;
			_clearHist();
			for (unsigned long i = 0; i < length; i++)
				hist[(int)(data[i])] ++;
			return true;
		}

		/*clear hist*/
		void _clearHist()
		{
			for (int i = 0; i < 256; i++)
				hist[i] = 0;
		}

		/*build huffman tree*/
		bool _buildHuffmanTree()
		{
			_destroyHuffmanTree();
			int nodeNum = 0;
			ZQ_HuffmanNode * m_huffmanNode[256] = { 0 }, *tmpNode = 0;
			for (int i = 0; i < 256; i++)
			{
				if (this->hist[i] != 0)
				{
					m_huffmanNode[nodeNum] = new ZQ_HuffmanNode();
					m_huffmanNode[nodeNum]->freq = this->hist[i];
					m_huffmanNode[nodeNum]->index = i;
					nodeNum++;
				}
			}
			if (nodeNum == 0)
				return false;
			while (nodeNum > 1)
			{
				//sort nodes
				for (int i = 0; i < nodeNum - 1; i++)
				{
					for (int j = 0; j < nodeNum - 1; j++)
					{
						if (m_huffmanNode[j]->freq < m_huffmanNode[j + 1]->freq)
						{
							tmpNode = m_huffmanNode[j];
							m_huffmanNode[j] = m_huffmanNode[j + 1];
							m_huffmanNode[j + 1] = tmpNode;
						}
					}
				}

				//merge the last two
				tmpNode = new ZQ_HuffmanNode();
				tmpNode->freq = m_huffmanNode[nodeNum - 2]->freq + m_huffmanNode[nodeNum - 1]->freq;
				tmpNode->leftChild = m_huffmanNode[nodeNum - 2];
				tmpNode->rightChild = m_huffmanNode[nodeNum - 1];
				m_huffmanNode[nodeNum - 2] = tmpNode;
				m_huffmanNode[nodeNum - 1] = 0;
				nodeNum--;
			}
			huffmanTree = m_huffmanNode[0];
			return true;
		}

		/*destroy huffman tree*/
		void _destroyHuffmanTree(){ ZQ_HuffmanNode::ReleaseNode(&(this->huffmanTree)); }

	public:
		/*print hist*/
		void PrintHist()
		{
			for (int i = 0; i < 256; i++)
			{
				if (hist[i] != 0)
				{
					printf("%4d\t%10d\n", i, hist[i]);
				}
			}
		}
	};

	class ZQ_HuffmanEncoder
	{
	public:
		ZQ_HuffmanEncoder(ZQ_HuffmanCode& m_huffcode)
		{
			for (int i = 0; i < 256; i++)
			{
				this->code[i] = 0;
				this->len[i] = 0;
				if (m_huffcode.code[i] != 0)
				{
					this->len[i] = strlen(m_huffcode.code[i]);
					if (this->len[i] > 0)
					{
						this->code[i] = (char*)malloc(sizeof(char) * (this->len[i] + 1));
						strcpy(this->code[i], m_huffcode.code[i]);
					}
					else //there is only one value, and its code is empty,but we have to store it, so we use a '0' to represent it
					{
						this->len[i] = 1;
						this->code[i] = (char*)malloc(sizeof(char) * (this->len[i] + 1));
						strcpy(this->code[i], "0");
					}
				}
			}
		}

		~ZQ_HuffmanEncoder()
		{
			for (int i = 0; i < 256; i++)
			{
				if (this->code[i] != 0)
				{
					free(this->code[i]);
					this->code[i] = 0;
				}
			}
		}

	private:
		char* code[256];
		unsigned long len[256];

	public:
		/*out is of '0' and '1'*/
		bool EncodeOneByte(unsigned char value, char out[], unsigned long * out_len)
		{
			if (this->code[value] == 0)
				return false;
			strcpy(out, this->code[value]);
			*out_len = this->len[value];
			return true;
		}
	};

	class ZQ_HuffmanDecoder
	{
	public:
		ZQ_HuffmanDecoder(ZQ_HuffmanCode& m_huffcode)
		{
			this->huffmanTree = new ZQ_HuffmanNode();
			for (int i = 0; i < 256; i++)
			{
				if (m_huffcode.code[i] != 0)
				{
					int len = strlen(m_huffcode.code[i]);
					ZQ_HuffmanNode* root = this->huffmanTree;
					for (int j = 0; j < len; j++)
					{
						if ((m_huffcode.code[i])[j] == '0')
						{
							if (root->leftChild == 0)
							{
								root->leftChild = new ZQ_HuffmanNode();
							}
							root = root->leftChild;
						}
						else
						{
							if (root->rightChild == 0)
							{
								root->rightChild = new ZQ_HuffmanNode();
							}
							root = root->rightChild;
						}
					}
					root->index = i;
				}
			}
		}

		~ZQ_HuffmanDecoder(){ ZQ_HuffmanNode::ReleaseNode(&(this->huffmanTree)); }

	private:
		ZQ_HuffmanNode* huffmanTree;

	public:
		bool DecoderBitStream(const unsigned char* bitStream,unsigned long bytelen, unsigned long offsetbitlen, unsigned char** out, unsigned long* out_len)
		{
			ZQ_BitStream* m_bitstream = new ZQ_BitStream(bytelen);
			unsigned long reallen;
			m_bitstream->SetData(bitStream, bytelen, &reallen);
			unsigned char * m_byte = 0;
			if (bytelen * 8 * 8 * sizeof(unsigned char) >= MAX_HUFFMAN_BYTES_LENGTH)
				m_byte = (unsigned char*)malloc(MAX_HUFFMAN_BYTES_LENGTH);
			else
				m_byte = (unsigned char*)malloc(sizeof(unsigned char)*bytelen * 8 * 8);

			m_bitstream->BackToStart();
			unsigned long m_outlen = 0;
			unsigned long bitCount = 0;
			unsigned long maxbits = 8 * (bytelen - 1) + (offsetbitlen + 7) % 8 + 1;
			ZQ_HuffmanNode* root = this->huffmanTree;
			unsigned long i = 0;
			for (i = 0; i < maxbits; i++)
			{
				bool bv;
				if (!m_bitstream->GetBit(&bv))
				{
					delete m_bitstream;
					free(m_byte);
					return false;
				}
				if (bv == true)	/*bit 1*/
				{
					root = root->rightChild;
				}
				else			/*bit 0*/
				{
					root = root->leftChild;
				}

				/*when there is only one value, there will be a node whose leftchild or rightChild not both 0*/
				if (root->leftChild == 0 && root->rightChild == 0)
				{
					if (m_outlen >= bytelen * 8 * 8)
						printf("error!\n");
					m_byte[m_outlen++] = root->index;
					root = this->huffmanTree;
				}
			}
			if (i == maxbits && root != this->huffmanTree)
			{
				delete m_bitstream;
				free(m_byte);
				return false;
			}

			*out_len = m_outlen;
			*out = (unsigned char*)malloc(sizeof(unsigned char)*m_outlen);
			memcpy(*out, m_byte, m_outlen);

			delete m_bitstream;
			free(m_byte);
			return true;
		}
	};

	class ZQ_HuffmanEndec
	{
	public:
		/****************************************************************************************/
		/*a .zqhuff file is of the following format:
		/*huffcodeLen		:size(unsigned long) ,	the length of huffcode hearder
		/*huffcodebitStream	:size(unsigned char) * huffcodeLen,	 store the huffcode bitstream
		/*filenameLen		:size(unsigned long) ,	the length of filename including '\0'
		/*filename			:size(unsigned char) * filenameLen  ,	store the filename
		/*fileBytesLen		:size(unsigned long) , the length of filebits in bytes 
		/*offbit			:size(unsigned char) , the last useful bit of filebits offset(0-7)
		/*filebits			:size(unsigned char) * fileBytsLength ,		store the filebits
		/****************************************************************************************/

		/*encode a file*/
		static bool ZQ_HuffmanEncodeFile(const char infile[], const char outfile[])
		{
			FILE * in = fopen(infile, "rb");
			if (in == 0)
				return false;
			FILE * out = fopen(outfile, "wb");
			if (out == 0)
			{
				fclose(in);
				return false;
			}

			fseek(in, 0, SEEK_END);
			unsigned long inlen = ftell(in);
			unsigned long readlen = 0;
			unsigned char *m_bytes = (unsigned char*)malloc(sizeof(unsigned char)*inlen);
			rewind(in);
			if ((readlen = fread(m_bytes, sizeof(unsigned char), inlen, in)) != inlen)
			{
				printf("error!readlen = %ld,inlen = %ld\n", readlen, inlen);
			}

			/*close in*/
			fclose(in);

			ZQ_Huffman* m_huffman = new ZQ_Huffman();
			m_huffman->SetDataLength(inlen);
			m_huffman->SetData(m_bytes);
			if (!(m_huffman->HuffmanEncode()))
				printf("error!\n");
			ZQ_HuffmanCode* m_huffcode = m_huffman->ExportHuffmanCode();

			/*delete m_huffman*/
			delete m_huffman;
			m_huffman = 0;


			if (m_huffcode == 0)
			{
				free(m_bytes);
				fclose(out);
				remove(outfile);
				return false;
			}
			unsigned char *huffcodeBitStream = 0;
			unsigned long huffcodeLen = 0;
			m_huffcode->ExportToBitStream(&huffcodeBitStream, &huffcodeLen);

			/*write huffcodeLen*/
			fwrite(&huffcodeLen, sizeof(unsigned long), 1, out);

			/*write huffcode bitSteam*/
			fwrite(huffcodeBitStream, sizeof(unsigned char), huffcodeLen, out);

			/*free huffcode-bitStream*/
			free(huffcodeBitStream);

			int infileNameLen = strlen(infile) + 1;

			/*write filenameLen*/
			fwrite(&infileNameLen, sizeof(int), 1, out);

			/*write filename*/
			fwrite(infile, sizeof(char), infileNameLen, out);

			ZQ_HuffmanEncoder* m_encoder = new ZQ_HuffmanEncoder(*m_huffcode);

			/*delete m_huffcode*/
			delete m_huffcode;
			m_huffcode = 0;

			if (m_encoder == 0)
			{
				free(m_bytes);
				fclose(out);
				remove(outfile);
				return false;
			}
			ZQ_BitStream* m_bitstream = new ZQ_BitStream(inlen);
			for (int i = 0; i < inlen; i++)
			{
				char outbits[257];
				unsigned long outbitslen = 0;
				if (!(m_encoder->EncodeOneByte(m_bytes[i], outbits, &outbitslen)))
				{
					free(m_bytes);
					delete m_encoder;
					delete m_bitstream;
					fclose(out);
					remove(outfile);
					return false;
				}
				for (int j = 0; j < outbitslen; j++)
				{
					if (!(m_bitstream->AddBit(outbits[j] != '0')))
					{
						free(m_bytes);
						delete m_encoder;
						delete m_bitstream;
						fclose(out);
						remove(outfile);
						return false;
					}
				}
			}

			/*free m_bytes*/
			free(m_bytes);
			/*delete m_encoder*/
			delete m_encoder;
			m_encoder = 0;

			unsigned char * m_outBytes = 0;
			unsigned long outBytesLen = 0;
			unsigned long outBitsLen = 0;
			unsigned char offbit = 0;
			m_bitstream->ExportData(&m_outBytes, &outBytesLen, &outBitsLen);

			offbit = (unsigned char)((outBitsLen - (outBytesLen - 1) * 8) % 8);

			/*write fileBytesLen*/
			fwrite(&outBytesLen, sizeof(unsigned long), 1, out);

			/*write offbit*/
			fwrite(&offbit, sizeof(unsigned char), 1, out);

			/*write filebits*/
			fwrite(m_outBytes, sizeof(unsigned char), outBytesLen, out);


			/*free m_outBytes*/
			free(m_outBytes);

			/*delete m_bitstream*/
			delete m_bitstream;
			m_bitstream = 0;

			/*fclose out*/
			fclose(out);

			return true;
		}

		/*decode a file. orignialName is the compressed file's name before compressed*/
		static bool ZQ_HuffmanDecodeFile(const char infile[], const char outfile[],char originalName[])
		{
			FILE* in = fopen(infile, "rb");
			if (in == 0)
				return false;

			/*read huffcodeLen*/
			unsigned long huffcodeLen = 0;
			if (fread(&huffcodeLen, sizeof(unsigned long), 1, in) != 1)
			{
				fclose(in);
				return false;
			}

			/*read huffcodeBitStream*/
			unsigned char* huffcodeBitStream = (unsigned char*)malloc(sizeof(unsigned char)*huffcodeLen);
			if (fread(huffcodeBitStream, sizeof(unsigned char), huffcodeLen, in) != huffcodeLen)
			{
				fclose(in);
				free(huffcodeBitStream);
				return false;
			}


			/*read filenameLen*/
			unsigned long filenameLen = 0;
			if (fread(&filenameLen, sizeof(unsigned long), 1, in) != 1)
			{
				fclose(in);
				free(huffcodeBitStream);
				return false;
			}

			/*read filename*/
			char* filename = (char*)malloc(sizeof(char)*filenameLen);
			if (fread(filename, sizeof(char), filenameLen, in) != filenameLen)
			{
				fclose(in);
				free(huffcodeBitStream);
				free(filename);
				return false;
			}

			/*read fileBytesLen*/
			unsigned long fileBytesLen = 0;
			if (fread(&fileBytesLen, sizeof(unsigned long), 1, in) != 1)
			{
				fclose(in);
				free(huffcodeBitStream);
				free(filename);
				return false;
			}

			/*read offbit*/
			unsigned char offbit = 0;
			if (fread(&offbit, sizeof(unsigned char), 1, in) != 1)
			{
				fclose(in);
				free(huffcodeBitStream);
				free(filename);
				return false;
			}

			/*read filebits*/
			unsigned char * filebits = (unsigned char*)malloc(sizeof(unsigned char) * fileBytesLen);
			unsigned long readlen = 0;
			if ((readlen = fread(filebits, sizeof(unsigned char), fileBytesLen, in)) != fileBytesLen)
			{
				printf("error!readlen = %ld,fileBytesLen = %ld\n", readlen, fileBytesLen);
				fclose(in);
				free(huffcodeBitStream);
				free(filename);
				free(filebits);
				return false;
			}

			/*close in*/
			fclose(in);

			ZQ_HuffmanCode* m_huffcode = new ZQ_HuffmanCode();
			if (!(m_huffcode->ImportFromBitStream(huffcodeBitStream, huffcodeLen)))
			{
				free(huffcodeBitStream);
				free(filename);
				free(filebits);
				delete m_huffcode;
			}

			/*free huffcode bitStream*/
			free(huffcodeBitStream);

			ZQ_HuffmanDecoder* m_decoder = new ZQ_HuffmanDecoder(*m_huffcode);

			/*delete m_huffcode*/
			delete m_huffcode;
			m_huffcode = 0;

			unsigned char * outBytes = 0;
			unsigned long outLen = 0;
			if (!(m_decoder->DecoderBitStream(filebits, fileBytesLen, offbit, &outBytes, &outLen)))
			{
				free(filename);
				free(filebits);
				delete m_decoder;
			}

			/*free filebits*/
			free(filebits);

			/*delete m_decoder*/
			delete m_decoder;
			m_decoder = 0;

			if (filename[filenameLen - 1] != '\0')
			{
				free(filename);
				free(outBytes);
				return false;
			}
			strcpy(originalName, filename);

			FILE* out = fopen(outfile, "wb");
			if (out == 0)
			{
				free(filename);
				free(outBytes);
				return false;
			}

			if (fwrite(outBytes, sizeof(unsigned char), outLen, out) != outLen)
			{
				fclose(out);
				remove(filename);
				free(filename);
				free(outBytes);
				return false;
			}

			/*fclose out*/
			fclose(out);

			/*free filename*/
			free(filename);

			/*free outBytes*/
			free(outBytes);

			return true;
		}

		/****************************************************************************************/
		/*a zqhuff bytestream is of the following format:
		/*bytestreamLen		:size(unsigned long) ,  the length of the bytestream in byte
		/*huffcodeLen		:size(unsigned long) ,	the length of huffcode hearder
		/*huffcodebitStream	:size(unsigned char) * huffcodeLen,	 store the huffcode bitstream
		/*fileBytesLen		:size(unsigned long) , the length of filebits in bytes 
		/*offbit			:size(unsigned char) , the last useful bit of filebits offset(0-7)
		/*filebits			:size(unsigned char) * fileBytsLength ,		store the filebits
		/****************************************************************************************/

		/*encode a byte-stream, the output contain all huffcode informations*/
		static bool ZQ_HuffmanEncodeByteStream(const unsigned char* input, unsigned long inlen, unsigned char** output, unsigned long* outlen)
		{
			ZQ_Huffman* m_huff = new ZQ_Huffman();
			m_huff->SetDataLength(inlen);
			m_huff->SetData(input);
			if (!(m_huff->HuffmanEncode()))
			{
				delete m_huff;
				return false;
			}

			ZQ_HuffmanCode* m_huffcode = m_huff->ExportHuffmanCode();

			/*delete m_huff*/
			delete m_huff;
			m_huff = 0;

			unsigned char* huffcodeBytes = 0;
			unsigned long huffcodeBytelen = 0;
			if (!(m_huffcode->ExportToBitStream(&huffcodeBytes, &huffcodeBytelen)))
			{
				delete m_huffcode;
				return false;
			}

			ZQ_HuffmanEncoder* m_encoder = new ZQ_HuffmanEncoder(*m_huffcode);

			/*delete m_huffcode*/
			delete m_huffcode;
			m_huffcode = 0;

			ZQ_BitStream* m_bitstream = new ZQ_BitStream(inlen);
			char s[258] = { 0 };
			for (int i = 0; i < inlen; i++)
			{
				unsigned long len = 0;
				m_encoder->EncodeOneByte(input[i], s, &len);
				for (int j = 0; j < len; j++)
					m_bitstream->AddBit(s[j] != '0');

			}

			/*delete m_encoder*/
			delete m_encoder;
			m_encoder = 0;

			unsigned char* encodeBytes = 0;
			unsigned long encodeByteLen = 0;
			unsigned long encodeBitLen = 0;
			unsigned char encodeOffset = 0;

			if (!(m_bitstream->ExportData(&encodeBytes, &encodeByteLen, &encodeBitLen)))
			{
				delete m_bitstream;
				free(huffcodeBytes);
				return false;
			}
			encodeOffset = (encodeByteLen == 0) ? 0 : (unsigned char)((encodeBitLen - ((encodeByteLen - 1) << 3)) % 8);

			/*delete m_bitstream*/
			delete m_bitstream;
			m_bitstream = 0;

			/*output*/
			*outlen = sizeof(unsigned long) * 3 + sizeof(unsigned char) * (huffcodeBytelen + encodeByteLen + 1);
			*output = (unsigned char*)malloc(*outlen);

			unsigned long cur = 0;
			memcpy(*output, outlen, sizeof(unsigned long));
			cur += sizeof(unsigned long);
			memcpy((*output) + cur, &huffcodeBytelen, sizeof(unsigned long));
			cur += sizeof(unsigned long);
			memcpy((*output) + cur, huffcodeBytes, huffcodeBytelen);
			cur += huffcodeBytelen;
			memcpy((*output) + cur, &encodeByteLen, sizeof(unsigned long));
			cur += sizeof(unsigned long);
			memcpy((*output) + cur, &encodeOffset, 1);
			cur += 1;
			memcpy((*output) + cur, encodeBytes, encodeByteLen);

			free(huffcodeBytes);
			free(encodeBytes);
			return true;
		}

		/*decode a byte-stream, the input must contain all huffcode information*/
		static bool ZQ_HuffmanDecodeByteStream(const unsigned char* input, unsigned long inlen, unsigned char** output, unsigned long* outlen)
		{

			unsigned long wholeLen = 0;
			unsigned long cur = 0;
			memcpy(&wholeLen, input, sizeof(unsigned long));
			cur += sizeof(unsigned long);
			if (wholeLen != inlen)
				return false;

			unsigned long huffcodeLen = 0;
			memcpy(&huffcodeLen, input + cur, sizeof(unsigned long));
			cur += sizeof(unsigned long);

			ZQ_HuffmanCode* m_huffcode = new ZQ_HuffmanCode();
			if (!(m_huffcode->ImportFromBitStream(input + cur, huffcodeLen)))
			{
				delete m_huffcode;
				return false;
			}
			cur += huffcodeLen;
			ZQ_HuffmanDecoder* m_decoder = new ZQ_HuffmanDecoder(*m_huffcode);

			/*delete m_huffcode*/
			delete m_huffcode;
			m_huffcode = 0;

			unsigned long fileBytesLen = 0;
			unsigned char fileBitOffset = 0;
			memcpy(&fileBytesLen, input + cur, sizeof(unsigned long));
			cur += sizeof(unsigned long);
			memcpy(&fileBitOffset, input + cur, 1);
			cur += 1;

			if (!(m_decoder->DecoderBitStream(input + cur, fileBytesLen, (unsigned long)fileBitOffset, output, outlen)))
			{
				delete m_decoder;
				return false;
			}

			/*delete m_decoder*/
			delete m_decoder;
			m_decoder = 0;

			return true;
		}

	};
}


#endif