
#ifndef _ZQ_OBJ_LOADER_H_
#define _ZQ_OBJ_LOADER_H_
#pragma once

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

/****************************************************************/
/* WARNING!!!: only simple obj files can be read.
/*
/*****************************************************************/
namespace ZQ
{

#define PARSER_MAXARGS 512

	enum SeparatorType
	{
		ST_DATA,        // is data
		ST_HARD,        // is a hard separator
		ST_SOFT,        // is a soft separator
		ST_EOS          // is a comment symbol, and everything past this character should be ignored
	};

	class ParserInterface
	{
	public:
		virtual bool ParseLine(int lineno,int argc,const char **argv, const int pase) =0;  // return TRUE to continue parsing, return FALSE to abort parsing process
		virtual ~ParserInterface() {}
	};

	class Parser
	{
	
	private:
		bool   mMyAlloc; // whether or not *I* allocated the buffer and am responsible for deleting it.
		char  *mData;  // ascii data to parse.
		int    mLen;   // length of data
		SeparatorType  mHard[256];
		char   mHardString[256*2];
		char   mQuoteChar;

	public:
		Parser()
		{
			Init();
		}
		~Parser()
		{

			if (mMyAlloc)
			{
				if (mData) delete[]mData;
			}
		}

		void Init()
		{
			mQuoteChar = 34;
			mData = 0;
			mLen = 0;
			mMyAlloc = false;
			for (int i = 0; i < 256; i++)
			{
				mHard[i] = ST_DATA;
				mHardString[i * 2] = i;
				mHardString[i * 2 + 1] = 0;
			}
			mHard[0] = ST_EOS;
			mHard[32] = ST_SOFT;
			mHard[9] = ST_SOFT;
			mHard[13] = ST_SOFT;
			mHard[10] = ST_SOFT;
		}

		bool SetFile(const char* fname)
		{
			if (mMyAlloc)
			{
				if (mData) delete[]mData;
			}

			mData = 0;
			mLen = 0;
			mMyAlloc = false;

			FILE* in = 0;
			if (0 != fopen_s(&in, fname, "r"))
				return false;


			fseek(in, 0L, SEEK_END);
			mLen = ftell(in);
			fseek(in, 0L, SEEK_SET);
			if (mLen)
			{
				mData = (char *)malloc(sizeof(char)*(mLen + 1));
				int ok = int(fread(mData, 1, mLen, in));
				if (!ok)
				{
					delete[]mData;
					mData = 0;
				}
				else
				{
					mData[mLen] = '\0';
					mMyAlloc = true;
				}
			}
			fclose(in);
			return true;
		}
		
		void SetSourceData(char *data, int len)
		{
			mData = data;
			mLen = len;
			mMyAlloc = false;
		}

		bool Parse(ParserInterface *callback, const int pass)
		{
			assert(callback);
			if (!mData) return false;

			int lineno = 0;

			char *foo = mData;
			char *begin = foo;

			while (*foo)
			{
				if (*foo == 10 || *foo == 13)
				{
					lineno++;
					*foo = 0;

					if (*begin) // if there is any data to parse at all...
					{
						if (!ProcessLine(lineno, begin, callback, pass))
							return false;
					}

					foo++;
					if (*foo == 10) foo++; // skip line feed, if it is in the carraige-return line-feed format...
					begin = foo;
				}
				else
				{
					foo++;
				}
			}

			lineno++; // last line.

			if (!ProcessLine(lineno, begin, callback, pass))
				return false;
			return true;
		}


		bool ProcessLine(int lineno, char *line, ParserInterface *callback, const int pass)
		{
			const char *argv[PARSER_MAXARGS];
			int argc = 0;

			char *foo = line;

			while (!EOS(*foo) && argc < PARSER_MAXARGS)
			{
				foo = _skipSpaces(foo); // skip any leading spaces

				if (EOS(*foo)) break;

				if (*foo == mQuoteChar) // if it is an open quote
				{
					foo++;
					if (argc < PARSER_MAXARGS)
					{
						argv[argc++] = foo;
					}
					while (!EOS(*foo) && *foo != mQuoteChar) foo++;
					if (!EOS(*foo))
					{
						*foo = 0; // replace close quote with zero byte EOS
						foo++;
					}
				}
				else
				{
					foo = _addHard(argc, argv, foo); // add any hard separators, skip any spaces

					if (_isNonSeparator(*foo))  // add non-hard argument.
					{
						bool quote = false;
						if (*foo == mQuoteChar)
						{
							foo++;
							quote = true;
						}

						if (argc < PARSER_MAXARGS)
						{
							argv[argc++] = foo;
						}

						if (quote)
						{
							while (*foo && *foo != mQuoteChar) foo++;
							if (*foo) *foo = 32;
						}

						// continue..until we hit an eos ..
						while (!EOS(*foo)) // until we hit EOS
						{
							if (_isWhiteSpace(*foo)) // if we hit a space, stomp a zero byte, and exit
							{
								*foo = 0;
								foo++;
								break;
							}
							else if (_isHard(*foo)) // if we hit a hard separator, stomp a zero byte and store the hard separator argument
							{
								const char *hard = &mHardString[*foo * 2];
								*foo = 0;
								if (argc < PARSER_MAXARGS)
								{
									argv[argc++] = hard;
								}
								foo++;
								break;
							}
							foo++;
						} // end of while loop...
					}
				}
			}

			if (argc)
			{
				return callback->ParseLine(lineno, argc, argv, pass);
			}

			return true;
		}
		
		const char ** GetArglist(char* line, int &count) // convert source string into an arg list, this is a destructive parse.
		{
			const char **ret = 0;

			static const char *argv[PARSER_MAXARGS];
			int argc = 0;

			char *foo = line;

			while (!EOS(*foo) && argc < PARSER_MAXARGS)
			{

				foo = _skipSpaces(foo); // skip any leading spaces

				if (EOS(*foo)) break;

				if (*foo == mQuoteChar) // if it is an open quote
				{
					foo++;
					if (argc < PARSER_MAXARGS)
					{
						argv[argc++] = foo;
					}
					while (!EOS(*foo) && *foo != mQuoteChar) foo++;
					if (!EOS(*foo))
					{
						*foo = 0; // replace close quote with zero byte EOS
						foo++;
					}
				}
				else
				{

					foo = _addHard(argc, argv, foo); // add any hard separators, skip any spaces

					if (_isNonSeparator(*foo))  // add non-hard argument.
					{
						bool quote = false;
						if (*foo == mQuoteChar)
						{
							foo++;
							quote = true;
						}

						if (argc < PARSER_MAXARGS)
						{
							argv[argc++] = foo;
						}

						if (quote)
						{
							while (*foo && *foo != mQuoteChar) foo++;
							if (*foo) *foo = 32;
						}

						// continue..until we hit an eos ..
						while (!EOS(*foo)) // until we hit EOS
						{
							if (_isWhiteSpace(*foo)) // if we hit a space, stomp a zero byte, and exit
							{
								*foo = 0;
								foo++;
								break;
							}
							else if (_isHard(*foo)) // if we hit a hard separator, stomp a zero byte and store the hard separator argument
							{
								const char *hard = &mHardString[*foo * 2];
								*foo = 0;
								if (argc < PARSER_MAXARGS)
								{
									argv[argc++] = hard;
								}
								foo++;
								break;
							}
							foo++;
						} // end of while loop...
					}
				}
			}

			count = argc;
			if (argc)
			{
				ret = argv;
			}

			return ret;
		}

		void SetHardSeparator(char c) // add a hard separator
		{
			mHard[(unsigned char)c] = ST_HARD;
		}

		void SetHard(char c) // add a hard separator
		{
			mHard[(unsigned char)c] = ST_HARD;
		}


		void SetCommentSymbol(char c) // comment character, treated as 'end of string'
		{
			mHard[(unsigned char)c] = ST_EOS;
		}

		void ClearHardSeparator(char c)
		{
			mHard[(unsigned char)c] = ST_DATA;
		}


		void DefaultSymbols()
		{
			SetHardSeparator(',');
			SetHardSeparator('(');
			SetHardSeparator(')');
			SetHardSeparator('=');
			SetHardSeparator('[');
			SetHardSeparator(']');
			SetHardSeparator('{');
			SetHardSeparator('}');
			SetCommentSymbol('#');
		}


		bool EOS(char c)
		{
			return mHard[(unsigned char)c] == ST_EOS;
		}

		void SetQuoteChar(char c)
		{
			mQuoteChar = c;
		}

		

	private:
		bool _isHard(char c)
		{
			return mHard[(unsigned char)c] == ST_HARD;
		}

		char * _addHard(int &argc, const char **argv, char *foo)
		{
			while (_isHard(*foo))
			{
				const char *hard = &mHardString[*foo * 2];
				if (argc < PARSER_MAXARGS)
				{
					argv[argc++] = hard;
				}
				foo++;
			}
			return foo;
		}

		bool _isWhiteSpace(char c)
		{
			return mHard[(unsigned char)c] == ST_SOFT;
		}

		char * _skipSpaces(char *foo)
		{
			while (!EOS(*foo) && _isWhiteSpace(*foo)) foo++;
			return foo;
		}

		bool _isNonSeparator(char c) // non seperator,neither hard nor soft
		{
			return !_isHard(c) && !_isWhiteSpace(c) && c != 0;
		}
	};


	class ZQ_ObjLoader : ParserInterface
	{
	private:
		int mVertexCount;
		int mTriCount;
		int mVertNormalNum;
		int mVertTexNum;
		unsigned int* mIndices;
		float* mVertices;
		float* mTexCoords;
		float* mNormals;
		unsigned int* mTexIndexForTriangle;
		unsigned int* mNormalIndexForTriangle;
		bool hasNormal;
		bool hasTexCoord;

	public:
		ZQ_ObjLoader()
		{
			mVertexCount = 0;
			mTriCount = 0;
			mVertNormalNum = 0;
			mVertTexNum = 0;
			mIndices = 0;
			mVertices = 0;
			mTexCoords = 0;
			mNormals = 0;
			mTexIndexForTriangle = 0;
			mNormalIndexForTriangle = 0;
			hasTexCoord = false;
			hasNormal = false;
		}

		~ZQ_ObjLoader()
		{
			_clear();
		}

		bool HasNormal(){return hasNormal;}
		bool HasTexCoord () {return hasTexCoord;}

		int GetVertexNum(){return mVertexCount;}
		int GetTriangleNum() {return mTriCount;}
		int GetVertexNormalNum() {return mVertNormalNum;}
		int GetVertexTexNum() {return mVertTexNum;}

		float* GetVertexPtr() {return mVertices;}
		unsigned int* GetIndexPtr() {return mIndices;}
		float* GetNormalPtr() {return mNormals;}
		float* GetTexCoordPtr() {return mTexCoords;}
		unsigned int* GetTexIndexForTriangle() {return mTexIndexForTriangle;}
		unsigned int* GetNormalIndexForTriangle() {return mNormalIndexForTriangle;}
		
		bool LoadFromObjFile(const char *fname)
		{
			_clear();

			Parser ipp;

			if (!ipp.SetFile(fname))
			{
				return false;
			}

			if (!ipp.Parse(this, 0))
				return false;

			if (!ipp.SetFile(fname))
			{
				return false;
			}

			mVertices = new float[mVertexCount * 3];
			mIndices = new unsigned int[mTriCount * 3];
			mTexCoords = new float[mVertTexNum * 2];
			mNormals = new float[mVertNormalNum * 3];
			mTexIndexForTriangle = new unsigned int[mTriCount * 3];
			mNormalIndexForTriangle = new unsigned int[mTriCount * 3];

			mVertexCount = 0;
			mVertNormalNum = 0;
			mVertTexNum = 0;
			mTriCount = 0;
			hasTexCoord = false;
			hasNormal = false;

			if (!ipp.Parse(this, 1))
			{
				return false;
			}

			for (int i = 0; i < mTriCount * 3; i++)
			{
				mIndices[i] -= 1;
			}

			if (!hasTexCoord)
			{
				delete[]mTexIndexForTriangle;
				mTexIndexForTriangle = 0;
			}
			else
			{
				for (int i = 0; i < mTriCount * 3; i++)
				{
					mTexIndexForTriangle[i] -= 1;
				}

			}
			if (!hasNormal)
			{
				delete[]mNormalIndexForTriangle;
				mNormalIndexForTriangle = 0;
			}
			else
			{
				for (int i = 0; i < mTriCount * 3; i++)
				{
					mNormalIndexForTriangle[i] -= 1;
				}
			}


			return true;
		}

	private:
		void _clear()
		{
			mVertexCount = 0;
			mTriCount = 0;
			mVertNormalNum = 0;
			mVertTexNum = 0;

			hasNormal = false;
			hasTexCoord = false;

			if (mVertices)
			{
				delete[]mVertices;
				mVertices = 0;
			}
			if (mIndices)
			{
				delete[]mIndices;
				mIndices = 0;
			}
			if (mNormals)
			{
				delete[]mNormals;
				mNormals = 0;
			}
			if (mTexCoords)
			{
				delete[]mTexCoords;
				mTexCoords = 0;
			}
			if (mTexIndexForTriangle)
			{
				delete[]mTexIndexForTriangle;
				mTexIndexForTriangle = 0;
			}
			if (mNormalIndexForTriangle)
			{
				delete[]mNormalIndexForTriangle;
				mNormalIndexForTriangle = 0;
			}
		}

	public:
		bool ParseLine(int lineno,int argc,const char **argv, const int pass)
		{
			/*printf("line %d : ",lineno);
			for(int i = 0;i < argc;i++)
			printf("%s ",argv[i]);
			printf("\n");*/

			if (argc >= 1)
			{
				const char *foo = argv[0];
				if (*foo != '#')
				{
					if (strcmp(argv[0], "v") == 0 && argc == 4)
					{
						if (pass > 0)
						{
							float vx = (float)atof(argv[1]);
							float vy = (float)atof(argv[2]);
							float vz = (float)atof(argv[3]);
							mVertices[mVertexCount * 3 + 0] = vx;
							mVertices[mVertexCount * 3 + 1] = vy;
							mVertices[mVertexCount * 3 + 2] = vz;
						}
						mVertexCount++;
					}
					else if (strcmp(argv[0], "vt") == 0 && (argc == 3 || argc == 4))
					{
						if (pass > 0)
						{
							float tx = (float)atof(argv[1]);
							float ty = (float)atof(argv[2]);
							mTexCoords[mVertTexNum * 2 + 0] = tx;
							mTexCoords[mVertTexNum * 2 + 1] = ty;
						}
						mVertTexNum++;
					}
					else if (strcmp(argv[0], "vn") == 0 && argc == 4)
					{
						if (pass > 0)
						{
							float normalx = (float)atof(argv[1]);
							float normaly = (float)atof(argv[2]);
							float normalz = (float)atof(argv[3]);
							mNormals[mVertNormalNum * 3 + 0] = normalx;
							mNormals[mVertNormalNum * 3 + 1] = normaly;
							mNormals[mVertNormalNum * 3 + 2] = normalz;
						}
						mVertNormalNum++;
					}
					else if (strcmp(argv[0], "f") == 0 && argc >= 4)
					{
						int vcount = argc - 1;
						static int v_id[32] = { 0 };
						static int vn_id[32] = { 0 };
						static int vt_id[32] = { 0 };

						if (pass > 0)
						{
							bool has_vt_id = false;
							bool has_vn_id = false;
							if (!_getFaceIndexInfo(argc - 1, argv + 1, v_id, vt_id, vn_id, has_vt_id, has_vn_id))
							{
								return false;
							}

							if (has_vt_id)
								hasTexCoord = true;
							if (has_vn_id)
								has_vn_id = true;

							for (int tn = 0; tn < vcount - 2; tn++)
							{
								mIndices[mTriCount * 3 + 0] = v_id[0];
								mIndices[mTriCount * 3 + 1] = v_id[tn + 1];
								mIndices[mTriCount * 3 + 2] = v_id[tn + 2];

								if (has_vt_id)
								{
									mTexIndexForTriangle[mTriCount * 3 + 0] = vt_id[0];
									mTexIndexForTriangle[mTriCount * 3 + 1] = vt_id[tn + 1];
									mTexIndexForTriangle[mTriCount * 3 + 2] = vt_id[tn + 2];
								}

								if (has_vn_id)
								{
									mNormalIndexForTriangle[mTriCount * 3 + 0] = vn_id[0];
									mNormalIndexForTriangle[mTriCount * 3 + 1] = vn_id[tn + 1];
									mNormalIndexForTriangle[mTriCount * 3 + 2] = vn_id[tn + 2];
								}

								mTriCount++;
							}
						}
						else
						{
							mTriCount += vcount - 2;
						}
					}
				}
			}

			return true;
		}

		bool _getFaceIndexInfo(const int argc, const char** argv,int* v_id,int* vt_id,int* vn_id,bool& has_vt_id,bool& has_vn_id)
		{
			_getEachVertInfoForFace(argv[0], &v_id[0], &vt_id[0], &vn_id[0], has_vt_id, has_vn_id);

			for (int i = 1; i < argc; i++)
			{
				bool tmp_has_vt_id = false;
				bool tmp_has_vn_id = false;
				_getEachVertInfoForFace(argv[i], &v_id[i], &vt_id[i], &vn_id[i], tmp_has_vt_id, tmp_has_vn_id);

				if (tmp_has_vn_id != has_vn_id || tmp_has_vt_id != has_vt_id)
					return false;
			}

			return true;
		}

		void _getEachVertInfoForFace(const char* argv, int* v_id, int* vt_id, int* vn_id, bool& has_vt_id, bool& has_vn_id)
		{

			char buf[2000] = { 0 };
			int i = 0, j;

			j = i;
			while (argv[j] != '\0' && argv[j] != '/') j++;
			int len = j - i;
			if (len == 0)
			{
				printf(buf, "something wrong:%s:%d\n", __FILE__, __LINE__);
				assert(buf);
			}
			strncpy_s(buf, argv + i, len);
			buf[len] = '\0';
			sscanf_s(buf, "%d", v_id);

			if (argv[j] == '\0')
			{
				has_vt_id = false;
				has_vn_id = false;
				return;
			}

			i = j + 1;
			j = i;
			while (argv[j] != '\0' && argv[j] != '/') j++;
			len = j - i;
			if (argv[j] == '\0')
			{
				if (len == 0)
				{
					has_vn_id = false;
					has_vt_id = false;
					return;
				}
				else
				{
					sscanf_s(argv + i, "%d", vt_id);
					has_vt_id = true;
					has_vn_id = false;
					return;
				}
			}

			if (len == 0)
			{
				has_vt_id = false;
			}
			else
			{
				strncpy_s(buf, argv + i, len);
				buf[len] = 0;
				sscanf_s(buf, "%d", vt_id);
				has_vt_id = true;
			}

			i = j + 1;
			j = i;
			if (argv[j] != '\0')
			{
				has_vn_id = true;
				sscanf_s(argv + i, "%d", vn_id);
			}
			else
			{
				has_vn_id = false;
			}
		}
	};

	class ZQ_RawMesh
	{
	protected:

		int mVertexCount;
		int mTriCount;
		unsigned int* mIndices;
		float* mVertices;
		float* mTexCoords;
		float* mNormals;

		bool hasNormal;
		bool hasTexCoord;

	public:

		ZQ_RawMesh()
		{
			mVertexCount = 0;
			mTriCount = 0;
			mIndices = 0;
			mVertices = 0;
			mTexCoords = 0;
			mNormals = 0;
			hasTexCoord = false;
			hasNormal = false;

		}

		~ZQ_RawMesh()
		{
			_clear();
		}

		const int GetVertexNum() const {return mVertexCount;}
		const int GetTriangleNum() const {return mTriCount;}
		const bool HasNormal() const {return hasNormal;}
		const bool HasTexCoord() const {return hasTexCoord;}

		const float* GetVerticesPtr() const {return mVertices;}
		const unsigned int* GetTriangleIndicesPtr() const {return mIndices;}
		const float* GetNormalPtr() const {return mNormals;}
		const float* GetTexCoordPtr() const {return mTexCoords;}

		bool LoadFromObjFile(const char* fname)
		{
			ZQ_ObjLoader obj;
			if (!obj.LoadFromObjFile(fname))
				return false;

			_clear();

			int vCount = obj.GetVertexNum();
			int fCount = obj.GetTriangleNum();
			int vnCount = obj.GetVertexNormalNum();
			int vtCount = obj.GetVertexNormalNum();

			float* vPtr = obj.GetVertexPtr();
			unsigned int* idPtr = obj.GetIndexPtr();
			float* vnPtr = obj.GetNormalPtr();
			float* vtPtr = obj.GetTexCoordPtr();
			unsigned int* vt_idPtr = obj.GetTexIndexForTriangle();
			unsigned int* vn_idPtr = obj.GetNormalIndexForTriangle();


			hasNormal = obj.HasNormal();
			hasTexCoord = obj.HasTexCoord();

			mTriCount = fCount;
			mVertexCount = fCount * 3;

			mVertices = new float[mVertexCount * 3];
			mIndices = new unsigned int[mTriCount * 3];
			if (hasNormal)
				mNormals = new float[mVertexCount * 3];
			if (hasTexCoord)
				mTexCoords = new float[mVertexCount * 2];

			for (int i = 0; i < fCount; i++)
			{
				int v_id0 = idPtr[3 * i + 0];
				int v_id1 = idPtr[3 * i + 1];
				int v_id2 = idPtr[3 * i + 2];

				int cur_v_id0 = 3 * i + 0;
				int cur_v_id1 = 3 * i + 1;
				int cur_v_id2 = 3 * i + 2;

				memcpy(mVertices + cur_v_id0 * 3, vPtr + v_id0 * 3, sizeof(float) * 3);
				memcpy(mVertices + cur_v_id1 * 3, vPtr + v_id1 * 3, sizeof(float) * 3);
				memcpy(mVertices + cur_v_id2 * 3, vPtr + v_id2 * 3, sizeof(float) * 3);

				mIndices[3 * i + 0] = cur_v_id0;
				mIndices[3 * i + 1] = cur_v_id1;
				mIndices[3 * i + 2] = cur_v_id2;

				if (hasTexCoord)
				{
					int vt_id0 = vt_idPtr[3 * i + 0];
					int vt_id1 = vt_idPtr[3 * i + 1];
					int vt_id2 = vt_idPtr[3 * i + 2];

					memcpy(mTexCoords + cur_v_id0 * 2, vtPtr + vt_id0 * 2, sizeof(float) * 2);
					memcpy(mTexCoords + cur_v_id1 * 2, vtPtr + vt_id1 * 2, sizeof(float) * 2);
					memcpy(mTexCoords + cur_v_id2 * 2, vtPtr + vt_id2 * 2, sizeof(float) * 2);
				}

				if (hasNormal)
				{
					int vn_id0 = vn_idPtr[3 * i + 0];
					int vn_id1 = vn_idPtr[3 * i + 1];
					int vn_id2 = vn_idPtr[3 * i + 2];

					memcpy(mNormals + cur_v_id0 * 3, vnPtr + vn_id0 * 3, sizeof(float) * 3);
					memcpy(mNormals + cur_v_id1 * 3, vnPtr + vn_id1 * 3, sizeof(float) * 3);
					memcpy(mNormals + cur_v_id2 * 3, vnPtr + vn_id2 * 3, sizeof(float) * 3);
				}
			}

			return true;
		}

		bool LoadFromBinaryFile(const char* fname)
		{
			_clear();
			FILE* in = 0;
			if(0 != fopen_s(&in, fname, "rb"))
			{
				return false;
			}

			fread(&mVertexCount, sizeof(unsigned int), 1, in);
			fread(&mTriCount, sizeof(unsigned int), 1, in);
			fread(&hasNormal, sizeof(bool), 1, in);
			fread(&hasTexCoord, sizeof(bool), 1, in);

			if (mVertexCount <= 0 || mTriCount < 0)
			{
				_clear();
				fclose(in);
				return false;
			}
			mVertices = new float[mVertexCount * 3];
			if (fread(mVertices, sizeof(float), mVertexCount * 3, in) != mVertexCount * 3)
			{
				_clear();
				fclose(in);
				return false;
			}

			if (mTriCount > 0)
			{
				mIndices = new unsigned int[mTriCount * 3];
				if (fread(mIndices, sizeof(unsigned int), mTriCount * 3, in) != mTriCount * 3)
				{
					_clear();
					fclose(in);
					return false;
				}
			}

			if (hasNormal)
			{
				mNormals = new float[mVertexCount * 3];
				if (fread(mNormals, sizeof(float), mVertexCount * 3, in) != mVertexCount * 3)
				{
					_clear();
					fclose(in);
					return false;
				}
			}

			if (hasTexCoord)
			{
				mTexCoords = new float[mVertexCount * 2];
				if (fread(mNormals, sizeof(float), mVertexCount * 2, in) != mVertexCount * 2)
				{
					_clear();
					fclose(in);
					return false;
				}
			}

			return true;
		}

		bool SaveToBinaryFile(const char* fname)
		{
			if (mVertexCount <= 0)
				return false;


			FILE* out = 0;
			if(0 != fopen_s(&out, fname, "wb"))
			{
				return false;
			}


			fwrite(&mVertexCount, sizeof(unsigned int), 1, out);
			fwrite(&mTriCount, sizeof(unsigned int), 1, out);
			fwrite(&hasNormal, sizeof(bool), 1, out);
			fwrite(&hasTexCoord, sizeof(bool), 1, out);

			if (fwrite(mVertices, sizeof(float), mVertexCount * 3, out) != mVertexCount * 3)
			{
				fclose(out);
				return false;
			}

			if (mTriCount > 0)
			{
				if (fwrite(mIndices, sizeof(unsigned int), mTriCount * 3, out) != mTriCount * 3)
				{
					fclose(out);
					return false;
				}
			}

			if (hasNormal)
			{
				if (fwrite(mNormals, sizeof(float), mVertexCount * 3, out) != mVertexCount * 3)
				{
					fclose(out);
					return false;
				}
			}

			if (hasTexCoord)
			{
				if (fwrite(mNormals, sizeof(float), mVertexCount * 2, out) != mVertexCount * 2)
				{
					fclose(out);
					return false;
				}
			}

			fclose(out);
			return true;
		}

	private:
		void _clear()
		{
			mVertexCount = 0;
			mTriCount = 0;
			hasNormal = false;
			hasTexCoord = false;

			if (mVertices)
			{
				delete[]mVertices;
				mVertices = 0;
			}
			if (mIndices)
			{
				delete[]mIndices;
				mIndices = 0;
			}
			if (mNormals)
			{
				delete[]mNormals;
				mNormals = 0;
			}
			if (mTexCoords)
			{
				delete[]mTexCoords;
				mTexCoords = 0;
			}
		}
	};
}




#endif
