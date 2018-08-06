#ifndef _ZQ_GLSL_SHADER_H_
#define _ZQ_GLSL_SHADER_H_
#pragma once

#include <GL/glew.h>

#include <iostream>
#include <map>
#include <vector>
#include <string>


namespace ZQ
{
	class ZQ_GLSLshader
	{
	public:
		ZQ_GLSLshader(){}
		~ZQ_GLSLshader(){ _deleteAllShaders(); }

	private:

		std::map<std::string,GLuint> programs;
		std::map<std::string,GLuint> vertShaders;
		std::map<std::string,GLuint> fragShaders;

	public:
		bool CreateShaderFromFile(const char* shaderName, const char* vertFileName,const char* fragFileName)
		{
			if (shaderName == 0 || vertFileName == 0 || fragFileName == 0)
				return false;

			GLchar* vsSrc = 0;
			GLchar* psSrc = 0;
			if (!_textFileRead(vertFileName, vsSrc))
				return false;
			if (!_textFileRead(fragFileName, psSrc))
			{
				delete[]vsSrc;
				return false;
			}

			bool flag = CreateShaderFromBytes(shaderName, vsSrc, psSrc);

			delete[]vsSrc;
			delete[]psSrc;
			return flag;
		}

		bool CreateShaderFromBytes(const char* shaderName, const char* vsSrc, const char* psSrc)
		{
			if (shaderName == 0 || vsSrc == 0 || psSrc == 0)
				return false;
			std::string programName(shaderName);
			GLuint vshader, pshader, program;

			if (programs.find(programName) != programs.end())
				return false;


			vshader = glCreateShader(GL_VERTEX_SHADER);
			pshader = glCreateShader(GL_FRAGMENT_SHADER);

			glShaderSource(vshader, 1, (const char**)&vsSrc, NULL);
			glShaderSource(pshader, 1, (const char**)&psSrc, NULL);

			GLint compiled, linked;
			glCompileShader(vshader);
			glGetShaderiv(vshader, GL_COMPILE_STATUS, &compiled);
			if (!compiled)
			{
				GLint length;
				GLchar* log;
				glGetShaderiv(vshader, GL_INFO_LOG_LENGTH, &length);
				log = new GLchar[length];
				glGetShaderInfoLog(vshader, length, &length, log);
				fprintf(stderr, "%s\n", log);
				delete[]log;
				glDeleteShader(vshader);
				glDeleteShader(pshader);
				return false;
			}
			glCompileShader(pshader);
			glGetShaderiv(pshader, GL_COMPILE_STATUS, &compiled);
			if (!compiled)
			{
				GLint length;
				GLchar* log;
				glGetShaderiv(pshader, GL_INFO_LOG_LENGTH, &length);
				log = new GLchar[length];
				glGetShaderInfoLog(pshader, length, &length, log);
				fprintf(stderr, "%s\n", log);
				delete[]log;
				glDeleteShader(vshader);
				glDeleteShader(pshader);
				return false;
			}
			program = glCreateProgram();
			glAttachShader(program, pshader);
			glAttachShader(program, vshader);

			glLinkProgram(program);
			glGetProgramiv(program, GL_LINK_STATUS, &linked);
			if (!linked)
			{
				GLint length;
				GLchar* log;
				glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
				log = new GLchar[length];
				glGetProgramInfoLog(program, length, &length, log);
				fprintf(stderr, "%s\n", log);
				delete[]log;
				glDeleteShader(vshader);
				glDeleteShader(pshader);
				glDeleteProgram(program);
				return false;
			}
			programs.insert(std::make_pair(programName, program));
			vertShaders.insert(std::make_pair(programName, vshader));
			fragShaders.insert(std::make_pair(programName, pshader));
			return true;
		}

		GLuint GetShaderHandler(const char* shaderName)
		{
			if (shaderName == 0)
				return 0;
			std::string name(shaderName);
			std::map<std::string, GLuint>::iterator it = programs.find(name);
			if (it == programs.end())
				return 0;
			return it->second;

		}

		bool UseShader(const char* shaderName)
		{
			if (shaderName == 0)
			{
				glUseProgram(0);
				return true;
			}
			std::string name(shaderName);
			std::map<std::string, GLuint>::iterator it = programs.find(name);
			if (it == programs.end())
				return false;
			glUseProgram(it->second);
			return true;
		}

	private:
		void _deleteAllShaders()
		{
			for (std::map<std::string, GLuint>::iterator it = vertShaders.begin(); it != vertShaders.end(); ++it)
				glDeleteShader(it->second);
			for (std::map<std::string, GLuint>::iterator it = fragShaders.begin(); it != fragShaders.end(); ++it)
				glDeleteShader(it->second);
			for (std::map<std::string, GLuint>::iterator it = programs.begin(); it != programs.end(); ++it)
				glDeleteProgram(it->second);
		}

		bool _textFileRead(const char* _fn, GLchar* &_shader)
		{
			if (NULL == _fn)
				return false;

			FILE *fp;
			int count = 0;

			fp = fopen(_fn, "rt");
			if (NULL == fp)
				return false;


			fseek(fp, 0, SEEK_END);

			count = ftell(fp);

			rewind(fp);

			if (count <= 0)
				return false;

			_shader = new GLchar[count];
			count = fread(_shader, sizeof(GLchar), count, fp);
			_shader[count] = '\0';
			fclose(fp);

			return true;
		}
	};

	
}

#endif