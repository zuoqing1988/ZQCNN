// tools/gl_stub/GL/glew.h —— **只给 `-fsyntax-only` 用的桩**，不是真的 glew。
//
// 为什么需要它（附录 IZ.1）
// ----------------------
// `ZQlib/ZQ_GLSLShader.h` 是 8 个「本机编不了」的头里**唯一一个不是平台特有的**：
// 另外 7 个要 `windows.h`（ZQ_WinSock*），这一个只要 OpenGL ——
// 而本机没有装 glew，于是它被 `probe_zqlib_headers.py` 归进 BROKEN。
// 于是「不修」的理由又变成了一句没验证过的话。
//
// 这里给出它用到的**全部 17 个 GL 符号**（GL 枚举 + 类型 + 函数），
// 声明与 glew 的签名一致，于是
//     g++ -fsyntax-only -I tools/gl_stub  ZQ_GLSLShader.h
// 能真的把这个头编一遍，**验证它自身是不是自足的**（有没有漏 include、
// 有没有打错字、有没有用到未声明的东西）。
//
// 它**不能**证明可链接或可运行 —— 桩里没有实现。真要跑必须有真 glew + GL 上下文。
// 所以本桩只进 `-fsyntax-only`，不进任何链接路径。
#ifndef _ZQ_GL_STUB_GLEW_H_
#define _ZQ_GL_STUB_GLEW_H_

#include <stddef.h>

typedef unsigned int   GLenum;
typedef unsigned char  GLboolean;
typedef unsigned int   GLbitfield;
typedef signed char    GLbyte;
typedef short          GLshort;
typedef int            GLint;
typedef unsigned char  GLubyte;
typedef unsigned short GLushort;
typedef unsigned int   GLuint;
typedef int            GLsizei;
typedef float          GLfloat;
typedef double         GLdouble;
typedef char           GLchar;
typedef ptrdiff_t      GLintptr;
typedef ptrdiff_t      GLsizeiptr;

#define GL_VERTEX_SHADER   0x8B31
#define GL_FRAGMENT_SHADER 0x8B30
#define GL_COMPILE_STATUS  0x8B81
#define GL_LINK_STATUS     0x8B82
#define GL_INFO_LOG_LENGTH 0x8B84

#ifdef __cplusplus
extern "C" {
#endif

GLuint glCreateShader(GLenum type);
void   glDeleteShader(GLuint shader);
void   glShaderSource(GLuint shader, GLsizei count, const GLchar* const* string, const GLint* length);
void   glCompileShader(GLuint shader);
void   glGetShaderiv(GLuint shader, GLenum pname, GLint* params);
void   glGetShaderInfoLog(GLuint shader, GLsizei bufSize, GLsizei* length, GLchar* infoLog);
GLuint glCreateProgram(void);
void   glDeleteProgram(GLuint program);
void   glAttachShader(GLuint program, GLuint shader);
void   glLinkProgram(GLuint program);
void   glGetProgramiv(GLuint program, GLenum pname, GLint* params);
void   glGetProgramInfoLog(GLuint program, GLsizei bufSize, GLsizei* length, GLchar* infoLog);
void   glUseProgram(GLuint program);

#ifdef __cplusplus
}
#endif

#endif
