#ifndef _ZQ_LOGGER_H_
#define _ZQ_LOGGER_H_
#pragma once

#include <windows.h>
#include <map>
#include <string>
#include <ostream>
#include <tchar.h>

#define ZQERR ZQ::ZQ_Logger::Instance()->Error
#define ZQWARNING ZQ::ZQ_Logger::Instance()->Warning
#ifdef UNICODE
#define _TSTRING std::wstring
#else
#define _TSTRING std::string
#endif
#define _TCOUT std::cout

namespace ZQ
{

	class ZQ_Logger
	{
	public:
		enum LogModeEnum{ LOG_MODE_CONSOLE = 0, LOG_MODE_MSGBOX, LOG_MODE_OSTREAM };


		~ZQ_Logger(void)
		{
			DeleteCriticalSection(&m_cs);
		}

		static ZQ_Logger *Instance(void)
		{
			static ZQ_Logger instance;
			return &instance;
		}
		
		void Error(const wchar_t *msg, ...)
		{
			wchar_t szBuf[512] = L"Error: ";
			va_list args;
			va_start(args, msg);
			vswprintf_s(szBuf + 7, 512 - 7, msg, args);
			va_end(args);
			szBuf[511] = L'\0';
#ifdef _UNICODE
			TextLine(szBuf);
#else
			char szBufAsc[512];
			WCharToChar(szBufAsc, szBuf);
			TextLine(szBufAsc);
#endif
		}


		void Error(const char *msg, ...)
		{
			char szBuf[512] = "Error: ";
			va_list args;
			va_start(args, msg);
			vsprintf_s(szBuf + 7, 512 - 7, msg, args);
			va_end(args);
			szBuf[511] = '\0';
#ifdef _UNICODE
			wchar_t szBufW[512];
			CharToWChar(szBufW, szBuf);
			TextLine(szBufW);
#else
			TextLine(szBuf);
#endif
		}

		void Error(const HRESULT hr)
		{
			std::map<HRESULT, _TSTRING>::iterator iter = m_hrMsg.find(hr);
			std::map<HRESULT, _TSTRING>::iterator iterf = m_hrFacMsg.find(HRESULT_FACILITY(hr));
			if (iter != m_hrMsg.end())
			{
				if (iterf != m_hrFacMsg.end())
				{
					Error(_T("[%s] %s"), iterf->second.c_str(), iter->second.c_str());
				}
				else
				{
					Error(iter->second.c_str());
				}
			}
			else
				Error(_T("Unknown error!"));
		}

		void LastError(void)
		{
			DWORD err = GetLastError();
			LPVOID lpMsgBuf;
			FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
				NULL,
				err,
				MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
				(LPTSTR)&lpMsgBuf,
				0,
				NULL);
			Error((TCHAR *)lpMsgBuf);
			LocalFree(lpMsgBuf);
		}

		void Warning(const wchar_t *msg, ...)
		{
			wchar_t szBuf[512] = L"Warning: ";
			va_list args;
			va_start(args, msg);
			vswprintf_s(szBuf + 9, 512 - 9, msg, args);
			va_end(args);
			szBuf[511] = L'\0';
#ifdef _UNICODE
			TextLine(szBuf);
#else
			char szBufAsc[512];
			WCharToChar(szBufAsc, szBuf);
			TextLine(szBufAsc);
#endif
		}

		void Warning(const char *msg, ...)
		{
			char szBuf[512] = "Warning: ";
			va_list args;
			va_start(args, msg);
			vsprintf_s(szBuf + 9, 512 - 9, msg, args);
			va_end(args);
			szBuf[511] = '\0';
#ifdef _UNICODE
			wchar_t szBufW[512];
			CharToWChar(szBufW, szBuf);
			TextLine(szBufW);
#else
			TextLine(szBuf);
#endif
		}

		void Text(const TCHAR *msg)
		{
			switch (m_logMode)
			{
			case LOG_MODE_CONSOLE:
				WriteConsole(GetStdHandle(STD_OUTPUT_HANDLE), msg, wcslen(msg), NULL, NULL);
				break;
			case LOG_MODE_OSTREAM:

				if (m_pOutStream)
					(*m_pOutStream) << msg;
				break;
			case LOG_MODE_MSGBOX:
				MessageBox(NULL, msg, _T("Log"), MB_OK);
				break;
			default:
				break;
			}
		}

		void TextLine(const TCHAR *msg)
		{
			switch (m_logMode)
			{
			case LOG_MODE_CONSOLE:
				WriteConsole(GetStdHandle(STD_OUTPUT_HANDLE), msg, wcslen(msg), NULL, NULL);
				break;
			case LOG_MODE_OSTREAM:
			{
									 if (m_pOutStream)
										 (*m_pOutStream) << msg << std::endl;
			}
				break;
			case LOG_MODE_MSGBOX:
			{
									MessageBox(NULL, msg, _T("Log"), MB_OK);
			}
				break;
			default:
				break;
			}
		}

		inline std::ostream *&OutStreamPtr(void) {return m_pOutStream;}
		inline LogModeEnum &LogMode(void) {return m_logMode;}

	protected:
		ZQ_Logger(void):
			m_logMode(LOG_MODE_CONSOLE),
			m_pOutStream(0)
		{
			//HRESULT to string
			typedef std::pair<HRESULT, _TSTRING> ItemType;

			m_hrMsg.insert(ItemType(E_ABORT, _T("Operation aborted!")));
			m_hrMsg.insert(ItemType(E_ACCESSDENIED, _T("General access denied error!")));
			m_hrMsg.insert(ItemType(E_FAIL, _T("Unspecified failure!")));
			m_hrMsg.insert(ItemType(E_HANDLE, _T("Handle that is not valid!")));
			m_hrMsg.insert(ItemType(E_INVALIDARG, _T("One or more arguments are not valid!")));
			m_hrMsg.insert(ItemType(E_NOINTERFACE, _T("No such interface supported!")));
			m_hrMsg.insert(ItemType(E_NOTIMPL, _T("Not implemented!")));
			m_hrMsg.insert(ItemType(E_OUTOFMEMORY, _T("Failed to allocate necessary memory!")));
			m_hrMsg.insert(ItemType(E_POINTER, _T("Pointer that is not valid!")));
			m_hrMsg.insert(ItemType(E_UNEXPECTED, _T("Unexpected failure!")));

			m_hrFacMsg.insert(ItemType(FACILITY_WINDOWS_CE, _T("WINDOWS_CE")));
			m_hrFacMsg.insert(ItemType(FACILITY_WINDOWS, _T("WINDOWS")));
			m_hrFacMsg.insert(ItemType(FACILITY_URT, _T("URT")));
			m_hrFacMsg.insert(ItemType(FACILITY_UMI, _T("UMI")));
			m_hrFacMsg.insert(ItemType(FACILITY_SXS, _T("SXS")));
			m_hrFacMsg.insert(ItemType(FACILITY_STORAGE, _T("STORAGE")));
			m_hrFacMsg.insert(ItemType(FACILITY_STATE_MANAGEMENT, _T("STATE_MANAGEMENT")));
			m_hrFacMsg.insert(ItemType(FACILITY_SCARD, _T("SCARD")));
			m_hrFacMsg.insert(ItemType(FACILITY_SETUPAPI, _T("SETUPAPI")));
			m_hrFacMsg.insert(ItemType(FACILITY_SECURITY, _T("SECURITY")));
			m_hrFacMsg.insert(ItemType(FACILITY_RPC, _T("RPC")));
			m_hrFacMsg.insert(ItemType(FACILITY_WIN32, _T("WIN32")));
			m_hrFacMsg.insert(ItemType(FACILITY_CONTROL, _T("CONTROL")));
			m_hrFacMsg.insert(ItemType(FACILITY_NULL, _T("GENERIC")));
			m_hrFacMsg.insert(ItemType(FACILITY_METADIRECTORY, _T("METADIRECTORY")));
			m_hrFacMsg.insert(ItemType(FACILITY_MSMQ, _T("MSMQ")));
			m_hrFacMsg.insert(ItemType(FACILITY_MEDIASERVER, _T("MEDIASERVER")));
			m_hrFacMsg.insert(ItemType(FACILITY_INTERNET, _T("INTERNET")));
			m_hrFacMsg.insert(ItemType(FACILITY_ITF, _T("COM")));
			m_hrFacMsg.insert(ItemType(FACILITY_HTTP, _T("HTTP")));
			m_hrFacMsg.insert(ItemType(FACILITY_DPLAY, _T("DPLAY")));
			m_hrFacMsg.insert(ItemType(FACILITY_DISPATCH, _T("DISPATCH")));
			m_hrFacMsg.insert(ItemType(FACILITY_CONFIGURATION, _T("CONFIGURATION")));
			m_hrFacMsg.insert(ItemType(FACILITY_COMPLUS, _T("COMPLUS")));
			m_hrFacMsg.insert(ItemType(FACILITY_CERT, _T("CERT")));
			m_hrFacMsg.insert(ItemType(FACILITY_BACKGROUNDCOPY, _T("BACKGROUNDCOPY")));
			m_hrFacMsg.insert(ItemType(FACILITY_ACS, _T("ACS")));
			m_hrFacMsg.insert(ItemType(FACILITY_AAF, _T("AAF")));

			InitializeCriticalSection(&m_cs);
		}

		std::map<HRESULT, _TSTRING> m_hrMsg;
		std::map<HRESULT, _TSTRING> m_hrFacMsg;
		CRITICAL_SECTION m_cs;
		std::ostream *m_pOutStream;
		LogModeEnum m_logMode;

	private:

		static void CharToWChar(wchar_t *out, const char *in)
		{
			UINT len = MultiByteToWideChar(CP_ACP, 0, in, -1, 0, 0);
			MultiByteToWideChar(CP_ACP, 0, in, -1, out, len);
		}

		static void WCharToChar(char *out, const wchar_t *in)
		{
			UINT len = WideCharToMultiByte(CP_ACP, 0, in, -1, 0, 0, 0, 0);
			WideCharToMultiByte(CP_ACP, 0, in, -1, out, len, 0, 0);
		}

		static void WStrToStr(std::string &out, const std::wstring &in)
		{
			char szTmp[512] = { 0 };
#ifdef _DEBUG
			if (in.length() >= MAX_PATH) return;
#endif

			WideCharToMultiByte(CP_ACP, 0, in.c_str(), -1, szTmp, MAX_PATH, 0, 0);
			out = szTmp;
		}

		static void StrToWStr(std::wstring &out, const std::string &in)
		{
			wchar_t szTmp[512] = { 0 };
#ifdef _DEBUG
			if (in.length() >= MAX_PATH) return;
#endif

			MultiByteToWideChar(CP_ACP, 0, in.c_str(), -1, szTmp, MAX_PATH);
			out = szTmp;
		}
	};

}

#endif