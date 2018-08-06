#ifndef _ZQ_WIN_SOCK_CLIENT_SIMPLE_H_
#define _ZQ_WIN_SOCK_CLIENT_SIMPLE_H_
#pragma once

#include "ZQ_WinSockClientImpl.h"
#include "ZQ_WinSockIOPoolSimple.h"

namespace ZQ
{
	class ZQ_WinSockClientSimple: public ZQ_EventSocketClientImpl
	{
	public:
		ZQ_WinSockClientSimple()
		{
			buf_len = 1024;
			buf = (char*)malloc(buf_len);
			BindIOObj(&pool);
		}

		~ZQ_WinSockClientSimple()
		{
			Close();
			BindIOObj(0);
			free(buf);
		}

		bool Send(/*_in_*/ std::string& msg)
		{
			size_t len = msg.length();
			if (len == 0)
				return false;
			unsigned int header_size = sizeof(ZQ_WinSockIOPoolSimple::ProtocalHeader);
			size_t need_len = len + header_size;
			char* buf = (char*)malloc(need_len);

			if (buf)
			{
				ZQ_WinSockIOPoolSimple::ProtocalHeader header;
				header.length = need_len;
				memcpy(buf, &header, header_size);
				memcpy(buf + header_size, &msg[0], len);
				bool ret = m_pIO->Send(m_socket, buf, need_len);
				//ret = ret && SendBuffered();
				free(buf);
				return ret;
				
			}
			return false;
		}

		//blocking reader function, should be called in a while(1){} loop
		bool ConsumePacket(/*_out_*/ std::string& msg, /*_in_*/ UINT waittime)
		{
			SOCKET sock;
			char* read_buf = 0;
			UINT read_buf_len = 0;
			if (!m_pIO->ConsumePacket(&sock, &read_buf, &read_buf_len, waittime))
				return false;
			else
			{
				int header_size = sizeof(ZQ_WinSockIOPoolSimple::ProtocalHeader);
				ZQ_WinSockIOPoolSimple::ProtocalHeader * ptr = (ZQ_WinSockIOPoolSimple::ProtocalHeader*)read_buf;
				int need_len = ptr->length - header_size;
				bool ret = false;
				if (need_len > 0)
				{
					msg.resize(need_len);
					memcpy(&msg[0], read_buf + header_size, need_len);
					ret = true;
				}
				
				free(read_buf);
				return ret;
			}
		}

	private:	
		ZQ_WinSockIOPoolSimple pool;
		long long buf_len;
		char* buf;
	};
}

#endif