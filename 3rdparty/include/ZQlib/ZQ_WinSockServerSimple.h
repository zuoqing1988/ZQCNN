#ifndef _ZQ_WIN_SOCK_SERVER_SIMPLE_H_
#define _ZQ_WIN_SOCK_SERVER_SIMPLE_H_
#pragma once

#include "ZQ_WinSockServerImpl.h"
#include "ZQ_WinSockIOPoolSimple.h"
namespace ZQ
{
	class ZQ_WinSockServerSimple : public ZQ_EventSocketServerImpl
	{
	public:
		ZQ_WinSockServerSimple()
		{
			BindIOObj(&pool);
		}

		~ZQ_WinSockServerSimple()
		{
			Stop();
			BindIOObj(0);
		}

		void BroadCast(/*_in_*/ const std::string& msg)
		{
			for (int i = 0; i < m_nSocket; i++)
			{
				Send(m_sockets[i], msg);
			}
		}

		bool Send(/*_in_*/ const SOCKET sock, /*_in_*/ const std::string& msg)
		{
			size_t len = msg.length();
			if (len == 0)
				return false;
			size_t header_size = sizeof(ZQ_WinSockIOPoolSimple::ProtocalHeader);
			size_t need_len = len + header_size;
			char* buf = (char*)malloc(need_len);
			
			if (buf)
			{
				ZQ_WinSockIOPoolSimple::ProtocalHeader header;
				header.length = need_len;
				memcpy(buf, &header, header_size);
				memcpy(buf + header_size, &msg[0], len);
				bool ret = m_pIO->Send(sock, buf, need_len);
				free(buf);
				return ret;
			}
			return false;
		}

		//blocking reader function, should be called in a while(1){} loop
		bool ConsumePacket(/*_out_*/ SOCKET& sock, /*_out_*/ std::string& msg, /*_in_*/ UINT waittime)
		{
			if (m_pIO == 0)
				return false;

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

		bool GetIPPort(/*_in_*/ SOCKET sock, std::string& ip, int& port)
		{
			struct sockaddr_in addr;
			int addrlen = sizeof(addr);
			if (getpeername(sock, (struct sockaddr*)&addr, &addrlen) == -1)
			{
				return false;
			}
			char buf[100];
			PCSTR ip_s = inet_ntop(AF_INET, &addr.sin_addr, buf, 100);
			//char* ip_s = inet_ntoa(addr.sin_addr);
			ip.assign(ip_s);
			port = addr.sin_port;
			return true;
		}

		bool IsConnecting(const std::string& ip, int port)
		{
			int count = m_nSocket;
			std::string tmp_ip;
			int tmp_port;
			for (int i = 0; i < count; i++)
			{
				if (GetIPPort(m_sockets[i], tmp_ip, tmp_port))
				{
					if (tmp_ip == ip && tmp_port == port)
						return true;
				}
			}
			return false;
		}

	private:
		ZQ_WinSockIOPoolSimple pool;
		
	};
}

#endif