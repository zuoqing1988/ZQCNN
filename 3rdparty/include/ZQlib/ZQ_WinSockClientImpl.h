#ifndef _ZQ_WIN32_SOCK_CLIENT_H_
#define _ZQ_WIN32_SOCK_CLIENT_H_
#pragma once

#include "ZQ_WinSockBase.h"
#include "ZQ_Logger.h"
#include <process.h>

namespace ZQ
{
	class ZQ_SocketClientBlocked : public ZQ_SocketBase
	{
	public:
		ZQ_SocketClientBlocked(void):m_socket(INVALID_SOCKET)
		{
			InitializeCriticalSection(&cs_socket);
		}
		~ZQ_SocketClientBlocked(void)
		{	
			Close(); 
			DeleteCriticalSection(&cs_socket);
		}

		virtual bool Connect(const char *szIP, USHORT port)
		{
			_scopeLock lock(&cs_socket);
			if (INVALID_SOCKET == (m_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP)))
			{
				ZQERR(_T("Unable to create socket!"));
				return false;
			}

			m_sockAddr.sin_family = AF_INET;
			//m_sockAddr.sin_addr.s_addr = inet_addr(szIP);
			inet_pton(AF_INET, szIP, &m_sockAddr.sin_addr.s_addr);
			m_sockAddr.sin_port = htons(port);

			if (SOCKET_ERROR == (connect(m_socket, (sockaddr *)&m_sockAddr, sizeof(m_sockAddr))))
			{
				ZQERR(_T("Failed to connect to server!\n"));
				closesocket(m_socket);
				return false;
			}

			ZQ_Logger::Instance()->Text(_T("Connected..."));
			return true;
		}

		virtual bool Send(const char *buf, UINT nBytes)
		{
			_scopeLock lock(&cs_socket);
			int lenSent;
			if (!buf || nBytes == 0 || INVALID_SOCKET == m_socket)
				return false;

			lenSent = send(m_socket, buf, nBytes, 0);
			if (SOCKET_ERROR == lenSent)
			{
				ZQERR(_T("Failed to send data!"));
				return false;
			}
			else if ((UINT)lenSent < nBytes)
			{
				ZQWARNING(_T("Sent less bytes than requested!"));
				return true;
			}

			return true;
		}

		virtual void Close(void)
		{
			_scopeLock lock(&cs_socket);
			if (INVALID_SOCKET != m_socket)
			{
				closesocket(m_socket);
				m_socket = INVALID_SOCKET;
			}
		}

	protected:
		CRITICAL_SECTION cs_socket;
		SOCKET m_socket;
		SOCKADDR_IN m_sockAddr;

		class _scopeLock
		{
		public:
			_scopeLock(CRITICAL_SECTION *cs)
			{
				EnterCriticalSection(cs);
				m_cs = cs;
			}
			~_scopeLock()
			{
				LeaveCriticalSection(m_cs);
			}

		private:
			CRITICAL_SECTION* m_cs;
		};
	};

	/*************************************************************/

	class ZQ_EventSocketClientImpl : public ZQ_SocketClientBlocked
	{
	protected:
		CRITICAL_SECTION cs_buffer; // first
		CRITICAL_SECTION cs_io; //second
		
		char *m_pBuffer;
		UINT m_nBufferSize;
		bool m_bConnected;
		HANDLE m_hSocketEvent;
		HANDLE m_hStartIO;
		ZQ_SocketIO *m_pIO;
		HANDLE m_hIOThread;
		bool m_terminate;
		bool m_bWriteReady;

	public:
		ZQ_EventSocketClientImpl(void) : m_bConnected(false),
			m_pIO(0),
			m_nBufferSize(0),
			m_pBuffer(0),
			m_hSocketEvent(WSA_INVALID_EVENT),
			m_hStartIO(INVALID_HANDLE_VALUE),
			m_hIOThread(INVALID_HANDLE_VALUE),
			m_terminate(false),
			m_bWriteReady(false)
		{
			InitializeCriticalSection(&cs_buffer);
			InitializeCriticalSection(&cs_io);
			ChangeBufferSize(1024 * 1024);
		}

		~ZQ_EventSocketClientImpl(void)
		{
			if (m_pBuffer)
				delete[]m_pBuffer;
			m_pBuffer = 0;
			DeleteCriticalSection(&cs_io);
			DeleteCriticalSection(&cs_buffer);
		}

		void BindIOObj(ZQ_SocketIO *pObj)
		{
			EnterCriticalSection(&cs_io);
			m_pIO = pObj;
			LeaveCriticalSection(&cs_io);
		}


		bool Connect(const char *szIP, USHORT port)
		{
			Close();
			EnterCriticalSection(&cs_socket);
			if (INVALID_SOCKET == (m_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP)))
			{
				ZQERR(_T("Unable to create socket!\n"));
				LeaveCriticalSection(&cs_socket);
				return false;
			}

			m_sockAddr.sin_family = AF_INET;
			//m_sockAddr.sin_addr.s_addr = inet_addr(szIP);
			inet_pton(AF_INET, szIP, &m_sockAddr.sin_addr.s_addr);
			m_sockAddr.sin_port = htons(port);

			if (WSA_INVALID_EVENT != m_hSocketEvent)
				WSACloseEvent(m_hSocketEvent);

			if (WSA_INVALID_EVENT == (m_hSocketEvent = WSACreateEvent()))
			{
				LeaveCriticalSection(&cs_socket);
				ZQERR(_T("Failed to create sync event for socket!\n"));
				return false;
			}

			WSAEventSelect(m_socket, m_hSocketEvent, FD_CONNECT | FD_READ | FD_WRITE | FD_CLOSE);

			unsigned long ul = 1;
			ioctlsocket(m_socket, FIONBIO, &ul);
			int eventId;
			eventId = connect(m_socket, (sockaddr *)&m_sockAddr, sizeof(m_sockAddr));

			LeaveCriticalSection(&cs_socket);
	
			if (m_hStartIO != INVALID_HANDLE_VALUE && m_hStartIO != 0)
				CloseHandle(m_hStartIO);

			if (0 == (m_hStartIO = CreateEvent(NULL, true, false, NULL)))
			{
				ZQ_Logger::Instance()->LastError();
				return false;
			}

			m_hIOThread = (HANDLE)_beginthreadex(0, 0, ZQ_EventSocketClientImpl::IOThreadProc, this, 0, NULL);
			ZQ_Logger::Instance()->Text(_T("Socket IO thread started...\n"));
			if (INVALID_HANDLE_VALUE == m_hIOThread)
			{
				ZQERR(_T("Failed to start socket IO threads!\n"));
				return false;
			}

			return true;
		}

		bool IsConnected() const { return m_bConnected; }

		void Close(void)
		{
			m_terminate = true;
			if (m_hIOThread != INVALID_HANDLE_VALUE)
			{
				WaitForSingleObject(m_hIOThread, INFINITE);
				m_hIOThread = INVALID_HANDLE_VALUE;
			}
			RemoveSocket();
		}

		bool ChangeBufferSize(UINT size)
		{
			char *tmp;
			tmp = new char[size];
			if (!tmp)
				return false;

			EnterCriticalSection(&cs_buffer);
			if (m_pBuffer)
				delete[]m_pBuffer;
			m_pBuffer = tmp;
			m_nBufferSize = size;
			LeaveCriticalSection(&cs_buffer);
			return true;
		}

	protected:
		static UINT _stdcall IOThreadProc(void *arg)
		{
			ZQ_EventSocketClientImpl *me = (ZQ_EventSocketClientImpl *)arg;
			WSANETWORKEVENTS networkEvents;
			UINT eventId;
			UINT nBytesReceived;
			me->m_terminate = false;

			while (!me->m_terminate)
			{
				me->SendBuffered();

				if (WSA_WAIT_FAILED ==
					(eventId = WSAWaitForMultipleEvents(1, &me->m_hSocketEvent, false, 5, false)))
				{
					ZQERR(_T("Failed to wait for network events\n"));
					ZQ_Logger::Instance()->LastError();
					return 1;
				}
				else if (WSA_WAIT_TIMEOUT == eventId)
				{
					continue;
				}
				else if (eventId - WSA_WAIT_EVENT_0 == 0) //IO event
				{
					if (SOCKET_ERROR ==
						WSAEnumNetworkEvents(me->m_socket, me->m_hSocketEvent, &networkEvents))
					{
						ZQERR(_T("Failed to enumerate network events!\n"));
						ZQ_Logger::Instance()->LastError();
						return 1;
					}

					if (networkEvents.lNetworkEvents & FD_CONNECT)
					{
						if (networkEvents.iErrorCode[FD_CONNECT_BIT] != 0)
						{
							ZQ_Logger::Instance()->Text(_T("Failed to connect!\n"));
							me->m_bConnected = false;
							me->m_terminate = true;
							continue;
						}
						else
						{
							ZQ_Logger::Instance()->Text(_T("Client socket connected!\n"));
							me->m_bConnected = true;
						}
						
					}

					if (networkEvents.lNetworkEvents & FD_READ)
					{
						EnterCriticalSection(&me->cs_buffer);
						nBytesReceived = recv(me->m_socket, me->m_pBuffer, me->m_nBufferSize, 0);
						if (nBytesReceived == SOCKET_ERROR && WSAEWOULDBLOCK != WSAGetLastError())
						{
							ZQERR(_T("Failed to receive data, removing socket.\n"));
							LeaveCriticalSection(&me->cs_buffer);
							break; //close
						}
						else if (nBytesReceived == SOCKET_ERROR || nBytesReceived == 0)
						{
							LeaveCriticalSection(&me->cs_buffer);
							continue;
						}

						//read handler
						EnterCriticalSection(&me->cs_io);
						if (me->m_pIO)
							me->m_pIO->OnRead(me->m_socket, me->m_pBuffer, nBytesReceived);
						LeaveCriticalSection(&me->cs_io);
						LeaveCriticalSection(&me->cs_buffer);
					}

					if (networkEvents.lNetworkEvents & FD_WRITE)
					{
						me->m_bWriteReady = true;
					}

					if (networkEvents.lNetworkEvents & FD_CLOSE)
					{
						me->RemoveSocket();
						break;
					}

					me->SendBuffered();
				}

			}

			me->RemoveSocket();
			ZQ_Logger::Instance()->Text(_T("Socket closed!\n"));
			return 0;
		}

		void SendBuffered(void)
		{
			
			while (true)
			{
				EnterCriticalSection(&cs_io);
				if (m_pIO == 0)
				{
					LeaveCriticalSection(&cs_io);
					break;
				}

				UINT socketId;
				SOCKET sock;
				UINT nBytes2Send;
				char* buffer = 0;
				if (!m_pIO->OnSendBuffered(&sock, &socketId /*ignored*/, &nBytes2Send, &buffer, &m_socket, &m_bWriteReady, 1))
				{
					LeaveCriticalSection(&cs_io);
					break;
				}

				if (!m_bWriteReady || nBytes2Send == 0 || !buffer)
				{
					LeaveCriticalSection(&cs_io);
					if(buffer) free(buffer);
					buffer = 0;
					continue;
				}

				int lenSent = send(m_socket, buffer, nBytes2Send, 0);
				//printf("!\n");
				if (SOCKET_ERROR == lenSent)
				{
					if (WSAEWOULDBLOCK != WSAGetLastError())
					{
						ZQERR(_T("Failed to send data! Socket being removed!"));
						RemoveSocket();
					}
					else
					{
						m_bWriteReady = false;
						m_pIO->ReSend(m_socket, buffer, nBytes2Send);
						ZQWARNING(_T("WSAEWOULDBLOCK!"));
					}
				}
				LeaveCriticalSection(&cs_io);
				if (buffer) free(buffer);
				buffer = 0;
			}
		}

		void RemoveSocket(void)
		{
			EnterCriticalSection(&cs_io);
			if (m_pIO)
			{
				m_pIO->ClearAllReceivedPackets();
				m_pIO->ClearAllSendingPackets();
			}
			LeaveCriticalSection(&cs_io);
			EnterCriticalSection(&cs_socket);
			if (INVALID_SOCKET != m_socket)
			{
				shutdown(m_socket, SD_SEND);
				closesocket(m_socket);
				m_socket = INVALID_SOCKET;
			}
			LeaveCriticalSection(&cs_socket);
			if (WSA_INVALID_EVENT != m_hSocketEvent)
			{
				WSACloseEvent(m_hSocketEvent);
				m_hSocketEvent = WSA_INVALID_EVENT;
			}
			m_bConnected = false;

			
		}
	};
}

#endif