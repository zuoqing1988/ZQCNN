#ifndef _ZQ_WIN_SOCK_SERVER_H_
#define _ZQ_WIN_SOCK_SERVER_H_
#pragma once

#include "ZQ_WinSockBase.h"
#include "ZQ_Logger.h"
#include <process.h>

namespace ZQ
{
	class ZQ_EventSocketServerImpl : public ZQ_SocketBase
	{
	protected:
		ZQ_SocketIO *m_pIO;
		char *m_pSharedBuffer;
		UINT m_nSharedBufferSize;
		
		CRITICAL_SECTION cs_sockets; // first
		CRITICAL_SECTION cs_buffer; // second
		CRITICAL_SECTION cs_io; // third
		
		bool first_listen;
		WSAEVENT m_listenEvent;
		HANDLE m_hListenerThread, m_hIOThread;
		WSAEVENT m_socketEvents[WSA_MAXIMUM_WAIT_EVENTS];
		SOCKET m_listenSocket;
		SOCKET m_sockets[WSA_MAXIMUM_WAIT_EVENTS];
		UINT m_nSocket;
		bool listen_should_end;
		bool listen_has_ended;
		bool io_should_end;
		bool io_has_ended;
		HANDLE m_hStartIO;
		bool m_bWriteReady[WSA_MAXIMUM_WAIT_EVENTS];
		

	public:
		ZQ_EventSocketServerImpl(void) :
			m_pIO(0),
			m_pSharedBuffer(0),
			m_nSharedBufferSize(0),
			first_listen(true),
			m_listenEvent(WSA_INVALID_EVENT),
			m_listenSocket(INVALID_SOCKET),
			m_nSocket(0),
			listen_should_end(false),
			listen_has_ended(true),
			io_should_end(false),
			io_has_ended(true),
			m_hListenerThread(INVALID_HANDLE_VALUE),
			m_hIOThread(INVALID_HANDLE_VALUE),
			m_hStartIO(0)
		{
			for (UINT i = 0; i < WSA_MAXIMUM_WAIT_EVENTS; i++)
			{
				m_socketEvents[i] = WSA_INVALID_EVENT;
				m_sockets[i] = INVALID_SOCKET;
				m_bWriteReady[i] = false;
			}
			InitializeCriticalSection(&cs_io);
			InitializeCriticalSection(&cs_buffer);
			InitializeCriticalSection(&cs_sockets);
			ChangeBufferSize(1024 * 1024);
		}

		~ZQ_EventSocketServerImpl(void)
		{
			Stop();
			DeleteCriticalSection(&cs_io);
			DeleteCriticalSection(&cs_buffer);
			DeleteCriticalSection(&cs_sockets);	
		}

		void BindIOObj(ZQ_SocketIO *pObj)
		{
			EnterCriticalSection(&cs_io);
			m_pIO = pObj;
			LeaveCriticalSection(&cs_io);
		}

		bool ChangeBufferSize(const UINT size)
		{
			char *tmp;
			tmp = new char[size];
			if (!tmp)
				return false;

			EnterCriticalSection(&cs_buffer);
			if (m_pSharedBuffer)
				delete[]m_pSharedBuffer;
			m_pSharedBuffer = tmp;
			m_nSharedBufferSize = size;
			LeaveCriticalSection(&cs_buffer);
			return true;
		}

		bool StartListening(USHORT port)
		{
			Stop();
			sockaddr_in inetAddr;
			if (INVALID_SOCKET == (m_listenSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP)))
			{
				ZQERR(_T("Unable to create listening socket!"));
				return false;
			}

			inetAddr.sin_family = AF_INET;
			inetAddr.sin_addr.s_addr = htonl(INADDR_ANY);
			inetAddr.sin_port = htons(port);

			int eventId;
			eventId = bind(m_listenSocket, (PSOCKADDR)&inetAddr, sizeof(sockaddr_in));
			if (SOCKET_ERROR == eventId)
			{
				ZQERR(_T("Failed to bind socket to local port!"));
				return false;
			}

			if (WSA_INVALID_EVENT == (m_listenEvent = WSACreateEvent()))
			{
				ZQERR(_T("Failed to create sync event for listening socket!"));
				return false;
			}

			WSAEventSelect(m_listenSocket, m_listenEvent, FD_ACCEPT | FD_CLOSE);

			if (m_hStartIO != INVALID_HANDLE_VALUE && m_hStartIO != 0)
				CloseHandle(m_hStartIO);


			if (listen(m_listenSocket, SOMAXCONN) == SOCKET_ERROR)
				return false;
		
			listen_should_end = false;
			io_should_end = false;
			m_hListenerThread = (HANDLE)_beginthreadex(0, 0, ZQ_EventSocketServerImpl::IOThreadProc, this, 0, NULL);
			m_hIOThread = (HANDLE)_beginthreadex(0, 0, ZQ_EventSocketServerImpl::ListenerThreadProc, this, 0, NULL);
			if (INVALID_HANDLE_VALUE == m_hListenerThread || INVALID_HANDLE_VALUE == m_hIOThread)
			{
				ZQERR(_T("Failed to start server threads!\n"));
				Stop();
				return false;
			}
			ZQ_Logger::Instance()->Text(_T("Start listening...\n"));

			return true;
		}

		void Stop(void)
		{
			listen_should_end = true;
			io_should_end = true;
			HANDLE handles[2];
			handles[0] = m_hListenerThread;
			handles[1] = m_hIOThread;
			DWORD res;
			res = WaitForMultipleObjects(2, handles, TRUE, INFINITE);
			RemoveAllSockets();
			if (m_listenSocket != INVALID_SOCKET)
			{
				closesocket(m_listenSocket);
				m_listenSocket = INVALID_SOCKET;
			}
			if (WSA_INVALID_EVENT != m_listenEvent)
			{
				WSACloseEvent(m_listenEvent);
				m_listenEvent = WSA_INVALID_EVENT;
			}
		}

		bool isConnected(const SOCKET sock)
		{
			_scopeLock lock(&cs_sockets);
			for (UINT i = 0; i < m_nSocket; i++)
			if (sock == m_sockets[i])
				return true;
			return false;
		}

		void RemoveSocket(const SOCKET sock)
		{
			_scopeLock lock(&cs_sockets);
			for (int i = 0; i < m_nSocket; i++)
			{
				if (sock == m_sockets[i])
				{
					EnterCriticalSection(&cs_io);
					if (m_pIO)
						m_pIO->RemoveSocket(sock);
					LeaveCriticalSection(&cs_io);
					closesocket(sock);
					WSACloseEvent(m_socketEvents[i]);
					for (UINT iii = i; iii < m_nSocket - 1; iii++)
					{
						m_sockets[iii] = m_sockets[iii + 1];
						m_socketEvents[iii] = m_socketEvents[iii + 1];
					}

					m_nSocket--;
				}
			}
		}

		void RemoveAllSockets()
		{
			_scopeLock lock(&cs_sockets);
			for (int i = 0; i < m_nSocket; i++)
			{
				EnterCriticalSection(&cs_io);
				if (m_pIO)
					m_pIO->RemoveSocket(m_sockets[i]);
				LeaveCriticalSection(&cs_io);
				closesocket(m_sockets[i]);
				WSACloseEvent(m_socketEvents[i]);
			}
			m_nSocket = 0;
		}

	private:
		static UINT _stdcall ListenerThreadProc(void *arg)
		{
			ZQ_EventSocketServerImpl *me = (ZQ_EventSocketServerImpl *)arg;
			WSANETWORKEVENTS networkEvents;
			UINT eventId;
			HANDLE hListenEvents[1];
			hListenEvents[0] = me->m_listenEvent;
			
			SOCKET acceptSocket;

			me->listen_has_ended = false;
			while (!me->listen_should_end)
			{
				if (WSA_WAIT_FAILED ==
					(eventId = WSAWaitForMultipleEvents(1, hListenEvents, false, 5000, false)))
				{
					ZQERR(_T("Failed to wait for network events!"));
					return 1;
				}
				else if (WSA_WAIT_TIMEOUT == eventId)
				{
					continue;
				}

				if (eventId - WSA_WAIT_EVENT_0 == 1) //stop signal
				{
					break;
				}
				else if (eventId - WSA_WAIT_EVENT_0 == 0)
				{
					if (SOCKET_ERROR ==
						WSAEnumNetworkEvents(me->m_listenSocket, me->m_listenEvent, &networkEvents))
					{
						ZQERR(_T("Failed to enumerate network events!"));
						break;
					}

					//actually got incoming events
					if (networkEvents.lNetworkEvents & FD_ACCEPT)
					{
						if (INVALID_SOCKET == (acceptSocket = accept(me->m_listenSocket, NULL, NULL)))
						{
							ZQERR(_T("Failed to accept incoming connection!"));
							continue;
						}

						if (me->m_nSocket > WSA_MAXIMUM_WAIT_EVENTS)
						{
							ZQERR(_T("Maximum connection available reached, new connection declined!"));
							closesocket(acceptSocket);
							continue;
						}

						me->AddSocket(acceptSocket);

						ZQ_Logger::Instance()->Text(_T("Connection accepted!\n"));
					}

					if (networkEvents.lNetworkEvents & FD_CLOSE)
					{
						ZQ_Logger::Instance()->Text(_T("Listener closed!\n"));
						shutdown(me->m_listenSocket, SD_SEND);
						closesocket(me->m_listenSocket);
						WSACloseEvent(me->m_listenEvent);
						me->m_listenSocket = INVALID_SOCKET;
						me->m_listenEvent = WSA_INVALID_EVENT;
						break;
					}
				}
			}

			shutdown(me->m_listenSocket, SD_SEND);
			closesocket(me->m_listenSocket);
			WSACloseEvent(me->m_listenEvent);
			me->m_listenSocket = INVALID_SOCKET;
			me->m_listenEvent = WSA_INVALID_EVENT;
			me->listen_has_ended = true; 
			return 0;
		}

		static UINT _stdcall IOThreadProc(void *arg)
		{
			ZQ_EventSocketServerImpl *me = (ZQ_EventSocketServerImpl *)arg;
			UINT eventId;
			WSANETWORKEVENTS networkEvents;
			int nByteReceived;
			UINT index;
			me->io_has_ended = false;
			while (!me->io_should_end)
			{
				if (me->m_nSocket == 0)
				{
					Sleep(5);
					continue;
				}

				me->SendBuffered();
				
				//Wait for network events on all sockets
				if (WSA_WAIT_FAILED ==
					(eventId = WSAWaitForMultipleEvents(me->m_nSocket, me->m_socketEvents, false, 5, false)))
				{
					ZQERR(_T("Failed to wait for network events!"));
					ZQ_Logger::Instance()->LastError();
					continue;
				}
				else if (WSA_WAIT_TIMEOUT == eventId)
				{
					continue;
				}

				index = eventId - WSA_WAIT_EVENT_0;

				//Iterate through all possible sockets
				for (UINT socketId = index; socketId < me->m_nSocket; socketId++)
				{
					SOCKET sock = me->m_sockets[socketId];
					eventId = WSAWaitForMultipleEvents(1, &me->m_socketEvents[socketId], false, 5, false);

					if (WSA_WAIT_FAILED == eventId)
					{
						me->RemoveSocket(sock);
						socketId--;
						ZQERR(_T("Connection closed due to unexpected error!\n"));
						continue;
					}
					else if (WSA_WAIT_TIMEOUT == eventId)
					{
						continue;
					}

					if (SOCKET_ERROR ==
						WSAEnumNetworkEvents(me->m_sockets[socketId],
						me->m_socketEvents[socketId],
						&networkEvents))
					{
						ZQERR(_T("Failed to enumerate network events!"));
						printf("return5\n");
						continue;
					}

					if (networkEvents.lNetworkEvents & FD_READ)
					{
						EnterCriticalSection(&me->cs_buffer);
						nByteReceived = recv(me->m_sockets[socketId],
							me->m_pSharedBuffer,
							me->m_nSharedBufferSize,
							0);
						
						if (nByteReceived == 0)
						{
							LeaveCriticalSection(&me->cs_buffer);
							ZQERR(_T("receive data 0 bytes! Socket being removed!"));
							me->RemoveSocket(sock);
							socketId--;
							continue;
						}
						else if (nByteReceived == SOCKET_ERROR)
						{
							int errono = WSAGetLastError();
							if (errono == WSAEINTR)
							{
								LeaveCriticalSection(&me->cs_buffer);
								printf("inter\n");
								continue;
							}
							else
							{
								LeaveCriticalSection(&me->cs_buffer);
								printf("receive error\n");
								continue;
							}
						}

						EnterCriticalSection(&me->cs_io);
						if (me->m_pIO)
						{
							me->m_pIO->OnRead(me->m_sockets[socketId], me->m_pSharedBuffer, nByteReceived);
						}
						LeaveCriticalSection(&me->cs_io);
						LeaveCriticalSection(&me->cs_buffer);
					}

					if (networkEvents.lNetworkEvents & FD_WRITE)
					{
						me->m_bWriteReady[socketId] = true;
					}

					if (networkEvents.lNetworkEvents & FD_CLOSE)
					{
						me->RemoveSocket(sock);
						socketId--;
						ZQ_Logger::Instance()->Text(_T("Connection closed!\n"));
						continue;
					}
				} //end of socket event iteration
			}
			me->io_has_ended = true;
			return 0;
		}

	protected:
		void AddSocket(const SOCKET sock)
		{
			_scopeLock lock(&cs_sockets);
			for (int i = 0; i < m_nSocket; i++)
			{
				if (m_sockets[i] == sock)
					return;
			}
			WSAEVENT acceptEvent;
			acceptEvent = WSACreateEvent();
			WSAEventSelect(sock, acceptEvent, FD_READ | FD_WRITE | FD_CLOSE);
			m_socketEvents[m_nSocket] = acceptEvent;
			m_sockets[m_nSocket] = sock;
			m_nSocket++;
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
				char *buffer = 0;
				if (!m_pIO->OnSendBuffered(&sock, &socketId, &nBytes2Send, &buffer, m_sockets, m_bWriteReady, WSA_MAXIMUM_WAIT_EVENTS))
				{
					LeaveCriticalSection(&cs_io);
					break;
				}

				if (false == m_bWriteReady[socketId] || nBytes2Send == 0 || buffer == 0)
				{
					LeaveCriticalSection(&cs_io);
					if (buffer) free(buffer);
					buffer = 0;
					continue;
				}


				int lenSent = send(m_sockets[socketId], buffer, nBytes2Send, 0);
				if (SOCKET_ERROR == lenSent)
				{
					if (WSAEWOULDBLOCK != WSAGetLastError())
					{
						ZQERR(_T("Failed to send data! Socket being removed!"));
						RemoveSocket(sock);
					}
					else
					{
						m_bWriteReady[socketId] = false;
						m_pIO->ReSend(socketId, buffer, nBytes2Send);
						TCHAR errMsg[50];
						_stprintf_s(errMsg, _T("WSAEWOULDBLOCK on socket %d!"), socketId);
						ZQWARNING(errMsg);
					}
				}
				
				LeaveCriticalSection(&cs_io);
				if (buffer) free(buffer);
			}
		}
	private:
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
}

#endif