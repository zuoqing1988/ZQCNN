#ifndef _ZQ_WIN_SOCK_BASE_H_
#define _ZQ_WIN_SOCK_BASE_H_
#pragma once

#include <winsock2.h>
#include <Ws2tcpip.h>

namespace ZQ
{
	class ZQ_SocketBase
	{
	public:
		static bool InitWinsock(void)
		{

			WSADATA wsaData;
			int err;

			err = WSAStartup(MAKEWORD(2, 2), &wsaData);
			return (err == 0);
		}
		
		static void ShutdownWinsock(void)
		{
			WSACleanup();
		}
	};

	class ZQ_SocketIO
	{
	public:
		ZQ_SocketIO(void) {}
		~ZQ_SocketIO(void) {}

		///////////////////////////////////////////////////////
		// INTERFACES FOR THE SOCKET SERVER CLASSES
		///////////////////////////////////////////////////////

		//called when data arrives at the server
		virtual bool OnRead(/*_in_*/ const SOCKET sock, /*_in_*/ const char *buffer, /*_in_*/ UINT nBytesReceived) = 0;

		//called when FD_WRITE occurs at the server
		virtual bool OnWrite(/*_in_*/ const SOCKET sock, /*_out_*/ UINT *nBytesToSend, /*_out_*/ char **pBuffer) = 0;

		/*called when the server wants to send buffered data.
		caller should free(buffer)
		*/
		virtual bool OnSendBuffered(/*_out_*/ SOCKET *sock, /*_out_*/ UINT *nBytesToSend, /*_out_*/ char **buffer) = 0;

		/*called when the server wants to send buffered data.
		caller should free(buffer)
		*/
		virtual bool OnSendBuffered(/*_out_*/ SOCKET *sock, /*_out_*/ UINT *socket_id, /*_out_*/ UINT *nBytesToSend, /*_out_*/ char **buffer,
			/*_in_*/ const SOCKET* allsocks, /*_in_*/ const bool *writeReady, /*_in_*/ UINT nSocket) = 0;

		//called when a socket disconnects
		virtual void RemoveSocket(/*_in_*/ const SOCKET sock) = 0;

		///////////////////////////////////////////////////////
		// INTERFACES FOR USER
		///////////////////////////////////////////////////////
		//called by the user to send data normally
		virtual bool Send(/*_in_*/ const SOCKET sock, /*_in_*/ const char *buffer, /*_in_*/ UINT nBytesToSend) = 0; //append to the end of the send list
		virtual bool ReSend(/*_in_*/ const SOCKET sock, /*_in_*/ const char *buffer, /*_in_*/ UINT nBytesToSend) = 0; //insert before the front of the send list

		//blocking reader function, should be called in a while(1){} loop
		virtual bool ConsumePacket(/*_out_*/ SOCKET *sock, /*_out_*/ char **buf, /*_out_*/ UINT *nBytes, /*_in_*/ UINT waittime) = 0;

		virtual void ClearAllSendingPackets() = 0;
		virtual void ClearAllReceivedPackets() = 0;
	};

}

#endif