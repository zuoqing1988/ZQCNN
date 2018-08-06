#ifndef _ZQ_WIN_SOCK_IO_POOL_SIMPLE_H_
#define _ZQ_WIN_SOCK_IO_POOL_SIMPLE_H_
#pragma once

#include "ZQ_WinSockBase.h"
#include "ZQ_Logger.h"
#include "ZQ_SemaphoreEx.h"
#include <map>
#include <list>
#include <tchar.h>

namespace ZQ
{
	class ZQ_WinSockIOPoolSimple : public ZQ_SocketIO
	{
		enum BUFFLEN{ BufLen = 1024 };

	protected:
		struct _Package
		{
			_Package(char *_p, UINT _length, SOCKET _sock, UINT _id)
			: p(_p), length(_length), sock(_sock), id(_id)
			{

			}
			char* p;
			UINT length;
			SOCKET sock;
			UINT id;
		};

		struct _BuffOffset
		{
			_BuffOffset(UINT _pos, UINT _length) : pos(_pos), length(_length) {}

			UINT pos;
			UINT length;
		};

		typedef std::map<SOCKET, char *> BuffMap;
		typedef std::map<SOCKET, _BuffOffset> BuffOffsetMap;
		typedef std::list<_Package> PackageList;

		class _scopeLock
		{
		public:
			_scopeLock(CRITICAL_SECTION* cs)
			{
				m_cs = cs;
				EnterCriticalSection(m_cs);
			}
			~_scopeLock()
			{
				LeaveCriticalSection(m_cs);
			}
			CRITICAL_SECTION* m_cs;
		};

	protected:
		CRITICAL_SECTION cs_receiving; // first
		CRITICAL_SECTION cs_received; // second
		CRITICAL_SECTION cs_sending; //third

		BuffMap m_recvBufs; //temporary buffers for each socket
		BuffOffsetMap m_recvBuffOffset; //offset info of temporary buffers
		PackageList m_recvPackages; //memory list for received data packets
		PackageList m_sendPackages; //memory list for data waiting to send
	
	public:
		struct ProtocalHeader
		{
			unsigned int length;
		};
		ZQ_WinSockIOPoolSimple()
		{
			InitializeCriticalSection(&cs_receiving);
			InitializeCriticalSection(&cs_received);
			InitializeCriticalSection(&cs_sending);
		}

		~ZQ_WinSockIOPoolSimple()
		{
			ClearAllSendingPackets();
			ClearAllReceivedPackets();

			if(1)
			{
				_scopeLock lock(&cs_receiving);
				for (BuffMap::iterator iter = m_recvBufs.begin(); iter != m_recvBufs.end(); ++iter)
				{
					char* tmpBuf = iter->second;
					if (tmpBuf)
					{
						free(tmpBuf);
					}
				}
				m_recvBufs.clear();
				m_recvBuffOffset.clear();
			}

			DeleteCriticalSection(&cs_receiving);
			DeleteCriticalSection(&cs_received);
			DeleteCriticalSection(&cs_sending);
		}

		virtual bool OnRead(/*_in_*/ const SOCKET sock, /*_in_*/ const char *buffer, /*_in_*/ UINT nBytesReceived)
		{
			_scopeLock lock(&cs_receiving);

			if (!buffer || !nBytesReceived)
				return false;

			char *recvBuf = GetRecvBuffer(sock); //allocate buffer if necessary
			if (!recvBuf)
				return false;

			BuffOffsetMap::iterator iter = m_recvBuffOffset.find(sock);
			UINT curPos = (iter->second).pos;
			UINT curLength = (iter->second).length;

			UINT header_size = sizeof(ProtocalHeader);

			ProtocalHeader *header = (ProtocalHeader*)recvBuf;
			UINT packetLength;
			UINT restLength;
			UINT nNewChunk;

			//iterate through all packets arrived
			while (1)
			{
				//incomplete header, incoming size unknown
				if (curPos < header_size)
				{
					if (curPos + nBytesReceived < header_size) //still incomplete
					{
						memcpy(recvBuf + curPos, buffer, nBytesReceived);
						curPos += nBytesReceived;
						iter->second.pos = curPos;
						break;
					}
					else //complete header
					{
						//fill header struct first
						memcpy(recvBuf + curPos, buffer, header_size - curPos);
						//parse packet length
						packetLength = header->length;
						nNewChunk = (packetLength + BufLen - 1) / BufLen;
						//reallocate memory if overflow
						if (packetLength > curLength)
						{
							recvBuf = ReAllocRecvBuffer(sock, nNewChunk);
							curLength = nNewChunk * BufLen;
						}
					}
				}
				//complete header, incoming size known
				else
				{
					packetLength = header->length;
					nNewChunk = (packetLength + BufLen - 1) / BufLen;
				}

				if (curPos + nBytesReceived > packetLength)
				{
					//fill packet
					restLength = packetLength - curPos;
					memcpy(recvBuf + curPos, buffer, restLength);
					//write to pool
					WriteToPool(sock, recvBuf, packetLength);
					//reset variables for the next loop
					nBytesReceived -= restLength;
					curPos = 0;
					iter->second.pos = 0;
					buffer += restLength;
					//next loop
					continue;
				}
				else if (curPos + nBytesReceived == packetLength)
				{
					//fill packet
					memcpy(recvBuf + curPos, buffer, nBytesReceived);
					//write to pool
					WriteToPool(sock, recvBuf, packetLength);
					//reset variable for next read
					iter->second.pos = 0;
					break;
				}
				else //curPos + nBytesReceived < packetLength
				{
					memcpy(recvBuf + curPos, buffer, nBytesReceived);
					iter->second.pos = curPos + nBytesReceived;
					break;
				}
			}// end of parsing loop
			return true;
		}

		//called when FD_WRITE occurs at the server
		virtual bool OnWrite(/*_in_*/ const SOCKET sock, /*_out_*/ UINT *nBytesToSend, /*_out_*/ char **pBuffer){ return true; }

		//called when the server wants to send buffered data
		virtual bool OnSendBuffered(/*_out_*/ SOCKET *sock, /*_out_*/ UINT *nBytesToSend, /*_out_*/ char **buffer)
		{
			_scopeLock lock(&cs_sending);

			if (m_sendPackages.size() == 0)
				return false;

			PackageList::iterator iter;
			while (m_sendPackages.size() != 0) //find a valid chunk
			{
				iter = m_sendPackages.begin();
				if (0 != iter->length) //not empty
					break;

				//empty, invalid chunk
				if (iter->p)
				{
					free(iter->p);
				}
				m_sendPackages.pop_front();
			}

			bool fRet;
			switch (m_sendPackages.size())
			{
			case 0: //empty already
				return false;
			case 1: //last one
			{
						//fRet = false;
						fRet = true;
			}
				break;
			default: //there are more
			{
						 fRet = true;
			}
				break;
			}

			*sock = iter->sock;
			*nBytesToSend = iter->length;
			*buffer = iter->p;
			//free pool chunk
			m_sendPackages.pop_front();
			return fRet;
		}

		virtual bool OnSendBuffered(/*_out_*/ SOCKET *sock, /*_out_*/ UINT *socket_id, /*_out_*/ UINT *nBytesToSend, /*_out_*/ char **buffer,
			/*_in_*/ const SOCKET* allsocks, /*_in_*/ const bool *writeReady, /*_in_*/ UINT nSocket)
		{
			_scopeLock lock(&cs_sending);

			if (m_sendPackages.size() == 0 || writeReady == 0)
				return false;

			PackageList::iterator iter;
			size_t size = m_sendPackages.size();
			while (size != 0) //erase invalid chunks
			{
				iter = m_sendPackages.begin();
				if (0 != iter->length && 0 != iter->p) //not empty
					break;

				//empty, invalid chunk
				if (iter->p)
				{
					free(iter->p);
				}
				m_sendPackages.pop_front();
				size = m_sendPackages.size();
			}

			if (m_sendPackages.size() == 0) //no valid chunk left
				return false;

			std::map<SOCKET, UINT> sock_to_id_map;
			std::map<SOCKET, UINT>::iterator sock_to_id_it;
			int sockid;
			for (UINT i = 0; i < nSocket; i++)
				sock_to_id_map.insert(std::make_pair(allsocks[i], i));

			//count writable chunks
			bool fRet = false;
			for (iter = m_sendPackages.begin(); iter != m_sendPackages.end(); ++iter)
			{
				sock_to_id_it = sock_to_id_map.find(iter->sock);
				if (sock_to_id_it != sock_to_id_map.end())
				{
					sockid = sock_to_id_it->second;
					if (writeReady[sockid] == true)
					{
						fRet = true;
						break;
					}
				}
			}
			if (!fRet) return fRet; //no writable chunks left

			*sock = iter->sock;
			*socket_id = sockid;
			*nBytesToSend = iter->length;
			*buffer = iter->p;
			m_sendPackages.erase(iter);
			return fRet;
		}

		//called when a socket disconnects
		virtual void RemoveSocket(/*_in_*/ const SOCKET sock)
		{
			if (1)
			{
				_scopeLock lock(&cs_receiving);
				BuffMap::iterator iter = m_recvBufs.find(sock);
				if (iter != m_recvBufs.end())
				{
					if (iter->second)
						free(iter->second);
					m_recvBufs.erase(iter);
				}
				BuffOffsetMap::iterator iter_off = m_recvBuffOffset.find(sock);
				if (iter_off != m_recvBuffOffset.end())
				{
					m_recvBuffOffset.erase(iter_off);
				}
			}

			if (1)
			{
				_scopeLock lock(&cs_received);
				for (PackageList::iterator list_iter = m_recvPackages.begin(); list_iter != m_recvPackages.end();)
				{
					if (list_iter->sock == sock)
					{
						char* tmpBuf = list_iter->p;
						free(tmpBuf);
						list_iter = m_recvPackages.erase(list_iter);
					}
					else
					{
						++list_iter;
					}
				}
			}

			if (1)
			{
				_scopeLock lock(&cs_sending);

				for (PackageList::iterator list_iter = m_sendPackages.begin(); list_iter != m_sendPackages.end();)
				{
					if (list_iter->sock == sock)
					{
						char* tmpBuf = list_iter->p;
						free(tmpBuf);
						list_iter = m_sendPackages.erase(list_iter);
					}
					else
					{
						++list_iter;
					}
				}
			}
		}


		///////////////////////////////////////////////////////
		// INTERFACES FOR USER
		///////////////////////////////////////////////////////
		//called by the user to send data normally
		//append to the end of the send list
		virtual bool Send(/*_in_*/ const SOCKET sock, /*_in_*/ const char *buffer, /*_in_*/ UINT nBytesToSend)
		{
			_scopeLock lock(&cs_sending);

			if (0 == nBytesToSend) return false;

			char *tmpBuf = (char*)malloc(nBytesToSend);
			if (0 == tmpBuf)
			{
				ZQERR(_T("Failed to allocate pool memory!"));
				return false;
			}

			memcpy(tmpBuf, buffer, nBytesToSend);
			static UINT send_id = 0;
			send_id++;
			//printf("s_id:%d\n",send_id);
			m_sendPackages.push_back(_Package(tmpBuf, nBytesToSend, sock, send_id));
			return true;
		}

		//insert before the front of the send list
		virtual bool ReSend(/*_in_*/ const SOCKET sock, /*_in_*/ const char *buffer, /*_in_*/ UINT nBytesToSend)
		{
			_scopeLock lock(&cs_sending);

			if (0 == nBytesToSend) return false;
			char *tmpBuf = (char*)malloc(nBytesToSend);
			if (0 == tmpBuf)
			{
				ZQERR(_T("Failed to allocate pool memory!"));
				return false;
			}
			memcpy(tmpBuf, buffer, nBytesToSend);
			static UINT resend_id = 0;
			resend_id++;
			m_sendPackages.push_front(_Package(tmpBuf, nBytesToSend, sock, resend_id));
			return true;
		}

		//blocking reader function, should be called in a while(1){} loop
		virtual bool ConsumePacket(/*_out_*/ SOCKET *sock, /*_out_*/ char **buf, /*_out_*/ UINT *nBytes, /*_in_*/ UINT waittime)
		{
			_scopeLock lock(&cs_received);

			PackageList::iterator iter = m_recvPackages.begin();
			if (iter == m_recvPackages.end())
			{
				return false;
			}

			*nBytes = iter->length;
			*sock = iter->sock;
			*buf = iter->p;
			m_recvPackages.pop_front();
			return true;
		}

		virtual void ClearAllSendingPackets()
		{
			_scopeLock lock(&cs_sending);
			for (PackageList::iterator iter = m_sendPackages.begin(); iter != m_sendPackages.end(); ++iter)
			{
				char* tmpBuf = iter->p;
				if (tmpBuf)
					free(tmpBuf);
			}
			m_sendPackages.clear();

		}

		virtual void ClearAllReceivedPackets()
		{
			_scopeLock lock(&cs_received);
			for (PackageList::iterator iter = m_recvPackages.begin(); iter != m_recvPackages.end(); ++iter)
			{
				char* tmpBuf = iter->p;
				if (tmpBuf)
					free(tmpBuf);
			}
			m_recvPackages.clear();
		}

		virtual void ClearAllPackets()
		{
			ClearAllReceivedPackets();
			ClearAllSendingPackets();
		}

	protected:
		char *GetRecvBuffer(const SOCKET sock) //allocate if not already exists
		{
			_scopeLock lock(&cs_receiving);
			BuffMap::iterator iter = m_recvBufs.find(sock);
			if (iter != m_recvBufs.end()) //already exists
				return iter->second;
			else //allocate new
			{
				char *tmpBuf = (char*)malloc(BufLen);
				if (!tmpBuf)
				{
					TCHAR msg[70];
					_stprintf_s(msg, _T("Failed to allocate reception buffer for incoming socket !"));
					ZQERR(msg);
					return 0;
				}

				m_recvBufs.insert(std::make_pair(sock, tmpBuf));
				m_recvBuffOffset.insert(std::make_pair(sock, _BuffOffset(0, BufLen)));
				return tmpBuf;
			}
		}

		char *ReAllocRecvBuffer(const SOCKET sock, UINT n) //extend the length of a certain receive buffer to (n * BufLen)
		{
			_scopeLock lock(&cs_receiving);
			BuffMap::iterator iterBuf = m_recvBufs.find(sock);
			BuffOffsetMap::iterator iterMeta = m_recvBuffOffset.find(sock);

			if (iterBuf == m_recvBufs.end())//doesn't exist
			{
				char *tmpBuf = (char*)malloc(n * BufLen);
				if (!tmpBuf)
				{
					TCHAR msg[70];
					_stprintf_s(msg, _T("Failed to allocate reception buffer for incoming socket !"));
					ZQERR(msg);
					return 0;
				}

				m_recvBufs.insert(std::make_pair(sock, tmpBuf));
				m_recvBuffOffset.insert(std::make_pair(sock, _BuffOffset(0, n*BufLen)));
				return tmpBuf;
			}
			else //already exists
			{
				char *oldBuf = iterBuf->second;
				UINT pos = iterMeta->second.pos;

				char *newBuf = (char*)malloc(n * BufLen);
				if (newBuf == 0)
				{
					TCHAR msg[70];
					_stprintf_s(msg, _T("Failed to allocate reception buffer for incoming socket!"));
					ZQERR(msg);
					return 0;
				}

				//preserve old content
				if (pos != 0)
					memcpy(newBuf, oldBuf, pos);
				free(oldBuf);
				iterBuf->second = newBuf;
				iterMeta->second.length = n * BufLen;
				return newBuf;
			}
		}

		bool WriteToPool(const SOCKET sock, const char *buf, UINT length)
		{
			_scopeLock lock(&cs_received);
			char *tmpBuf = (char*)malloc(length);
			if (tmpBuf == 0)
			{
				ZQERR(_T("Failed to allocate pool memory!"));
				return false;
			}

			memcpy(tmpBuf, buf, length);

			static UINT count_id = 0;
			count_id++;
			//printf("w_id %d: %p,%d\n",count_id,tmpBuf,length);
			m_recvPackages.push_back(_Package(tmpBuf, length, sock, count_id));

			return true;
		}

	
	};

}


#endif