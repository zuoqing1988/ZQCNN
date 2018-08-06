//////////////////////////////////////////////////////////////////////
//The same code as:
// SemaphoreEx.h
//
// SHEN Fangyang
// me@shenfy.com
//
// Copyright (C) SHEN Fangyang, 2010, All rights reserved.
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
//  Brief Description
//
//	Extended semaphore supporting +-*/ on its value
//
//////////////////////////////////////////////////////////////////////
#ifndef _ZQ_SEMAPHOREEX_H_
#define _ZQ_SEMAPHOREEX_H_
#pragma once

#include "ZQ_ProtectedData.h"
#include "ZQ_Logger.h"
#include <tchar.h>

namespace ZQ
{
	class ZQ_SemaphoreEx : public ZQ_ProtectedData<int>
	{
	public:
		ZQ_SemaphoreEx(void)
		{
			if (0 == (m_hEvent = CreateEvent(NULL, TRUE, FALSE, NULL)))
			{
				ZQERR(_T("Failed to create event for _CProtectedData object!"));
			}
		}

		ZQ_SemaphoreEx(const int& val)
		{
			m_data = val;
			if (0 == (m_hEvent = CreateEvent(NULL, TRUE, FALSE, NULL)))
			{
				ZQERR(_T("Failed to create event for _CProtectedData object!"));
			}
		}

		~ZQ_SemaphoreEx(void)
		{
			if (m_hEvent)
				CloseHandle(m_hEvent);
		}


		virtual void OnValChange(int oldVal, int newVal)
		{
			if (oldVal <= 0 && newVal > 0)
				SetEvent(m_hEvent);
			else if (oldVal > 0 && newVal <= 0)
				ResetEvent(m_hEvent);
		}

		ZQ_SemaphoreEx &operator= (const int& rhs)
		{
			if (m_data != rhs)
			{
				int oldVal = m_data;
				EnterCriticalSection(&m_cs);
				m_data = rhs;
				OnValChange(oldVal, rhs);
				LeaveCriticalSection(&m_cs);
			}
			return (*this);
		}

		void Release(int n = 1)
		{
			int oldVal = m_data;
			EnterCriticalSection(&m_cs);
			m_data -= n;
			OnValChange(oldVal, m_data);
			LeaveCriticalSection(&m_cs);
		}

		inline operator HANDLE() {return m_hEvent;}

	protected:
		HANDLE m_hEvent;
	};

}

#endif