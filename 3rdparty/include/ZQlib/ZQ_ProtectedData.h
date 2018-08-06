//////////////////////////////////////////////////////////////////////
// The same as code :
// ProtectedData.h
//
// SHEN Fangyang
// me@shenfy.com
//
// Copyright (C) SHEN Fangyang, 2010, All rights reserved.
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
//  Brief Description
//
//	Thread-safe, managed data with OnChange callback
//
//////////////////////////////////////////////////////////////////////
#ifndef _ZQ_PROTECTED_DATA_H_
#define _ZQ_PROTECTED_DATA_H_
#pragma once
#include <windows.h>

namespace ZQ
{
	template<typename T>
	class ZQ_ProtectedData
	{
	public:
		ZQ_ProtectedData(void);
		virtual ~ZQ_ProtectedData(void);

		virtual void OnValChange(T valOld, T valNew) = 0;
		const T& GetValue(void);

		//overload operators
#define _ZQ_PROTECTEDDATA_UNARY_OPERATOR_DECL(op) \
	inline ZQ_ProtectedData<T> &operator op##= (const T& rhs);

		_ZQ_PROTECTEDDATA_UNARY_OPERATOR_DECL(+)
			_ZQ_PROTECTEDDATA_UNARY_OPERATOR_DECL(-)
			_ZQ_PROTECTEDDATA_UNARY_OPERATOR_DECL(*)
			_ZQ_PROTECTEDDATA_UNARY_OPERATOR_DECL(/)

#undef _ZQ_PROTECTEDDATA_UNARY_OPERATOR_DECL

	protected:
		T m_data;
		CRITICAL_SECTION m_cs;
	};

	template<typename T>
	ZQ_ProtectedData<T>::ZQ_ProtectedData()
	{
		InitializeCriticalSection(&m_cs);
	}

	template<typename T>
	ZQ_ProtectedData<T>::~ZQ_ProtectedData()
	{
		DeleteCriticalSection(&m_cs);
	}

	//overload operators
#define _ZQ_PROTECTEDDATA_UNARY_OPERATOR(op) \
	template<typename T> \
	inline ZQ_ProtectedData<T> &ZQ_ProtectedData<T>::operator op##= (const T& rhs) \
	{ \
	T oldVal = m_data; \
	EnterCriticalSection(&m_cs); \
	m_data op##= rhs; \
	this->OnValChange(oldVal, m_data); \
	LeaveCriticalSection(&m_cs); \
	return (*this); \
	}

	_ZQ_PROTECTEDDATA_UNARY_OPERATOR(+)
		_ZQ_PROTECTEDDATA_UNARY_OPERATOR(-)
		_ZQ_PROTECTEDDATA_UNARY_OPERATOR(*)
		_ZQ_PROTECTEDDATA_UNARY_OPERATOR(/)

#undef _ZQ_PROTECTEDDATA_UNARY_OPERATOR

	template<typename T>
	const T &ZQ_ProtectedData<T>::GetValue()
	{
		return m_data;
	}
}

#endif