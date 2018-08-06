#ifndef _ZQ_PIXEL_FORMAT_H_
#define _ZQ_PIXEL_FORMAT_H_
#pragma once
namespace ZQ
{
	enum ZQ_PixelFormat {
		ZQ_PIXEL_FMT_GRAY = 0,
		ZQ_PIXEL_FMT_BGR,
		ZQ_PIXEL_FMT_RGB,
		ZQ_PIXEL_FMT_BGRX,
		ZQ_PIXEL_FMT_RGBX,
		ZQ_PIXEL_FMT_XBGR,
		ZQ_PIXEL_FMT_XRGB
	};
}

#endif