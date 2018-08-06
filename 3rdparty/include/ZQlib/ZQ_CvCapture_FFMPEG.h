
#ifndef _ZQ_CV_CAPTURE_FFMPEG_H_
#define _ZQ_CV_CAPTURE_FFMPEG_H_

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libswscale/swscale.h"
}

#include "math.h"
#include "ZQ_DoubleImage.h"

#pragma comment(lib, "avcodec.lib")
#pragma comment(lib, "avformat.lib  ")
#pragma comment(lib, "avutil.lib    ")
#pragma comment(lib, "avdevice.lib  ")
#pragma comment(lib, "avfilter.lib  ")
#pragma comment(lib, "postproc.lib  ")
#pragma comment(lib, "swresample.lib")
#pragma comment(lib, "swscale.lib   ")

namespace ZQ
{
	enum
	{
		ZQ_CV_FFMPEG_CAP_PROP_POS_MSEC = 0,
		ZQ_CV_FFMPEG_CAP_PROP_POS_FRAMES = 1,
		ZQ_CV_FFMPEG_CAP_PROP_POS_AVI_RATIO = 2,
		ZQ_CV_FFMPEG_CAP_PROP_FRAME_WIDTH = 3,
		ZQ_CV_FFMPEG_CAP_PROP_FRAME_HEIGHT = 4,
		ZQ_CV_FFMPEG_CAP_PROP_FPS = 5,
		ZQ_CV_FFMPEG_CAP_PROP_FOURCC = 6,
		ZQ_CV_FFMPEG_CAP_PROP_FRAME_COUNT = 7
	};

	struct ZQ_Image_FFMPEG
	{
		unsigned char* data;
		int step;
		int width;
		int height;
		int cn;
	};

	/* B-frame is !NOT! supported*/
	class ZQ_CvCapture_FFMPEG
	{
	public:
		ZQ_CvCapture_FFMPEG(const char* filename)
		{
			_init();
			is_opened = _open(filename);
		}

		~ZQ_CvCapture_FFMPEG()
		{
			_close();
		}

		bool IsOpened(){ return is_opened; }

		/*!DONOT! release the pointer of fr */
		bool Read(ZQ_Image_FFMPEG& fr)
		{
			if (!GrabFrame())
				return false;
			if (!RetrieveFrame(fr))
				return false;
			return true;
		}

		bool GrabFrame()
		{
			bool valid = false;
			int got_picture;

			int count_errs = 0;
			const int max_number_of_attempts = 1 << 9;

			if (!ic || !video_st)  return false;

			if (ic->streams[video_stream]->nb_frames > 0 &&
				frame_number > ic->streams[video_stream]->nb_frames)
				return false;

			picture_pts = AV_NOPTS_VALUE;

			// get the next frame
			while (!valid)
			{
				av_free_packet(&packet);

				int ret = av_read_frame(ic, &packet);
				if (ret == AVERROR(EAGAIN)) continue;

				/* else if (ret < 0) break; */

				if (packet.stream_index != video_stream)
				{
					av_free_packet(&packet);
					count_errs++;
					if (count_errs > max_number_of_attempts)
						break;
					continue;
				}

				// Decode video frame

				avcodec_decode_video2(video_st->codec, picture, &got_picture, &packet);

				// Did we get a video frame?
				if (got_picture)
				{
					//picture_pts = picture->best_effort_timestamp;
					if (picture_pts == AV_NOPTS_VALUE)
						picture_pts = packet.pts != AV_NOPTS_VALUE && packet.pts != 0 ? packet.pts : packet.dts;
					
					frame_number++;
					
					valid = true;
				}
				else
				{
					count_errs++;
					if (count_errs > max_number_of_attempts)
						break;
				}
			}

			if (valid && first_frame_number < 0)
				first_frame_number = _dts_to_frame_number(picture_pts);

			// return if we have a new picture or not
			return valid;
		}

		/*!DONOT! release the pointer of fr */
		bool RetrieveFrame(ZQ_Image_FFMPEG& fr)
		{
			if (!video_st || !picture->data[0])
				return false;

			if (img_convert_ctx == NULL ||
				frame.width != video_st->codec->width ||
				frame.height != video_st->codec->height ||
				frame.data == NULL)
			{
				// Some sws_scale optimizations have some assumptions about alignment of data/step/width/height
				// Also we use coded_width/height to workaround problem with legacy ffmpeg versions (like n0.8)
				int buffer_width = video_st->codec->coded_width, buffer_height = video_st->codec->coded_height;
				img_convert_ctx = sws_getCachedContext(
					img_convert_ctx,
					buffer_width, buffer_height,
					video_st->codec->pix_fmt,
					buffer_width, buffer_height,
					AV_PIX_FMT_BGR24,
					SWS_BICUBIC,
					NULL, NULL, NULL
					);

				if (img_convert_ctx == NULL)
					return false;//CV_Error(0, "Cannot initialize the conversion context!");

				int aligns[AV_NUM_DATA_POINTERS];
				avcodec_align_dimensions2(video_st->codec, &buffer_width, &buffer_height, aligns);
				rgb_picture.data[0] = (uint8_t*)realloc(rgb_picture.data[0],
					avpicture_get_size(AV_PIX_FMT_BGR24,
					buffer_width, buffer_height));
				avpicture_fill((AVPicture*)&rgb_picture, rgb_picture.data[0],
					AV_PIX_FMT_BGR24, buffer_width, buffer_height);

				frame.width = video_st->codec->width;
				frame.height = video_st->codec->height;
				frame.cn = 3;
				frame.data = rgb_picture.data[0];
				frame.step = rgb_picture.linesize[0];
			}

			sws_scale(
				img_convert_ctx,
				picture->data,
				picture->linesize,
				0, video_st->codec->coded_height,
				rgb_picture.data,
				rgb_picture.linesize
				);

			fr.data = frame.data;
			fr.step = frame.step;
			fr.width = frame.width;
			fr.height = frame.height;
			fr.cn = frame.cn;

			return true;
		}

		double GetProperty(int property_id)
		{
			if (!video_st) return 0;

			switch (property_id)
			{
			case ZQ_CV_FFMPEG_CAP_PROP_POS_MSEC:
				return 1000.0*(double)frame_number / _get_fps();
			case ZQ_CV_FFMPEG_CAP_PROP_POS_FRAMES:
				return (double)frame_number;
			case ZQ_CV_FFMPEG_CAP_PROP_POS_AVI_RATIO:
				return _r2d(ic->streams[video_stream]->time_base);
			case ZQ_CV_FFMPEG_CAP_PROP_FRAME_COUNT:
				return (double)_get_total_frames();
			case ZQ_CV_FFMPEG_CAP_PROP_FRAME_WIDTH:
				return (double)frame.width;
			case ZQ_CV_FFMPEG_CAP_PROP_FRAME_HEIGHT:
				return (double)frame.height;
			case ZQ_CV_FFMPEG_CAP_PROP_FPS:
				return av_q2d(video_st->r_frame_rate);
			case ZQ_CV_FFMPEG_CAP_PROP_FOURCC:
				return (double)video_st->codec->codec_tag;
			default:
				break;
			}

			return 0;
		}

		bool SetProperty(int property_id, double value)
		{
			if (!video_st) return false;
			switch (property_id)
			{
			case ZQ_CV_FFMPEG_CAP_PROP_POS_MSEC:
			case ZQ_CV_FFMPEG_CAP_PROP_POS_FRAMES:
			case ZQ_CV_FFMPEG_CAP_PROP_POS_AVI_RATIO:
				switch (property_id)
				{
				case ZQ_CV_FFMPEG_CAP_PROP_POS_FRAMES:
					Seek((int64_t)((long long)value));
					break;

				case ZQ_CV_FFMPEG_CAP_PROP_POS_MSEC:
					Seek(value / 1000.0);
					break;

				case ZQ_CV_FFMPEG_CAP_PROP_POS_AVI_RATIO:
					Seek((int64_t)(value*ic->duration));
					break;
				}

				picture_pts = (int64_t)value;

				break;
			default:
				return false;

			}

			return true;
		}

		void Seek(int64_t _frame_number)
		{
			//printf("frame_number = %d\n", _frame_number);
			_frame_number = __min(_frame_number, _get_total_frames());
			int delta = 16;

			// if we have not grabbed a single frame before first seek, let's read the first frame
			// and get some valuable information during the process
			if (first_frame_number < 0 && _get_total_frames() > 1)
				GrabFrame();

			for (;;)
			{
				int64_t _frame_number_temp = __max(_frame_number - delta, (int64_t)0);
				double sec = (double)_frame_number_temp / _get_fps();
				int64_t time_stamp = ic->streams[video_stream]->start_time;
				double  time_base = _r2d(ic->streams[video_stream]->time_base);
				time_stamp += (int64_t)(sec / time_base + 0.5);
				if (_get_total_frames() > 1) av_seek_frame(ic, video_stream, time_stamp, AVSEEK_FLAG_BACKWARD);
				avcodec_flush_buffers(ic->streams[video_stream]->codec);
				if (_frame_number > 0)
				{
					GrabFrame();

					if (_frame_number > 1)
					{
						frame_number = _dts_to_frame_number(picture_pts) - first_frame_number;
						//printf("_frame_number = %d, frame_number = %d, delta = %d\n",
						//       (int)_frame_number, (int)frame_number, delta);

						if (frame_number < 0 || frame_number > _frame_number - 1)
						{
							if (_frame_number_temp == 0 || delta >= INT_MAX / 4)
								break;
							delta = delta < 16 ? delta * 2 : delta * 3 / 2;
							continue;
						}
						while (frame_number < _frame_number - 1)
						{
							if (!GrabFrame())
								break;
						}
						frame_number++;
						break;
					}
					else
					{
						frame_number = 1;
						break;
					}
				}
				else
				{
					frame_number = 0;
					break;
				}
			}
		}

		void Seek(double sec)
		{
			Seek((int64_t)(sec * _get_fps() + 0.5));
		}
	
	private:
		bool is_opened;
		AVFormatContext * ic;
		AVCodec         * avcodec;
		int               video_stream;
		AVStream        * video_st;
		AVFrame         * picture;
		AVFrame           rgb_picture;
		int64_t           picture_pts;

		AVPacket          packet;
		ZQ_Image_FFMPEG      frame;
		struct SwsContext *img_convert_ctx;

		int64_t frame_number, first_frame_number;

		double eps_zero;
		///*
		//'filename' contains the filename of the videosource,
		//'filename==NULL' indicates that ffmpeg's seek support works
		//for the particular file.
		//'filename!=NULL' indicates that the slow fallback function is used for seeking,
		//and so the filename is needed to reopen the file on backward seeking.
		//*/
		//const char*		filename;
	private:
		bool _open(const char* filename)
		{
			_close();
			av_register_all();

			if (avformat_open_input(&ic, filename, NULL, NULL) < 0)
			{
				return false;
			}
			if (avformat_find_stream_info(ic, 0) < 0)
			{
				return false;
			}

			////////////////
			unsigned i;
			bool valid = false;


			for (i = 0; i < ic->nb_streams; i++)
			{
				AVCodecContext *enc = ic->streams[i]->codec;
				enc->thread_count = 1;// get_number_of_cpus();

				if (AVMEDIA_TYPE_VIDEO == enc->codec_type && video_stream < 0)
				{
					// backup encoder' width/height
					int enc_width = enc->width;
					int enc_height = enc->height;

					AVCodec *codec = avcodec_find_decoder(enc->codec_id);
					if (!codec || avcodec_open2(enc, codec, NULL) < 0)
						goto exit_func;

					// checking width/height (since decoder can sometimes alter it, eg. vp6f)
					if (enc_width && (enc->width != enc_width)) { enc->width = enc_width; }
					if (enc_height && (enc->height != enc_height)) { enc->height = enc_height; }

					video_stream = i;
					video_st = ic->streams[i];

					picture = av_frame_alloc();

					frame.width = enc->width;
					frame.height = enc->height;
					frame.cn = 3;
					frame.step = 0;
					frame.data = NULL;
					break;
				}
			}

			if (video_stream >= 0) valid = true;

		exit_func:


			if (!valid)
				_close();

			return valid;
		}

		void _close()
		{
			if (img_convert_ctx)
			{
				sws_freeContext(img_convert_ctx);
				img_convert_ctx = 0;
			}

			if (picture)
			{
				av_free(picture);
			}

			if (video_st)
			{
				avcodec_close(video_st->codec);
				video_st = NULL;
			}

			if (ic)
			{
				avformat_close_input(&ic);
				ic = NULL;
			}

			if (rgb_picture.data[0])
			{
				free(rgb_picture.data[0]);
				rgb_picture.data[0] = 0;
			}
			// free last packet if exist
			if (packet.data) {
				av_free_packet(&packet);
				packet.data = NULL;
			}

			_init();
		}

		void _init()
		{
			is_opened = false;
			ic = 0;
			video_stream = -1;
			video_st = 0;
			picture = 0;
			picture_pts = AV_NOPTS_VALUE;
			first_frame_number = -1;
			memset(&rgb_picture, 0, sizeof(rgb_picture));
			memset(&frame, 0, sizeof(frame));
			memset(&packet, 0, sizeof(packet));
			av_init_packet(&packet);
			img_convert_ctx = 0;

			avcodec = 0;
			frame_number = 0;
			eps_zero = 0.000025;
		}

		int64_t _get_total_frames()
		{
			int64_t nbf = ic->streams[video_stream]->nb_frames;

			if (nbf == 0)
			{
				nbf = (int64_t)floor(_get_duration_sec() * _get_fps() + 0.5);
			}
			return nbf;
		}

		double  _get_duration_sec()
		{
			double sec = (double)ic->duration / (double)AV_TIME_BASE;

			if (sec < eps_zero)
			{
				sec = (double)ic->streams[video_stream]->duration * _r2d(ic->streams[video_stream]->time_base);
			}

			if (sec < eps_zero)
			{
				sec = (double)ic->streams[video_stream]->duration * _r2d(ic->streams[video_stream]->time_base);
			}

			return sec;
		}

		double  _get_fps()
		{
			double fps = _r2d(ic->streams[video_stream]->r_frame_rate);
			if (fps < eps_zero)
			{
				fps = 1.0 / _r2d(ic->streams[video_stream]->codec->time_base);
			}

			return fps;
		}
		int     get_bitrate(){ return ic->bit_rate; }

		double  _r2d(AVRational r)
		{
			return r.num == 0 || r.den == 0 ? 0. : (double)r.num / (double)r.den;
		}
		int64_t _dts_to_frame_number(int64_t dts)
		{
			double sec = _dts_to_sec(dts);
			return (int64_t)(_get_fps() * sec + 0.5);
		}

		double  _dts_to_sec(int64_t dts)
		{
			return (double)(dts - ic->streams[video_stream]->start_time) *
				_r2d(ic->streams[video_stream]->time_base);
		}

	};
}

#endif