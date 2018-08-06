#ifndef _ZQ_TEXTURE_SYNTHESIS_PROBE_H_
#define _ZQ_TEXTURE_SYNTHESIS_PROBE_H_
#pragma once 

#include <stdlib.h>

namespace ZQ
{
	class ZQ_TextureSynthesisProbe
	{
	public:
		ZQ_TextureSynthesisProbe(const int width, const int height, const float random_factor = 0.1)
		{
			Reset(width,height,random_factor);
		}
		~ZQ_TextureSynthesisProbe(){}

	private:
		int counter;
		int width;
		int height;
		float random_factor;
		int max_random_count;

	public:
		void Reset()
		{
			counter = 0;
		}
		void Reset(const int width, const int height, const float random_factor = 0.1)
		{
			counter = 0;
			this->width = width;
			this->height = height;
			this->random_factor = random_factor;
			max_random_count = random_factor * width * height;
		}
		void Next(int& x, int& y, bool& completed)
		{
			y = counter / width;
			x = counter % width;
			if(counter >= width*height)
				completed = true;
			else
				completed = false;
			counter ++;
		}
		void NextRandom(int& x, int& y, bool& completed)
		{
			int i = rand()%(width*height);
			y = i / width;
			x = i % width;
			if (counter >= max_random_count)
				completed = true;
			else
				completed = false;
			counter++;
		}
	};
}


#endif