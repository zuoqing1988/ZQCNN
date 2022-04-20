#include "ZQ_CNN_SSDDetectorPytorch.h"
#include <opencv2/opencv.hpp>
#include "ZQ_CNN_CompileConfig.h"
#if ZQ_CNN_USE_BLAS_GEMM
#include <openblas/cblas.h>
#pragma comment(lib,"libopenblas.lib")
#elif ZQ_CNN_USE_MKL_GEMM
#include <mkl/mkl.h>
#pragma comment(lib,"mklml.lib")
#else
#pragma comment(lib,"ZQ_GEMM.lib")
#endif

using namespace ZQ;

void draw_objects(cv::Mat& bgr, const std::vector<ZQ_CNN_SSDDetectorUtils::BBox>& objects);

int main(int argc, const char** argv)
{
	if (argc != 2)
	{
		printf("%s image_file\n", argv[0]);
		return EXIT_FAILURE;
	}

	ZQ_CNN_SSDDetectorPytorch ssd;
	ssd.SetParam(0.5);
	ssd.SetShowDebugInfo(true);
	if (!ssd.LoadModel("model/model-face.zqparams", "model/model-face.nchwbin", "model/model-face.cfg"))
	{
		printf("failed to load model!\n");
		return EXIT_FAILURE;
	}

	cv::Mat image = cv::imread(argv[1], 1);
	cv::Mat input_image = image;
	if (ssd.UseGray())
	{
		if (image.channels() == 3)
			cv::cvtColor(image, input_image, cv::COLOR_BGR2GRAY);
	}
	else
	{
		if (image.channels() == 1)
			cv::cvtColor(image, input_image, cv::COLOR_GRAY2BGR);
	}
	
	std::vector<ZQ_CNN_SSDDetectorUtils::BBox> output;
	ssd.Detect(input_image.data, input_image.cols, input_image.rows, input_image.step[0], output);

	cv::Mat show_img;
	if (image.channels() == 3)
		image.copyTo(show_img);
	else
	{
		cv::cvtColor(image, show_img, cv::COLOR_GRAY2BGR);
	}

	draw_objects(show_img, output);
	cv::namedWindow("ssd output");
	cv::imshow("ssd output", show_img);
	cv::waitKey(0);
	return EXIT_SUCCESS;
}

void draw_objects(cv::Mat& image, const std::vector<ZQ_CNN_SSDDetectorUtils::BBox>& objects)
{
	static const char* class_names[] = { "background",
		"face"
	};

	for (size_t i = 0; i < objects.size(); i++)
	{
		const ZQ_CNN_SSDDetectorUtils::BBox& obj = objects[i];

		int x1 = obj.xmin + 0.5;
		int y1 = obj.ymin + 0.5;
		int x2 = obj.xmax + 0.5;
		int y2 = obj.ymax + 0.5;
		fprintf(stderr, "%d = %.5f at (%d, %d) (%d, %d)\n", obj.class_id, obj.prob,
			x1, y1, x2, y2);

		cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0));

		char text[256];
		sprintf(text, "%s %.1f%%", class_names[obj.class_id], obj.prob * 100);

		int baseLine = 0;
		cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

		int x = x1;
		int y = y1 - label_size.height - baseLine;
		if (y < 0)
			y = 0;
		if (x + label_size.width > image.cols)
			x = image.cols - label_size.width;

		cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
			cv::Scalar(255, 255, 255), -1);

		cv::putText(image, text, cv::Point(x, y + label_size.height),
			cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
	}
}
