#include "SampleMouthDetector.h"
#include <cblas.h>

using namespace cv;
using namespace std;
using namespace ZQ;

int SampleDetectMouth_cam();
int SampleDetectMouth_fig();

int main(int argc, const char** argv)
{
	openblas_set_num_threads(1);
	if (argc == 2)
	{
		if (_strcmpi(argv[1], "cam") == 0)
		{
			return SampleDetectMouth_cam();
		}
		else if (_strcmpi(argv[1], "fig") == 0)
		{
			return SampleDetectMouth_fig();
		}
	}
	return EXIT_FAILURE;
}

int SampleDetectMouth_cam()
{
	SampleMouthDetector detector;
	SampleMouthDetector::InitialArgs init_args;
	SampleMouthDetector::DetectArgs detect_args;
	SampleMouthDetector::DetectedResult detected_result;
	SampleMouthDetector::SimpleDetectedResult simple_detected_result;


	detect_args.enable_rot = true;
	detect_args.mtcnn_min_size = 60;
	detect_args.ssd_mouth_thresh = 0.3;
	if (!detector.Initialize(&init_args))
	{
		cout << "init failed\n";
		return EXIT_FAILURE;
	}

	Mat image;
	VideoCapture cap(0);

	if (!cap.isOpened())
	{
		cout << "fail to open camera!" << endl;
		return EXIT_FAILURE;
	}

	//cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	//cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
	cap >> image;

	if (image.empty()) 
	{
		cout << "failed to load video" << endl;
		return EXIT_FAILURE;
	}

	double t0 = omp_get_wtime();

	int stop = 1200;
	int fr_id = 0;
	while (fr_id++ < stop) {
		cap >> image;

		if (image.empty())
		{
			cout << "empty image\n";
			break;
		}
		//cv::resize(image, image, Size(1920, 1080));
		double t1 = omp_get_wtime();
		printf("fr_id:[%05d], timecode: %.3f\n", fr_id, t1 - t0);
		/*if (detector.Detect(image.ptr<unsigned char>(0), image.cols, image.rows, image.step[0], &detect_args, &detected_result))
		{
			cout << "!!\n";
			double t2 = omp_get_wtime();
			detector.DrawResult(image.ptr<unsigned char>(0), image.cols, image.rows, image.step[0], &detected_result);
			double t3 = omp_get_wtime();
			printf("detect : %.3f s, draw: %.3f s\n", (t2 - t1), (t3 - t2));

		}*/
		if (detector.DetectSimpleResult(image.ptr<unsigned char>(0), image.cols, image.rows, image.step[0], &detect_args, &simple_detected_result))
		{
			double t2 = omp_get_wtime();
			detector.DrawSimpleResult(image.ptr<unsigned char>(0), image.cols, image.rows, image.step[0], &simple_detected_result);
			double t3 = omp_get_wtime();
			printf("detect : %.3f ms, draw: %.3f ms\n", 1000 * (t2 - t1), 1000 * (t3 - t2));
		}
		imshow("result", image);
		if (waitKey(1) >= 0) break;
	}
	image.release();
	return EXIT_SUCCESS;

}

int SampleDetectMouth_fig()
{
	SampleMouthDetector detector;
	SampleMouthDetector::InitialArgs init_args;
	SampleMouthDetector::DetectArgs detect_args;
	SampleMouthDetector::DetectedResult detected_result;
	SampleMouthDetector::SimpleDetectedResult simple_detected_result;

	
	detect_args.enable_rot = true;
	detect_args.mtcnn_min_size = 60;
	detect_args.mtcnn_thresh_p = 0.8;
	detect_args.mtcnn_thresh_r = 0.7;
	detect_args.mtcnn_thresh_o = 0.7;
	detect_args.ssd_mouth_thresh = 0.5;

	//detect_args.enlarge_border = 1;
	if (!detector.Initialize(&init_args))
	{
		cout << "init failed\n";
		return EXIT_FAILURE;
	}
	std::string input = "data\\4";
	Mat image = cv::imread(input + ".jpg");


	if (image.empty()) 
	{
		cout << "failed to load image" << endl;
		return EXIT_FAILURE;
	}
	Mat copy;
	for (int i = 0; i < 10; i++)
	{
		
		image.copyTo(copy);
		double t1 = omp_get_wtime();
		if (detector.Detect(copy.ptr<unsigned char>(0), copy.cols, copy.rows, copy.step[0], &detect_args, &detected_result))
			//if (detector.DetectSimpleResult(copy.ptr<unsigned char>(0), copy.cols, copy.rows, copy.step[0], &detect_args, &simple_detected_result))
		{
			double t2 = omp_get_wtime();
			detector.DrawResult(copy.ptr<unsigned char>(0), copy.cols, copy.rows, copy.step[0], &detected_result);
			//detector.DrawSimpleResult(copy.ptr<unsigned char>(0), copy.cols, copy.rows, copy.step[0], &simple_detected_result);
			double t3 = omp_get_wtime();
			printf("detect : %.3f ms, draw: %.3f ms\n", 1000 * (t2 - t1), 1000 * (t3 - t2));

		}
	}

	cv::imshow("show", copy);
	cv::waitKey(0);
	cv::imwrite(input + "_.jpg", copy);

	return EXIT_SUCCESS;

}