#include "ZQ_CNN_MouthDetector.h"
#include "ZQ_CNN_ComplieConfig.h"
#if ZQ_CNN_USE_BLAS_GEMM
#include <cblas.h>
#pragma comment(lib,"libopenblas.lib")
#endif

using namespace cv;
using namespace std;
using namespace ZQ;

int SampleDetectMouth_cam(int argc, const char** argv);
int SampleDetectMouth_fig(int argc, const char** argv);

int main(int argc, const char** argv)
{
#if ZQ_CNN_USE_BLAS_GEMM
	openblas_set_num_threads(1);
#endif
	if (argc >= 2)
	{
		if (_strcmpi(argv[1], "cam") == 0)
		{
			return SampleDetectMouth_cam(argc,argv);
		}
		else if (_strcmpi(argv[1], "fig") == 0)
		{
			return SampleDetectMouth_fig(argc,argv);
		}
		else
		{
			return EXIT_FAILURE;
		}
	}
	return EXIT_FAILURE;
}

int SampleDetectMouth_cam(int argc, const char** argv)
{
	ZQ_CNN_MouthDetector detector;
	ZQ_CNN_MouthDetector::InitialArgs init_args;
	ZQ_CNN_MouthDetector::DetectArgs detect_args;
	ZQ_CNN_MouthDetector::DetectedResult detected_result;
	ZQ_CNN_MouthDetector::SimpleDetectedResult simple_detected_result;


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

int SampleDetectMouth_fig(int argc, const char** argv)
{
	int min_size = 60;
	float ssd_mouth_thresh = 0.5;
	if (argc < 5)
	{
		printf("%s %s model_root image_file out_file [min_size] [thresh] [draw_image_file]\n", min_size, ssd_mouth_thresh);
		return EXIT_FAILURE;
	}
	std::string model_root = argv[2];
	std::string image_file = argv[3];
	std::string out_file = argv[4];
	bool should_save_draw = false;
	std::string draw_image_file;
	if (argc > 5)
		min_size = __max(12, atoi(argv[5]));
	if (argc > 6)
		ssd_mouth_thresh = __max(0.15, atoi(argv[6]));
	if (argc > 7)
	{
		should_save_draw = true;
		draw_image_file = argv[7];
	}

	ZQ_CNN_MouthDetector detector;
	ZQ_CNN_MouthDetector::InitialArgs init_args;
	init_args.mtcnn_pnet_proto = model_root + "\\det1.zqparams";
	init_args.mtcnn_pnet_model = model_root + "\\det1.nchwbin";
	init_args.mtcnn_rnet_proto = model_root + "\\det2.zqparams";
	init_args.mtcnn_rnet_model = model_root + "\\det2.nchwbin";
	init_args.mtcnn_onet_proto = model_root + "\\det3.zqparams";
	init_args.mtcnn_onet_model = model_root + "\\det3.nchwbin";
	init_args.ssd_proto = model_root + "\\MobileNetSSD_deploy-face.zqparams";
	init_args.ssd_model = model_root + "\\MobileNetSSD_deploy-face.nchwbin";
	init_args.ssd_class_names_file = model_root + "\\MobileNetSSD_deploy-face.names";
	ZQ_CNN_MouthDetector::DetectArgs detect_args;
	ZQ_CNN_MouthDetector::DetectedResult detected_result;
	ZQ_CNN_MouthDetector::SimpleDetectedResult simple_detected_result;

	
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
	
	Mat image = cv::imread(image_file);


	if (image.empty()) 
	{
		cout << "failed to load image" << endl;
		return EXIT_FAILURE;
	}
	
	if (0)
	{
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
	}
	else
	{
		if (!detector.DetectSimpleResult(image.ptr<unsigned char>(0), image.cols, image.rows, image.step[0], &detect_args, &simple_detected_result))
		{
			//printf("failed to detect\n");
			FILE* out = 0;
			if (0 != fopen_s(&out, out_file.c_str(), "w"))
			{
				printf("failed to create file %s\n", out_file.c_str());
				return EXIT_FAILURE;
			}
			fprintf(out, "0\n");
			fclose(out);
			return EXIT_SUCCESS;
		}

		if (should_save_draw)
		{
			detector.DrawSimpleResult(image.ptr<unsigned char>(0), image.cols, image.rows, image.step[0], &simple_detected_result);
			if (!cv::imwrite(draw_image_file, image))
			{
				printf("failed to save %s\n", draw_image_file.c_str());
				return EXIT_FAILURE;
			}
		}
		
		FILE* out = 0;
		if (0 != fopen_s(&out, out_file.c_str(), "w"))
		{
			printf("failed to create file %s\n", out_file.c_str());
			return EXIT_FAILURE;
		}
		
		int count = simple_detected_result.faces.size();
		fprintf(out, "%d\n", count);

		for (int i = 0; i < count; i++)
		{
			ZQ_CNN_MouthDetector::SimpleFaceInfo& face = simple_detected_result.faces[i];
			fprintf(out, "%d %d %d %d %f\n", face.face_off_x, face.face_off_y, face.face_width, face.face_height, face.mouth_prob);
		}
		fclose(out);
	}

	return EXIT_SUCCESS;

}