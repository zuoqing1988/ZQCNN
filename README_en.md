English | [中文](README.md)

# Introduction

ZQCNN is an inference framework that can run on Windows, Linux and ARM-Linux. It also includes some face detection and recognition related demos.

## Build and run (since 2026-10)

CMake is the only build entry point (the VS .sln/.vcxproj files were removed):

```shell
# Windows (VS2022)
cmake -S . -B build_x64 -G"Visual Studio 17 2022" -A x64
cmake --build build_x64 --config Release -j8
cd cmake-out-win32-x64/release/Release

# Linux
mkdir -p build && cd build && cmake .. && make -j8
```

* The output directory gets the `data/`, `model/` links and the third-party dlls, so samples run there directly.
* **MKL is no longer a default dependency** (ZQCNN never calls `cblas_*`); edit `ZQ_CNN_USE_MKL_GEMM` in `ZQCNN/ZQ_CNN_CompileConfig.h` to switch back.
* `imshow`/`waitKey` are commented out in all samples, so nothing blocks.
* Only the MTCNN/SSD weights ship in `model/`; other samples need the Model Zoo downloads.
* `SamplesZQGEMM` compares the hand-written assembly GEMM against the intrinsic one; `SamplesZQBLAS/SampleGEMMCompare` additionally loads MKL / OpenBLAS at runtime.

## Main Development Environment: [VS2015 with Update 3](https://pan.baidu.com/s/1zoREccOxVsggV-iI2z4HTg)

  MKL Download: [Download Here](https://pan.baidu.com/s/1d75IIf6fgTZ5oeumd0vtTw)

## Core Modules Support Linux:

  If you cannot compile completely following [build-with-cmake.md](https://github.com/zuoqing1988/ZQCNN/blob/master/build-with-cmake.md), you can only compile ZQ_GEMM, ZQCNN, and other programs you want to test

## Core Modules Support ARM-Linux:

  If you cannot compile completely following [build-with-cmake.md](https://github.com/zuoqing1988/ZQCNN/blob/master/build-with-cmake.md), you can only compile ZQ_GEMM, ZQCNN, and other programs you want to test
  
**BUG:** cmake .. -DSIMD_ARCH_TYPE=arm64 -DBLAS_TYPE=openblas_zq_gemm 

In ideal case, the faster one between OpenBLAS and ZQ_GEMM will be used for convolution computation (I choose the branch by testing time on Cortex-A72). However, this option cannot achieve the expected effect currently. You need to manually comment in ZQ_CNN_CompileConfig.h:
  
	#define ZQ_CNN_USE_ZQ_GEMM 1
	#define ZQ_CNN_USE_BLAS_GEMM 1
	
You can comment out:
  
	line 67: #if defined(ZQ_CNN_USE_BOTH_BLAS_ZQ_GEMM)
	line 70: #endif

## Training Related

  PyTorch Training SSD: https://github.com/zuoqing1988/pytorch-ssd-for-ZQCNN

  Training Gender & Age: https://github.com/zuoqing1988/train-GenderAge
	
  Training MTCNN: https://github.com/zuoqing1988/train-mtcnn
	
  Training SSD: https://github.com/zuoqing1988/train-ssd
	
  Training MTCNN for Head Detection: https://github.com/zuoqing1988/train-mtcnn-head


# Update Log

**Update on Aug 18, 2022: Optimized 106-point facial landmark pipeline in video mode, added a new head pose and gaze estimation model**

Demo program is in SampleVideoFaceDetection_Interface.cpp

The original 106-point pb model and head pose gaze pb model are in TensorFlow_to_ZQCNN

**Update on Apr 20, 2022: Support SSD models trained with pytorch-ssd-for-ZQCNN**

**Update on May 8, 2020: Added text recognition sample SampleOCR**

	No text detection capability yet, input images need to be pre-cropped
	
	Model download address
	
	Link: https://pan.baidu.com/s/1O75LRBjXWwPXqAshLMJV3w Extraction code: f2q8
	
**Update on Mar 22, 2020: Provide MTCNN models that can detect faces with masks**

	model\det1-dw20-plus.zqparams
	model\det1-dw20-plus.nchwbin

	model\det2-dw24-p0.zqparams	
	model\det2-dw24-p0.nchwbin
	
	model\det3-dw48-p0.zqparams
	model\det3-dw48-p0.nchwbin

**Update on Jul 8, 2019: Code for converting ZQCNN models to MNN models**

[Click to read](https://github.com/zuoqing1988/ZQCNN/tree/master/ZQCNN_to_MNN)

**Update on May 28, 2019: Open source a quasi-commercial 106-point model**

ZQCNN format: in model folder det5-dw112

mxnet format: Link: https://pan.baidu.com/s/19DTG3rmkct8AiEu0l3DYjw Code: qjzk 


**Update on Mar 16, 2019: Reached 800 stars, release more accurate 106-point landmark model**

[ZQCNN format:det5-dw96-v2s](https://github.com/zuoqing1988/ZQCNN/tree/master/model) in model folder det5-dw96-v2s.zqparams, det5-dw96-v2s.nchwbin

[mxnet format:Lnet106_96_v2s](https://pan.baidu.com/s/1iuuAHgJBsdWsUoAdU5H58Q) Extraction code: r5h2

**Update on Feb 14, 2019: Reached 700 stars, release selected face detection models**

[ZQCNN format: Selected 6 Pnet, 2 Rnet, 2 Onet, 2 Lnet](https://pan.baidu.com/s/1X2U9Y-6MJw3md8WuYxaotw)

| Six Pnet Types                                                  | Input Size   | Computation (excluding bbox) | Notes                        |
| --------                                                        | ------       | ------------                 | --------------------         |
| [Pnet20_v00](https://pan.baidu.com/s/1g7JnOxnbXIbNWPXGI-IzrQ)   | 320x240      | 8.5 M                        | Benchmark: libfacedetection  |
| [Pnet20_v0](https://pan.baidu.com/s/1r3VcmEX1a2C5gKlGKnC4kw)    | 320x240      | 11.6 M                       | Benchmark: libfacedetection  |
| [Pnet20_v1](https://pan.baidu.com/s/1qVU3_nporbOUzXYu7giZkA)    | 320x240      | 14.6 M                       |                              |
| [Pnet20_v2](https://pan.baidu.com/s/1bXzdmsTgfqU_TJHsozSmrQ)    | 320x240      | 18.4 M                       | Benchmark: original pnet     |
| [Pnet16_v0](https://pan.baidu.com/s/1s5eZLeAKnqp1ZDTrzaOD_w)    | 256x192      | 7.5 M                        | stride=4                     |
| [Pnet16_v1](https://pan.baidu.com/s/1Lf0z6rRq5WUKE_DMze_C7w)    | 256x192      | 9.8 M                        | stride=4                     |

| Two Rnet Types                                                | Input Size   | Computation      | Notes                        |
| --------                                                      | ------       | ------------     | --------------------         |
| [Rnet_v1](https://pan.baidu.com/s/1SEIolnvmtPvdqbHxU1vPWQ)    | 24x24        | 0.5 M            | Benchmark: original Rnet     |
| [Rnet_v2](https://pan.baidu.com/s/1APWYGcFC5MAn6Ba5vWo80w)    | 24x24        | 1.4 M            |                              |

| Two Onet Types                                                | Input Size   | Computation      | Notes                        |
| --------                                                      | ------       | ------------     | --------------------         |
| [Onet_v1](https://pan.baidu.com/s/1UTvSKErOul2wkT5EMxXgVA)    | 48x48        | 2.0 M            | No landmark                  |
| [Onet_v2](https://pan.baidu.com/s/19QomSIy3Py516OEIBFDcVg)    | 48x48        | 3.2 M            | No landmark                  |

| Two Lnet Types                                                | Input Size   | Computation      | Notes                        |
| --------                                                      | ------       | ------------     | --------------------         |
| [Lnet_v2](https://pan.baidu.com/s/1W6bxNeD0psxwxbou_xwK-g)    | 48x48        |  3.5 M           | lnet_basenum=16              |
| [Lnet_v2](https://pan.baidu.com/s/1e3tuwrR3AoU_zRKkIFK8xg)    | 48x48        | 10.8 M           | lnet_basenum=32              |

**Update on Jan 31, 2019: Reached 600 stars, release MTCNN head detection model**

Trained on hollywoodheads data, the effect is average, barely usable

Head detection mtcnn-head [mxnet-v0](https://pan.baidu.com/s/11I-ZnW3AAijlijtroyxClQ)&[zqcnn-v0](https://pan.baidu.com/s/1Xh27qm_LmuV6ZIDLBUXfPQ)



**Update on Jan 24, 2019: Core modules support Linux**

If you cannot compile completely following [build-with-cmake.md](https://github.com/zuoqing1988/ZQCNN/blob/master/build-with-cmake.md), you can only compile ZQ_GEMM, ZQCNN, and other programs you want to test

**Update on Jan 17, 2019**

Modified ZQ_CNN_MTCNN.h

(1) When thread_num is set to less than 1 during init, Pnet_stage can be forced to run multi-threaded, which will be divided into blocks, preventing memory explosion for large images with small faces

(2) The size of rnet/onet/lnet can be other than 24/48/48, but only supports equal width and height

(3) rnet/onet/lnet batch processing can reduce memory usage when there are many faces

**Update on Jan 15, 2019: Celebrate reaching 500 stars, release 106-point landmark model**

[mxnet format & zqcnn format](https://pan.baidu.com/s/18VTMfChnAEyeU_9vE9GJaw)


**Update on Jan 4, 2019: Celebrate reaching 400 stars, release fast face model**

[mxnet format](https://pan.baidu.com/s/1pOvAaXncbarNfD0G-4BwlQ)

[zqcnn format](https://pan.baidu.com/s/18FLOduY4SoHjXHBCXWQ5LQ)

v3 version is not good enough, v4 version will be released later, roughly as shown in the diagram below

![MTCNN-v4 Diagram](https://github.com/zuoqing1988/ZQCNN/blob/master/mtcnn%E7%A4%BA%E6%84%8F%E5%9B%BE.jpg)

**~~Update on Dec 25, 2018: Not open-sourced 106-point landmark~~**

~~Life is quite tight, making some extra money.~~

~~landmark106-normal-1000.jpg is the landmark generated by model\det5-dw48-1000.nchwbin~~
	
~~landmark106-normal.jpg and landmark106-big.jpg are two models I trained that are not open-sourced~~
	
~~The normal model is 2.1M, computation 11.4M, PC single-thread takes 0.6-0.7ms, big model is 7.56M, computation 36.4M, PC single-thread takes 1.5-1.6ms~~

**Update on Dec 20, 2018: Add MTCNN 106-point landmark model**

Try it in SampleMTCNN (the released one is not very good, better ones are waiting to be sold)

SampleLnet106 has timing, single-thread takes about 0.6~0.7ms (E5-1650V4, 3.6GHz)

**Update on Dec 3, 2018: Compile models into code**

model2code in SamplesZQCNN can compile models into code (build it with CMake, see build-with-cmake.md)

	model2code.exe param_file model_file code_file prefix
	
Then add to your project

	#include"code_file"
	
Use the following function to load the model

	LoadFromBuffer(prefix_param, prefix_param_len, prefix_model, prefix_model_len)


**Update on Nov 21, 2018**

Support mxnet-ssd trained models, mean_val needs to be set to 127.5 to run correctly in SampleSSD.

But training with ReLU seems incorrect. I trained one with PReLU from scratch, only mAP=0.48, barely usable, [Click to download](https://pan.baidu.com/s/1-wfpuvGLBGPtlqicdO1raw).

After changing the model, you must first train a classification model with ImageNet, and then train SSD to improve mAP.

**Update on Nov 14, 2018**

(1) Optimized ZQ_GEMM, on a 3.6GHz machine MKL peak is about 46GFLOPS, ZQ_GEMM is about 32GFLOPS. Using ZQ_GEMM face model overall time is about 1.5 times that of using MKL.

Note: ZQ_GEMM compiled with VS2017 is faster than VS2015, but SampleMTCNN multi-threaded run is wrong (maybe different OpenMP support rules?).

(2) Very small weights can be removed when loading the model. When you find the model is much slower than expected, it is mostly caused by too small weight values.

**Update on Nov 6, 2018**

(1) Removed all omp multi-threaded code in layers, computation is too small, slower than single-threaded

(2) cblas_gemm can choose MKL, but the mkl in 3rdparty is very slow on my machine, the dll is quite large, I didn't put it in 3rdparty\bin, please [download here](https://pan.baidu.com/s/1d75IIf6fgTZ5oeumd0vtTw).

**Update 2 on Oct 30, 2018: MTCNN recommend using Gaussian filter first for finding small faces in large images**

**Update on Oct 30, 2018: BatchNorm eps issue**

(1) The default eps for BatchNorm and BatchNormScale are both 0

(2) If using mxnet2zqcnn to convert models from mxnet, eps will be added to var as the new var during conversion

(3) If converting models from other platforms, either manually add eps to var, or add eps=? after BatchNorm and BatchNormScale (? is the eps value of this layer on that platform)

Note: To prevent division by zero, when dividing by var it is calculated as sqrt(__max(var+eps,1e-32)), which means if var+eps is less than 1e-32, it will be slightly different from the theoretical value.
However, after today's modification, the LFW accuracy of the following face models is exactly the same as minicaffe's results.

**Update on Oct 26, 2018**

MTCNN supports multi-threading. For large images with small faces and many faces, 8 threads can achieve more than 4 times the effect of single thread. Please test with data\test2.jpg

**Update on Oct 15, 2018**

Improved MTCNN's nms strategy: 1. The local maximum of nms for each scale's Pnet must cover a certain number of non-maximum, the number is set in the parameters; 2. When Pnet's resolution is too large, nms performs block processing.

**Update on Sep 25, 2018**

Support insightface's GNAP, automatic model conversion using mxnet2zqcnn, see [mxnet2zqcnn](https://github.com/zuoqing1988/ZQCNN-v0.0/wiki/mxnet2zqcnn). You can try [MobileFaceNet-GNAP](https://pan.baidu.com/s/1hv4lbYwSLlLiGK07FuJM5Q)

**Update on Sep 20, 2018**

(1) Updated the test method for tar-far accuracy of face recognition models. You can follow the steps [How-to-evaluate-TAR-FAR-on-your-dataset](https://github.com/zuoqing1988/ZQCNN-v0.0/wiki/How-to-evaluate-TAR-FAR-on-your-dataset) to construct a test set and test model accuracy.

(2) According to (1), I cleaned CASIA-Webface and constructed two test sets [webface1000X50](https://pan.baidu.com/s/1AoJkj_IhydkiyD1UGm8rDQ), [webface5000X20](https://pan.baidu.com/s/1AoJkj_IhydkiyD1UGm8rDQ), and tested the accuracy of several main face recognition models I open-sourced.

**Update on Sep 13, 2018**

(1) Support loading models from memory

(2) Added compilation configuration ZQ_CNN_CompileConfig.h, you can choose whether to use _mm_fmadd_ps, _mm256_fmadd_ps (you can test the speed to see if it is faster or slower).

**Update on Sep 12, 2018: Steps for training 112*96 (sphereface size) using [insightface](https://github.com/deepinsight/insightface):** [InsightFace: how to train 112*96](https://github.com/zuoqing1988/ZQCNN-v0.0/wiki/InsightFace%EF%BC%9A-how-to-train-112*96)

**Update on Aug 15, 2018**

(1) Added natural scene text detection, model converted from [TextBoxes](https://github.com/MhLiao/TextBoxes). Personally, I think the speed is too slow and accuracy is not high.

Note that the PriorBoxLayer used in this project is different from the PriorBoxLayer in SSD. To export ZQCNN format weights, I modified deploy.prototxt and saved it as deploy_tmp.prototxt.
Download the model from [here](https://pan.baidu.com/s/1XOREgRzyimx_AMC9bg8MgQ).

(2) Added NSFW image detection, model converted from [open_nsfw](https://github.com/yahoo/open_nsfw), I haven't tested how accurate it is.

Download the model from [here](https://pan.baidu.com/s/1asjZFr3iTliQ4xlNbtKUtw).

**Update on Aug 10, 2018**

Successfully converted [GenderAge-r50 model](https://pan.baidu.com/s/1f8RyNuQd7hl2ItlV-ibBNQ) and [Arcface-LResNet100E-IR](https://pan.baidu.com/s/1wuRTf2YIsKt76TxFufsRNA) from mxnet, same steps as converting MobileFaceNet model.
See [mxnet2zqcnn](https://github.com/zuoqing1988/ZQCNN-v0.0/wiki/mxnet2zqcnn)

The Model Zoo below has models I converted, which should be slightly faster than automatically converted ones.

Build with CMake and run SampleGenderAge to see the effect. On my E5-1650V4 CPU, single-thread time fluctuates greatly, average is about 1900-2000ms, four threads is over 400ms.

**Update on Aug 9, 2018**

Added mxnet2zqcnn, successfully converted MobileFaceNet from mxnet to ZQCNN format (cannot guarantee other models can be converted successfully, ZQCNN does not support many layers yet). See [mxnet2zqcnn](https://github.com/zuoqing1988/ZQCNN-v0.0/wiki/mxnet2zqcnn)

**Update on Aug 7, 2018**

Bug Fix: Previously Convolution, DepthwiseConvolution, InnerProduct, BatchNormScale/Scale default with_bias=true, now changed to default with_bias=false. That is, previous code cannot load these layers without bias.

Example: As shown below, a Layer like this would default to having bias_term before, now defaults to no bias_term

Convolution name=conv1 bottom=data top=conv1 num_output=10 kernel_size=3 stride=1 

**Update on Aug 6, 2018**

Added face recognition accuracy testing on LFW database. The related samples are in the SamplesZQlibFaceID directory.

Since the calculation accuracy of C++ code is slightly different from matlab, the calculated accuracy also has some differences, but the difference is within 0.1%.

**Update on Aug 3, 2018**

Support multi-threading (accelerated through openmp). **Please note, currently multi-threading is slower than single-threading**

**Update on Jul 26, 2018**

Support MobileNet-SSD. For caffemodel conversion, refer to export_mobilenet_SSD_caffemodel_to_nchw_binary.m. You need to compile matcaffe.
You can try this version [caffe-ZQ](https://github.com/zuoqing1988/caffe-ZQ)

**Update on Jun 5, 2018**

Follow the trend, release source code.
Forgot to mention that it depends on openblas. I directly used the version in mini-caffe, the one compiled by myself is very slow.



# Model Zoo

**Face Detection**

[MTCNN-author-version](https://pan.baidu.com/s/1lWLKDYv8YQ6Th6KRiKvgug) Converted format from [MTCNN](https://github.com/kpzhang93/MTCNN_face_detection_alignment)

[MTCNN-ZQ-version](https://pan.baidu.com/s/1j1WqkwbUCf_9f4hCQukoFg)

**Face Recognition (unless otherwise specified, models are trained with ms1m-refine-v2)**

|Model                                                                                |LFW Accuracy (ZQCNN)                                 | LFW Accuracy (OpenCV3.4.2)                        | LFW Accuracy (minicaffe)                         |Time (ZQCNN)                       |Notes           
|------------                                                                         | -------------                                       |----------------------                             | ------------                                     |---------------------              | ------------- 
|[MobileFaceNet-res2-6-10-2-dim128](https://pan.baidu.com/s/1AQEad5Zp2cag4UA5KtpbYQ) |99.67%-99.55%(matlab crop), 99.72-99.60%(C++ crop)  |99.63%-99.65%(matlab crop), 99.68-99.70%(C++ crop) |99.62%-99.65%(matlab crop), 99.68-99.60%(C++ crop)|Time similar to dim256             |Network structure same as dim256, only output dimension differs
|[MobileFaceNet-res2-6-10-2-dim256](https://pan.baidu.com/s/143j7eULc2AqpNcSugFdTxA) |99.60%-99.60%(matlab crop), 99.62-99.62%(C++ crop)  |99.73%-99.68%(matlab crop), 99.78-99.68%(C++ crop) |99.55%-99.63%(matlab crop), 99.60-99.62%(C++ crop)|Single-thread ~21-22ms, four threads ~11-12ms, 3.6GHz |Network structure in download link, trained with faces_emore 
|[MobileFaceNet-res2-6-10-2-dim512](https://pan.baidu.com/s/1_0O3kJ5dMmD-HdRwNR0Hpw) |99.52%-99.60%(matlab crop), 99.63-99.72%(C++ crop)  |99.70%-99.67%(matlab crop), 99.77-99.77%(C++ crop) |99.55%-99.62%(matlab crop), 99.62-99.68%(C++ crop)|Time similar to dim256             |Network structure same as dim256, only output dimension differs. Thanks to [moli](https://github.com/moli232777144) for training this model

|Model                                                                                |LFW Accuracy (ZQCNN)                                 | LFW Accuracy (OpenCV3.4.2)                        | LFW Accuracy (minicaffe)                         |Time (ZQCNN)                       |Notes           
|------------                                                                         | -------------                                       |----------------------                             | ------------                                     |---------------------              | ------------- 
|[MobileFaceNet-res4-8-16-4-dim128](https://pan.baidu.com/s/1z6H5p4b3aVun2-1dZGDXkg) |99.72%-99.72%(matlab crop), 99.72-99.68%(C++ crop)  |99.82%-99.83%(matlab crop), 99.80-99.78%(C++ crop) |99.72%-99.72%(matlab crop), 99.72-99.68%(C++ crop)|Time similar to dim256             |Network structure same as dim256, only output dimension differs
|[MobileFaceNet-res4-8-16-4-dim256](https://pan.baidu.com/s/1f_VtqNRxDNe972h8UrOsPw) |99.78%-99.78%(matlab crop), 99.75-99.75%(C++ crop)  |99.82%-99.82%(matlab crop), 99.80-99.82%(C++ crop) |99.78%-99.78%(matlab crop), 99.73-99.73%(C++ crop)|Single-thread ~32-33ms, four threads ~16-19ms, 3.6GHz |Network structure in download link, trained with faces_emore 
|[MobileFaceNet-res4-8-16-4-dim512](https://pan.baidu.com/s/14ukmtAWDhIJC6312WBhZhA) |99.80%-99.73%(matlab crop), 99.85-99.83%(C++ crop)  |99.83%-99.82%(matlab crop), 99.87-99.83%(C++ crop) |99.80%-99.73%(matlab crop), 99.85-99.82%(C++ crop)|Time similar to dim256             |Network structure same as dim256, only output dimension differs. Thanks to [moli](https://github.com/moli232777144) for training this model

|Model\Test Set webface1000X50                                                      |thresh@ FAR=1e-7|TAR@ FAR=1e-7|thresh@ FAR=1e-6|TAR@ FAR=1e-6|thresh@ FAR=1e-5|TAR@ FAR=1e-5
|------------                                                                       | -------------  | ----------  |--------------- |-------      | ------------   |-----------                         
|[MobileFaceNet-res2-6-10-2-dim128](https://pan.baidu.com/s/1AQEad5Zp2cag4UA5KtpbYQ)|0.78785         |9.274%       |0.66616         |40.459%      |0.45855         |92.716%
|[MobileFaceNet-res2-6-10-2-dim256](https://pan.baidu.com/s/143j7eULc2AqpNcSugFdTxA)|0.77708         |7.839%       |0.63872         |40.934%      |0.43182         |92.605%
|[MobileFaceNet-res2-6-10-2-dim512](https://pan.baidu.com/s/1_0O3kJ5dMmD-HdRwNR0Hpw)|0.76699         |8.197%       |0.63452         |38.774%      |0.41572         |93.000%
|[MobileFaceNet-res4-8-16-4-dim128](https://pan.baidu.com/s/1z6H5p4b3aVun2-1dZGDXkg)|0.79268         |9.626%       |0.65770         |48.252%      |0.45431         |95.576%
|[MobileFaceNet-res4-8-16-4-dim256](https://pan.baidu.com/s/1f_VtqNRxDNe972h8UrOsPw)|0.76858         |9.220%       |0.62852         |46.195%      |0.40010         |96.929%
|[MobileFaceNet-res4-8-16-4-dim512](https://pan.baidu.com/s/14ukmtAWDhIJC6312WBhZhA)|0.76287         |9.296%       |0.62555         |44.775%      |0.39047         |97.347%

|Model\Test Set webface5000X20                                                      |thresh@ FAR=1e-7|TAR@ FAR=1e-7|thresh@ FAR=1e-6|TAR@ FAR=1e-6|thresh@ FAR=1e-5|TAR@ FAR=1e-5
|------------                                                                       | -------------  | ----------  |--------------- |-------      | ------------   |-----------                         
|[MobileFaceNet-res2-6-10-2-dim128](https://pan.baidu.com/s/1AQEad5Zp2cag4UA5KtpbYQ)|0.70933         |29.558%      |0.51732         |85.160%      |0.45108         |94.313%
|[MobileFaceNet-res2-6-10-2-dim256](https://pan.baidu.com/s/143j7eULc2AqpNcSugFdTxA)|0.68897         |28.376%      |0.48820         |85.278%      |0.42386         |94.244%
|[MobileFaceNet-res2-6-10-2-dim512](https://pan.baidu.com/s/1_0O3kJ5dMmD-HdRwNR0Hpw)|0.68126         |27.708%      |0.47260         |85.840%      |0.40727         |94.632%
|[MobileFaceNet-res4-8-16-4-dim128](https://pan.baidu.com/s/1z6H5p4b3aVun2-1dZGDXkg)|0.71238         |32.153%      |0.51391         |89.525%      |0.44667         |96.583%
|[MobileFaceNet-res4-8-16-4-dim256](https://pan.baidu.com/s/1f_VtqNRxDNe972h8UrOsPw)|0.68490         |30.639%      |0.46092         |91.900%      |0.39198         |97.696%
|[MobileFaceNet-res4-8-16-4-dim512](https://pan.baidu.com/s/14ukmtAWDhIJC6312WBhZhA)|0.67303         |32.404%      |0.45216         |92.453%      |0.38344         |98.003%

|Model\Test Set TAO ids:6606,ims:87210                                              |thresh@ FAR=1e-7|TAR@ FAR=1e-7|thresh@ FAR=1e-6|TAR@ FAR=1e-6|thresh@ FAR=1e-5|TAR@ FAR=1e-5
|------------                                                                       | -------------  |-------------|--------------- |-------------| -------------- |-----------                         
|[MobileFaceNet-res2-6-10-2-dim128](https://pan.baidu.com/s/1AQEad5Zp2cag4UA5KtpbYQ)|0.92204         |01.282%      |0.88107         |06.837%      |0.78302         |41.740%
|[MobileFaceNet-res2-6-10-2-dim256](https://pan.baidu.com/s/143j7eULc2AqpNcSugFdTxA)|0.91361         |01.275%      |0.86750         |07.081%      |0.76099         |42.188%
|[MobileFaceNet-res2-6-10-2-dim512](https://pan.baidu.com/s/1_0O3kJ5dMmD-HdRwNR0Hpw)|0.90657         |01.448%      |0.86061         |07.299%      |0.75488         |41.956%
|[MobileFaceNet-res4-8-16-4-dim128](https://pan.baidu.com/s/1z6H5p4b3aVun2-1dZGDXkg)|0.92098         |01.347%      |0.88233         |06.795%      |0.78711         |41.856%
|[MobileFaceNet-res4-8-16-4-dim256](https://pan.baidu.com/s/1f_VtqNRxDNe972h8UrOsPw)|0.90862         |01.376%      |0.86397         |07.083%      |0.75975         |42.430%
|[MobileFaceNet-res4-8-16-4-dim512](https://pan.baidu.com/s/14ukmtAWDhIJC6312WBhZhA)|0.90710         |01.353%      |0.86190         |06.948%      |0.75518         |42.241%


|Model\Test Set ZQCNN-Face_5000_X_20                                                     |thresh@ FAR=1e-8|TAR@ FAR=1e-8|thresh@ FAR=1e-7|TAR@ FAR=1e-7|thresh@ FAR=1e-6|TAR@ FAR=1e-6
|------------                                                                             | -------------  | ----------  |--------------- |-------      | ------------   |-----------                         
|[MobileFaceNet-GNAP](https://pan.baidu.com/s/1UL4Am0R2MYQOH6lZnPsvTg)                    |0.73537         |11.722%      |0.69903         |20.110%      |0.65734         |33.189%
|[MobileFaceNet-res2-6-10-2-dim128](https://pan.baidu.com/s/1AQEad5Zp2cag4UA5KtpbYQ)      |0.64772         |40.527%      |0.60485         |55.345%      |0.55571         |70.986%
|[MobileFaceNet-res2-6-10-2-dim256](https://pan.baidu.com/s/143j7eULc2AqpNcSugFdTxA)      |0.61647         |42.046%      |0.57561         |55.801%      |0.52852         |70.622%
|[MobileFaceNet-res2-6-10-2-dim512](https://pan.baidu.com/s/1_0O3kJ5dMmD-HdRwNR0Hpw)      |0.59725         |44.651%      |0.55690         |58.220%      |0.51134         |72.294%
|[MobileFaceNet-res4-8-16-4-dim128](https://pan.baidu.com/s/1z6H5p4b3aVun2-1dZGDXkg)      |0.64519         |47.735%      |0.60247         |62.882%      |0.55342         |77.777%
|[MobileFaceNet-res4-8-16-4-dim256](https://pan.baidu.com/s/1f_VtqNRxDNe972h8UrOsPw)      |0.58229         |56.977%      |0.54582         |69.118%      |0.49763         |82.161%
|[MobileFaceNet-res4-8-16-4-dim512](https://pan.baidu.com/s/14ukmtAWDhIJC6312WBhZhA)      |0.58296         |54.731%      |0.54219         |68.613%      |0.49174         |82.812%
|[MobileFaceNet-res8-16-32-8-dim512](https://pan.baidu.com/s/1On5BfcrOB5jrTrRD40vLkw)     |0.58058         |61.826%      |0.53841         |75.281%      |0.49098         |86.554%

|Model\Test Set ZQCNN-Face_5000_X_20                                                               |thresh@ FAR=1e-8|TAR@ FAR=1e-8|thresh@ FAR=1e-7|TAR@ FAR=1e-7|thresh@ FAR=1e-6|TAR@ FAR=1e-6
|------------                                                                                       | -------------  | ----------  |--------------- |-------      | ------------   |-----------   
|[ArcFace-r34-v2](https://pan.baidu.com/s/1q3ZqQdjabDBESqbsxC7ESQ)(Not trained by me)               |0.61953         |47.103%      |0.57375         |62.207%      |0.52226         |76.758%
|[ArcFace-r50](https://pan.baidu.com/s/1qOIhCauwZNTOCIM9eojPrA) (ms1m-refine-v1, not trained by me)|0.61299         |50.594%      |0.56658         |65.757%      |0.51637         |79.207%
|[ArcFace-r100](https://pan.baidu.com/s/1PeujQbIqFfgARIYAdRt3pw) (Not trained by me)                |0.57350         |67.434%      |0.53136         |79.944%      |0.48164         |90.147%


|Model\Test Set ZQCNN-Face_12000_X_10-40                                                 |thresh@ FAR=1e-8|TAR@ FAR=1e-8|thresh@ FAR=1e-7|TAR@ FAR=1e-7|thresh@ FAR=1e-6|TAR@ FAR=1e-6
|------------                                                                             | -------------  | ----------  |--------------- |-------      | ------------   |-----------                         
|[MobileFaceNet-res2-6-10-2-dim128](https://pan.baidu.com/s/1AQEad5Zp2cag4UA5KtpbYQ)      |0.64507         |39.100%      |0.60347         |53.638%      |0.55492         |69.516%
|[MobileFaceNet-res2-6-10-2-dim256](https://pan.baidu.com/s/143j7eULc2AqpNcSugFdTxA)      |0.61589         |39.864%      |0.57402         |54.179%      |0.52596         |69.658%
|[MobileFaceNet-res2-6-10-2-dim512](https://pan.baidu.com/s/1_0O3kJ5dMmD-HdRwNR0Hpw)      |0.60030         |41.309%      |0.55806         |55.676%      |0.50984         |70.979%
|[MobileFaceNet-res4-8-16-4-dim128](https://pan.baidu.com/s/1z6H5p4b3aVun2-1dZGDXkg)      |0.64443         |45.764%      |0.60060         |61.564%      |0.55168         |76.776%
|[MobileFaceNet-res4-8-16-4-dim256](https://pan.baidu.com/s/1f_VtqNRxDNe972h8UrOsPw)      |0.58879         |52.542%      |0.54497         |67.597%      |0.49547         |81.495%
|[MobileFaceNet-res4-8-16-4-dim512](https://pan.baidu.com/s/14ukmtAWDhIJC6312WBhZhA)      |0.58492         |51.752%      |0.54085         |67.104%      |0.49010         |81.836%
|[MobileFaceNet-res8-16-32-8-dim512](https://pan.baidu.com/s/1On5BfcrOB5jrTrRD40vLkw)     |0.58119         |61.412%      |0.53700         |75.520%      |0.48997         |86.647%

|Model\Test Set ZQCNN-Face_12000_X_10-40                                                         |thresh@ FAR=1e-8|TAR@ FAR=1e-8|thresh@ FAR=1e-7|TAR@ FAR=1e-7|thresh@ FAR=1e-6|TAR@ FAR=1e-6
|------------                                                                                     | -------------  | ----------  |--------------- |-------      | ------------   |-----------                         
|[ArcFace-r34-v2](https://pan.baidu.com/s/1q3ZqQdjabDBESqbsxC7ESQ) (Not trained by me)            |0.61904         |45.072%      |0.57173         |60.964%      |0.52062         |75.789%
|[ArcFace-r50](https://pan.baidu.com/s/1qOIhCauwZNTOCIM9eojPrA)(ms1m-refine-v1, not trained by me)|0.61412         |48.155%      |0.56749         |63.676%      |0.51537         |78.138%
|[ArcFace-r100](https://pan.baidu.com/s/1PeujQbIqFfgARIYAdRt3pw) (Not trained by me)              |0.57891         |63.854%      |0.53337         |78.129%      |0.48079         |89.579%



For more face models, please see [Model-Zoo-for-Face-Recognition](https://github.com/zuoqing1988/ZQCNN-v0.0/wiki/Model-Zoo-for-Face-Recognition)

**Facial Expression Recognition**

[FacialEmotion](https://pan.baidu.com/s/1zJtRYv-kSGSCTgpvqc4Iug) Seven types of expressions trained with Fer2013

**Gender and Age Recognition**

[GenderAge-ZQ](https://pan.baidu.com/s/1igSpmFt8XBoMk5d4GiXONg) Model trained using [train-GenderAge](https://github.com/zuoqing1988/train-GenderAge)

**Object Detection**

[MobileNetSSD](https://pan.baidu.com/s/1cyly_17cTOJBaCRiiQtWkQ) Converted format from [MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD)

[MobileNetSSD-Mouth](https://pan.baidu.com/s/1_l0Z1R34sOv2R73DB_zXyg) Used for SampleDetectMouth

**Text Detection**

[TextBoxes](https://pan.baidu.com/s/1XOREgRzyimx_AMC9bg8MgQ) Converted format from [TextBoxes](https://github.com/MhLiao/TextBoxes)

**NSFW Image Detection**

[NSFW](https://pan.baidu.com/s/1asjZFr3iTliQ4xlNbtKUtw) Converted format from [open_nsfw](https://github.com/yahoo/open_nsfw)

# Related Articles

(1)[How much precision is lost storing face feature vectors as integers?](https://zhuanlan.zhihu.com/p/35904005)

(2)[Accelerating similarity calculation for tens of millions of face feature vectors?](https://zhuanlan.zhihu.com/p/35955061)

(3)[Building a Forward library faster than mini-caffe](https://zhuanlan.zhihu.com/p/36410185)

(4)[Precision issues in vector dot product](https://zhuanlan.zhihu.com/p/36488847)

(5)[ZQCNN supports Depthwise Convolution and modified SphereFaceNet-10 with MobileNet](https://zhuanlan.zhihu.com/p/36630082)

(6)[Following the trend, releasing some source code](https://zhuanlan.zhihu.com/p/37708639)

(7)[ZQCNN supports SSD, about 30% faster than mini-caffe](https://zhuanlan.zhihu.com/p/40634934)

(8)[ZQCNN's SSD supports freely changing resolution for the same model](https://zhuanlan.zhihu.com/p/40676503)

(9)[99.78% accuracy face recognition model in ZQCNN format](https://zhuanlan.zhihu.com/p/41197488)

(10)[ZQCNN adds face recognition testing code on LFW dataset](https://zhuanlan.zhihu.com/p/41381883)

(11)[Embracing mxnet, starting to write mxnet2zqcnn](https://zhuanlan.zhihu.com/p/41667828)

(12)[Large-scale face test set, and how to build your own face test set](https://zhuanlan.zhihu.com/p/45441865)

(13)[Matrix description of regular convolution, MobileNet convolution, and global average pooling](https://zhuanlan.zhihu.com/p/45536594)

(14)[ZQ_FastFaceDetector: faster and more accurate face detection library](https://zhuanlan.zhihu.com/p/51561288)

**Android Compilation Instructions**

1. Modify the ndk path and opencv Android SDK path in build.sh

2. Modify CMakeLists.txt
   
   From original:
   
    #add_definitions(-march=native)
    add_definitions(-mfpu=neon)
    add_definitions(-mfloat-abi=hard)
    
    Change to:
    
    #add_definitions(-march=native)
    add_definitions(-mfpu=neon)
    add_definitions(-mfloat-abi=softfp)

3. This way you should be able to compile two libraries ZQ_GEMM and ZQCNN. If you want to compile SampleMTCNN, you can modify the parts that cannot be compiled according to the error prompts, mainly OpenMP and timing functions.
