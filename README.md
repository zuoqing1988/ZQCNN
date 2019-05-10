# 简介

ZQCNN是ZuoQing参照mini-caffe写的forward库，ZQCNN性能远超mini-caffe、opencv。

## 主开发环境 ：[VS2015 with Update 3](https://pan.baidu.com/s/1zoREccOxVsggV-iI2z4HTg)

  MKL下载地址:[此处下载](https://pan.baidu.com/s/1d75IIf6fgTZ5oeumd0vtTw)

## 核心模块支持linux:

  如果按照[build-with-cmake.md](https://github.com/zuoqing1988/ZQCNN/blob/master/build-with-cmake.md)不能完全编译，可以只编译ZQ_GEMM，ZQCNN，和其他你想测试的程序

## 核心模块支持arm-linux:

  如果按照[build-with-cmake.md](https://github.com/zuoqing1988/ZQCNN/blob/master/build-with-cmake.md)不能完全编译，可以只编译ZQ_GEMM，ZQCNN，和其他你想测试的程序
  
**BUG:** cmake .. -DSIMD_ARCH_TYPE=arm64 -DBLAS_TYPE=openblas_zq_gemm 

理想情况下会使用openblas和ZQ_GEMM较快的一方来计算卷积（我通过在cortex-A72上测试时间来选择分支）。然而目前这个选项并不能达到预期效果，
  需要手工注在ZQ_CNN_CompileConfig.h里定义
  
	#define ZQ_CNN_USE_ZQ_GEMM 1
	#define ZQ_CNN_USE_BLAS_GEMM 1
	
可以注释掉
  
	line 67: #if defined(ZQ_CNN_USE_BOTH_BLAS_ZQ_GEMM)
	line 70: #endif

## 训练相关

  训练性别年龄：https://github.com/zuoqing1988/train-GenderAge
	
  训练MTCNN：https://github.com/zuoqing1988/train-mtcnn
	
  训练SSD: https://github.com/zuoqing1988/train-ssd
	
  训练MTCNN用于人头检测：https://github.com/zuoqing1988/train-mtcnn-head


# 更新日志

**2019-03-16日更新：达到800星，公布更准的106点landmark模型**

[ZQCNN格式:det5-dw96-v2s](https://github.com/zuoqing1988/ZQCNN/tree/master/model)model文件夹中det5-dw96-v2s.zqparams, det5-dw96-v2s.nchwbin

[mxnet格式:Lnet106_96_v2s](https://pan.baidu.com/s/1iuuAHgJBsdWsUoAdU5H58Q)提取码：r5h2

**2019-02-14日更新：达到700星，公布人脸检测精选模型**

[ZQCNN格式：精选6种Pnet、2种Rnet、2种Onet、2种Lnet](https://pan.baidu.com/s/1X2U9Y-6MJw3md8WuYxaotw)

| 六种Pnet                                                        | 输入尺寸     | 计算量（不计bbox）|  备注                |
| --------                                                        | ------       | ------------      | -------------------- |
| [Pnet20_v00](https://pan.baidu.com/s/1g7JnOxnbXIbNWPXGI-IzrQ)   | 320x240      | 8.5 M             | 对标libfacedetection |
| [Pnet20_v0](https://pan.baidu.com/s/1r3VcmEX1a2C5gKlGKnC4kw)    | 320x240      | 11.6 M            | 对标libfacedetection |
| [Pnet20_v1](https://pan.baidu.com/s/1qVU3_nporbOUzXYu7giZkA)    | 320x240      | 14.6 M            |                      |
| [Pnet20_v2](https://pan.baidu.com/s/1bXzdmsTgfqU_TJHsozSmrQ)    | 320x240      | 18.4 M            | 对标原版pnet         |
| [Pnet16_v0](https://pan.baidu.com/s/1s5eZLeAKnqp1ZDTrzaOD_w)    | 256x192      | 7.5 M             |         stride=4     |
| [Pnet16_v1](https://pan.baidu.com/s/1Lf0z6rRq5WUKE_DMze_C7w)    | 256x192      | 9.8 M             |         stride=4     |

| 两种Rnet                                                      | 输入尺寸   | 计算量           |  备注                |
| --------                                                      | ------     | ------------     | -------------------- |
| [Rnet_v1](https://pan.baidu.com/s/1SEIolnvmtPvdqbHxU1vPWQ)    | 24x24      | 0.5 M            | 对标原版Rnet         |
| [Rnet_v2](https://pan.baidu.com/s/1APWYGcFC5MAn6Ba5vWo80w)    | 24x24      | 1.4 M            |                      |

| 两种Onet                                                      | 输入尺寸   | 计算量           |  备注                |
| --------                                                      | ------     | ------------     | -------------------- |
| [Onet_v1](https://pan.baidu.com/s/1UTvSKErOul2wkT5EMxXgVA)    | 48x48      | 2.0 M            | 不含landmark         |
| [Onet_v2](https://pan.baidu.com/s/19QomSIy3Py516OEIBFDcVg)    | 48x48      | 3.2 M            | 不含landmark         |

| 两种Lnet                                                      | 输入尺寸   | 计算量           |  备注                |
| --------                                                      | ------     | ------------     | -------------------- |
| [Lnet_v2](https://pan.baidu.com/s/1W6bxNeD0psxwxbou_xwK-g)    | 48x48      |  3.5 M           | lnet_basenum=16      |
| [Lnet_v2](https://pan.baidu.com/s/1e3tuwrR3AoU_zRKkIFK8xg)    | 48x48      | 10.8 M           | lnet_basenum=32      |

**2019-01-31日更新：达到600星，公布MTCNN人头检测模型**

hollywoodheads数据训练的，效果一般，凑合用吧

人头检测mtcnn-head[mxnet-v0](https://pan.baidu.com/s/1eqzgeoszon_6bNgS1psa7w)&[zqcnn-v0](https://pan.baidu.com/s/1Xh27qm_LmuV6ZIDLBUXfPQ)



**2019-01-24日更新：核心模块支持linux**

如果按照[build-with-cmake.md](https://github.com/zuoqing1988/ZQCNN/blob/master/build-with-cmake.md)不能完全编译，可以只编译ZQ_GEMM，ZQCNN，和其他你想测试的程序

**2019-01-17日更新**

更改了ZQ_CNN_MTCNN.h

(1)init时设置thread_num小于1时可以强制Pnet_stage执行多线程，也就是会分块，对于大图找小脸来说可以防止内存爆掉

(2)rnet/onet/lnet的尺寸可以不是24/48/48，但是只支持宽高相等

(3)rnet/onet/lnet分批处理，在脸非常多时可以减小内存占用

**2019-01-15日更新：庆祝达到500星，发放106点landmark模型**

[mxnet格式&zqcnn格式](https://pan.baidu.com/s/18VTMfChnAEyeU_9vE9GJaw)


**2019-01-04日更新：庆祝达到400星，发放快速人脸模型**

[mxnet格式](https://pan.baidu.com/s/1pOvAaXncbarNfD0G-4BwlQ)

[zqcnn格式](https://pan.baidu.com/s/18FLOduY4SoHjXHBCXWQ5LQ)

v3版本还不够好，后面还将出v4版本，大概就是下面这个图的意思

![MTCNN-v4示意图](https://github.com/zuoqing1988/ZQCNN/blob/master/mtcnn%E7%A4%BA%E6%84%8F%E5%9B%BE.jpg)

**~~2018-12-25日更新:不开源的106点landmark~~**

~~生活比较拮据，挣点外快。~~

~~landmark106-normal-1000.jpg是model\det5-dw48-1000.nchwbin生成的landmark~~
	
~~landmark106-normal.jpg，与landmark106-big.jpg是我训练的两个没开源的模型~~
	
~~其中normal模型2.1M，计算量11.4M，PC单线程耗时0.6-0.7ms，big模型7.56M，计算量36.4M，PC单线程耗时1.5-1.6ms~~

**2018-12-20日更新：添加MTCNN106点landmark模型**

在SampleMTCNN里试用（放出来的只是一个不太好的，更好的等着卖钱）

SampleLnet106有计时，单线程约0.6~0.7ms (E5-1650V4, 3.6GHz)

**2018-12-03日更新：将模型编译到代码里面**

ZQCNN.sln里 model2code 可以将模型编译成代码

	model2code.exe param_file model_file code_file prefix
	
然后在你的工程里面添加

	#include"code_file"
	
使用下面的函数加载模型

	LoadFromBuffer(prefix_param, prefix_param_len, prefix_model, prefix_model_len)


**2018-11-21日更新**

支持mxnet-ssd训练的模型，mean_val需要设成127.5才能在SampleSSD里面正确运行。

但是使用ReLU训练的好像不正确，我用PReLU训练一个，重头训练的，只有mAP=0.48凑合着看吧，[点此下载](https://pan.baidu.com/s/1-wfpuvGLBGPtlqicdO1raw)。

更改模型之后必须用imagenet先训练分类模型，然后再训练SSD，才能把mAP弄上去。

**2018-11-14日更新**

(1)优化ZQ_GEMM，3.6GHz的机器上MKL峰值约46GFLOPS， ZQ_GEMM约32GFLOPS。使用ZQ_GEMM人脸模型总体时间约为使用MKL时1.5倍。

注意：使用VS2017编译出来的ZQ_GEMM比VS2015快，但是SampleMTCNN多线程运行是错的（可能OpenMP的支持规则不同？）。

(2)加载模型时可以去掉非常小的权重。当你发现模型比预料中慢很多时，多半是由于权重值太小造成的。

**2018-11-06日更新**

(1)去掉layers里所有omp多线程的代码，计算量太小，速度比单线程更慢

(2)cblas_gemm可以选择MKL，不过3rdparty带的mkl在我机器上很慢，dll比较大，我没放在3rdparty\bin里，请从[此处下载](https://pan.baidu.com/s/1d75IIf6fgTZ5oeumd0vtTw)。

**2018-10-30日更新2：MTCNN大图找小脸建议先用高斯滤波**

**2018-10-30日更新：BatchNorm的eps问题**

(1)BatchNorm、BatchNormScale的默认eps都是0

(2)如果是用mxnet2zqcnn从mxnet转过来模型，转的过程中会把eps加到var上面当做新的var

(3)如果是从其他平台转过来模型，要么手工把eps加到var上面，要么在BatchNorm、BatchNormScale后面加上eps=?(?为该平台这个层的eps值)

注意：为了防止除0错，在除var的时候是这么计算的sqrt(__max(var+eps,1e-32))，也就是说如果var+eps小于1e-32，会与理论值略有不同。
不过今天修改之后下面几个人脸模型的LFW的精度反而与minicaffe的结果一模一样了。

**2018-10-26日更新**

MTCNN支持多线程，大图找小脸而且脸多的情况下，8线程可以取得单线程4倍以上效果，请用data\test2.jpg来测试

**2018-10-15日更新**

改进MTCNN的nms策略：1.每个scale的Pnet的nms的局部极大必须覆盖一定数量的非极大，数量在参数中设置; 2.当Pnet的分辨率太大时，nms进行分块处理。

**2018-09-25日更新**

支持insightface的GNAP，自动转模型使用mxnet2zqcnn，查看[mxnet2zqcnn](https://github.com/zuoqing1988/ZQCNN-v0.0/wiki/mxnet2zqcnn)。可以试用[MobileFaceNet-GNAP](https://pan.baidu.com/s/1hv4lbYwSLlLiGK07FuJM5Q)

**2018-09-20日更新**

(1)更新人脸识别模型tar-far精度的测试方法，可以按照步骤[How-to-evaluate-TAR-FAR-on-your-dataset](https://github.com/zuoqing1988/ZQCNN-v0.0/wiki/How-to-evaluate-TAR-FAR-on-your-dataset)自行构造测试集测试模型精度。

(2)按照(1)我清洗CASIA-Webface构造了两个测试集[webface1000X50](https://pan.baidu.com/s/1AoJkj_IhydkiyD1UGm8rDQ)、[webface5000X20](https://pan.baidu.com/s/1AoJkj_IhydkiyD1UGm8rDQ)，并测试了我开源的几个主要人脸识别模型的精度。

**2018-09-13日更新**

(1)支持从内存加载模型

(2)增加编译配置ZQ_CNN_CompileConfig.h，可以选择是否使用_mm_fmadd_ps, _mm256_fmadd_ps (可以测一下速度看看到底快了还是慢了)。

**2018-09-12日更新 利用[insightface](https://github.com/deepinsight/insightface)训练112*96(即sphereface的尺寸)步骤：** [InsightFace： how to train 112*96](https://github.com/zuoqing1988/ZQCNN-v0.0/wiki/InsightFace%EF%BC%9A-how-to-train-112*96)

**2018-08-15日更新**

(1)添加自然场景文本检测，模型从[TextBoxes](https://github.com/MhLiao/TextBoxes)转过来的。我个人觉得速度太慢，而且准确度不高。

注意这个项目里用的PriorBoxLayer与SSD里的PriorBoxLayer是不同的，为了导出ZQCNN格式的权重我修改了deploy.prototxt保存为deploy_tmp.prototxt。
从[此处](https://pan.baidu.com/s/1XOREgRzyimx_AMC9bg8MgQ)下载模型。

(2)添加图片鉴黄，模型从[open_nsfw](https://github.com/yahoo/open_nsfw)转过来的，准确度高不高我也没测过。

从[此处](https://pan.baidu.com/s/1asjZFr3iTliQ4xlNbtKUtw)下载模型。

**2018-08-10日更新**

成功转了mxnet上的[GenderAge-r50模型](https://pan.baidu.com/s/1f8RyNuQd7hl2ItlV-ibBNQ) 以及[Arcface-LResNet100E-IR](https://pan.baidu.com/s/1wuRTf2YIsKt76TxFufsRNA)，与转MobileFaceNet模型步骤一样。
查看[mxnet2zqcnn](https://github.com/zuoqing1988/ZQCNN-v0.0/wiki/mxnet2zqcnn)

下面Model Zoo 有我转好的模型，比自动转出来的应该略快。

打开ZQCNN.sln运行SampleGenderAge查看效果。我E5-1650V4的CPU，单线程时间波动很大，均值约1900-2000ms，四线程400多ms。

**2018-08-09日更新**

添加mxnet2zqcnn，成功将mxnet上的MobileFaceNet转成ZQCNN格式（不能保证其他模型也能转成功，ZQCNN还不支持很多Layer）。查看[mxnet2zqcnn](https://github.com/zuoqing1988/ZQCNN-v0.0/wiki/mxnet2zqcnn)

**2018-08-07日更新**

BUG修复：之前Convolution, DepthwiseConvolution, InnerProduct, BatchNormScale/Scale默认with_bias=true， 现在改成默认with_bias=false。也就是之前的代码无法加载不带bias的这几个Layer。

示例，如下这样一个Layer，以前会默认为有bias_term，现在默认没有bias_term

Convolution name=conv1 bottom=data top=conv1 num_output=10 kernel_size=3 stride=1 

**2018-08-06日更新**

增加人脸识别在LFW数据库的精度测试。打开ZQlibFaceID.sln可以看到相关Project。

由于C++代码的计算精度与matlab略有差距，统计出的精度也有一些差别，但是相差在0.1%以内。

**2018-08-03日更新**

支持多线程（通过openmp加速）。**请注意，目前多线程反而比单线程慢**

**2018-07-26日更新**

支持MobileNet-SSD。caffemodel转我用的模型参考export_mobilenet_SSD_caffemodel_to_nchw_binary.m。需要编译出matcaffe才行。
你可以试试这个版本[caffe-ZQ](https://github.com/zuoqing1988/caffe-ZQ)

**2018-06-05日更新**

跟上时代潮流、发布源码。
忘了说需要依赖openblas，我是直接用的mini-caffe里面的那个版本，自己编译出来的很慢。



# Model Zoo

**人脸检测**

[MTCNN-author-version](https://pan.baidu.com/s/1lWLKDYv8YQ6Th6KRiKvgug) 从[MTCNN](https://github.com/kpzhang93/MTCNN_face_detection_alignment)转的格式

[MTCNN-ZQ-version](https://pan.baidu.com/s/1j1WqkwbUCf_9f4hCQukoFg)

**人脸识别(如无说明，模型都是ms1m-refine-v2训练的)**

|模型                                                                               |LFW精度(ZQCNN)                                      | LFW精度(OpenCV3.4.2)                              | LFW精度(minicaffe)                               |耗时 (ZQCNN)                       |备注           
|------------                                                                       | -------------                                      |----------------------                             | ------------                                     |---------------------              | ------------- 
|[MobileFaceNet-res2-6-10-2-dim128](https://pan.baidu.com/s/1AQEad5Zp2cag4UA5KtpbYQ)|99.67%-99.55%(matlab crop), 99.72-99.60%(C++ crop)  |99.63%-99.65%(matlab crop), 99.68-99.70%(C++ crop) |99.62%-99.65%(matlab crop), 99.68-99.60%(C++ crop)|时间与dim256接近 |网络结构与dim256一样，只不过输出维数不同
|[MobileFaceNet-res2-6-10-2-dim256](https://pan.baidu.com/s/143j7eULc2AqpNcSugFdTxA)|99.60%-99.60%(matlab crop), 99.62-99.62%(C++ crop)  |99.73%-99.68%(matlab crop), 99.78-99.68%(C++ crop) |99.55%-99.63%(matlab crop), 99.60-99.62%(C++ crop)|单线程约21-22ms，四线程约11-12ms, 3.6GHz |网络结构在下载链接里,用faces_emore训练的 
|[MobileFaceNet-res2-6-10-2-dim512](https://pan.baidu.com/s/1_0O3kJ5dMmD-HdRwNR0Hpw)|99.52%-99.60%(matlab crop), 99.63-99.72%(C++ crop)  |99.70%-99.67%(matlab crop), 99.77-99.77%(C++ crop) |99.55%-99.62%(matlab crop), 99.62-99.68%(C++ crop)|时间与dim256接近 |网络结构与dim256一样，只不过输出维数不同。感谢[moli](https://github.com/moli232777144)训练此模型

|模型                                                                               |LFW精度(ZQCNN)                                      | LFW精度(OpenCV3.4.2)                              | LFW精度(minicaffe)                               |耗时 (ZQCNN)                       |备注           
|------------                                                                       | -------------                                      |----------------------                             | ------------                                     |---------------------              | ------------- 
|[MobileFaceNet-res4-8-16-4-dim128](https://pan.baidu.com/s/1z6H5p4b3aVun2-1dZGDXkg)|99.72%-99.72%(matlab crop), 99.72-99.68%(C++ crop)  |99.82%-99.83%(matlab crop), 99.80-99.78%(C++ crop) |99.72%-99.72%(matlab crop), 99.72-99.68%(C++ crop)|时间与dim256接近 |网络结构与dim256一样，只不过输出维数不同
|[MobileFaceNet-res4-8-16-4-dim256](https://pan.baidu.com/s/1f_VtqNRxDNe972h8UrOsPw)|99.78%-99.78%(matlab crop), 99.75-99.75%(C++ crop)  |99.82%-99.82%(matlab crop), 99.80-99.82%(C++ crop) |99.78%-99.78%(matlab crop), 99.73-99.73%(C++ crop)|单线程约32-33ms，四线程约16-19ms, 3.6GHz |网络结构在下载链接里,用faces_emore训练的 
|[MobileFaceNet-res4-8-16-4-dim512](https://pan.baidu.com/s/14ukmtAWDhIJC6312WBhZhA)|99.80%-99.73%(matlab crop), 99.85-99.83%(C++ crop)  |99.83%-99.82%(matlab crop), 99.87-99.83%(C++ crop) |99.80%-99.73%(matlab crop), 99.85-99.82%(C++ crop)|时间与dim256接近 |网络结构与dim256一样，只不过输出维数不同。感谢[moli](https://github.com/moli232777144)训练此模型

|模型\测试集webface1000X50                                                          |thresh@ FAR=1e-7|TAR@ FAR=1e-7|thresh@ FAR=1e-6|TAR@ FAR=1e-6|thresh@ FAR=1e-5|TAR@ FAR=1e-5
|------------                                                                       | -------------  | ----------  |--------------- |-------      | ------------   |-----------                         
|[MobileFaceNet-res2-6-10-2-dim128](https://pan.baidu.com/s/1AQEad5Zp2cag4UA5KtpbYQ)|0.78785         |9.274%       |0.66616         |40.459%      |0.45855         |92.716%
|[MobileFaceNet-res2-6-10-2-dim256](https://pan.baidu.com/s/143j7eULc2AqpNcSugFdTxA)|0.77708         |7.839%       |0.63872         |40.934%      |0.43182         |92.605%
|[MobileFaceNet-res2-6-10-2-dim512](https://pan.baidu.com/s/1_0O3kJ5dMmD-HdRwNR0Hpw)|0.76699         |8.197%       |0.63452         |38.774%      |0.41572         |93.000%
|[MobileFaceNet-res4-8-16-4-dim128](https://pan.baidu.com/s/1z6H5p4b3aVun2-1dZGDXkg)|0.79268         |9.626%       |0.65770         |48.252%      |0.45431         |95.576%
|[MobileFaceNet-res4-8-16-4-dim256](https://pan.baidu.com/s/1f_VtqNRxDNe972h8UrOsPw)|0.76858         |9.220%       |0.62852         |46.195%      |0.40010         |96.929%
|[MobileFaceNet-res4-8-16-4-dim512](https://pan.baidu.com/s/14ukmtAWDhIJC6312WBhZhA)|0.76287         |9.296%       |0.62555         |44.775%      |0.39047         |97.347%

|模型\测试集webface5000X20                                                          |thresh@ FAR=1e-7|TAR@ FAR=1e-7|thresh@ FAR=1e-6|TAR@ FAR=1e-6|thresh@ FAR=1e-5|TAR@ FAR=1e-5
|------------                                                                       | -------------  | ----------  |--------------- |-------      | ------------   |-----------                         
|[MobileFaceNet-res2-6-10-2-dim128](https://pan.baidu.com/s/1AQEad5Zp2cag4UA5KtpbYQ)|0.70933         |29.558%      |0.51732         |85.160%      |0.45108         |94.313%
|[MobileFaceNet-res2-6-10-2-dim256](https://pan.baidu.com/s/143j7eULc2AqpNcSugFdTxA)|0.68897         |28.376%      |0.48820         |85.278%      |0.42386         |94.244%
|[MobileFaceNet-res2-6-10-2-dim512](https://pan.baidu.com/s/1_0O3kJ5dMmD-HdRwNR0Hpw)|0.68126         |27.708%      |0.47260         |85.840%      |0.40727         |94.632%
|[MobileFaceNet-res4-8-16-4-dim128](https://pan.baidu.com/s/1z6H5p4b3aVun2-1dZGDXkg)|0.71238         |32.153%      |0.51391         |89.525%      |0.44667         |96.583%
|[MobileFaceNet-res4-8-16-4-dim256](https://pan.baidu.com/s/1f_VtqNRxDNe972h8UrOsPw)|0.68490         |30.639%      |0.46092         |91.900%      |0.39198         |97.696%
|[MobileFaceNet-res4-8-16-4-dim512](https://pan.baidu.com/s/14ukmtAWDhIJC6312WBhZhA)|0.67303         |32.404%      |0.45216         |92.453%      |0.38344         |98.003%

|模型\测试集TAO ids:6606,ims:87210                                                  |thresh@ FAR=1e-7|TAR@ FAR=1e-7|thresh@ FAR=1e-6|TAR@ FAR=1e-6|thresh@ FAR=1e-5|TAR@ FAR=1e-5
|------------                                                                       | -------------  |-------------|--------------- |-------------| -------------- |-----------                         
|[MobileFaceNet-res2-6-10-2-dim128](https://pan.baidu.com/s/1AQEad5Zp2cag4UA5KtpbYQ)|0.92204         |01.282%      |0.88107         |06.837%      |0.78302         |41.740%
|[MobileFaceNet-res2-6-10-2-dim256](https://pan.baidu.com/s/143j7eULc2AqpNcSugFdTxA)|0.91361         |01.275%      |0.86750         |07.081%      |0.76099         |42.188%
|[MobileFaceNet-res2-6-10-2-dim512](https://pan.baidu.com/s/1_0O3kJ5dMmD-HdRwNR0Hpw)|0.90657         |01.448%      |0.86061         |07.299%      |0.75488         |41.956%
|[MobileFaceNet-res4-8-16-4-dim128](https://pan.baidu.com/s/1z6H5p4b3aVun2-1dZGDXkg)|0.92098         |01.347%      |0.88233         |06.795%      |0.78711         |41.856%
|[MobileFaceNet-res4-8-16-4-dim256](https://pan.baidu.com/s/1f_VtqNRxDNe972h8UrOsPw)|0.90862         |01.376%      |0.86397         |07.083%      |0.75975         |42.430%
|[MobileFaceNet-res4-8-16-4-dim512](https://pan.baidu.com/s/14ukmtAWDhIJC6312WBhZhA)|0.90710         |01.353%      |0.86190         |06.948%      |0.75518         |42.241%


|模型\测试集ZQCNN-Face_5000_X_20                                                     |thresh@ FAR=1e-8|TAR@ FAR=1e-8|thresh@ FAR=1e-7|TAR@ FAR=1e-7|thresh@ FAR=1e-6|TAR@ FAR=1e-6
|------------                                                                        | -------------  | ----------  |--------------- |-------      | ------------   |-----------                         
|[MobileFaceNet-GNAP](https://pan.baidu.com/s/1UL4Am0R2MYQOH6lZnPsvTg)               |0.73537         |11.722%      |0.69903         |20.110%      |0.65734         |33.189%
|[MobileFaceNet-res2-6-10-2-dim128](https://pan.baidu.com/s/1AQEad5Zp2cag4UA5KtpbYQ) |0.64772         |40.527%      |0.60485         |55.345%      |0.55571         |70.986%
|[MobileFaceNet-res2-6-10-2-dim256](https://pan.baidu.com/s/143j7eULc2AqpNcSugFdTxA) |0.61647         |42.046%      |0.57561         |55.801%      |0.52852         |70.622%
|[MobileFaceNet-res2-6-10-2-dim512](https://pan.baidu.com/s/1_0O3kJ5dMmD-HdRwNR0Hpw) |0.59725         |44.651%      |0.55690         |58.220%      |0.51134         |72.294%
|[MobileFaceNet-res4-8-16-4-dim128](https://pan.baidu.com/s/1z6H5p4b3aVun2-1dZGDXkg) |0.64519         |47.735%      |0.60247         |62.882%      |0.55342         |77.777%
|[MobileFaceNet-res4-8-16-4-dim256](https://pan.baidu.com/s/1f_VtqNRxDNe972h8UrOsPw) |0.58229         |56.977%      |0.54582         |69.118%      |0.49763         |82.161%
|[MobileFaceNet-res4-8-16-4-dim512](https://pan.baidu.com/s/14ukmtAWDhIJC6312WBhZhA) |0.58296         |54.731%      |0.54219         |68.613%      |0.49174         |82.812%
|[MobileFaceNet-res8-16-32-8-dim512](https://pan.baidu.com/s/1On5BfcrOB5jrTrRD40vLkw)|0.58058         |61.826%      |0.53841         |75.281%      |0.49098         |86.554%

|模型\测试集ZQCNN-Face_5000_X_20                                                               |thresh@ FAR=1e-8|TAR@ FAR=1e-8|thresh@ FAR=1e-7|TAR@ FAR=1e-7|thresh@ FAR=1e-6|TAR@ FAR=1e-6
|------------                                                                                  | -------------  | ----------  |--------------- |-------      | ------------   |-----------   
|[ArcFace-r34-v2](https://pan.baidu.com/s/1q3ZqQdjabDBESqbsxC7ESQ)(非本人训练)                 |0.61953         |47.103%      |0.57375         |62.207%      |0.52226         |76.758%
|[ArcFace-r50](https://pan.baidu.com/s/1qOIhCauwZNTOCIM9eojPrA) (ms1m-refine-v1非本人训练)     |0.61299         |50.594%      |0.56658         |65.757%      |0.51637         |79.207%
|[ArcFace-r100](https://pan.baidu.com/s/1PeujQbIqFfgARIYAdRt3pw) (非本人训练)                  |0.57350         |67.434%      |0.53136         |79.944%      |0.48164         |90.147%


|模型\测试集ZQCNN-Face_12000_X_10-40                                                 |thresh@ FAR=1e-8|TAR@ FAR=1e-8|thresh@ FAR=1e-7|TAR@ FAR=1e-7|thresh@ FAR=1e-6|TAR@ FAR=1e-6
|------------                                                                        | -------------  | ----------  |--------------- |-------      | ------------   |-----------                         
|[MobileFaceNet-res2-6-10-2-dim128](https://pan.baidu.com/s/1AQEad5Zp2cag4UA5KtpbYQ) |0.64507         |39.100%      |0.60347         |53.638%      |0.55492         |69.516%
|[MobileFaceNet-res2-6-10-2-dim256](https://pan.baidu.com/s/143j7eULc2AqpNcSugFdTxA) |0.61589         |39.864%      |0.57402         |54.179%      |0.52596         |69.658%
|[MobileFaceNet-res2-6-10-2-dim512](https://pan.baidu.com/s/1_0O3kJ5dMmD-HdRwNR0Hpw) |0.60030         |41.309%      |0.55806         |55.676%      |0.50984         |70.979%
|[MobileFaceNet-res4-8-16-4-dim128](https://pan.baidu.com/s/1z6H5p4b3aVun2-1dZGDXkg) |0.64443         |45.764%      |0.60060         |61.564%      |0.55168         |76.776%
|[MobileFaceNet-res4-8-16-4-dim256](https://pan.baidu.com/s/1f_VtqNRxDNe972h8UrOsPw) |0.58879         |52.542%      |0.54497         |67.597%      |0.49547         |81.495%
|[MobileFaceNet-res4-8-16-4-dim512](https://pan.baidu.com/s/14ukmtAWDhIJC6312WBhZhA) |0.58492         |51.752%      |0.54085         |67.104%      |0.49010         |81.836%
|[MobileFaceNet-res8-16-32-8-dim512](https://pan.baidu.com/s/1On5BfcrOB5jrTrRD40vLkw)|0.58119         |61.412%      |0.53700         |75.520%      |0.48997         |86.647%

|模型\测试集ZQCNN-Face_12000_X_10-40                                                         |thresh@ FAR=1e-8|TAR@ FAR=1e-8|thresh@ FAR=1e-7|TAR@ FAR=1e-7|thresh@ FAR=1e-6|TAR@ FAR=1e-6
|------------                                                                                | -------------  | ----------  |--------------- |-------      | ------------   |-----------                         
|[ArcFace-r34-v2](https://pan.baidu.com/s/1q3ZqQdjabDBESqbsxC7ESQ) (非本人训练)              |0.61904         |45.072%      |0.57173         |60.964%      |0.52062         |75.789%
|[ArcFace-r50](https://pan.baidu.com/s/1qOIhCauwZNTOCIM9eojPrA)(ms1m-refine-v1非本人训练)    |0.61412         |48.155%      |0.56749         |63.676%      |0.51537         |78.138%
|[ArcFace-r100](https://pan.baidu.com/s/1PeujQbIqFfgARIYAdRt3pw) (非本人训练)                |0.57891         |63.854%      |0.53337         |78.129%      |0.48079         |89.579%



更多人脸模型请查看[Model-Zoo-for-Face-Recognition](https://github.com/zuoqing1988/ZQCNN-v0.0/wiki/Model-Zoo-for-Face-Recognition)

**表情识别**

[FacialEmotion](https://pan.baidu.com/s/1zJtRYv-kSGSCTgpvqc4Iug) 七类表情用Fer2013训练

**性别年龄识别**

[GenderAge-ZQ](https://pan.baidu.com/s/1igSpmFt8XBoMk5d4GiXONg) 使用[train-GenderAge](https://github.com/zuoqing1988/train-GenderAge)训练出来的模型

**目标检测**

[MobileNetSSD](https://pan.baidu.com/s/1cyly_17cTOJBaCRiiQtWkQ) 从[MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD)转的格式

[MobileNetSSD-Mouth](https://pan.baidu.com/s/1_l0Z1R34sOv2R73DB_zXyg) 用于SampleDetectMouth

**文字检测**

[TextBoxes](https://pan.baidu.com/s/1XOREgRzyimx_AMC9bg8MgQ) 从[TextBoxes](https://github.com/MhLiao/TextBoxes)转的格式

**图片鉴黄**

[NSFW](https://pan.baidu.com/s/1asjZFr3iTliQ4xlNbtKUtw) 从[open_nsfw](https://github.com/yahoo/open_nsfw)转的格式

# 相关文章

(1)[人脸特征向量用整数存储精度损失多少？](https://zhuanlan.zhihu.com/p/35904005)

(2)[千万张脸的特征向量，计算相似度提速？](https://zhuanlan.zhihu.com/p/35955061)

(3)[打造一款比mini-caffe更快的Forward库](https://zhuanlan.zhihu.com/p/36410185)

(4)[向量点积的精度问题](https://zhuanlan.zhihu.com/p/36488847)

(5)[ZQCNN支持Depthwise Convolution并用mobilenet改了一把SphereFaceNet-10](https://zhuanlan.zhihu.com/p/36630082)

(6)[跟上时代潮流，发布一些源码](https://zhuanlan.zhihu.com/p/37708639)

(7)[ZQCNN支持SSD，比mini-caffe快大概30%](https://zhuanlan.zhihu.com/p/40634934)

(8)[ZQCNN的SSD支持同一个模型随意改分辨率](https://zhuanlan.zhihu.com/p/40676503)

(9)[ZQCNN格式的99.78%精度的人脸识别模型](https://zhuanlan.zhihu.com/p/41197488)

(10)[ZQCNN增加人脸识别在LFW数据集上的测试代码](https://zhuanlan.zhihu.com/p/41381883)

(11)[抱紧mxnet的大腿，着手写mxnet2zqcnn](https://zhuanlan.zhihu.com/p/41667828)

(12)[大规模人脸测试集，及如何打造自己的人脸测试集](https://zhuanlan.zhihu.com/p/45441865)

(13)[普通卷积、mobilenet卷积、全局平均池化的矩阵描述](https://zhuanlan.zhihu.com/p/45536594)

(14)[ZQ_FastFaceDetector更快更准的人脸检测库](https://zhuanlan.zhihu.com/p/51561288)

**Android编译说明**
1. 修改build.sh中的ndk路径和opencv安卓sdk的路径
2. 修改CMakeLists.txt
   从原来的
    #add_definitions(-march=native)
    add_definitions(-mfpu=neon)
    add_definitions(-mfloat-abi=hard)
    改为
    #add_definitions(-march=native)
    add_definitions(-mfpu=neon)
    add_definitions(-mfloat-abi=softfp)
3. 这样应该可以编译两个库ZQ_GEMM和ZQCNN了.如果要编译SampleMTCNN可以按照错误提示修改不能编译的部分,主要是openmp和计时函数.
