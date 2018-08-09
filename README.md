ZQCNN-v0.0是ZuoQing参照mini-caffe写的forward库，随便用用

# 更新日志
**2018-08-09日更新**

添加mxnet2zqcnn，成功将mxnet上的MobileFaceNet转成ZQCNN格式。

第一步：编译出mxnet2zqcnn.exe

第二步：下载[model-y1.zip](https://pan.baidu.com/s/1If28BkHde4fiuweJrbicVA)然后解压

第三步：在刚才解压的目录下运行命令行 mxnet2zqcnn.exe model-symbol.json model-0000.params test.zqparams test.nchwbin

第四步：用记事本打开test.zqparams, 在第一行（Input Layer）后面加上 C=3 H=112 W=112 然后保存

第五步：把test.zqparams和test.nchwbin复制到model文件夹下，然后在VS2015里运行SampleMobileNet.exe，注意工作目录是$(SolutionDir)

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

[MTCNN](https://pan.baidu.com/s/1f6_wQ2kXiTZFyH6PFIDc2Q) 从[MTCNN](https://github.com/kpzhang93/MTCNN_face_detection_alignment)转的格式

**人脸识别**

[SeetaFace](https://pan.baidu.com/s/17GySgiI8EASfOCuRizMAOw) LFW约97.8-97.9%，每次提取时间约110ms，3.6GHz

[SphereFace04bn256](https://pan.baidu.com/s/1YXt2PLbbUg9-VZITcMw5mQ) LFW约97.8%-97.9%，速度最快

[SphereFace04](https://pan.baidu.com/s/1-Bb6yuU3eAN6U2ZdVsC5Mg) LFW约98.2%

[SphereFace04bn](https://pan.baidu.com/s/18uvL3p7PWRpJcHm00-7ABg) LFW约98.5%

[SphereFace06bn](https://pan.baidu.com/s/1LXjAoJWkWp-CT0sTgIHqfg) LFW约98.7%-99.8%

[SphereFace20](https://pan.baidu.com/s/1fGJU9PfPNBot6qGVeGlcug) LFW约99.2%-99.3%

[Mobile-SphereFace10bn512](https://pan.baidu.com/s/1BEP1pg5s3yJCLA2elqTB0A) LFW约98.6%-98.7%，性价比高

[ArcFace-r50](https://pan.baidu.com/s/1qOIhCauwZNTOCIM9eojPrA) LFW约99.75%-99.78%,精度最高，但是很慢

[ArcFace-r34](https://pan.baidu.com/s/1tRt6PxDg4UNv7yf9pMZ_LA) LFW约99.65%-99.70%,比r-50稍微快一点

[ArcFace-MobileFaceNet-v0](https://pan.baidu.com/s/1f-Mfad-7zRvWcy3wYoPrUg) 从[model-y1.zip](https://pan.baidu.com/s/1If28BkHde4fiuweJrbicVA)转的格式，转完之后在LFW上只有99.13%-99.23%，单线程14-15ms

**表情识别**

[FacialEmotion](https://pan.baidu.com/s/1zJtRYv-kSGSCTgpvqc4Iug) 七类表情用Fer2013训练

**目标检测**

[MobileNetSSD](https://pan.baidu.com/s/1cyly_17cTOJBaCRiiQtWkQ) 从[MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD)转的格式

[MobileNetSSD-Mouth](https://pan.baidu.com/s/1_l0Z1R34sOv2R73DB_zXyg) 用于SampleDetectMouth

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
