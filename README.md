ZQCNN-v0.0是ZuoQing参照mini-caffe写的forward库，随便用用

# 更新日志

2018-08-03日更新
支持多线程（通过openmp加速）

2018-07-26日更新

支持MobileNet-SSD。caffemodel转我用的模型参考export_mobilenet_SSD_caffemodel_to_nchw_binary.m。需要编译出matcaffe才行。你可以试试这个https://github.com/zuoqing1988/caffe-ZQ

2018-06-05日更新

跟上时代潮流、发布源码。
忘了说需要依赖openblas，我是直接用的mini-caffe里面的那个版本，自己编译出来的很慢。



# Model Zoo

**人脸检测**

MTCNN：https://pan.baidu.com/s/1f6_wQ2kXiTZFyH6PFIDc2Q

**人脸识别**

SphereFace04bn256(LFW约97.8%，速度最快)：https://pan.baidu.com/s/1YXt2PLbbUg9-VZITcMw5mQ

SphereFace04（LFW约98.2%）：https://pan.baidu.com/s/1-Bb6yuU3eAN6U2ZdVsC5Mg

SphereFace04bn (LFW约98.5%)：https://pan.baidu.com/s/18uvL3p7PWRpJcHm00-7ABg

SphereFace06bn (LFW约98.8%)：https://pan.baidu.com/s/1LXjAoJWkWp-CT0sTgIHqfg

SphereFace20 (LFW约99.2%)：https://pan.baidu.com/s/1fGJU9PfPNBot6qGVeGlcug

Mobile-SphereFace10bn512(LFW约98.6%，性价比高)：https://pan.baidu.com/s/1BEP1pg5s3yJCLA2elqTB0A

ArcFace-r50(LFW约99.75%-99.78%)：https://pan.baidu.com/s/1ORhYzZkggSBgSRQ3BbTbDA

**表情识别**

FacialEmotion(七类表情用Fer2013训练)：https://pan.baidu.com/s/1zJtRYv-kSGSCTgpvqc4Iug

**目标检测**

MobileNetSSD: https://pan.baidu.com/s/1ddkVzjQ0kFqUS7atTgrMrw

# 相关文章

(1)人脸特征向量用整数存储精度损失多少？
https://zhuanlan.zhihu.com/p/35904005

(2)千万张脸的特征向量，计算相似度提速？
https://zhuanlan.zhihu.com/p/35955061

(3)打造一款比mini-caffe更快的Forward库
https://zhuanlan.zhihu.com/p/36410185

(4)向量点积的精度问题
https://zhuanlan.zhihu.com/p/36488847

(5)ZQCNN支持Depthwise Convolution并用mobilenet改了一把SphereFaceNet-10
https://zhuanlan.zhihu.com/p/36630082

(6)跟上时代潮流，发布一些源码
https://zhuanlan.zhihu.com/p/37708639

(7)ZQCNN支持SSD，比mini-caffe快大概30%
https://zhuanlan.zhihu.com/p/40634934

(8)ZQCNN的SSD支持同一个模型随意改分辨率
https://zhuanlan.zhihu.com/p/40676503

(9)ZQCNN格式的99.78%精度的人脸识别模型
https://zhuanlan.zhihu.com/p/41197488