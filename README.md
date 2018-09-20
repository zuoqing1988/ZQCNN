ZQCNN-v0.0是ZuoQing参照mini-caffe写的forward库，随便用用

# 更新日志

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

[MTCNN](https://pan.baidu.com/s/1f6_wQ2kXiTZFyH6PFIDc2Q) 从[MTCNN](https://github.com/kpzhang93/MTCNN_face_detection_alignment)转的格式

**人脸识别**

|模型                                                                               |LFW精度(ZQCNN)                                      | LFW精度(OpenCV3.4.2)                              | LFW精度(minicaffe)                               |耗时 (ZQCNN)                       |备注           
|------------                                                                       | -------------                                      |----------------------                             | ------------                                     |---------------------              | ------------- 
|[MobileFaceNet-res2-6-10-2-dim128](https://pan.baidu.com/s/1AQEad5Zp2cag4UA5KtpbYQ)|99.67%-99.55%(matlab crop), 99.72-99.60%(C++ crop)  |99.63%-99.65%(matlab crop), 99.68-99.70%(C++ crop) |99.62%-99.65%(matlab crop), 99.68-99.60%(C++ crop)|时间与dim256接近 |网络结构与dim256一样，只不过输出维数不同
|[MobileFaceNet-res2-6-10-2-dim256](https://pan.baidu.com/s/143j7eULc2AqpNcSugFdTxA)|99.60%-99.60%(matlab crop), 99.62-99.62%(C++ crop)  |99.73%-99.68%(matlab crop), 99.78-99.68%(C++ crop) |99.55%-99.63%(matlab crop), 99.60-99.62%(C++ crop)|单线程约85ms，四线程约30ms, 3.6GHz |网络结构在下载链接里,用faces_emore训练的 
|[MobileFaceNet-res2-6-10-2-dim512](https://pan.baidu.com/s/1_0O3kJ5dMmD-HdRwNR0Hpw)|99.52%-99.60%(matlab crop), 99.63-99.72%(C++ crop)  |99.70%-99.67%(matlab crop), 99.77-99.77%(C++ crop) |99.55%-99.62%(matlab crop), 99.62-99.68%(C++ crop)|时间与dim256接近 |网络结构与dim256一样，只不过输出维数不同。感谢[moli](https://github.com/moli232777144)训练此模型

|模型                                                                               |LFW精度(ZQCNN)                                      | LFW精度(OpenCV3.4.2)                              | LFW精度(minicaffe)                               |耗时 (ZQCNN)                       |备注           
|------------                                                                       | -------------                                      |----------------------                             | ------------                                     |---------------------              | ------------- 
|[MobileFaceNet-res4-8-16-4-dim128](https://pan.baidu.com/s/1z6H5p4b3aVun2-1dZGDXkg)|99.72%-99.72%(matlab crop), 99.72-99.68%(C++ crop)  |99.82%-99.83%(matlab crop), 99.80-99.78%(C++ crop) |99.72%-99.72%(matlab crop), 99.72-99.68%(C++ crop)|时间与dim256接近 |网络结构与dim256一样，只不过输出维数不同
|[MobileFaceNet-res4-8-16-4-dim256](https://pan.baidu.com/s/1f_VtqNRxDNe972h8UrOsPw)|99.78%-99.78%(matlab crop), 99.75-99.75%(C++ crop)  |99.82%-99.82%(matlab crop), 99.80-99.82%(C++ crop) |99.78%-99.78%(matlab crop), 99.73-99.73%(C++ crop)|单线程约135ms，四线程约42ms, 3.6GHz |网络结构在下载链接里,用faces_emore训练的 
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

|模型\私有测试集TAO (ids:6606,imgs:87210)                                           |thresh@ FAR=1e-7|TAR@ FAR=1e-7|thresh@ FAR=1e-6|TAR@ FAR=1e-6|thresh@ FAR=1e-5|TAR@ FAR=1e-5
|------------                                                                       | -------------  | ----------  |--------------- |-------      | ------------   |-----------                         
|[MobileFaceNet-res2-6-10-2-dim128](https://pan.baidu.com/s/1AQEad5Zp2cag4UA5KtpbYQ)|0.92204         |1.282%       |0.88107         |6.837%       |0.78302         |41.740%
|[MobileFaceNet-res2-6-10-2-dim256](https://pan.baidu.com/s/143j7eULc2AqpNcSugFdTxA)|0.91361         |1.275%       |0.86750         |7.081%       |0.76099         |42.188%
|[MobileFaceNet-res2-6-10-2-dim512](https://pan.baidu.com/s/1_0O3kJ5dMmD-HdRwNR0Hpw)|0.90657         |1.448%       |0.86061         |7.299%       |0.75488         |41.956%
|[MobileFaceNet-res4-8-16-4-dim128](https://pan.baidu.com/s/1z6H5p4b3aVun2-1dZGDXkg)|0.92098         |1.347%       |0.88233         |6.795%       |0.78711         |41.856%
|[MobileFaceNet-res4-8-16-4-dim256](https://pan.baidu.com/s/1f_VtqNRxDNe972h8UrOsPw)|0.90862         |1.376%       |0.86397         |7.083%       |0.75975         |42.430%
|[MobileFaceNet-res4-8-16-4-dim512](https://pan.baidu.com/s/14ukmtAWDhIJC6312WBhZhA)|0.90710         |1.353%       |0.86190         |6.948%       |0.75518         |42.241%

更多人脸模型请查看[Model-Zoo-for-Face-Recognition](https://github.com/zuoqing1988/ZQCNN-v0.0/wiki/Model-Zoo-for-Face-Recognition)

**表情识别**

[FacialEmotion](https://pan.baidu.com/s/1zJtRYv-kSGSCTgpvqc4Iug) 七类表情用Fer2013训练

**性别年龄识别**

[GenderAge-r50](https://pan.baidu.com/s/1FeuMLNG9jQ0CeD0ZANrY0g)从[insightface](https://github.com/deepinsight/insightface/wiki/Model-Zoo)的[gamodel-r50](https://pan.baidu.com/s/1f8RyNuQd7hl2ItlV-ibBNQ)转的格式。

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
