ZQCNN-v0.0是ZuoQing参照mini-caffe写的forward库，随便用用

# 更新日志

**2018-09-13日更新**

(1)支持从内存加载模型

(2)增加编译配置ZQ_CNN_CompileConfig.h，可以选择是否使用_mm_fmadd_ps, _mm256_fmadd_ps (可以测一下速度看看到底快了还是慢了)。

**2018-09-12日更新 利用[insightface](https://github.com/deepinsight/insightface)训练112*96(即sphereface的尺寸)步骤：**

(1)修改insightface\src\image_iter.py： def net_sample(sef) line124 return之前加一句img=img[:,8:104,:]

修改前

	header, img = recordio.unpack(s)

	return header.label, img, None, None

修改后

	header, img = recordio.unpack(s)

	img = img[:,8:104,:]#added by zuoqing

	return header.label, img, None, None

(2)修改insightface\src\image_iter.py： def next(self)  line215 self.postprocess_data(datum)之前加一句datum=datum[:,8:104,:]

修改前

	#print(datum.shape)

	batch_data[i][:] = self.postprocess_data(datum)

修改后

	#print(datum.shape)

	datum = datum[:,8:104,:]# added by zuoqing

	batch_data[i][:] = self.postprocess_data(datum)

(3)修改datasets\faces_emore\property：112,112,改成了112,96 （同理，你用其他数据训练也得改）

(4)修改insightface\src\eval\verification.py： def load_bin(path,image_size) line193 tranpose之前加一句img=img[:,8:104,:]

修改前
    
	img = mx.image.imdecode(_bin)
    
	img = nd.transpose(img, axes=(2, 0, 1))

修改后

    img = mx.image.imdecode(_bin)

    img = img[:,8:104,:]#added by zuoqing

    img = nd.transpose(img, axes=(2, 0, 1))

(5)修改insightface\src\symbols\symbol_utils.py里面get_fc1把kernel改成(7,6) 

改完之后显存会少使用一些，可以使用更大的batch_size

**2018-08-15日更新**

(1)添加自然场景文本检测，模型从[TextBoxes](https://github.com/MhLiao/TextBoxes)转过来的。我个人觉得速度太慢，而且准确度不高。

注意这个项目里用的PriorBoxLayer与SSD里的PriorBoxLayer是不同的，为了导出ZQCNN格式的权重我修改了deploy.prototxt保存为deploy_tmp.prototxt。
从[此处](https://pan.baidu.com/s/1XOREgRzyimx_AMC9bg8MgQ)下载模型。

(2)添加图片鉴黄，模型从[open_nsfw](https://github.com/yahoo/open_nsfw)转过来的，准确度高不高我也没测过。

从[此处](https://pan.baidu.com/s/1asjZFr3iTliQ4xlNbtKUtw)下载模型。

**2018-08-10日更新**

成功转了mxnet上的[GenderAge-r50模型](https://pan.baidu.com/s/1f8RyNuQd7hl2ItlV-ibBNQ) 以及[Arcface-LResNet100E-IR](https://pan.baidu.com/s/1wuRTf2YIsKt76TxFufsRNA)，与转MobileFaceNet模型步骤一样。

下面Model Zoo 有我转好的模型，比自动转出来的应该略快。

打开ZQCNN.sln运行SampleGenderAge查看效果。我E5-1650V4的CPU，单线程时间波动很大，均值约1900-2000ms，四线程400多ms。

**2018-08-09日更新**

添加mxnet2zqcnn，成功将mxnet上的MobileFaceNet转成ZQCNN格式（不能保证其他模型也能转成功，ZQCNN还不支持很多Layer）。

第一步：编译出mxnet2zqcnn.exe

第二步：下载[model-y1.zip](https://pan.baidu.com/s/1If28BkHde4fiuweJrbicVA)然后解压

第三步：在刚才解压的目录下运行命令行 mxnet2zqcnn.exe model-symbol.json model-0000.params test.zqparams test.nchwbin

第四步：用记事本打开test.zqparams, 在第一行（Input Layer）后面加上 C=3 H=112 W=112 然后保存

第五步：把test.zqparams和test.nchwbin复制到model文件夹下，然后在VS2015里运行SampleMobileFaceNet.exe，注意工作目录是$(SolutionDir)

自动转出来的速度慢了不少，可以手工修改test.zqparams，可以参考[ArcFace-MobileFaceNet-v0](https://pan.baidu.com/s/1f-Mfad-7zRvWcy3wYoPrUg)

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

|模型名称                                                                    |LFW精度        |耗时                               |备注           
|------------                                                                | ------------- | ------------                      | ------------- 
|[SphereFace04](https://pan.baidu.com/s/1-Bb6yuU3eAN6U2ZdVsC5Mg)             | 98.2%         | -                                 |不建议使用  
|[SphereFace04bn](https://pan.baidu.com/s/18uvL3p7PWRpJcHm00-7ABg)           | 98.5%         | -                                 |不建议使用
|[SphereFace06bn](https://pan.baidu.com/s/1LXjAoJWkWp-CT0sTgIHqfg)           | 98.7%-98.8%   | -                                 |不建议使用
|[SphereFace20](https://pan.baidu.com/s/1fGJU9PfPNBot6qGVeGlcug)             | 99.2%-99.3%    |单线程约195ms， 3.6GHz            |不建议使用


|模型名称                                                                    |LFW精度        |耗时                               |备注           
|------------                                                                | ------------- | ------------                      | ------------- 
|[SphereFace04bn256](https://pan.baidu.com/s/1YXt2PLbbUg9-VZITcMw5mQ)        | 97.8%-97.9%   |单线程6-7ms, 3.6GHz                |速度最快
|[Mobile-SphereFace10bn](https://pan.baidu.com/s/1BEP1pg5s3yJCLA2elqTB0A)    | 98.6%-98.7%   |单线程15ms, 3.6GHz                 |性价比高 
|[MobileFaceNet-v0](https://pan.baidu.com/s/1f-Mfad-7zRvWcy3wYoPrUg)         |99.13%-99.23%  |单线程33-35ms，4线程14-15ms, 3.6GHz|从[model-y1.zip](https://pan.baidu.com/s/1If28BkHde4fiuweJrbicVA)转的格式 
|[MobileFaceNet-v1](https://pan.baidu.com/s/1b1g-hH7IWYxplY-XAvSz-Q)         |99.17%-99.37%  |单线程33-35ms，4线程14-15ms, 3.6GHz|我自己用insightface训练了一把 
|[MobileFaceNet-res2-6-10-2-dim128](https://pan.baidu.com/s/1AQEad5Zp2cag4UA5KtpbYQ)|99.67%-99.55%(matlab crop), 99.72-99.60%(C++ crop)  |时间与dim256接近 |网络结构与dim256一样，只不过输出维数不同
|[MobileFaceNet-res2-6-10-2-dim256](https://pan.baidu.com/s/143j7eULc2AqpNcSugFdTxA)|99.60%-99.60%(matlab crop), 99.62-99.62%(C++ crop)  |单线程约85ms，四线程约30ms, 3.6GHz |网络结构在下载链接里,用faces_emore训练的 
|[MobileFaceNet-res2-6-10-2-dim512](https://pan.baidu.com/s/1_0O3kJ5dMmD-HdRwNR0Hpw)|99.52%-99.60%(matlab crop), 99.63-99.72%(C++ crop)  |时间与dim256接近 |网络结构与dim256一样，只不过输出维数不同。感谢[moli](https://github.com/moli232777144)训练此模型
|[MobileFaceNet-res4-8-16-4-dim256](https://pan.baidu.com/s/1f_VtqNRxDNe972h8UrOsPw)|99.78%-99.78%(matlab crop), 99.75-99.75%(C++ crop)  |单线程约135ms，四线程约42ms, 3.6GHz |网络结构在下载链接里,用faces_emore训练的 
|[MobileFaceNet-res4-8-16-4-dim512](https://pan.baidu.com/s/14ukmtAWDhIJC6312WBhZhA)|99.80%-99.72%(matlab crop), 99.85-99.82%(C++ crop)  |时间与dim256接近 |网络结构与dim256一样，只不过输出维数不同。感谢[moli](https://github.com/moli232777144)训练此模型
|[ArcFace-r34](https://pan.baidu.com/s/1tRt6PxDg4UNv7yf9pMZ_LA)              |99.65%-99.70%  |单线程500ms+,3.6GHz                |-            
|[ArcFace-r34-v2](https://pan.baidu.com/s/1q3ZqQdjabDBESqbsxC7ESQ)           |99.73%-99.77%(matlab crop), 99.68-99.78%(C++ crop) |单线程500ms+,3.6GHz                |-            
|[ArcFace-r50](https://pan.baidu.com/s/1qOIhCauwZNTOCIM9eojPrA)              |99.75%-99.78%  |单线程700ms+,3.6GHz                |-            
|[ArcFace-r100](https://pan.baidu.com/s/1PeujQbIqFfgARIYAdRt3pw)             |99.80%-99.82%  |单线程1900ms+，四线程480ms, 3.6GHz |时间波动很大 

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
