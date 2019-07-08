# ZQCNN格式转MNN格式

**1.把converter文件夹里的内容覆盖MNN/tools/converter**

**2.重新编译converter里的内容（注意清除之前的cmake cache）**

**3.转换命令为**

	./ZQCNN_to_MNN in.zqparams in.nchwbin out.mnn
	
**4.示例代码**

	./SampleOnet.out model.mnn input.jpg
	
## 目前支持的Layer

**BatchNormScale**

**Concat**

**Convolution**

**DepthwiseConvolution**

**Eltwise**

**InnerProduct**

**Pooling**

**PReLU**

**ReLU**

**Softmax**



# 还有很多BUG

**比如：本目录里的det3-dw48-p0.mnn 有两个输出"cls_prob"和"conv6_2"，但是getSessionOutput的name,必须第一个填"cls_prob"、第二个填NULL，才能正确运行**

**可能不同MNN版本生成的模型不通用，请用自己的代码重新转换和运行**