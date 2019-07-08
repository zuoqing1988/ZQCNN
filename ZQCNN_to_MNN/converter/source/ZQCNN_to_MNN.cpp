#include "MNN_generated.h"
#include "addBizCode.hpp"
#include "optimizer.hpp"
#include "writeFb.hpp"
#include "ZQ_CNN_Net.h"
using namespace ZQ;

bool ZQCNN_to_MNNNet(ZQ_CNN_Net& zq_net, const std::string bizCode, std::unique_ptr<MNN::NetT>& netT);

int main(int argc, const char** argv)
{
	if (argc != 4)
	{
		printf("%s in.zqparams in.nchwbin out.mnn\n", argv[0]);
		return 1;
	}

	ZQ_CNN_Net zq_net;
	if (!zq_net.LoadFrom(argv[1], argv[2], false, 1e-9, false))
	{
		printf("failed to load ZQ_CNN_Net\n");
		return 1;
	}

	printf("suc to load ZQ_CNN_Net\n");

	std::cout << "Start to Convert Other Model Format To MNN Model..." << std::endl;
	std::unique_ptr<MNN::NetT> netT = std::unique_ptr<MNN::NetT>(new MNN::NetT());
	if (!ZQCNN_to_MNNNet(zq_net, "MNN", netT))
	{
		printf("failed to convert ZQCNN to MNN\n");
		return 1;
	}
	std::cout << "Start to Optimize the MNN Net..." << std::endl;
	std::unique_ptr<MNN::NetT> newNet = optimizeNet(netT);
	writeFb(newNet, argv[3], false);
	
	std::cout << "Converted Done!" << std::endl;
	return 0;
}

bool ZQCNN_to_MNNNet(ZQ_CNN_Net& zq_net, const std::string bizCode, std::unique_ptr<MNN::NetT>& netT) 
{
	netT->tensorName.resize(zq_net.blobs.size());
        netT->tensorName[0] = "name";
        auto iter = zq_net.map_name_to_blob_idx.begin();
        for(;iter != zq_net.map_name_to_blob_idx.end();++iter)
        {
            int id = iter->second;
            if(id > 0)
                netT->tensorName[id] = iter->first;
        }
        
        
	//input
	{
		MNN::OpT* op = new MNN::OpT;
		op->name = "data";
		op->type = MNN::OpType_Input;
		op->main.type = MNN::OpParameter_Input;
		auto inputT = new MNN::InputT;
                
                inputT->dims.push_back(1);		
		inputT->dims.push_back(zq_net.input_C);
		inputT->dims.push_back(zq_net.input_H);
		inputT->dims.push_back(zq_net.input_W);
	
		op->main.value = inputT;
		op->outputIndexes.push_back(0);

		netT->oplists.emplace_back(op);
		netT->tensorName.push_back(op->name);
		
	}
	

	for (int l = 1; l < zq_net.layers.size(); l++)
	{
		MNN::OpT* op = new MNN::OpT;
		op->name = zq_net.layers[l]->name;
                for(int i = 0;i < zq_net.bottoms[l].size();i++)
                    op->inputIndexes.push_back(zq_net.bottoms[l][i]);
                for(int i = 0;i < zq_net.tops[l].size();i++)
                    op->outputIndexes.push_back(zq_net.tops[l][i]);
		if (ZQ_CNN_Layer::_my_strcmpi(zq_net.layer_type_names[l].c_str(), "Convolution") == 0)
		{
			auto convolution2D = new MNN::Convolution2DT;
			op->main.value = convolution2D;

			convolution2D->common = std::unique_ptr<MNN::Convolution2DCommonT>(new MNN::Convolution2DCommonT);
			auto& common = convolution2D->common;

			common->group = 1;

			ZQ_CNN_Layer_Convolution* cur_layer = (ZQ_CNN_Layer_Convolution*)zq_net.layers[l];
			ZQ_CNN_Tensor4D* filters = cur_layer->filters;
			common->outputCount = filters->GetN();
			common->inputCount = filters->GetC();
			
			common->kernelX = cur_layer->kernel_W;
			common->kernelY = cur_layer->kernel_H;
			
			common->dilateX = cur_layer->dilate_W;
			common->dilateY = cur_layer->dilate_H;

			
			common->strideX = cur_layer->stride_W;
			common->strideY = cur_layer->stride_H;
			
			common->padX = cur_layer->pad_W;
			common->padY = cur_layer->pad_H;
			common->padMode = MNN::PadMode_CAFFE;


			int filter_size = filters->GetN()*filters->GetC()*filters->GetH()*filters->GetW();
			std::vector<float> raw_filters(filter_size);
			filters->ConvertToCompactNCHW(&raw_filters[0]);
			convolution2D->weight = raw_filters;

			ZQ_CNN_Tensor4D* bias = cur_layer->bias;
			std::vector<float> raw_bias(outputCount,0.0f);
			if (bias == 0)
			{
			}
			else
			{
				bias->ConvertToCompactNCHW(&raw_bias[0]);
			}
			convolution2D->bias = raw_bias;


			op->type = MNN::OpType::OpType_Convolution;
			op->main.type = MNN::OpParameter::OpParameter_Convolution2D;
			
		}
		else if (ZQ_CNN_Layer::_my_strcmpi(zq_net.layer_type_names[l].c_str(), "DepthwiseConvolution") == 0)
		{
			auto convolution2D = new MNN::Convolution2DT;
			op->main.value = convolution2D;

			convolution2D->common = std::unique_ptr<MNN::Convolution2DCommonT>(new MNN::Convolution2DCommonT);
			auto& common = convolution2D->common;

			ZQ_CNN_Layer_DepthwiseConvolution* cur_layer = (ZQ_CNN_Layer_DepthwiseConvolution*)zq_net.layers[l];
			ZQ_CNN_Tensor4D* filters = cur_layer->filters;
			common->group = filters->GetC();
			common->outputCount = filters->GetC();
			common->inputCount = 1;

			common->kernelX = cur_layer->kernel_W;
			common->kernelY = cur_layer->kernel_H;

			common->dilateX = cur_layer->dilate_W;
			common->dilateY = cur_layer->dilate_H;


			common->strideX = cur_layer->stride_W;
			common->strideY = cur_layer->stride_H;

			common->padX = cur_layer->pad_W;
			common->padY = cur_layer->pad_H;
			common->padMode = MNN::PadMode_CAFFE;


			int filter_size = filters->GetN()*filters->GetC()*filters->GetH()*filters->GetW();
			std::vector<float> raw_filters(filter_size);
			filters->ConvertToCompactNCHW(&raw_filters[0]);
			convolution2D->weight = raw_filters;

			ZQ_CNN_Tensor4D* bias = cur_layer->bias;
			std::vector<float> raw_bias(outputCount, 0.0f);
			if (bias == 0)
			{
			}
			else
			{
				bias->ConvertToCompactNCHW(&raw_bias[0]);
			}
			convolution2D->bias = raw_bias;


			op->type = MNN::OpType::OpType_ConvolutionDepthwise;
			op->main.type = MNN::OpParameter::OpParameter_Convolution2D;

		}
		else if (ZQ_CNN_Layer::_my_strcmpi(zq_net.layer_type_names[l].c_str(), "InnerProduct") == 0)
		{
			auto innerproduct = new MNN::InnerProductT;
			op->main.value = innerproduct;
			ZQ_CNN_Layer_InnerProduct* cur_layer = (ZQ_CNN_Layer_InnerProduct*)zq_net.layers[l];
			ZQ_CNN_Tensor4D* filters = cur_layer->filters;
			innerproduct->outputCount = filters->GetN();
			innerproduct->axis = 1;
			
			innerproduct->transpose = false;
			
			ZQ_CNN_Tensor4D* bias = cur_layer->bias;
			innerproduct->biasTerm = cur_layer->bias != 0;
			if (bias != 0)
			{
				innerproduct->bias.resize(bias->GetC());
				bias->ConvertToCompactNCHW(&innerproduct->bias[0]);
			}
			
			innerproduct->weightSize = filters->GetN()*filters->GetC()*filters->GetH()*filters->GetW();
			innerproduct->weight.resize(innerproduct->weightSize);
			filters->ConvertToCompactNCHW(&innerproduct->weight[0]);

			op->type = MNN::OpType::OpType_InnerProduct;
			op->main.type = MNN::OpParameter::OpParameter_InnerProduct;
		}
		else if (ZQ_CNN_Layer::_my_strcmpi(zq_net.layer_type_names[l].c_str(), "Pooling") == 0)
		{
			auto pool = new MNN::PoolT;
			op->main.value = pool;
			ZQ_CNN_Layer_Pooling* cur_layer = (ZQ_CNN_Layer_Pooling*)zq_net.layers[l];
			if (cur_layer->type == ZQ_CNN_Layer_Pooling::TYPE_MAXPOOLING)
			{
				pool->type = MNN::PoolType_MAXPOOL;
			}
			else if (cur_layer->type == ZQ_CNN_Layer_Pooling::TYPE_AVGPOOLING)
			{
				pool->type = MNN::PoolType_AVEPOOL;
			}
			else
			{
				printf("unsupported pool type: %d\n", cur_layer->type);
				return false;
			}
			
			pool->kernelY = cur_layer->kernel_H;
			pool->kernelX = cur_layer->kernel_W;

			
			pool->strideY = cur_layer->stride_H;
			pool->strideX = cur_layer->stride_W;

			pool->padY = cur_layer->pad_H;
			pool->padX = cur_layer->pad_W;

			pool->isGlobal = cur_layer->global_pool;
			op->type = MNN::OpType::OpType_Pooling;
			op->main.type = MNN::OpParameter::OpParameter_Pool;
		}
		else if (ZQ_CNN_Layer::_my_strcmpi(zq_net.layer_type_names[l].c_str(), "PReLU") == 0)
		{
			auto relu = new MNN::PReluT;

			ZQ_CNN_Layer_PReLU* cur_layer = (ZQ_CNN_Layer_PReLU*)zq_net.layers[l];
			ZQ_CNN_Tensor4D* slope = cur_layer->slope;
			
			relu->slopeCount = slope->GetC();
			relu->slope.resize(relu->slopeCount);
			slope->ConvertToCompactNCHW(&relu->slope[0]);
			op->main.value = relu;
			op->type = MNN::OpType::OpType_PReLU;
			op->main.type = MNN::OpParameter::OpParameter_PRelu;
		}
		else if (ZQ_CNN_Layer::_my_strcmpi(zq_net.layer_type_names[l].c_str(), "ReLU") == 0)
		{
			auto relu = new MNN::ReluT;
			ZQ_CNN_Layer_ReLU* cur_layer = (ZQ_CNN_Layer_ReLU*)zq_net.layers[l];
			relu->slope = cur_layer->slope;
			op->main.value = relu;
			op->type = MNN::OpType::OpType_ReLU;
			op->main.type = MNN::OpParameter::OpParameter_Relu;
		}
		else if (ZQ_CNN_Layer::_my_strcmpi(zq_net.layer_type_names[l].c_str(), "Softmax") == 0)
		{
			auto axisT = new MNN::AxisT;
			axisT->axis = 1;
			op->main.value = axisT;
			op->type = MNN::OpType::OpType_Softmax;
			op->main.type = MNN::OpParameter::OpParameter_Axis;
		}
		else if (ZQ_CNN_Layer::_my_strcmpi(zq_net.layer_type_names[l].c_str(), "BatchNormScale") == 0)
		{
			auto bn = new MNN::BatchNormT;
			op->main.value = bn;
			ZQ_CNN_Layer_BatchNormScale* cur_layer = (ZQ_CNN_Layer_BatchNormScale*)zq_net.layers[l];
			ZQ_CNN_Tensor4D* mean = cur_layer->mean;
			ZQ_CNN_Tensor4D* var = cur_layer->var;
			ZQ_CNN_Tensor4D* scale = cur_layer->scale;
			ZQ_CNN_Tensor4D* bias = cur_layer->bias;
			

			bn->channels = mean->GetC();
			
			bn->slopeData.resize(bn->channels);
			scale->ConvertToCompactNCHW(&bn->slopeData[0]);

			bn->varData.resize(bn->channels);
			bn->meanData.resize(bn->channels);

			mean->ConvertToCompactNCHW(&bn->meanData[0]);
			var->ConvertToCompactNCHW(&bn->varData[0]);
			bn->biasData.resize(bn->channels);
			if (bias)
			{
				bias->ConvertToCompactNCHW(&bn->biasData[0]);
			}
			else
			{
				for (int i = 0;i < bn->channels;i++)
					bn->biasData[i] = 0;
			}
			op->type = MNN::OpType::OpType_BatchNorm;
			op->main.type = MNN::OpParameter::OpParameter_BatchNorm;
		}
		else if (ZQ_CNN_Layer::_my_strcmpi(zq_net.layer_type_names[l].c_str(), "Eltwise") == 0)
		{
			static const int ELTWISE_MUL = 0;
			static const int ELTWISE_SUM = 1;
			static const int ELTWISE_MAX = 2;
			auto elt = new MNN::EltwiseT;
			op->main.value = elt;
			ZQ_CNN_Layer_Eltwise* cur_layer = (ZQ_CNN_Layer_Eltwise*)zq_net.layers[l];
			if (cur_layer->operation == ELTWISE_SUM)
			{
				elt->type = MNN::EltwiseType_SUM;
			}
			else if(cur_layer->operation == ELTWISE_MUL)
			{
				elt->type = MNN::EltwiseType_PROD;
			}
			else if(cur_layer->operation == ELTWISE_MAX)
			{
				elt->type = MNN::EltwiseType_MAXIMUM;
			}
			
			const int coffSize = cur_layer->weight.size();
			elt->coeff.resize(coffSize);
			for (int i = 0; i < coffSize; ++i) 
			{
				elt->coeff[i] = cur_layer->weight[i];
			}
			
			op->type = MNN::OpType::OpType_Eltwise;
			op->main.type = MNN::OpParameter::OpParameter_Eltwise;
		}
		else if (ZQ_CNN_Layer::_my_strcmpi(zq_net.layer_type_names[l].c_str(), "Concat") == 0)
		{
			auto axisT        = new MNN::AxisT;
			op->main.value = axisT;
			ZQ_CNN_Layer_Concat* cur_layer = (ZQ_CNN_Layer_Concat*)zq_net.layers[l];
			axisT->axis = cur_layer->axis;
			
			op->type = MNN::OpType::OpType_Concat;
			op->main.type = MNN::OpParameter::OpParameter_Axis;
		}
		else
		{
			printf("unsupported layer type: %s\n", zq_net.layer_type_names[l].c_str());
			return false;
		}

		netT->oplists.emplace_back(op);
	}
	
	netT->sourceType = MNN::NetSource_CAFFE;
	netT->bizCode = bizCode;

	return true;
}
