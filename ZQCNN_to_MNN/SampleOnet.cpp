
#include <stdio.h>
#include "ImageProcess.hpp"
#include "Interpreter.hpp"
#define MNN_OPEN_TIME_TRACE
#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <vector>
#include "AutoTime.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

using namespace MNN;
using namespace MNN::CV;

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        MNN_PRINT("Usage: ./SampleOnet.out model.mnn input.jpg\n");
        return 0;
    }
    std::shared_ptr<Interpreter> net(Interpreter::createFromFile(argv[1]));
    ScheduleConfig config;
    config.type  = MNN_FORWARD_AUTO;
    auto session = net->createSession(config);

    auto input = net->getSessionInput(session, NULL);
    auto shape = input->shape();
    shape[0]   = 1;
    net->resizeTensor(input, shape);
    net->resizeSession(session);
    auto output = net->getSessionOutput(session, "cls_prob");
    auto output2 = net->getSessionOutput(session, NULL);
    std::vector<std::string> words;
    if (argc >= 4) {
        std::ifstream inputOs(argv[3]);
        std::string line;
        while (std::getline(inputOs, line)) {
            words.emplace_back(line);
        }
    }
    {
        auto dims    = input->shape();
        int inputDim = 0;
        int size_w   = 0;
        int size_h   = 0;
        int bpp      = 0;
        bpp          = input->channel();
        size_h       = input->height();
        size_w       = input->width();
        if (bpp == 0)
            bpp = 1;
        if (size_h == 0)
            size_h = 1;
        if (size_w == 0)
            size_w = 1;
        MNN_PRINT("input: w:%d , h:%d, bpp: %d\n", size_w, size_h, bpp);

        auto inputPatch = argv[2];
        int width, height, channel;
        auto inputImage = stbi_load(inputPatch, &width, &height, &channel, 4);
        if (nullptr == inputImage) {
            MNN_ERROR("Can't open %s\n", inputPatch);
            return 0;
        }
        MNN_PRINT("origin size: %d, %d\n", width, height);
        Matrix trans;
        // Set scale, from dst scale to src
        trans.setScale((float)(width-1) / (size_w-1), (float)(height-1) / (size_h-1));
        ImageProcess::Config config;
        config.filterType = BILINEAR;
        float mean[3]     = {127.5f, 127.5f, 127.5f};
        float normals[3] = {0.00785f, 0.00785f, 0.00785f};
        ::memcpy(config.mean, mean, sizeof(mean));
        ::memcpy(config.normal, normals, sizeof(normals));
        config.sourceFormat = RGBA;
        config.destFormat   = BGR;

        std::shared_ptr<ImageProcess> pretreat(ImageProcess::create(config));
        pretreat->setMatrix(trans);
        pretreat->convert((uint8_t*)inputImage, width, height, 0, input);
        stbi_image_free(inputImage);
    }
    net->runSession(session);
    {
        auto dimType = output->getDimensionType();
        auto dimType2 = output2->getDimensionType();
        if (output->getType().code != halide_type_float) {
            dimType = Tensor::TENSORFLOW;
        }
        if (output2->getType().code != halide_type_float) {
            dimType2 = Tensor::TENSORFLOW;
        }
        std::shared_ptr<Tensor> outputUser(new Tensor(output, dimType));
        std::shared_ptr<Tensor> outputUser2(new Tensor(output2, dimType2));
        MNN_PRINT("output  size:%d\n", outputUser->elementSize());
        MNN_PRINT("output2 size:%d\n", outputUser2->elementSize());
        output->copyToHostTensor(outputUser.get());
        output2->copyToHostTensor(outputUser2.get());
        auto type = outputUser->getType();
        auto type2 = outputUser2->getType();

        auto size = outputUser->elementSize();
        auto size2 = outputUser2->elementSize();
        std::vector<std::pair<int, float>> tempValues(size), tempValues2(size2);
        if (type.code == halide_type_float) {
            auto values = outputUser->host<float>();
            for (int i = 0; i < size; ++i) {
                tempValues[i] = std::make_pair(i, values[i]);
            }
        }
        if (type.code == halide_type_uint && type.bytes() == 1) {
            auto values = outputUser->host<uint8_t>();
            for (int i = 0; i < size; ++i) {
                tempValues[i] = std::make_pair(i, values[i]);
            }
        }
        if (type2.code == halide_type_float) {
             auto values = outputUser2->host<float>();
             for (int i = 0;i < size2; ++i) {
                 tempValues2[i] = std::make_pair(i, values[i]);
             }
        }
        if (type2.code == halide_type_uint && type2.bytes() == 1) {
             auto values = outputUser2->host<uint8_t>();
             for (int i = 0;i < size2; ++i) {
                 tempValues2[i] = std::make_pair(i, values[i]);
             }
        }
        
        int length = size;
        if (words.empty()) {
            for (int i = 0; i < length; ++i) {
                MNN_PRINT("%d, %f\n", tempValues[i].first, tempValues[i].second);
            }
        } else {
            for (int i = 0; i < length; ++i) {
                MNN_PRINT("%s: %f\n", words[tempValues[i].first].c_str(), tempValues[i].second);
            }
        }
        int length2 = size2;//
        for (int i = 0;i < length2; ++ i) {
                MNN_PRINT("%d, %f\n", tempValues2[i].first, tempValues2[i].second);
        }
    }
    return 0;
}
