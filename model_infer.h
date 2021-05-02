//
// Created by YH-Mac on 2020/9/2.
//

#ifndef REDLIGHT_MODEL_INFER_H
#define REDLIGHT_MODEL_INFER_H

#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>
#include "opencv2/opencv.hpp"
//#include "const.h"
#include <MNN/ImageProcess.hpp>


class ModelInfer
{
public:
    ModelInfer(const std::string &proto_model_dir, int thread, float mean[], float normal[])
    {
        detect_model_ = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(proto_model_dir.c_str()));
        _config.type = (MNNForwardType)0;
        _config.numThread = thread;
        MNN::BackendConfig backendConfig;
        backendConfig.precision = MNN::BackendConfig::Precision_High;
        backendConfig.power = MNN::BackendConfig::Power_Low;
        _config.backendConfig = &backendConfig;
        this->mean = mean;
        this->normal = normal;
        if (detect_model_ == nullptr) {
            std::cout << "NIUBI Interpreter loading model failed." << std::endl;
        } else {
            std::cout << "Interpreter loads model successfully." << std::endl;
        }
    }

    ModelInfer(){}

    void Init(const std::string &input, const std::string &output ,int width, int height)
    {
        sess = detect_model_->createSession(_config);
        tensor_shape_.resize(4);
        tensor_shape_ = {1, 3, height, width};
        input_ = detect_model_->getSessionInput(sess, input.c_str());
        output_ = detect_model_->getSessionOutput(sess, output.c_str());
        width_ = width;
        height_ = height;
    }

    std::vector<float> Infer(const cv::Mat &mat)
    {
        assert(mat.rows == height_);
        assert(mat.cols == width_);
        MNN::CV::ImageProcess::Config config;
        config.destFormat = MNN::CV::ImageFormat::BGR;
        config.sourceFormat = MNN::CV::BGR;
        for (int i = 0; i < 3; i++)
        {
            config.mean[i] = mean[i];
            config.normal[i] = normal[i];
        }
        std::unique_ptr<MNN::CV::ImageProcess> process(MNN::CV::ImageProcess::create(config));
        process->convert(mat.data, mat.cols, mat.rows, (int)mat.step1(), input_);
        detect_model_->runSession(sess);
        output_ = detect_model_->getSessionOutput(sess, "detection_out");
        auto dimType = output_->getDimensionType();
        if (output_->getType().code != halide_type_float) {
            dimType = MNN::Tensor::TENSORFLOW;
        }
        std::shared_ptr<MNN::Tensor> outputUser(new MNN::Tensor(output_, dimType));
        output_->copyToHostTensor(outputUser.get());

//        // add
//        MNN::Tensor output_host(output_, output_->getDimensionType());
//        output_->copyToHostTensor(&output_host);

        auto type = outputUser->getType();
        auto size = outputUser->elementSize();
        std::vector<float> tempValues(size);
        if (type.code == halide_type_float) {
            auto values = outputUser->host<float>();

            std::cout << size << std::endl;
            for (int i = 0; i < size; ++i) {
                tempValues[i] =  values[i];

            }
        }
        return tempValues;
    }
    float* mean;
    float* normal;
private:
    std::shared_ptr<MNN::Interpreter> detect_model_;
    MNN::Tensor *input_;
    MNN::Tensor *output_;
    MNN::Session *sess;
    std::vector<int> tensor_shape_;
    MNN::ScheduleConfig _config;
    int width_;
    int height_;
};

#endif //REDLIGHT_MODEL_INFER_H
