#ifndef NCNN_MOBILEFACENET_FACE_LANDMARK_H
#define NCNN_MOBILEFACENET_FACE_LANDMARK_H

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
        for (int i = 0; i < 3; i++){
            this->mean[i] = mean[i];
            this->normal[i] = normal[i];
        }
}

    //ModelInfer(){}

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

//        ::memcpy(config.mean, mean, sizeof(mean));
//        ::memcpy(config.normal, normal, sizeof(mean));
//        std::cout << std::endl;
//        std::cout << "mean: " << this->mean[0] << " " << this->mean[1] << " " <<this->mean[2] <<std::endl;
//        std::cout << "normal: " << this->normal[0] << " " << this->normal[1] << " "<<this->normal[2] <<std::endl;

        std::unique_ptr<MNN::CV::ImageProcess> process(MNN::CV::ImageProcess::create(config));
        process->convert(mat.data, mat.cols, mat.rows, (int)mat.step1(), input_);
        detect_model_->runSession(sess);
        auto dimType = input_->getDimensionType();

        // debug
//        auto input_v = input_->host<float>();
//        for (int i = 0; i < 112 * 112 * 3; ++i) {
//            std::cout << input_v[i] << std::endl;
//        }

        if (output_->getType().code != halide_type_float) {
            dimType = MNN::Tensor::TENSORFLOW;
        }
        std::shared_ptr<MNN::Tensor> outputUser(new MNN::Tensor(output_, dimType));
        output_->copyToHostTensor(outputUser.get());
        auto type = outputUser->getType();
        auto size = outputUser->elementSize();
        std::vector<float> tempValues(size);
        if (type.code == halide_type_float) {
            auto values = outputUser->host<float>();
            for (int i = 0; i < size; ++i) {
                tempValues[i] =  values[i];
            }
        }
        return tempValues;
    }
    float mean[3];
    float normal[3];
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

#endif //NCNN_MOBILEFACENET_FACE_LANDMARK_H
