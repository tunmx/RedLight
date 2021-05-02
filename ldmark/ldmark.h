//
// Created by YH-Mac on 2020/9/9.
//

#ifndef REDLIGHT_LDMARK_H
#define REDLIGHT_LDMARK_H


#include "MNN/Interpreter.hpp"
#include "MNN/ImageProcess.hpp"
#include "MNN/MNNDefine.h"
#include "MNN/Tensor.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "../common/common.h"
#include "iostream"

using namespace std;

class ldmark{
public:
    ldmark(const std::string &models_path){
        Init(models_path);
    }

    ldmark(){

    }

    ~ldmark(){
        m_interpreter->releaseModel();
        m_interpreter->releaseSession(m_session);
    }

    int Init(const std::string &models_path){
        std::cout << "start init person detect interpreter." << std::endl;
//        std::string model_file = models_path;
        m_interpreter = std::unique_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(models_path.c_str()));
        if (m_interpreter == nullptr) {
            std::cout << "Interpreter loading model failed." << std::endl;
            return 10000;
        } else {
            std::cout << "niubi Interpreter loads model successfully." << std::endl;
        }

        // create session
        MNN::ScheduleConfig schedule_config;
        schedule_config.type = MNN_FORWARD_CPU;
        schedule_config.numThread = 4;

        MNN::BackendConfig backend_config;
        backend_config.power = MNN::BackendConfig::Power_High;
        backend_config.precision = MNN::BackendConfig::Precision_High;
        schedule_config.backendConfig = &backend_config;
        m_session = m_interpreter->createSession(schedule_config);

        // image processer
        MNN::CV::Matrix trans;
        trans.setScale(1.0f, 1.0f);
        MNN::CV::ImageProcess::Config img_config;
        img_config.filterType = MNN::CV::BICUBIC;
        ::memcpy(img_config.mean, meanVals, sizeof(meanVals));
        ::memcpy(img_config.normal, normVals, sizeof(normVals));
        img_config.sourceFormat = MNN::CV::BGR;
        img_config.destFormat = MNN::CV::RGB;
        m_process = std::shared_ptr<MNN::CV::ImageProcess>(MNN::CV::ImageProcess::create(img_config));
        m_process->setMatrix(trans);

        std::string input_name = "input_1";
        m_input_tensor = m_interpreter->getSessionInput(m_session, input_name.c_str());
        m_interpreter->resizeTensor(m_input_tensor, dims);
        m_interpreter->resizeSession(m_session);

        initialized_ = true;

//    std::cout << "Successful initialization." << std::endl;

        return 0;
    }

    int detect(const cv::Mat& img_src, std::vector<float>& out){
        if (!initialized_) {
            std::cout << "model uninitialized." << std::endl;
            return 10000;
        }
        if (img_src.empty()) {
            std::cout << "input empty." << std::endl;
            return 10001;
        }
        int width = img_src.cols;
        int height = img_src.rows;

        // img preprocess
        cv::Mat img_resized;
        cv::resize(img_src, img_resized, inputSize);
        m_process->convert(img_resized.data, inputSize.width, inputSize.height, 0, m_input_tensor);

        m_interpreter->runSession(m_session);
        std::string output_name = "strided_slice";
        MNN::Tensor *output_tensor = m_interpreter->getSessionOutput(m_session, output_name.c_str());

        auto dimType = output_tensor->getDimensionType();
        if (output_tensor->getType().code != halide_type_float) {
            dimType = MNN::Tensor::TENSORFLOW;
        }

        // copy to host
        MNN::Tensor output_host(output_tensor, output_tensor->getDimensionType());
        output_tensor->copyToHostTensor(&output_host);

        auto type = output_host.getType();
        auto size = output_host.elementSize();
//        std::vector<float> tempValues(size);
        if (type.code == halide_type_float) {
            auto values = output_host.host<float>();
            for (int i = 0; i < size; ++i) {
                out.push_back(values[i]);
            }
        }

    }

    bool initialized_;
    cv::Size inputSize = {112, 112};
    std::vector<int> dims = {1, 3, 112, 112};
    float meanVals[3] = {0, 0, 0};
    float normVals[3] = {1, 1, 1};

    std::shared_ptr<MNN::Interpreter> m_interpreter = nullptr;
    MNN::Session *m_session = nullptr;
    MNN::Tensor *m_input_tensor = nullptr;
    std::shared_ptr<MNN::CV::ImageProcess> m_process = nullptr;

    int target = 15;
};

#endif //REDLIGHT_LDMARK_H
