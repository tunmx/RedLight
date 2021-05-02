//
// Created by YH-Mac on 2020/9/2.
//

#ifndef REDLIGHT_MODEL_PERSON_H
#define REDLIGHT_MODEL_PERSON_H


#include "MNN/Interpreter.hpp"
#include "MNN/ImageProcess.hpp"
#include "MNN/MNNDefine.h"
#include "MNN/Tensor.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "common/common.h"
#include "iostream"

using namespace std;

class model_person {
public:
    model_person();
    model_person(const std::string &models_path){
        initialized_ = false;
        Init(models_path);
    }

    ~model_person(){
        m_interpreter->releaseModel();
        m_interpreter->releaseSession(m_session);
    }

    int Init(const std::string &models_path){
        std::cout << "start init person detect interpreter." << std::endl;
        std::string model_file = models_path + "/ssd.mnn";
        m_interpreter = std::unique_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_file.c_str()));
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

        std::string input_name = "data";
        m_input_tensor = m_interpreter->getSessionInput(m_session, input_name.c_str());
        m_interpreter->resizeTensor(m_input_tensor, dims);
        m_interpreter->resizeSession(m_session);

        initialized_ = true;

//    std::cout << "Successful initialization." << std::endl;

        return 0;
    }

    int detect(const cv::Mat& img_src, std::vector<ObjectInfo>& objects){
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
        std::string output_name = "detection_out";
        MNN::Tensor *output_tensor = m_interpreter->getSessionOutput(m_session, output_name.c_str());

        // copy to host
        MNN::Tensor output_host(output_tensor, output_tensor->getDimensionType());
        output_tensor->copyToHostTensor(&output_host);

        auto output_ptr = output_host.host<float>();
        std::vector<ObjectInfo> objects_tmp;
        for (int i = 0; i < output_host.height(); ++i) {
            int index = i * output_host.width();
            if (int(output_ptr[index + 0]) == target) {
                ObjectInfo object;
                object.name_ = class_names[int(output_ptr[index + 0])];
                object.score_ = output_ptr[index + 1];
                object.location_.x = output_ptr[index + 2] * width;
                object.location_.y = output_ptr[index + 3] * height;
                object.location_.width = output_ptr[index + 4] * width - object.location_.x;
                object.location_.height = output_ptr[index + 5] * height - object.location_.y;
//            object.location_ = rectCenterScale(object.location_, 0.15);
                objects_tmp.push_back(object);
            }
//        std::cout << object.location_.x << ", " << object.location_.y << ", " << object.location_.width << ", " << object.location_.height << std::endl;
        }
        NMS(objects_tmp, &objects, nmsThreshold);
        return 0;
    }

private:
    bool initialized_;
    cv::Size inputSize = {300, 300};
    std::vector<int> dims = {1, 3, 300, 300};
    float meanVals[3] = {0.5f, 0.5f, 0.5f};
    float normVals[3] = {0.007843f, 0.007843f, 0.007843f};
    float nmsThreshold = 0.5f;

    std::shared_ptr<MNN::Interpreter> m_interpreter = nullptr;
    MNN::Session *m_session = nullptr;
    MNN::Tensor *m_input_tensor = nullptr;
    std::shared_ptr<MNN::CV::ImageProcess> m_process = nullptr;

    int target = 15;

public:
    std::vector<std::string> class_names = {
            "background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair",
            "cow", "diningtable", "dog", "horse",
            "motorbike", "person", "pottedplant",
            "sheep", "sofa", "train", "tvmonitor"
    };
};

#endif //REDLIGHT_MODEL_PERSON_H
