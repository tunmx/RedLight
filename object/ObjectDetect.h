//
// Created by Tunm on 2020/8/12.
//

#ifndef REDLIGHT_PERSONDETECT_H
#define REDLIGHT_PERSONDETECT_H

#include "MNN/Interpreter.hpp"
#include "MNN/ImageProcess.hpp"
#include "MNN/MNNDefine.h"
#include "MNN/Tensor.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "../common/common.h"


class ObjectDetect {
public:
    ObjectDetect();
    ObjectDetect(const std::string &models_path);

    ~ObjectDetect();

    int Init(const std::string &models_path);

    int detect(const cv::Mat& img_src, std::vector<ObjectInfo>& objects);

private:
    bool initialized_;
    cv::Size inputSize = {360, 360};
    std::vector<int> dims = {1, 3, 360, 360};
    float meanVals[3] = {0.5f, 0.5f, 0.5f};
    float normVals[3] = {0.007843f, 0.007843f, 0.007843f};
    float nmsThreshold = 0.5f;
    float clsThreshold = 0.5f;

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


#endif //REDLIGHT_PERSONDETECT_H
