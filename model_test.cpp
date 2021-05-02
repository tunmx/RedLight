//
// Created by YH-Mac on 2020/9/2.
//

#include "model_test.h"
#include "opencv2/opencv.hpp"

model_test::model_test() {
    float mean[3] = {0.5f, 0.5f, 0.5f};
    float normal[3] = {0.007843f, 0.007843f, 0.007843f};
    m_modelInfer = new ModelInfer("/Users/yh-mac/CLionProjects/RedLight/models/ssd.mnn", 1, mean, normal);
    m_modelInfer->Init("data", "detection_out", 300, 300);
}

int model_test::infer(const cv::Mat img) {
    std::vector<float> s = m_modelInfer->Infer(img);
    for (int i = 0; i < s.size(); ++i) {
        std::cout << s[i] << std::endl;
    }
}

model_test::~model_test() {

}