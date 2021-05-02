//
// Created by YH-Mac on 2020/9/2.
//

#ifndef REDLIGHT_MODEL_TEST_H
#define REDLIGHT_MODEL_TEST_H
#include "model_infer.h"

class model_test {
public:
    model_test();
    ~model_test();
    int infer(const cv::Mat img);

private:
    ModelInfer * m_modelInfer;
};


#endif //REDLIGHT_MODEL_TEST_H
