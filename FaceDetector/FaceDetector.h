//
// Created by dl on 19-7-19.
//

#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H

#include <string>
#include <stack>
#include "net.h"
#include "opencv2/opencv.hpp"
#include <chrono>
#include "../common/common.h"
using namespace std::chrono;

class Timer
{
public:
    std::stack<high_resolution_clock::time_point> tictoc_stack;

    void tic()
    {
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        tictoc_stack.push(t1);
    }

    double toc(std::string msg = "", bool flag = true)
    {
        double diff = duration_cast<milliseconds>(high_resolution_clock::now() - tictoc_stack.top()).count();
        if(msg.size() > 0){
            if (flag)
                printf("%s time elapsed: %f ms\n", msg.c_str(), diff);
        }

        tictoc_stack.pop();
        return diff;
    }
    void reset()
    {
        tictoc_stack = std::stack<high_resolution_clock::time_point>();
    }
};



class Detector
{

public:
    Detector();

    void Init(const std::string &model_param, const std::string &model_bin);
    void Init(const std::string &model_path);

    Detector(const std::string &model_param, const std::string &model_bin, bool retinaface = false);
    Detector(const std::string &models_path, bool retinaface = false);

    inline void Release();

    void nms(std::vector<bbox> &input_boxes, float NMS_THRESH);

    void Detect(cv::Mat& bgr, std::vector<bbox>& boxes);

    void create_anchor(std::vector<box> &anchor, int w, int h);

    void create_anchor_retinaface(std::vector<box> &anchor, int w, int h);

    inline void SetDefaultParams();

    static inline bool cmp(bbox a, bbox b);

    ~Detector();

public:
    float _nms;
    float _threshold;
    float _mean_val[3];
    bool _retinaface;

    ncnn::Net *Net;
};
#endif //
