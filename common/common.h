//
// Created by YH-Mac on 2020/8/13.
//
#include <cmath>
#include <string>
#include <vector>

#include "opencv2/imgproc.hpp"
//#include "opencv2/opencv.hpp"

#ifndef REDLIGHT_COMMON_H
#define REDLIGHT_COMMON_H


struct ObjectInfo {
    std::string name_;
    cv::Rect location_;
    float score_;
};

struct Point{
    float _x;
    float _y;
};
struct bbox{
    float x1;
    float y1;
    float x2;
    float y2;
    float s;
    Point point[5];
};

struct box{
    float cx;
    float cy;
    float sx;
    float sy;
};

float InterRectArea(const cv::Rect &a, const cv::Rect &b);

int ComputeIOU(const cv::Rect &rect1, const cv::Rect &rect2, float *iou, const std::string &type = "UNION");

int GenerateAnchors(const int &width, const int &height, const std::vector<std::vector<float>> &min_boxes,
                    const std::vector<float> &strides, std::vector<std::vector<float>> *anchors);

template<typename T>
int const NMS(const std::vector<T> &inputs, std::vector<T> *result,
              const float &threshold, const std::string &type = "UNION") {
    result->clear();
    if (inputs.size() == 0)
        return -1;

    std::vector<T> inputs_tmp;
    inputs_tmp.assign(inputs.begin(), inputs.end());
    std::sort(inputs_tmp.begin(), inputs_tmp.end(),
              [](const T &a, const T &b) {
                  return a.score_ > b.score_;
              });

    std::vector<int> indexes(inputs_tmp.size());

    for (int i = 0; i < indexes.size(); i++) {
        indexes[i] = i;
    }

    while (indexes.size() > 0) {
        int index_good = indexes[0];
        std::vector<int> indexes_tmp = indexes;
        indexes.clear();
        std::vector<int> indexes_nms;
        indexes_nms.push_back(index_good);
        float total = exp(inputs_tmp[index_good].score_);
        for (int i = 1; i < static_cast<int>(indexes_tmp.size()); ++i) {
            int index_tmp = indexes_tmp[i];
            float iou = 0.0f;
            ComputeIOU(inputs_tmp[index_good].location_, inputs_tmp[index_tmp].location_, &iou, type);
            if (iou <= threshold) {
                indexes.push_back(index_tmp);
            } else {
                indexes_nms.push_back(index_tmp);
                total += exp(inputs_tmp[index_tmp].score_);
            }
        }
        if ("BLENDING" == type) {
            T t;
            memset(&t, 0, sizeof(t));
            for (auto index : indexes_nms) {
                float rate = exp(inputs_tmp[index].score_) / total;
                t.score_ += rate * inputs_tmp[index].score_;
                t.location_.x += rate * inputs_tmp[index].location_.x;
                t.location_.y += rate * inputs_tmp[index].location_.y;
                t.location_.width += rate * inputs_tmp[index].location_.width;
                t.location_.height += rate * inputs_tmp[index].location_.height;
            }
            result->push_back(t);
        } else {
            result->push_back(inputs_tmp[index_good]);
        }

    }
    return 0;
}

cv::Rect rectCenterScale(cv::Rect rect, cv::Size size);
cv::Rect rectCenterScale(cv::Rect rect, float scale);


int DrawFaceDetInfo(cv::Mat& img, const std::vector<bbox>& boxes, float scale);

int DrawPersonDetInfo(cv::Mat& img, const std::vector<ObjectInfo>& objects);


#endif //REDLIGHT_COMMON_H
