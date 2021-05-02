//
// Created by YH-Mac on 2020/8/13.
//

#include "common.h"
#include <algorithm>
#include <iostream>

float InterRectArea(const cv::Rect& a, const cv::Rect& b) {
    cv::Point left_top = cv::Point(MAX(a.x, b.x), MAX(a.y, b.y));
    cv::Point right_bottom = cv::Point(MIN(a.br().x, b.br().x), MIN(a.br().y, b.br().y));
    cv::Point diff = right_bottom - left_top;
    return (MAX(diff.x + 1, 0) * MAX(diff.y + 1, 0));
}

int ComputeIOU(const cv::Rect& rect1,
               const cv::Rect& rect2, float* iou,
               const std::string& type) {

    float inter_area = InterRectArea(rect1, rect2);
    if (type == "MIN") {
        *iou = inter_area / MIN(rect1.area(), rect2.area());
    }
    else {
        *iou = inter_area / (rect1.area() + rect2.area() - inter_area);
    }
    return 0;
}

int DrawFaceDetInfo(cv::Mat& img, const std::vector<bbox>& boxes, float scale){
    for (int j = 0; j < boxes.size(); ++j) {
        cv::Rect rect(boxes[j].x1 / scale, boxes[j].y1 / scale, boxes[j].x2 / scale - boxes[j].x1 / scale,
                      boxes[j].y2 / scale - boxes[j].y1 / scale);
        cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 1, 8, 0);
        char test[80];
        sprintf(test, "%f", boxes[j].s);

//        cv::putText(img, test, cv::Size((boxes[j].x1 / scale), boxes[j].y1 / scale), cv::FONT_HERSHEY_COMPLEX, 0.5,
//                    cv::Scalar(0, 255, 255));
//        cv::circle(img, cv::Point(boxes[j].point[0]._x / scale, boxes[j].point[0]._y / scale), 1,
//                   cv::Scalar(0, 0, 225), 4);
//        cv::circle(img, cv::Point(boxes[j].point[1]._x / scale, boxes[j].point[1]._y / scale), 1,
//                   cv::Scalar(0, 255, 225), 4);
//        cv::circle(img, cv::Point(boxes[j].point[2]._x / scale, boxes[j].point[2]._y / scale), 1,
//                   cv::Scalar(255, 0, 225), 4);
//        cv::circle(img, cv::Point(boxes[j].point[3]._x / scale, boxes[j].point[3]._y / scale), 1,
//                   cv::Scalar(0, 255, 0), 4);
//        cv::circle(img, cv::Point(boxes[j].point[4]._x / scale, boxes[j].point[4]._y / scale), 1,
//                   cv::Scalar(255, 0, 0), 4);
    }

    return 0;
}

int DrawPersonDetInfo(cv::Mat& img, const std::vector<ObjectInfo>& objects){
    int num_objects = static_cast<int>(objects.size());
    std::cout << "num_objects: " << objects.size() << std::endl;
    for (int i = 0; i < num_objects; ++i) {
        std::cout << "location: " << objects[i].name_ << ", " << objects[i].location_ << "," << objects[i].score_ << std::endl;
        cv::rectangle(img, objects[i].location_, cv::Scalar(255, 0, 255), 2);
//        char text[256];
//        sprintf(text, "%s %.1f%%", objects[i].name_.c_str(), objects[i].score_ * 100);
//        int baseLine = 0;
//        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
//        cv::putText(img, text, cv::Point(objects[i].location_.x, objects[i].location_.y + label_size.height),
//                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    return 0;
}

cv::Rect rectCenterScale(cv::Rect rect, cv::Size size)
{
    rect = rect + size;
    cv::Point pt;
    pt.x = cvRound(size.width/2.0);
    pt.y = cvRound(size.height/2.0);
    return (rect-pt);
}

cv::Rect rectCenterScale(cv::Rect rect, float scale)
{
    cv::Size size(rect.width * scale, rect.height * scale);
    rect = rect + size;
    cv::Point pt;
    pt.x = cvRound(size.width/2.0);
    pt.y = cvRound(size.height/2.0);
    return (rect-pt);
}