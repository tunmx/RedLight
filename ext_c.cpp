//
// Created by YH-Mac on 2020/8/14.
//

#include <iostream>
#include "opencv2/opencv.hpp"
#include "persondet/PersonDetect.h"
#include "FaceDetector/FaceDetector.h"

using namespace std;

class TotalDetector {
public:
    TotalDetector(const std::string &dir) {
        face_detector = new Detector(dir);
        person_detector = new PersonDetect(dir);
//        std::string require= GetMD5(get_mac()+"%$####@");
        std::string require = "tunmniubi";
        std::string key_read;
        std::ifstream infile;
        infile.open("/Users/yh-mac/CLionProjects/RedLight/license.lic");
        if (!infile)
            cout << "Load license file error." << endl;
        else
            cout << "Load license file successful." << endl;
        getline(infile, key_read);

        if (require != key_read) {
            cout << "Not a release" << endl;
        } else {
            cout << "release." << endl;
            is_release = true;
        }


    }

    ~TotalDetector() {
        delete face_detector;
        delete person_detector;
    }

    int face_detect(char *data, int w, int h, int c, std::vector<bbox> &boxes) {
        cv::Mat img_scale(h, w, CV_8UC3, data);
        face_detector->Detect(img_scale, boxes);
    }

    int face_detect(cv::Mat img, std::vector<bbox> &boxes) {
        face_detector->Detect(img, boxes);
    }

    int person_detect(char *data, int w, int h, int c, std::vector<ObjectInfo> &objects) {
        cv::Mat img_scale(h, w, CV_8UC3, data);
        person_detector->detect(img_scale, objects);
        if (!is_release) {
            if (limit < 20) {
                cout << "Exceeding the maximum number of calls" << endl;
            }
            limit -= 1;
        }
    }

    int person_detect(cv::Mat img, std::vector<ObjectInfo> &objects) {
        person_detector->detect(img, objects);
    }

private:
    Detector *face_detector;
    PersonDetect *person_detector;
    bool is_release = false;
    int limit = 20;
};

extern "C" long InitModelsSession(char *models_path) {
    std::string dir(models_path);
    TotalDetector *totalDetector = new TotalDetector(dir);
    return (long) totalDetector;
}

extern "C" void FaceDetect(long session, char *image, int w, int h, int minSize, float **output, int *outputNum) {
    TotalDetector *session_instance = (TotalDetector *) session;
    std::vector<bbox> faceInfo;
    session_instance->face_detect(image, w, h, minSize, faceInfo);
    *outputNum = faceInfo.size();
    *output = new float[faceInfo.size() * 15];
    float *data = *output;
    for (int i = 0; i < faceInfo.size(); i++) {
        data[i * 15 + 0] = faceInfo[i].s;
        data[i * 15 + 1] = faceInfo[i].x1;
        data[i * 15 + 2] = faceInfo[i].y1;
        data[i * 15 + 3] = faceInfo[i].x2;
        data[i * 15 + 4] = faceInfo[i].y2;
        data[i * 15 + 5] = faceInfo[i].point[0]._x;
        data[i * 15 + 6] = faceInfo[i].point[0]._y;
        data[i * 15 + 7] = faceInfo[i].point[1]._x;
        data[i * 15 + 8] = faceInfo[i].point[1]._y;
        data[i * 15 + 9] = faceInfo[i].point[2]._x;
        data[i * 15 + 10] = faceInfo[i].point[2]._y;
        data[i * 15 + 11] = faceInfo[i].point[3]._x;
        data[i * 15 + 12] = faceInfo[i].point[3]._y;
        data[i * 15 + 13] = faceInfo[i].point[4]._x;
        data[i * 15 + 14] = faceInfo[i].point[4]._y;
    }

}

extern "C" void PersonDetect(long session, char *image, int w, int h, int c, float **output, int *outputNum) {
    TotalDetector *session_instance = (TotalDetector *) session;
    std::vector<ObjectInfo> objects;
    session_instance->person_detect(image, w, h, c, objects);
    *outputNum = objects.size();
    *output = new float[objects.size() * 5];
    float *data = *output;
    for (int i = 0; i < objects.size(); ++i) {
        data[i * 5 + 0] = objects[i].score_;
        data[i * 5 + 1] = objects[i].location_.x;
        data[i * 5 + 2] = objects[i].location_.y;
        data[i * 5 + 3] = objects[i].location_.x + objects[i].location_.width;
        data[i * 5 + 4] = objects[i].location_.y + objects[i].location_.height;
    }
}

extern "C" void ReleaseModelsSession(long session) {
    TotalDetector *totalDetector = (TotalDetector *) session;
    delete totalDetector;
}

extern "C" void FreeFloat(float **memblock) {
    delete[]*memblock;
}

extern "C" void FreeFloat1D(float **memblock) {
    delete[]*memblock;
}


