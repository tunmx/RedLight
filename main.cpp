#include <iostream>
#include <stdio.h>
#include "opencv2/opencv.hpp"
#include "common/common.h"


#include "FaceDetector/FaceDetector.h"
#include "object/ObjectDetect.h"


using namespace std;

#if 0
int video(){
//    string param = "../models/face.param";
//    string bin = "../models/face.bin";
    string models_path = "/Users/yh-mac/CLionProjects/RedLight/models/";
    PersonDetect detect(models_path);
    Detector detector(models_path, false);


    cv::VideoCapture cap;
    cap.open("/Users/yh-mac/dataset/1593966516767525.mp4");
    cv::Mat frame;
    cv::namedWindow("video test");
    int frame_num = cap.get(cv::CAP_PROP_FRAME_COUNT);
    std::cout << "total frame number is: " << frame_num << std::endl;

    cv::Mat img_scale;
    float scale = 0.5;

    for (int i = 0; i < frame_num - 1; ++i)
    {
        cap >> frame;
        std::vector<ObjectInfo> objects;
        detect.detect(frame, objects);

        cv::resize(frame, img_scale, cv::Size(frame.cols * scale, frame.rows * scale));
        std::vector<bbox> boxes;

        detector.Detect(img_scale, boxes);


        DrawPersonDetInfo(frame, objects);
        DrawFaceDetInfo(frame, boxes, scale);

        imshow("video test", frame);
        if (cv::waitKey(20) == 'q')
        {
            break;
        }
    }

    cv::destroyWindow("video test");
    cap.release();
    return 0;
}
#endif

#if 0
int main() {

    const char *img_path = "/Users/yh-mac/Desktop/timg.jpeg";
    cv::Mat img_src = cv::imread(img_path);
    string models_path = "/Users/yh-mac/CLionProjects/RedLight/models/";
    ObjectDetect detect(models_path);
    Detector detector(models_path, false);

//    detect.Init("/Users/yh-mac/CLionProjects/RedLight/models/");
    std::vector<ObjectInfo> objects;
    detect.detect(img_src, objects);

    cv::Mat img_scale;
    float scale = 0.5;
    cv::resize(img_src, img_scale, cv::Size(img_src.cols * scale, img_src.rows * scale));
    std::vector<bbox> boxes;
    detector.Detect(img_scale, boxes);

    DrawPersonDetInfo(img_src, objects);
    DrawFaceDetInfo(img_src, boxes, scale);

    cv::Rect rect;
    rect = rectCenterScale(objects[0].location_, 0.15);

        for (int i = 0; i < boxes.size(); ++i) {
            cout << boxes[i].x1 << ", " << boxes[i].y1 << endl;
        }

    cv::imshow("result", img_src);
    cv::waitKey(0);


//    video();


    return 0;
}
#endif

#if 0
int main() {
    // Object det
    const char *img_path = "/Users/yh-mac/Desktop/iv-real.jpeg";
    cv::Mat img_src = cv::imread(img_path);
    string models_path = "/Users/yh-mac/CLionProjects/RedLight/models/";
    ObjectDetect detect(models_path);

    std::vector<ObjectInfo> objects;
    detect.detect(img_src, objects);

    DrawPersonDetInfo(img_src, objects);


    cv::imshow("result", img_src);
    cv::waitKey(0);

//    cv::wechat_qrcode
    return 0;
}
#endif


#if 0
class TotalDetector{
public:
    TotalDetector(const std::string &dir){
        face_detector = new Detector(dir);
        person_detector = new PersonDetect(dir);
//        std::string require= GetMD5(get_mac()+"%$####@");
        std::string require= "tunmniubia";
        std::string key_read;
        std::ifstream infile;
        infile.open("/Users/yh-mac/CLionProjects/RedLight/license.lic");
        if (!infile)
            cout << "Load license file error." << endl;
        else
            cout << "Load license file successful." << endl;
        getline (infile, key_read);

        if(require != key_read){
            cout << "Not a release" << endl;
        } else{
            cout << "release." << endl;
            is_release = true;
        }


    }
    ~TotalDetector(){
        delete face_detector;
        delete person_detector;
    }

    int face_detect(char* data,int w,int h, int c, std::vector<bbox> &boxes){
        cv::Mat img_scale(h,w,CV_8UC3 , data);
        face_detector->Detect(img_scale, boxes);
    }

    int face_detect(cv::Mat img, std::vector<bbox> &boxes){
        face_detector->Detect(img, boxes);
    }

    int person_detect(char* data,int w,int h, int c, std::vector<ObjectInfo> &objects){
        cv::Mat img_scale(h,w,CV_8UC3 , data);
        person_detector->detect(img_scale, objects);
        if(!is_release){
            if (limit < 20){
                cout << "Exceeding the maximum number of calls" << endl;
            }
            limit -=1;
        }
    }

    int person_detect(cv::Mat img, std::vector<ObjectInfo> &objects){
        person_detector->detect(img, objects);
        if(!is_release){
            if (limit < 20){
                cout << "Exceeding the maximum number of calls" << endl;
            }
            limit -=1;
        }
    }

private:
    Detector *face_detector;
    PersonDetect *person_detector;
    bool is_release = false;
    int limit = 20;
};
#endif

#if 0
int main(){
    string models_path = "/Users/yh-mac/CLionProjects/RedLight/models/";
    TotalDetector totalDetector(models_path);
    cv::Mat img_src = cv::imread("/Users/yh-mac/CLionProjects/RedLight/imgs/malu.png");
    std::vector<ObjectInfo> objects;
    for (int i = 0; i < 30; ++i) {
        totalDetector.person_detect(img_src, objects);
    }


    return 0;
}

#endif

#if 0
int main(){
    string models_path = "/Users/yh-mac/CLionProjects/RedLight/models/";
    TotalDetector totalDetector(models_path);

    cv::Mat img_src = cv::imread("/Users/yh-mac/CLionProjects/RedLight/imgs/malu.png");
    cv::Mat img_scale;
    float scale = 0.5;
    cv::resize(img_src, img_scale, cv::Size(img_src.cols * scale, img_src.rows * scale));

    std::vector<ObjectInfo> objects;
    std::vector<bbox> boxes;
//    totalDetector.person_detect(img_src, objects);
    totalDetector.face_detect(img_scale, boxes);
    DrawFaceDetInfo(img_src, boxes, 0.5);
//    DrawPersonDetInfo(img_src, objects);
    cv::imshow("result", img_src);
    cv::waitKey(0);

    for (int i = 0; i < boxes.size(); ++i) {
        cout << boxes[i].x1 << ", " << boxes[i].y1 << endl;
    }


    return 0;
}

#endif

#include "model_test.h"
#include "model_person.h"
//#include "persondet/PersonDetect.h"

#if 0
int main() {
//    model_test * obj = new model_test();
    cv::Mat img_src = cv::imread("/Users/yh-mac/CLionProjects/RedLight/imgs/malu.png");
    cv::Mat img_scale;
    cv::resize(img_src, img_scale, cv::Size(300, 300));
//    obj->infer(img_scale);

    string models_path = "/Users/yh-mac/CLionProjects/RedLight/models/";
    model_person detect(models_path);
    std::vector<ObjectInfo> objects;
    detect.detect(img_src, objects);
    DrawPersonDetInfo(img_src, objects);
    cv::imshow("result", img_src);
    cv::waitKey(0);

    return 0;
}
#endif


#include "ldmark/ldmark.h"

#if 0
int main() {
    ldmark ld("/Users/yh-mac/Downloads/source/tools/pack/models/zzlmk.mnn");
//    ld->Init("/Users/yh-mac/Downloads/source/tools/pack/models/zzlmk.mnn");
    cv::Mat image = cv::imread("/Users/yh-mac/Desktop/ss.png");
    cv::resize(image, image, cv::Size(112, 112));
    std::vector<float> out;
    ld.detect(image, out);
//    for (int i = 0; i < out.size(); ++i) {
//        std::cout << out[i] << std::endl;
//    }
    for (int i = 0; i < 106; i++) {
        float x = out[i * 2];
        float y = out[i * 2 + 1];
        printf("x:%f ,y:%f \n", x, y);
        cv::circle(
                image,
                cv::Point(static_cast<float>(x * 112), static_cast<float>(y * 112)),
                1,
                cv::Scalar(255, 0, 0),
                2);
    }
    cv::imshow("image", image);
    cv::waitKey(0);

    return 0;
}

#endif