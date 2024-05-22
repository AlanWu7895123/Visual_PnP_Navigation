#ifndef CAMERA_H
#define CAMERA_H

#include <opencv2/opencv.hpp>

class Camera
{
public:
    Camera(int cameraIndex = 0);
    ~Camera();
    bool open();
    void close();
    bool getFrame(cv::Mat &frame);

private:
    int cameraIndex;
    cv::VideoCapture cap;
};

#endif // CAMERA_H