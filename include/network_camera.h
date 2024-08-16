#ifndef NETWORK_CAMERA_H
#define NETWORK_CAMERA_H

#include <opencv2/opencv.hpp>

class NetworkCamera
{
public:
    NetworkCamera();
    NetworkCamera(std::string &url);
    ~NetworkCamera();
    bool open();
    void close();
    void setUrl(std::string &url);
    bool getFrame(cv::Mat &frame);
    bool isOpen();

private:
    std::string rtsp_url;
    cv::VideoCapture cap;
    bool isOpened;
};

#endif // NETWORK_CAMERA_H