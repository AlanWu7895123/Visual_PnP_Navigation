#include "camera.h"

Camera::Camera(int cameraIndex) : cameraIndex(cameraIndex) {}

Camera::~Camera()
{
    close();
}

bool Camera::open()
{
    cap.open(cameraIndex);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 2048);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1536);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open camera." << std::endl;
        return false;
    }
    return true;
}

void Camera::close()
{
    if (cap.isOpened())
    {
        cap.release();
    }
}

bool Camera::getFrame(cv::Mat &frame)
{
    if (!cap.isOpened())
    {
        std::cerr << "Error: Camera is not open." << std::endl;
        return false;
    }
    cap >> frame;
    if (frame.empty())
    {
        std::cerr << "Error: Captured frame is empty." << std::endl;
        return false;
    }
    return true;
}