#include "network_camera.h"

NetworkCamera::NetworkCamera(std::string &url) : rtsp_url(url), cap(), isOpened(false) {}

NetworkCamera::NetworkCamera() : rtsp_url(), cap(), isOpened(false) {}

NetworkCamera::~NetworkCamera()
{
    close();
}

bool NetworkCamera::open()
{
    cap.open(rtsp_url);
    isOpened = cap.isOpened();
    if (!isOpened)
    {
        std::cerr << "Error: Could not open RTSP stream." << std::endl;
    }
    return isOpened;
}

void NetworkCamera::close()
{
    if (isOpened)
    {
        cap.release();
        isOpened = false;
    }
}

void NetworkCamera::setUrl(std::string &url)
{
    rtsp_url = url;
}

bool NetworkCamera::getFrame(cv::Mat &frame)
{
    if (!isOpened)
    {
        std::cerr << "Error: RTSP stream is not opened." << std::endl;
        return false;
    }
    // cap >> frame;
    cap.read(frame);
    if (frame.empty())
    {
        std::cerr << "Error: Could not read frame from RTSP stream." << std::endl;
        return false;
    }
    return true;
}

bool NetworkCamera::isOpen()
{
    return isOpened;
}