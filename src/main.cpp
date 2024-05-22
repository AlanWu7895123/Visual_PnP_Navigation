#include <iostream>
#include <eigen3/Eigen/Dense>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "camera.h"
#include "image_processor.h"
#include "icp_registerer.h"
#include "read_file.h"

using namespace std;
using namespace cv;

std::mutex mtx;
std::condition_variable conditionVariable;
cv::Mat buffer;
bool bufferReady = false;
bool stopThreads = false;

void captureThread()
{
    Camera camera;
    if (!camera.open())
    {
        return;
    }

    cv::namedWindow("Camera", cv::WINDOW_AUTOSIZE);

    cv::Mat frame;
    cv::Mat yuvImage;
    cv::Mat rgbImage;

    while (!stopThreads)
    {
        if (!camera.getFrame(frame))
        {
            continue;
        }

        cvtColor(frame, yuvImage, COLOR_BGR2YUV_YVYU);
        // cvtColor(frame, yuvImage, COLOR_BGR2YUV_UYVY);

        cvtColor(yuvImage, rgbImage, COLOR_YUV2BGR_Y422);

        {
            std::lock_guard<std::mutex> lock(mtx);
            frame.copyTo(buffer);
            bufferReady = true;
        }
        conditionVariable.notify_one();
        cv::imshow("RGB Image", rgbImage);
        cv::imshow("frame", frame);

        cv::waitKey(100);
    }

    camera.close();
}

void processingThread()
{
    Matrix4 transformed = Matrix4::Zero();
    transformed << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;
    while (!stopThreads)
    {
        std::unique_lock<std::mutex> lock(mtx);
        conditionVariable.wait(lock, []
                               { return bufferReady || stopThreads; });

        if (stopThreads)
        {
            break;
        }

        if (bufferReady)
        {
            cv::Mat frame = buffer.clone();
            bufferReady = false;
            lock.unlock();

            ImageProcessor processor(frame);

            processor.convertToGray();
            processor.applyThreshold();
            processor.findContours();
            processor.detectCircles();
            // processor.saveResults();

            std::vector<std::pair<double, double>> points37;
            std::vector<std::pair<double, double>> pointsN;

            points37 = read37Points("../data/points.txt");
            pointsN = readNPoints("../data/circles.txt");

            ICPRegisterer ICPRegisterer(pointsN, points37, transformed);
            ICPRegisterer.computePCLICP();
            transformed = ICPRegisterer.getTransformed();

            cv::imshow("Processed Frame", processor.output);
            cv::waitKey(1);
        }
    }
}

int main()
{
    std::thread t1(captureThread);
    std::thread t2(processingThread);

    std::cout << "Press Enter to stop..." << std::endl;
    std::cin.get();

    {
        std::lock_guard<std::mutex> lock(mtx);
        stopThreads = true;
    }
    conditionVariable.notify_all();

    t1.join();
    t2.join();

    // Matrix4 transformed = Matrix4::Zero();
    // transformed << 1, 0, 0, 0,
    //     0, 1, 0, 0,
    //     0, 0, 1, 0,
    //     0, 0, 0, 1;

    // // transformed << 0.99894, 0.0460383, 0, -141.503,
    // //     0.0460383, 0.99894, 0, 171.779,
    // //     0, 0, 1, 1800,
    // //     0, 0, 0, 1;

    // // std::cout << "transformed1:" << std::endl
    // //           << transformed << std::endl;
    // std::vector<std::pair<double, double>> points37;
    // std::vector<std::pair<double, double>> pointsN;

    // points37 = read37Points("../data/points.txt");
    // pointsN = readNPoints("../data/circles2.txt");

    // ICPRegisterer ICPRegisterer(pointsN, points37, transformed);
    // ICPRegisterer.computePCLICP();
    // transformed = ICPRegisterer.getTransformed();
    // // std::cout << "Next transformed:" << std::endl
    // //           << transformed << std::endl;

    return 0;
}