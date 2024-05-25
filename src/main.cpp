#include <iostream>
#include <eigen3/Eigen/Dense>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <cmath>
#include <eigen3/Eigen/Core>
#include "camera.h"
#include "image_processor.h"
#include "icp_registerer.h"
#include "icp.h"
#include "read_file.h"

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
    // cv::Mat yuvImage;
    // cv::Mat rgbImage;

    while (!stopThreads)
    {
        if (!camera.getFrame(frame))
        {
            continue;
        }

        // cvtColor(frame, yuvImage, COLOR_BGR2YUV_YVYU);
        // cvtColor(yuvImage, rgbImage, COLOR_YUV2BGR_Y422);

        {
            std::lock_guard<std::mutex> lock(mtx);
            // rgbImage.copyTo(buffer);
            frame.copyTo(buffer);
            bufferReady = true;
        }
        conditionVariable.notify_one();
        // cv::imshow("RGB Image", rgbImage);
        cv::imshow("frame", frame);

        cv::waitKey(100);
    }

    camera.close();
}

void processingThread()
{

    Matrix3d rotationMatrix = Matrix3d::Identity();
    Vector2d translationVector = Vector2d::Zero();
    std::vector<std::pair<double, double>> points37;
    points37 = read37Points("../data/points.txt");
    ofstream out("../data/trajectory.txt", ios::out);
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

            vector<pair<double, double>> pointsN;
            vector<pair<double, double>> transformedPoints;

            pointsN = readNPoints("../data/circles.txt");
            if (pointsN.size() < 4)
            {
                cerr << "Not enough points in the source point cloud." << endl;
                continue;
            }

            ICPAlgorithm icpAlg;

            icpAlg.transformPointCloud(pointsN, rotationMatrix, translationVector, transformedPoints);

            vector<pair<double, double>> sourceCloud = transformedPoints;
            vector<pair<double, double>> targetCloud = points37;

            vector<int> correspondences;
            int flag = icpAlg.icp(sourceCloud, targetCloud, correspondences);

            cv::imshow("Processed Frame", processor.output);
            cv::waitKey(100);

            if (flag == 1)
                continue;

            icpAlg.calTransformed(pointsN, targetCloud, correspondences, rotationMatrix, translationVector);

            cv::Mat pose_inv = icpAlg.estimateCameraPose(sourceCloud, targetCloud, correspondences);
            cv::Mat camera_position = pose_inv(cv::Rect(3, 0, 1, 3));

            cout << "Camera Position in World Coordinates: " << endl
                 << "X: " << camera_position.at<double>(0) << ", "
                 << "Y: " << camera_position.at<double>(1) << ", "
                 << "Z: " << camera_position.at<double>(2) << endl;

            out << camera_position.at<double>(0) << " " << camera_position.at<double>(1) << endl;
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

    // string filename;
    // cin >> filename;
    // std::vector<std::pair<double, double>> points37;
    // points37 = read37Points("../data/points.txt");
    // Matrix3d rotationMatrix = Matrix3d::Identity();
    // Vector2d translationVector = Vector2d::Zero();

    // // rotationMatrix << 0.962266, -0.272109, 0,
    // //     0.272109, 0.962266, 0,
    // //     0, 0, 1;

    // translationVector << -26.8275, 361.368;

    // // 打印 Matrix3d
    // cout << "Rotation Matrix 0: " << endl;
    // cout << rotationMatrix << endl;

    // // 打印 Vector2d
    // cout << "Translation Vector 0: " << endl;
    // cout << translationVector << endl;

    // ImageProcessor processor(filename);
    // processor.readImage();
    // processor.convertToGray();
    // processor.applyThreshold();
    // processor.findContours();
    // processor.detectCircles();
    // processor.saveResults();

    // vector<pair<double, double>> pointsN;
    // vector<pair<double, double>> transformedPoints;

    // pointsN = readNPoints("../data/circles.txt");
    // if (pointsN.size() < 4)
    // {
    //     cerr << "Not enough points in the source point cloud." << endl;
    //     return 0;
    // }

    // ICPAlgorithm icpAlg;

    // icpAlg.transformPointCloud(pointsN, rotationMatrix, translationVector, transformedPoints);

    // vector<pair<double, double>> sourceCloud = transformedPoints;
    // vector<pair<double, double>> targetCloud = points37;

    // vector<int> correspondences;
    // icpAlg.icp(sourceCloud, targetCloud, correspondences);

    // icpAlg.calTransformed(sourceCloud, targetCloud, correspondences, rotationMatrix, translationVector);

    // // 打印 Matrix3d
    // cout << "Rotation Matrix 1: " << endl;
    // cout << rotationMatrix << endl;

    // // 打印 Vector2d
    // cout << "Translation Vector 1: " << endl;
    // cout << translationVector << endl;

    // cv::Mat pose_inv = icpAlg.estimateCameraPose(sourceCloud, targetCloud, correspondences);
    // cv::Mat camera_position = pose_inv(cv::Rect(3, 0, 1, 3));

    // cout << "Camera Position in World Coordinates: " << endl
    //      << "X: " << camera_position.at<double>(0) << ", "
    //      << "Y: " << camera_position.at<double>(1) << ", "
    //      << "Z: " << camera_position.at<double>(2) << endl;

    return 0;
}