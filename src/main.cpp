#include <iostream>
#include <eigen3/Eigen/Dense>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <cmath>
#include <eigen3/Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include "camera.h"
#include "image_processor.h"
#include "icp.h"
#include "utils.h"

using namespace cv;

std::mutex mtx;
std::condition_variable conditionVariable;
bool bufferReady = false;
bool stopThreads = false;

std::atomic<State> currentState(State::INIT);
std::vector<std::thread> threads;
Camera camera;
cv::Mat buffer;
std::deque<std::pair<double, double>> window;
std::pair<double, double> avg_position;
const size_t windowSize = 10;
std::vector<double> weights = {1, 1, 1, 1, 1, 2, 2, 2, 2, 3};
Matrix3d rotationMatrix = Matrix3d::Identity();
Vector2d translationVector = Vector2d::Zero();
std::vector<std::pair<double, double>> points37;
ofstream out("../data/trajectory.txt", ios::out);

void initThread()
{
    if (!camera.open())
    {
        currentState = State::FINISHED;
    }
    else
    {
        points37 = read37Points("../data/points.txt");
        translationVector << -1024,
            -768;
        currentState = State::CAPUTRE_IMAGE;
    }
}

void finishThread()
{
    camera.close();
}

void captureImageThread()
{
    cv::namedWindow("Camera", cv::WINDOW_AUTOSIZE);
    cv::Mat frame;
    if (!camera.getFrame(frame))
    {
        cout << "get camera image failed" << endl;
        currentState = State::CAPUTRE_IMAGE;
    }
    else
    {
        frame.copyTo(buffer);
        cv::imshow("frame", frame);
        cv::waitKey(1);
        currentState = State::FEATURE_DETECT;
    }
}

void featureDetectThread()
{
    cv::Mat frame = buffer.clone();
    ImageProcessor processor(frame);
    processor.convertToGray();
    processor.applyThreshold();
    processor.findContours();
    processor.detectCircles();
    // processor.saveResults();

    cv::imshow("Processed Frame", processor.output);
    cv::waitKey(1);
    currentState = State::COORDINATE_CALCULATION;
}

void coordinateCalculationThread()
{
    vector<pair<double, double>> pointsN;
    vector<pair<double, double>> transformedPoints;

    pointsN = readNPoints("../data/circles.txt");
    if (pointsN.size() > points37.size())
    {
        std::cout << "There are so many detect points..." << std::endl;
        currentState = State::CAPUTRE_IMAGE;
        return;
    }

    ICPAlgorithm icpAlg;

    icpAlg.transformPointCloud(pointsN, rotationMatrix, translationVector, transformedPoints);

    vector<pair<double, double>> sourceCloud = transformedPoints;
    vector<pair<double, double>> targetCloud = points37;

    vector<int> correspondences;
    // cout << "start icp" << endl;

    int errorFlag = icpAlg.icp(pointsN, sourceCloud, targetCloud, correspondences);

    if (errorFlag == 1)
    {
        cerr << "Error Matching." << endl;
    }
    else
    {
        // icpAlg.calTransformed(pointsN, targetCloud, correspondences, rotationMatrix, translationVector);

        cv::Mat pose_inv = icpAlg.estimateCameraPose(pointsN, targetCloud, correspondences);

        cv::Mat camera_position = pose_inv(cv::Rect(3, 0, 1, 3));

        // average filter
        window.push_back({camera_position.at<double>(0), camera_position.at<double>(1)});
        if (window.size() > windowSize)
        {
            // if (icpAlg.distance(avg_position, window[windowSize]) > 500)
            // {
            //     std::cout << "The new point is so far..." << std::endl;
            //     window.pop_back();
            //     currentState = State::CAPUTRE_IMAGE;
            //     return;
            // }
            // else
            // {
            cout << "rotationMatrix=" << rotationMatrix << endl;
            cout << "translationVector=" << translationVector << endl;
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    rotationMatrix(i, j) = pose_inv.at<double>(i, j);
                }
            }
            for (int i = 0; i < 2; ++i)
            {
                translationVector(i) = pose_inv.at<double>(i, 3);
            }
            window.pop_front();
            // }
        }
        if (window.size() == windowSize)
        {
            pair<double, double> filteredPoint = weightedMovingAverageFilter(window, weights);
            avg_position = filteredPoint;
            std::cout << "Filtered coordinates: (" << filteredPoint.first << ", " << filteredPoint.second << ")" << std::endl;
            out << filteredPoint.first
                << " " << filteredPoint.second << endl;
        }
        else
        {
            std::cout << "Insufficient data for filtering. Waiting for more points..." << std::endl;
        }
    }
    currentState = State::CAPUTRE_IMAGE;
}

void captureThread()
{

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

        cv::waitKey(1);
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

    // average filter
    std::deque<std::pair<double, double>> window;
    std::pair<double, double> avg_position;
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
            int flag = icpAlg.icp(pointsN, sourceCloud, targetCloud, correspondences);

            cv::imshow("Processed Frame", processor.output);
            cv::waitKey(1);

            if (flag == 1)
                continue;

            // icpAlg.calTransformed(pointsN, targetCloud, correspondences, rotationMatrix, translationVector);

            cv::Mat pose_inv = icpAlg.estimateCameraPose(pointsN, targetCloud, correspondences);
            cv::Mat camera_position = pose_inv(cv::Rect(3, 0, 1, 3));

            cout << "rotationMatrix=" << rotationMatrix << endl;
            cout << "translationVector=" << translationVector << endl;
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    rotationMatrix(i, j) = pose_inv.at<double>(i, j);
                }
            }
            for (int i = 0; i < 2; ++i)
            {
                translationVector(i) = pose_inv.at<double>(i, 3);
            }

            // average filter
            window.push_back({camera_position.at<double>(0), camera_position.at<double>(1)});
            if (window.size() > windowSize)
            {
                // if (icpAlg.distance(avg_position, window[windowSize]) > 500)
                // {
                //     window.pop_back();
                // }
                // else
                // {
                window.pop_front();
                // }
            }
            if (window.size() == windowSize)
            {
                pair<double, double> filteredPoint = weightedMovingAverageFilter(window, weights);
                avg_position = filteredPoint;
                std::cout << "Filtered coordinates: (" << filteredPoint.first << ", " << filteredPoint.second << ")" << std::endl;
                out << filteredPoint.first << " " << filteredPoint.second << endl;
            }
            else
            {
                std::cout << "Insufficient data for filtering. Waiting for more points..." << std::endl;
            }

            // cout << "Camera Position in World Coordinates: " << endl
            //      << "X: " << camera_position.at<double>(0) << ", "
            //      << "Y: " << camera_position.at<double>(1) << ", "
            //      << "Z: " << camera_position.at<double>(2) << endl;

            // out << camera_position.at<double>(0) << " " << camera_position.at<double>(1) << endl;
        }
    }
}

void testThread()
{
    string filename;
    cout << "Please enter the image name" << endl;
    cin >> filename;
    std::vector<std::pair<double, double>> points37;
    points37 = read37Points("../data/points.txt");
    Matrix3d rotationMatrix = Matrix3d::Identity();
    Vector2d translationVector = Vector2d::Zero();

    translationVector << -1024,
        -768;

    // translationVector << -932,
    //     -596;

    // translationVector << -1007,
    //     -795;

    // translationVector << -1065,
    //     -1045;

    // rotationMatrix << 0.9999853066492733, -0.005420244990833255, -8.619627686081539e-05,
    //     0.00542025113451311, 0.9999853077960045, 7.120230992889232e-05,
    //     8.580907648380496e-05, -7.166846919582479e-05, 0.9999999937502164;

    // // 03->04.jpg normal
    // rotationMatrix << 0.993147, 0.116876, 0,
    //     -0.116876, 0.993147, 0,
    //     0, 0, 1;

    // translationVector << -129.98,
    //     1066.62;

    // // 03->04.jpg delete points
    // rotationMatrix << 0.997393, 0.072167, 0,
    //     -0.072167, 0.997393, 0,
    //     0, 0, 1;

    // translationVector << -98.5932,
    //     992.045;

    // // 02->03.jpg
    // rotationMatrix << 0.999338, -0.0363679, 0,
    //     0.0363679, 0.999338, 0,
    //     0, 0, 1;

    // translationVector << -186.88,
    //     929.689;

    // // 01->02.jpg
    // rotationMatrix << 0.984061, -0.177832, 0,
    //     0.177832, 0.984061, 0,
    //     0, 0, 1;

    // translationVector << -84.9096,
    //     664.075;

    ImageProcessor processor(filename);
    processor.readImage();
    processor.convertToGray();
    processor.applyThreshold();
    processor.findContours();
    processor.detectCircles();
    processor.saveResults();

    vector<pair<double, double>> pointsN;
    vector<pair<double, double>> _pointsN;
    vector<pair<double, double>> transformedPoints;

    pointsN = readNPoints("../data/circles.txt");
    // _pointsN = readNPoints("../data/circles0.txt");
    // if (pointsN.size() < 4)
    // {
    //     cerr << "Not enough points in the source point cloud." << endl;
    //     return 0;
    // }

    ICPAlgorithm icpAlg;
    // cout << "rotationMatrix=" << rotationMatrix << endl;
    // cout << "translationVector=" << translationVector << endl;

    icpAlg.transformPointCloud(pointsN, rotationMatrix, translationVector, transformedPoints);

    // if (pointsN == transformedPoints)
    //     cout << "same" << endl;

    vector<pair<double, double>> sourceCloud = transformedPoints;
    vector<pair<double, double>> targetCloud = points37;

    vector<int> correspondences;
    icpAlg.icp(pointsN, sourceCloud, targetCloud, correspondences);

    if (pointsN.size() < 4)
    {
        cerr << "Not enough points in the source point cloud." << endl;
        return;
    }

    icpAlg.calTransformed(pointsN, targetCloud, correspondences, rotationMatrix, translationVector);

    // // 打印 Matrix3d
    // cout << "Rotation Matrix: " << endl;
    // cout << rotationMatrix << endl;

    // // 打印 Vector2d
    // cout << "Translation Vector: " << endl;
    // cout << translationVector << endl;

    cv::Mat pose_inv = icpAlg.estimateCameraPose(pointsN, targetCloud, correspondences);
    cout << "----pose_inv----" << endl;
    cout << pose_inv << endl;
    // cout << "test _pointsN position" << endl;
    // cv::Mat _pose_inv = icpAlg.estimateCameraPose(_pointsN, targetCloud, correspondences);
    // Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> eigen_map(pose_inv.ptr<double>(), 3, 3);
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            rotationMatrix(i, j) = pose_inv.at<double>(i, j);
        }
    }
    for (int i = 0; i < 2; ++i)
    {
        translationVector(i) = pose_inv.at<double>(i, 3);
    }
    cout << "----new rotationMatrix----" << endl;
    cout << rotationMatrix << endl;
    cout << "----new translationMatrix----" << endl;
    cout << translationVector << endl;

    cv::Mat camera_position = pose_inv(cv::Rect(3, 0, 1, 3));

    cout << "Camera Position in World Coordinates: " << endl
         << "X: " << camera_position.at<double>(0) << ", "
         << "Y: " << camera_position.at<double>(1) << ", "
         << "Z: " << camera_position.at<double>(2) << endl;
}

int main()
{
    // std::thread t1(captureThread);
    // std::thread t2(processingThread);

    // std::cout << "Press Enter to stop..." << std::endl;
    // std::cin.get();

    // {
    //     std::lock_guard<std::mutex> lock(mtx);
    //     stopThreads = true;
    // }
    // conditionVariable.notify_all();

    // t1.join();
    // t2.join();

    int mode;
    cout << "choose mode, 0 is test, 1 is normal\n";
    cin >> mode;
    if (mode == 0)
    {
        currentState = State::TEST;
    }
    else if (mode == 1)
    {
        currentState = State::INIT;
    }
    else
    {
        cout << "error mode, please run again" << endl;
        return 0;
    }

    while (currentState != State::FINISHED)
    {
        if (kbhit())
        {
            char ch = getchar();
            if (ch == 'q' || ch == 'Q')
            {
                std::cout << "\nQ key pressed. Stopping the system..." << std::endl;
                threads.emplace_back(finishThread);
                currentState = State::FINISHED;
                break;
            }
        }

        for (auto it = threads.begin(); it != threads.end();)
        {
            if (it->joinable() && it->get_id() == std::this_thread::get_id())
            {
                it->join();
                it = threads.erase(it);
            }
            else
            {
                ++it;
            }
        }

        cout << "currentState=" << currentState.load() << endl;
        switch (currentState.load())
        {
        case State::INIT:
            std::cout << "Initial state." << std::endl;
            threads.emplace_back(initThread);
            currentState = State::CAPUTRE_IMAGE;
            break;
        case State::CAPUTRE_IMAGE:
            std::cout << "Launching Capture Image." << std::endl;
            threads.emplace_back(captureImageThread);
            currentState = State::CAPUTRE_IMAGE_RUNNING;
            break;
        case State::FEATURE_DETECT:
            std::cout << "Launching Feature Detect." << std::endl;
            threads.emplace_back(featureDetectThread);
            currentState = State::FEATURE_DETECT_RUNNING;
            break;
        case State::COORDINATE_CALCULATION:
            std::cout << "Launching Coordinate Calculation." << std::endl;
            threads.emplace_back(coordinateCalculationThread);
            currentState = State::COORDINATE_CALCULATION_RUNNING;
            break;
        case State::TEST:
            threads.emplace_back(testThread);
            currentState = State::FINISHED;
            break;
        default:
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    for (auto &t : threads)
    {
        if (t.joinable())
        {
            t.join();
        }
    }

    std::cout << "System finished." << std::endl;

    return 0;
}