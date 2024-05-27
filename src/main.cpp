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

const size_t windowSize = 10;
std::vector<double> weights = {1, 1, 1, 1, 1, 2, 2, 2, 2, 3};
std::deque<Matrix3d> rotationHistory;
std::deque<Vector2d> translationHistory;

void addHistory(const Matrix3d &rotationMatrix, const Vector2d &translationVector)
{
    if (rotationHistory.size() >= 10)
    {
        rotationHistory.pop_front();
        translationHistory.pop_front();
    }
    rotationHistory.push_back(rotationMatrix);
    translationHistory.push_back(translationVector);
}

VectorXd predictNextValue(const std::deque<VectorXd> &history)
{
    int n = history.size();
    if (n < 2)
        return history.back();

    MatrixXd X(n, 2);
    MatrixXd Y(n, history.front().size());

    for (int i = 0; i < n; ++i)
    {
        X(i, 0) = 1;
        X(i, 1) = i;
        Y.row(i) = history[i].transpose();
    }

    MatrixXd beta = (X.transpose() * X).ldlt().solve(X.transpose() * Y);

    VectorXd nextX(2);
    nextX << 1, n;

    return (nextX.transpose() * beta).transpose();
}

Matrix3d predictNextRotation()
{
    std::deque<VectorXd> elementsHistory[9];
    for (const auto &mat : rotationHistory)
    {
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                elementsHistory[i * 3 + j].push_back(VectorXd::Constant(1, mat(i, j)));
            }
        }
    }

    Matrix3d predictedRotation;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            predictedRotation(i, j) = predictNextValue(elementsHistory[i * 3 + j])(0);
        }
    }

    return predictedRotation;
}

Vector2d predictNextTranslation()
{
    std::deque<VectorXd> elementsHistory[2];
    for (const auto &vec : translationHistory)
    {
        for (int i = 0; i < 2; ++i)
        {
            elementsHistory[i].push_back(VectorXd::Constant(1, vec(i)));
        }
    }

    Vector2d predictedTranslation;
    for (int i = 0; i < 2; ++i)
    {
        predictedTranslation(i) = predictNextValue(elementsHistory[i])(0);
    }

    return predictedTranslation;
}

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

        cv::waitKey(1);
    }

    camera.close();
}

std::pair<double, double> weightedMovingAverageFilter(
    const std::deque<std::pair<double, double>> &window,
    const std::vector<double> &weights)
{
    double sumWeights = 0;
    double weightedSumX = 0;
    double weightedSumY = 0;

    for (size_t i = 0; i < window.size(); ++i)
    {
        weightedSumX += window[i].first * weights[i];
        weightedSumY += window[i].second * weights[i];
        sumWeights += weights[i];
    }

    return {weightedSumX / sumWeights, weightedSumY / sumWeights};
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

            icpAlg.calTransformed(pointsN, targetCloud, correspondences, rotationMatrix, translationVector);

            addHistory(rotationMatrix, translationVector);
            rotationMatrix = predictNextRotation();
            translationVector = predictNextTranslation();

            cv::Mat pose_inv = icpAlg.estimateCameraPose(pointsN, targetCloud, correspondences);
            cv::Mat camera_position = pose_inv(cv::Rect(3, 0, 1, 3));

            // average filter
            window.push_back({camera_position.at<double>(0), camera_position.at<double>(1)});
            if (window.size() > windowSize)
            {
                if (icpAlg.distance(avg_position, window[windowSize]) > 500)
                {
                    window.pop_back();
                }
                else
                {
                    window.pop_front();
                }
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

    // // // 03->04.jpg normal
    // // rotationMatrix << 0.993147, 0.116876, 0,
    // //     -0.116876, 0.993147, 0,
    // //     0, 0, 1;

    // // translationVector << -129.98,
    // //     1066.62;

    // // // 03->04.jpg delete points
    // // rotationMatrix << 0.997393, 0.072167, 0,
    // //     -0.072167, 0.997393, 0,
    // //     0, 0, 1;

    // // translationVector << -98.5932,
    // //     992.045;

    // // // 02->03.jpg
    // // rotationMatrix << 0.999338, -0.0363679, 0,
    // //     0.0363679, 0.999338, 0,
    // //     0, 0, 1;

    // // translationVector << -186.88,
    // //     929.689;

    // // // 01->02.jpg
    // // rotationMatrix << 0.984061, -0.177832, 0,
    // //     0.177832, 0.984061, 0,
    // //     0, 0, 1;

    // // translationVector << -84.9096,
    // //     664.075;

    // ImageProcessor processor(filename);
    // processor.readImage();
    // processor.convertToGray();
    // processor.applyThreshold();
    // processor.findContours();
    // processor.detectCircles();
    // processor.saveResults();

    // vector<pair<double, double>> pointsN;
    // vector<pair<double, double>> _pointsN;
    // vector<pair<double, double>> transformedPoints;

    // pointsN = readNPoints("../data/circles.txt");
    // _pointsN = readNPoints("../data/circles0.txt");
    // // if (pointsN.size() < 4)
    // // {
    // //     cerr << "Not enough points in the source point cloud." << endl;
    // //     return 0;
    // // }

    // ICPAlgorithm icpAlg;

    // icpAlg.transformPointCloud(pointsN, rotationMatrix, translationVector, transformedPoints);

    // // if (pointsN == transformedPoints)
    // //     cout << "same" << endl;

    // vector<pair<double, double>> sourceCloud = transformedPoints;
    // vector<pair<double, double>> targetCloud = points37;

    // vector<int> correspondences;
    // icpAlg.icp(pointsN, sourceCloud, targetCloud, correspondences);

    // if (pointsN.size() < 4)
    // {
    //     cerr << "Not enough points in the source point cloud." << endl;
    //     return 0;
    // }

    // icpAlg.calTransformed(pointsN, targetCloud, correspondences, rotationMatrix, translationVector);

    // // 打印 Matrix3d
    // cout << "Rotation Matrix: " << endl;
    // cout << rotationMatrix << endl;

    // // 打印 Vector2d
    // cout << "Translation Vector: " << endl;
    // cout << translationVector << endl;

    // cv::Mat pose_inv = icpAlg.estimateCameraPose(pointsN, targetCloud, correspondences);
    // cout << "test _pointsN position" << endl;
    // cv::Mat _pose_inv = icpAlg.estimateCameraPose(_pointsN, targetCloud, correspondences);
    // cv::Mat camera_position = pose_inv(cv::Rect(3, 0, 1, 3));

    // cout << "Camera Position in World Coordinates: " << endl
    //      << "X: " << camera_position.at<double>(0) << ", "
    //      << "Y: " << camera_position.at<double>(1) << ", "
    //      << "Z: " << camera_position.at<double>(2) << endl;

    return 0;
}