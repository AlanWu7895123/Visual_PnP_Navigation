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
#include "read_file.h"

using namespace Eigen;
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

        // cvtColor(frame, yuvImage, COLOR_BGR2YUV_YVYU);
        // cvtColor(frame, yuvImage, COLOR_BGR2YUV_UYVY);

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

Vector2d computeCentroid(const vector<pair<double, double>> &points)
{
    Vector2d centroid(0.0, 0.0);
    for (const auto &p : points)
    {
        centroid[0] += p.first;
        centroid[1] += p.second;
    }
    centroid /= points.size();
    return centroid;
}

// 计算两点之间的距离
double distance(const pair<double, double> &p1, const pair<double, double> &p2)
{
    return sqrt(pow(p1.first - p2.first, 2) + pow(p1.second - p2.second, 2));
}

// 计算两点集之间的平均距离
double averageDistance(const vector<pair<double, double>> &A, const vector<pair<double, double>> &B, const vector<int> &correspondences)
{
    double totalDistance = 0.0;
    for (size_t i = 0; i < A.size(); ++i)
    {
        totalDistance += distance(A[i], B[correspondences[i]]);
    }
    return totalDistance / A.size();
}

// ICP 算法（带旋转匹配）
void icp(const vector<pair<double, double>> &A, const vector<pair<double, double>> &B, vector<int> &correspondences)
{
    // 初始化 correspondences
    correspondences.resize(A.size());

    // 初始时，将每个点与最近邻点匹配
    for (size_t i = 0; i < A.size(); ++i)
    {
        double minDist = distance(A[i], B[0]);
        int minIndex = 0;
        for (size_t j = 1; j < B.size(); ++j)
        {
            double dist = distance(A[i], B[j]);
            if (dist < minDist)
            {
                // 检查目标点是否已被选中
                bool alreadySelected = false;
                for (size_t k = 0; k < i; ++k)
                {
                    if (correspondences[k] == static_cast<int>(j))
                    {
                        alreadySelected = true;
                        break;
                    }
                }
                if (!alreadySelected)
                {
                    minDist = dist;
                    minIndex = static_cast<int>(j);
                }
            }
        }
        correspondences[i] = minIndex;
    }

    // // 输出初始匹配结果
    // cout << "初始匹配结果：" << endl;
    // for (size_t i = 0; i < A.size(); ++i)
    // {
    //     cout << "A[" << i << "] -> B[" << correspondences[i] << "]" << endl;
    // }

    // 迭代次数
    const int maxIterations = 50;
    double prevAvgDist = numeric_limits<double>::max();

    // 临时点集用于保存变换后的点
    vector<pair<double, double>> transformedA = A;

    for (int iter = 0; iter < maxIterations; ++iter)
    {
        // 计算最佳的刚性变换（包括平移和旋转）
        // 在这里，我们假设点集B是固定的，不会随着迭代而变化
        // 在实际情况下，您可能需要实现更复杂的变换优化算法
        Vector2d centroidA = computeCentroid(transformedA);
        Vector2d centroidB = computeCentroid(B);

        Matrix2d H = Matrix2d::Zero(); // 2x2 矩阵 H
        for (size_t i = 0; i < transformedA.size(); ++i)
        {
            Vector2d p1(transformedA[i].first, transformedA[i].second);
            Vector2d p2(B[correspondences[i]].first, B[correspondences[i]].second);
            Vector2d q1 = p1 - centroidA;
            Vector2d q2 = p2 - centroidB;
            H += q1 * q2.transpose();
        }

        JacobiSVD<Matrix2d> svd(H, ComputeFullU | ComputeFullV);
        Matrix2d R = svd.matrixV() * svd.matrixU().transpose();

        // 应用平移和旋转变换并更新临时点集
        for (size_t i = 0; i < transformedA.size(); ++i)
        {
            Vector2d p(transformedA[i].first, transformedA[i].second);
            Vector2d rotatedP = R * (p - centroidA) + centroidB;
            transformedA[i].first = rotatedP.x();
            transformedA[i].second = rotatedP.y();
        }

        // 重新进行匹配，确保不会多个源点匹配到同一个目标点
        for (size_t i = 0; i < transformedA.size(); ++i)
        {
            double minDist = numeric_limits<double>::max();
            int minIndex = -1;
            for (size_t j = 0; j < B.size(); ++j)
            {
                // 检查目标点是否已被选中
                bool alreadySelected = false;
                for (size_t k = 0; k < i; ++k)
                {
                    if (correspondences[k] == static_cast<int>(j))
                    {
                        alreadySelected = true;
                        break;
                    }
                }
                if (!alreadySelected)
                {
                    double dist = distance(transformedA[i], B[j]);
                    if (dist < minDist)
                    {
                        minDist = dist;
                        minIndex = static_cast<int>(j);
                    }
                }
            }
            correspondences[i] = minIndex;
        }

        // // 输出匹配结果
        // cout << "迭代 " << iter + 1 << " 匹配结果：" << endl;
        // for (size_t i = 0; i < transformedA.size(); ++i)
        // {
        //     cout << "A[" << i << "] -> B[" << correspondences[i] + 1 << "]" << endl;
        // }

        // 计算平均距离
        double avgDist = averageDistance(transformedA, B, correspondences);
        // cout << "平均距离：" << avgDist << endl;

        // 如果平均距离变化很小，停止迭代
        if (abs(prevAvgDist - avgDist) < 0.001)
        {
            // cout << "收敛于平均距离变化小于0.001, 迭代结束。" << endl;
            break;
        }
        prevAvgDist = avgDist;
    }
}

void convertToPointCloud(const std::vector<std::pair<double, double>> &points, PointCloud::Ptr cloud, float z)
{
    cloud->clear();
    for (const auto &point : points)
    {
        PointT pt;
        pt.x = point.first;
        pt.y = point.second;
        pt.z = z;
        cloud->push_back(pt);
    }
}

void calTransformed(const vector<pair<double, double>> &sourceCloud,
                    const vector<pair<double, double>> &targetCloud,
                    const vector<int> &correspondences,
                    Matrix3d &rotationMatrix,
                    Vector2d &translationVector)
{

    // 建立对应点的坐标矩阵
    MatrixXd srcMat(2, correspondences.size());
    MatrixXd tgtMat(2, correspondences.size());

    for (size_t i = 0; i < correspondences.size(); ++i)
    {
        srcMat.col(i) << sourceCloud[i].first, sourceCloud[i].second;
        tgtMat.col(i) << targetCloud[correspondences[i]].first, targetCloud[correspondences[i]].second;
    }

    // 计算源点云和目标点云的中心点
    Vector2d srcCentroid = srcMat.rowwise().mean();
    Vector2d tgtCentroid = tgtMat.rowwise().mean();

    // 去中心化
    MatrixXd srcCentered = srcMat.colwise() - srcCentroid;
    MatrixXd tgtCentered = tgtMat.colwise() - tgtCentroid;

    // 计算协方差矩阵
    MatrixXd covariance = srcCentered * tgtCentered.transpose();

    // 使用奇异值分解计算旋转矩阵
    JacobiSVD<MatrixXd> svd(covariance, ComputeFullU | ComputeFullV);
    MatrixXd U = svd.matrixU();
    MatrixXd V = svd.matrixV();

    // 计算旋转矩阵
    MatrixXd R = V * U.transpose();

    // 计算平移向量
    Vector2d t = tgtCentroid - R * srcCentroid;

    // 转换为 3x3 的旋转矩阵
    rotationMatrix.setIdentity();
    rotationMatrix.block<2, 2>(0, 0) = R;

    // 设置平移向量
    translationVector = t;
}

void transformPointCloud(const vector<pair<double, double>> &points,
                         const Matrix3d &rotationMatrix,
                         const Vector2d &translationVector,
                         vector<pair<double, double>> &transformedPoints)
{
    transformedPoints.clear();
    for (const auto &point : points)
    {
        Vector2d p(point.first, point.second);
        Vector2d p_transformed = rotationMatrix.block<2, 2>(0, 0) * p + translationVector;
        transformedPoints.emplace_back(p_transformed.x(), p_transformed.y());
    }
}

void processingThread()
{
    // Matrix4 transformed = Matrix4::Zero();
    // transformed << 1, 0, 0, 0,
    //     0, 1, 0, 0,
    //     0, 0, 1, 0,
    //     0, 0, 0, 1;

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

            std::vector<std::pair<double, double>> pointsN;
            std::vector<std::pair<double, double>> transformedPoints;

            pointsN = readNPoints("../data/circles.txt");
            if (pointsN.size() < 4)
                continue;

            transformPointCloud(pointsN, rotationMatrix, translationVector, transformedPoints);

            // for (int i = 0; i < transformedPoints.size(); i++)
            // {
            //     cout << transformedPoints[i].first << "," << transformedPoints[i].second << endl;
            // }

            // vector<pair<double, double>> sourceCloud = transformedPoints;
            vector<pair<double, double>> sourceCloud = pointsN;
            vector<pair<double, double>> targetCloud = points37;

            // 进行点云配准（带旋转）
            vector<int> correspondences;
            icp(sourceCloud, targetCloud, correspondences);

            // Matrix3d rotationMatrix;
            // Vector2d translationVector;

            // cout << "start cal transformed" << endl;
            calTransformed(sourceCloud, targetCloud, correspondences, rotationMatrix, translationVector);

            // // 输出结果
            // cout << "Rotation Matrix: " << endl
            //      << rotationMatrix << endl;
            // cout << "Translation Vector: " << endl
            //      << translationVector << endl;

            PointCloud::Ptr cloudA(new PointCloud);
            PointCloud::Ptr cloudB(new PointCloud);
            convertToPointCloud(pointsN, cloudA, 0);
            convertToPointCloud(points37, cloudB, 1800);
            // 提取匹配点对
            vector<cv::Point3f> objectPoints;
            vector<cv::Point2f> imagePoints;
            for (size_t i = 0; i < correspondences.size(); ++i)
            {
                const PointT &srcPoint = cloudA->points[i];
                const PointT &dstPoint = cloudB->points[correspondences[i]];
                objectPoints.push_back(cv::Point3f(dstPoint.x, dstPoint.y, dstPoint.z));
                imagePoints.push_back(cv::Point2f(srcPoint.x, srcPoint.y));
            }

            // 假设相机内参矩阵 K（这里使用单位矩阵，实际应根据具体相机参数设置）
            cv::Mat K = cv::Mat::eye(3, 3, CV_64F);

            // 求解相机位姿
            cv::Mat rvec, tvec;
            cv::solvePnP(objectPoints, imagePoints, K, cv::Mat(), rvec, tvec);

            // 将旋转向量转换为旋转矩阵
            cv::Mat R;
            cv::Rodrigues(rvec, R);
            // std::cout << "Rotation Matrix (R): " << std::endl
            //           << R << std::endl;

            // 构建位姿矩阵 (4x4)
            cv::Mat pose = cv::Mat::eye(4, 4, CV_64F);
            R.copyTo(pose(cv::Rect(0, 0, 3, 3)));
            tvec.copyTo(pose(cv::Rect(3, 0, 1, 3)));

            // std::cout << "Pose Matrix: " << std::endl
            //           << pose << std::endl;

            // 求逆变换
            cv::Mat pose_inv = pose.inv();
            // std::cout << "Inverse Pose Matrix: " << std::endl
            //           << pose_inv << std::endl;

            // 相机在世界坐标系中的位置 (取出平移部分)
            cv::Mat camera_position = pose_inv(cv::Rect(3, 0, 1, 3));
            std::cout << "Camera Position in World Coordinates: " << std::endl
                      << "X: " << camera_position.at<double>(0) << ", "
                      << "Y: " << camera_position.at<double>(1) << ", "
                      << "Z: " << camera_position.at<double>(2) << std::endl;

            out << camera_position.at<double>(0) << " " << camera_position.at<double>(1) << endl;

            // std::vector<std::pair<double, double>> pointsN;

            // pointsN = readNPoints("../data/circles.txt");

            // ICPRegisterer ICPRegisterer(pointsN, points37, transformed);
            // ICPRegisterer.computePCLICP();
            // transformed = ICPRegisterer.getTransformed();

            cv::imshow("Processed Frame", processor.output);
            cv::waitKey(1);
        }
    }
}

int main()
{
    // int mode;
    // cout << "please choose the mode, 1 is normal, 2 is generate the trajectory for test" << endl;
    // cin >> mode;
    // sleep(3);
    // if (mode == 1)
    // {
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
    // }
    // else
    // {
    // }

    // string filename;
    // cin >> filename;
    // ImageProcessor processor(filename);

    // processor.readImage();
    // processor.convertToGray();
    // processor.applyThreshold();
    // processor.findContours();
    // processor.detectCircles();
    // processor.saveResults();

    // Matrix4 transformed = Matrix4::Zero();
    // transformed << 1, 0, 0, 0,
    //     0, 1, 0, 0,
    //     0, 0, 1, 0,
    //     0, 0, 0, 1;

    // // transformed << 0.992645, 0.121066, 0, 101.66,
    // //     -0.121066, 0.992645, 0, 196.564,
    // //     0, 0, 1, 1800,
    // //     0, 0, 0, 1;

    // // std::cout << "transformed1:" << std::endl
    // //           << transformed << std::endl;
    // std::vector<std::pair<double, double>> points37;
    // std::vector<std::pair<double, double>> pointsN;
    // std::vector<std::pair<double, double>> transformedPoints;
    // Matrix3d rotationMatrix = Matrix3d::Identity();
    // Vector2d translationVector = Vector2d::Zero();

    // points37 = read37Points("../data/points.txt");
    // pointsN = readNPoints("../data/circles.txt");

    // transformPointCloud(pointsN, rotationMatrix, translationVector, transformedPoints);

    // for (int i = 0; i < transformedPoints.size(); i++)
    // {
    //     cout << transformedPoints[i].first << "," << transformedPoints[i].second << endl;
    // }

    // vector<pair<double, double>> sourceCloud = transformedPoints;
    // vector<pair<double, double>> targetCloud = points37;

    // // 进行点云配准（带旋转）
    // vector<int> correspondences;
    // icp(sourceCloud, targetCloud, correspondences);

    // // Matrix3d rotationMatrix;
    // // Vector2d translationVector;

    // cout << "start cal transformed" << endl;
    // calTransformed(sourceCloud, targetCloud, correspondences, rotationMatrix, translationVector);

    // // 输出结果
    // cout << "Rotation Matrix: " << endl
    //      << rotationMatrix << endl;
    // cout << "Translation Vector: " << endl
    //      << translationVector << endl;

    // PointCloud::Ptr cloudA(new PointCloud);
    // PointCloud::Ptr cloudB(new PointCloud);
    // convertToPointCloud(pointsN, cloudA, 0);
    // convertToPointCloud(points37, cloudB, 1800);
    // // 提取匹配点对
    // vector<cv::Point3f> objectPoints;
    // vector<cv::Point2f> imagePoints;
    // for (size_t i = 0; i < correspondences.size(); ++i)
    // {
    //     const PointT &srcPoint = cloudA->points[i];
    //     const PointT &dstPoint = cloudB->points[correspondences[i]];
    //     objectPoints.push_back(cv::Point3f(dstPoint.x, dstPoint.y, dstPoint.z));
    //     imagePoints.push_back(cv::Point2f(srcPoint.x, srcPoint.y));
    // }

    // // 假设相机内参矩阵 K（这里使用单位矩阵，实际应根据具体相机参数设置）
    // cv::Mat K = cv::Mat::eye(3, 3, CV_64F);

    // // 求解相机位姿
    // cv::Mat rvec, tvec;
    // cv::solvePnP(objectPoints, imagePoints, K, cv::Mat(), rvec, tvec);

    // // 将旋转向量转换为旋转矩阵
    // cv::Mat R;
    // cv::Rodrigues(rvec, R);
    // // std::cout << "Rotation Matrix (R): " << std::endl
    // //           << R << std::endl;

    // // 构建位姿矩阵 (4x4)
    // cv::Mat pose = cv::Mat::eye(4, 4, CV_64F);
    // R.copyTo(pose(cv::Rect(0, 0, 3, 3)));
    // tvec.copyTo(pose(cv::Rect(3, 0, 1, 3)));

    // // std::cout << "Pose Matrix: " << std::endl
    // //           << pose << std::endl;

    // // 求逆变换
    // cv::Mat pose_inv = pose.inv();
    // // std::cout << "Inverse Pose Matrix: " << std::endl
    // //           << pose_inv << std::endl;

    // // 相机在世界坐标系中的位置 (取出平移部分)
    // cv::Mat camera_position = pose_inv(cv::Rect(3, 0, 1, 3));
    // std::cout << "Camera Position in World Coordinates: " << std::endl
    //           << "X: " << camera_position.at<double>(0) << ", "
    //           << "Y: " << camera_position.at<double>(1) << ", "
    //           << "Z: " << camera_position.at<double>(2) << std::endl;

    // ICPRegisterer ICPRegisterer(pointsN, points37, transformed);
    // ICPRegisterer.computePCLICP();
    // transformed = ICPRegisterer.getTransformed();
    // std::cout << "Next transformed:" << std::endl
    //           << transformed << std::endl;

    // cout << "hungarian" << endl;
    // ICPRegisterer.computeHungarian();

    return 0;
}