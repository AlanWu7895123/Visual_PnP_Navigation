#include "pnp.h"
#include <pcl/registration/icp.h>

PNPAlgorithm::PNPAlgorithm() {}

PNPAlgorithm::PNPAlgorithm(const vector<pair<double, double>> &source, const vector<pair<double, double>> &target,
                           const vector<int> &correspondences) : source(source),
                                                                 target(target), correspondences(correspondences) {}

void PNPAlgorithm::convertToPointCloud(const vector<pair<double, double>> &points, PointCloud::Ptr &cloud, float z)
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

void PNPAlgorithm::estimateCameraPose()
{
    PointCloud::Ptr cloudA(new PointCloud);
    PointCloud::Ptr cloudB(new PointCloud);
    convertToPointCloud(source, cloudA, 0);
    convertToPointCloud(target, cloudB, 0);

    vector<cv::Point3f> objectPoints;
    vector<cv::Point2f> imagePoints;
    for (size_t i = 0; i < correspondences.size(); ++i)
    {
        const PointT &srcPoint = cloudA->points[i];
        const PointT &dstPoint = cloudB->points[correspondences[i]];
        objectPoints.push_back(cv::Point3f(dstPoint.x, dstPoint.y, dstPoint.z));
        imagePoints.push_back(cv::Point2f(srcPoint.x, srcPoint.y));
    }

    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat rvec, tvec;
    std::vector<int> inliers;
    bool useExtrinsicGuess = false;
    int iterationsCount = 300;      // Increased iterations
    float reprojectionError = 50.0; // Moderate reprojection error threshold
    double confidence = 0.99;       // High confidence

    // // 打印读取到的参数
    // std::cout << "Camera Matrix: " << cameraMatrix << std::endl;
    // std::cout << "Distortion Coefficients: " << distCoeffs << std::endl;

    cv::solvePnPRansac(objectPoints, imagePoints, K, cv::Mat(), rvec, tvec, useExtrinsicGuess,
                       iterationsCount, reprojectionError, confidence, inliers, cv::SOLVEPNP_ITERATIVE);
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    pose = cv::Mat::eye(4, 4, CV_64F);
    R.copyTo(pose(cv::Rect(0, 0, 3, 3)));
    tvec.copyTo(pose(cv::Rect(3, 0, 1, 3)));
}

cv::Mat PNPAlgorithm::getPose()
{
    return pose.inv();
}

void PNPAlgorithm::setCameraConfig(cv::Mat myCameraMatrix, cv::Mat myDistCoeffs)
{
    cameraMatrix = myCameraMatrix;
    distCoeffs = myDistCoeffs;
    return;
}