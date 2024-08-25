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

bool PNPAlgorithm::estimateCameraPose()
{
    PointCloud::Ptr cloudA(new PointCloud);
    PointCloud::Ptr cloudB(new PointCloud);
    convertToPointCloud(source, cloudA, 0);
    convertToPointCloud(target, cloudB, 1850);

    vector<cv::Point3f> objectPoints;
    vector<cv::Point2f> imagePoints;
    cout << "pnp solve point nums = " << correspondences.size() << endl;
    if (correspondences.size() < 6)
        return false;
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

    // // 打印读取到的参数
    // std::cout << "Camera Matrix: " << cameraMatrix << std::endl;
    // std::cout << "Distortion Coefficients: " << distCoeffs << std::endl;

    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 2763.52, 0.00, 1319.75,
                            0.00, 2901.47, 544.91,
                            0.00, 0.00, 1.00);

    cv::Mat distCoeffs = (cv::Mat_<double>(5, 1) << -2.90,
                          2.38,
                          -0.00,
                          -0.26,
                          1.39);
    distCoeffs = cv::Mat();
    cameraMatrix = cv::Mat::eye(3, 3, CV_64F);

    // cv::SOLVEPNP_ITERATIVE,cv::SOLVEPNP_EPNP,cv::SOLVEPNP_P3P
    cout << "start solve pnp" << endl;
    // bool flag = cv::solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, useExtrinsicGuess,
    //                                iterationsCount, reprojectionError, confidence, inliers, cv::SOLVEPNP_ITERATIVE);

    bool flag = cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, useExtrinsicGuess,
                             cv::SOLVEPNP_ITERATIVE);
    cout << "finish solve pnp" << endl;

    // cv::solvePnPRansac(objectPoints, imagePoints, K, cv::Mat(), rvec, tvec, useExtrinsicGuess,
    //                    iterationsCount, reprojectionError, confidence, inliers, cv::SOLVEPNP_ITERATIVE);
    if (flag)
    {
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        pose = cv::Mat::eye(4, 4, CV_64F);
        R.copyTo(pose(cv::Rect(0, 0, 3, 3)));
        tvec.copyTo(pose(cv::Rect(3, 0, 1, 3)));
        // cout << "----pose----" << endl;
        // cout << pose << endl;
    }
    else
    {
        cout << "cal pnp failed" << endl;
    }
    return flag;
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

void PNPAlgorithm::setSolveParams(int _iterationsCount, float _reprojectionError, double _confidence)
{
    iterationsCount = _iterationsCount;
    reprojectionError = _reprojectionError;
    confidence = _confidence;
}