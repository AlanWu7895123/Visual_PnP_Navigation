#include <eigen3/unsupported/Eigen/NonLinearOptimization>
#include <eigen3/unsupported/Eigen/NumericalDiff>
#include "icp_registerer.h"
#include "hungarian.h"

using namespace Eigen;

ICPRegisterer::ICPRegisterer(const std::vector<std::pair<double, double>> &A, const std::vector<std::pair<double, double>> &B, const Matrix4 transformed)
    : cloudA(new PointCloud), cloudB(new PointCloud), cloudARegistered(new PointCloud)
{
    // // config icp
    // icp.setMaxCorrespondenceDistance(0.05);
    // icp.setMaximumIterations(50);
    // icp.setTransformationEpsilon(1e-8);
    // icp.setEuclideanFitnessEpsilon(1);

    PointCloud::Ptr tmpCloud(new PointCloud);
    convertToPointCloud(A, tmpCloud, 0);

    // std::cout << "Origin points:" << std::endl;
    // for (const auto &point : tmpCloud->points)
    // {
    //     std::cout << "(" << point.x << ", " << point.y << ", " << point.z << ")" << std::endl;
    // }
    // std::cout << "transformed:" << std::endl
    //           << transformed << std::endl;

    transform(tmpCloud, transformed, cloudA);

    // std::cout << "Transformed points:" << std::endl;
    // for (const auto &point : cloudA->points)
    // {
    //     std::cout << "(" << point.x << ", " << point.y << ", " << point.z << ")" << std::endl;
    // }

    convertToPointCloud(B, cloudB, 1800);
}

void ICPRegisterer::transform(const PointCloud::Ptr tmpCloud, const Matrix4 transformed, PointCloud::Ptr &cloudA)
{
    for (const auto &point : tmpCloud->points)
    {
        Eigen::Vector4f homogenousPoint(point.x, point.y, point.z, 1.0f);
        Eigen::Vector4f transformedPoint = transformed * homogenousPoint;

        pcl::PointXYZ newPoint;
        newPoint.x = transformedPoint(0) / transformedPoint(3);
        newPoint.y = transformedPoint(1) / transformedPoint(3);
        newPoint.z = transformedPoint(2) / transformedPoint(3);

        cloudA->push_back(newPoint);
    }
}

void ICPRegisterer::convertToPointCloud(const std::vector<std::pair<double, double>> &points, PointCloud::Ptr cloud, float z)
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

// double ICPRegisterer::distance(const std::pair<double, double> &p1, const std::pair<double, double> &p2) const
// {
//     return sqrt(pow(p1.first - p2.first, 2) + pow(p1.second - p2.second, 2));
// }

// double ICPRegisterer::averageDistance(const std::vector<std::pair<double, double>> &A, const std::vector<std::pair<double, double>> &B, const std::vector<int> &correspondences) const
// {
//     double totalDistance = 0.0;
//     for (size_t i = 0; i < A.size(); ++i)
//     {
//         totalDistance += distance(A[i], B[correspondences[i]]);
//     }
//     return totalDistance / A.size();
// }

Matrix4 ICPRegisterer::getTransformed()
{
    return icp.getFinalTransformation();
}

void ICPRegisterer::computeHungarian()
{
    if (cloudA->points.size() < 6 || cloudA->points.size() > 12)
        return;

    // 构建代价矩阵
    int n = cloudA->points.size();
    MatrixXd costMatrix(n, n);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            double dx = cloudA->points[i].x - cloudB->points[j].x;
            double dy = cloudA->points[i].y - cloudB->points[j].y;
            double dz = cloudA->points[i].z - cloudB->points[j].z;
            costMatrix(i, j) = sqrt(dx * dx + dy * dy + dz * dz);
        }
    }

    // 使用匈牙利算法求解最优匹配
    HungarianAlgorithm hungarian;
    std::vector<int> assignment;
    double cost = hungarian.Solve(costMatrix, assignment);

    if (assignment.size() != n)
    {
        cout << "点云配准失败！" << endl;
        return;
    }

    // 输出匹配关系
    cout << "配准后的点云匹配关系：" << endl;
    for (size_t i = 0; i < assignment.size(); ++i)
    {
        cout << "点 " << i << " 在 B 中的对应点索引：" << assignment[i] + 1 << endl;
    }

    // 提取匹配点对
    vector<cv::Point3f> objectPoints;
    vector<cv::Point2f> imagePoints;
    for (size_t i = 0; i < assignment.size(); ++i)
    {
        const PointT &srcPoint = cloudA->points[i];
        const PointT &dstPoint = cloudB->points[assignment[i]];
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

    // 打印相机位置（即平移向量 tvec）
    cout << "相机坐标：" << endl
         << "X: " << tvec.at<double>(0) << ", Y: " << tvec.at<double>(1) << ", Z: " << tvec.at<double>(2) << endl;
}

void ICPRegisterer::computePCLICP()
{
    // pcl::IterativeClosestPoint<PointT, PointT> icp;
    if (cloudA->points.size() < 6 || cloudA->points.size() > 12)
        return;
    icp.setInputSource(cloudA);
    icp.setInputTarget(cloudB);

    // icp.setMaxCorrespondenceDistance(250);

    // 执行点云配准
    // PointCloud::Ptr cloudARegistered(new PointCloud);
    icp.align(*cloudARegistered);

    if (icp.hasConverged())
    {
        cout << "点云配准成功！" << endl;
        cout << "变换矩阵：" << endl
             << icp.getFinalTransformation() << endl;

        // 计算匹配关系
        pcl::registration::CorrespondenceEstimation<PointT, PointT> est;
        est.setInputSource(cloudA);
        est.setInputTarget(cloudB);
        pcl::Correspondences correspondences;
        est.determineCorrespondences(correspondences);

        // 输出匹配关系
        cout << "配准后的点云匹配关系：" << endl;
        for (size_t i = 0; i < correspondences.size(); ++i)
        {
            const auto &correspondence = correspondences[i];
            cout << "点 " << correspondence.index_query << " 在 B 中的对应点索引：" << correspondence.index_match << endl;
        }

        // 提取匹配点对
        vector<cv::Point3f> objectPoints;
        vector<cv::Point2f> imagePoints;
        for (const auto &correspondence : correspondences)
        {
            const PointT &srcPoint = cloudA->points[correspondence.index_query];
            const PointT &dstPoint = cloudB->points[correspondence.index_match];
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

        // 打印旋转矩阵和平移向量
        cout << "旋转矩阵：" << endl
             << R << endl;
        cout << "平移向量：" << endl
             << tvec << endl;

        // 构建位姿矩阵
        cv::Mat pose = cv::Mat::eye(4, 4, CV_64F);
        R.copyTo(pose(cv::Rect(0, 0, 3, 3)));
        tvec.copyTo(pose(cv::Rect(3, 0, 1, 3)));

        // 打印位姿矩阵
        cout << "位姿矩阵：" << endl
             << pose << endl;

        // 打印相机位置（即平移向量 tvec）
        cout << "相机坐标：" << endl
             << "X: " << tvec.at<double>(0) << ", Y: " << tvec.at<double>(1) << ", Z: " << tvec.at<double>(2) << endl;
    }
    else
    {
        cout << "点云配准失败！" << endl;
    }
}

// void ICPRegisterer::computeICP()
// {
//     // Initial correspondences: nearest neighbor
//     for (size_t i = 0; i < A.size(); ++i)
//     {
//         double minDist = distance(A[i], B[0]);
//         int minIndex = 0;
//         for (size_t j = 1; j < B.size(); ++j)
//         {
//             double dist = distance(A[i], B[j]);
//             if (dist < minDist)
//             {
//                 minDist = dist;
//                 minIndex = j;
//             }
//         }
//         correspondences[i] = minIndex;
//     }

//     cout << "Initial correspondences:" << endl;
//     for (size_t i = 0; i < A.size(); ++i)
//     {
//         cout << "A[" << i << "] -> B[" << correspondences[i] << "]" << endl;
//     }

//     // Iteration parameters
//     const int maxIterations = 10;
//     double _avgDist = -1;

//     for (int iter = 0; iter < maxIterations; ++iter)
//     {
//         // Compute the best rigid transformation (simplified to mean translation)
//         double offsetX = 0.0;
//         double offsetY = 0.0;
//         for (size_t i = 0; i < transformedA.size(); ++i)
//         {
//             offsetX += B[correspondences[i]].first - transformedA[i].first;
//             offsetY += B[correspondences[i]].second - transformedA[i].second;
//         }
//         offsetX /= transformedA.size();
//         offsetY /= transformedA.size();

//         // Apply translation
//         for (size_t i = 0; i < transformedA.size(); ++i)
//         {
//             transformedA[i].first += offsetX;
//             transformedA[i].second += offsetY;
//         }

//         // Update correspondences
//         for (size_t i = 0; i < transformedA.size(); ++i)
//         {
//             double minDist = numeric_limits<double>::max();
//             int minIndex = -1;
//             for (size_t j = 0; j < B.size(); ++j)
//             {
//                 if (find(correspondences.begin(), correspondences.begin() + i, j) != correspondences.begin() + i)
//                     continue;

//                 double dist = distance(transformedA[i], B[j]);
//                 if (dist < minDist)
//                 {
//                     minDist = dist;
//                     minIndex = j;
//                 }
//             }
//             correspondences[i] = minIndex;
//         }

//         cout << "Iteration " << iter + 1 << " correspondences:" << endl;
//         for (size_t i = 0; i < transformedA.size(); ++i)
//         {
//             cout << "A[" << i << "] -> B[" << correspondences[i] << "]" << endl;
//         }

//         double avgDist = averageDistance(transformedA, B, correspondences);
//         cout << "Average distance: " << avgDist << endl;

//         if (_avgDist == avgDist)
//             break;
//         _avgDist = avgDist;

//         if (avgDist < 0.01)
//         {
//             cout << "Converged with average distance < 0.01, stopping iteration." << endl;
//             break;
//         }
//     }
// }

// void ICPRegisterer::printResults() const
// {
//     cout << "Final correspondences:" << endl;
//     for (size_t i = 0; i < A.size(); ++i)
//     {
//         cout << "A[" << i << "] -> B[" << correspondences[i] + 1 << "]" << endl;
//     }
//     cout << "Final transformed" << endl;
//     for (int i = 0; i < transformedA.size(); i++)
//     {
//         cout << transformedA[i].first << "-" << transformedA[i].second << endl;
//     }
// }
