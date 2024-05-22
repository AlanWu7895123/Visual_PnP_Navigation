#ifndef ICP_REGISTERER_H
#define ICP_REGISTERER_H

#include <iostream>
#include <vector>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/correspondence_estimation.h>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::Registration<pcl::PointXYZ, pcl::PointXYZ, float>::Matrix4 Matrix4;

class ICPRegisterer
{
public:
    // Constructor
    ICPRegisterer(const std::vector<std::pair<double, double>> &A, const std::vector<std::pair<double, double>> &B, const Matrix4 transformed);

    // Methods
    void computePCLICP();
    Matrix4 getTransformed();
    // void computeICP();
    // void printResults() const;

private:
    void convertToPointCloud(const std::vector<std::pair<double, double>> &points, PointCloud::Ptr cloud, float z);
    void transform(const PointCloud::Ptr tmpCloud, const Matrix4 transformed, PointCloud::Ptr &cloudA);
    // double distance(const std::pair<double, double> &p1, const std::pair<double, double> &p2) const;
    // double averageDistance(const std::vector<std::pair<double, double>> &A, const std::vector<std::pair<double, double>> &B, const std::vector<int> &correspondences) const;

    // std::vector<std::pair<double, double>> A;
    // std::vector<std::pair<double, double>> B;
    // std::vector<int> correspondences;
    // std::vector<std::pair<double, double>> transformedA;

    PointCloud::Ptr cloudA;
    PointCloud::Ptr cloudB;
    PointCloud::Ptr cloudARegistered;
    pcl::IterativeClosestPoint<PointT, PointT> icp;
};

#endif // ICP_REGISTERER_H
