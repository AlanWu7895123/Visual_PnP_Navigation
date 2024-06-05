#ifndef PNPALGORITHM_H
#define PNPALGORITHM_H

#include <vector>
#include <utility>
#include <limits>
#include <cmath>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

using namespace std;
using namespace Eigen;
using PointT = pcl::PointXYZ;
using PointCloud = pcl::PointCloud<PointT>;

class PNPAlgorithm
{
public:
    PNPAlgorithm();
    PNPAlgorithm(const vector<pair<double, double>> &source, const vector<pair<double, double>> &target,
                 const vector<int> &correspondences);
    void convertToPointCloud(const vector<pair<double, double>> &points, PointCloud::Ptr &cloud, float z);
    void estimateCameraPose();
    cv::Mat getPose();

private:
    std::vector<std::pair<double, double>> source;
    std::vector<std::pair<double, double>> target;
    vector<int> correspondences;
    cv::Mat pose;
};

#endif // PNPALGORITHM_H
