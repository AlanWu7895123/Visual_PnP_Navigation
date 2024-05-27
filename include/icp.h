#ifndef ICPALGORITHM_H
#define ICPALGORITHM_H

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

class ICPAlgorithm
{
public:
    ICPAlgorithm();
    int icp(vector<pair<double, double>> &origin, const vector<pair<double, double>> &A,
            const vector<pair<double, double>> &B, vector<int> &correspondences);
    void calTransformed(const vector<pair<double, double>> &sourceCloud,
                        const vector<pair<double, double>> &targetCloud,
                        const vector<int> &correspondences,
                        Matrix3d &rotationMatrix,
                        Vector2d &translationVector);
    void transformPointCloud(const vector<pair<double, double>> &points,
                             const Matrix3d &rotationMatrix,
                             const Vector2d &translationVector,
                             vector<pair<double, double>> &transformedPoints);
    void convertToPointCloud(const vector<pair<double, double>> &points, PointCloud::Ptr cloud, float z);
    cv::Mat estimateCameraPose(const vector<pair<double, double>> &pointsN, const vector<pair<double, double>> &points37,
                               const vector<int> &correspondences);
    double distance(const pair<double, double> &p1, const pair<double, double> &p2);

private:
    Vector2d computeCentroid(const vector<pair<double, double>> &points);
    double averageDistance(const vector<pair<double, double>> &A, const vector<pair<double, double>> &B, const vector<int> &correspondences);
};

#endif // ICPALGORITHM_H
