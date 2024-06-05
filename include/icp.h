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
    ICPAlgorithm(vector<pair<double, double>> source, vector<pair<double, double>> target, Matrix3d R, Vector2d T);
    int pclIcp();
    void calTransformed();
    vector<int> getCorrespondences();

    // int icp(vector<pair<double, double>> &origin, const vector<pair<double, double>> &A,
    //         const vector<pair<double, double>> &B, vector<int> &correspondences);
    // double distance(const pair<double, double> &p1, const pair<double, double> &p2);

private:
    // Vector2d computeCentroid(const vector<pair<double, double>> &points);
    // double averageDistance(const vector<pair<double, double>> &A, const vector<pair<double, double>> &B,
    //                        const vector<int> &correspondences);

    vector<pair<double, double>> source;
    vector<pair<double, double>> target;
    vector<pair<double, double>> transformed;
    vector<int> correspondences;
    Matrix3d R;
    Vector2d T;
};

#endif // ICPALGORITHM_H
