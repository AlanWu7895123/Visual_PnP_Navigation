#include "icp.h"
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/icp.h>

ICPAlgorithm::ICPAlgorithm() {}

ICPAlgorithm::ICPAlgorithm(vector<pair<double, double>> source, vector<pair<double, double>> target,
                           Matrix3d R, Vector2d T) : source(source), target(target), R(R), T(T) {}

void ICPAlgorithm::calTransformed()
{
    transformed.clear();
    for (const auto &point : source)
    {
        Vector2d p(point.first, point.second);
        Vector2d p_transformed = R.block<2, 2>(0, 0) * p + T;
        transformed.emplace_back(p_transformed.x(), p_transformed.y());
    }
}

int ICPAlgorithm::pclIcp()
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_A(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_B(new pcl::PointCloud<pcl::PointXYZ>);

    for (const auto &point : transformed)
    {
        cloud_A->push_back(pcl::PointXYZ(point.first, point.second, 0.0));
    }
    for (const auto &point : target)
    {
        cloud_B->push_back(pcl::PointXYZ(point.first, point.second, 0.0));
    }

    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(cloud_A);
    icp.setInputTarget(cloud_B);
    icp.setMaxCorrespondenceDistance(180);
    icp.setMaximumIterations(500);

    pcl::PointCloud<pcl::PointXYZ> Final;
    icp.align(Final);
    if (icp.hasConverged())
    {
        std::cout << "ICP converged." << std::endl;
        std::cout << "The score is " << icp.getFitnessScore() << std::endl;
        Eigen::Matrix4f transformation = icp.getFinalTransformation();
        // std::cout << "Final transformation matrix: \n"
        //           << transformation << std::endl;

        correspondences.clear();
        for (size_t i = 0; i < cloud_A->size(); ++i)
        {
            float min_dist = std::numeric_limits<float>::max();
            int min_index = -1;
            for (size_t j = 0; j < cloud_B->size(); ++j)
            {
                float dist = (cloud_A->points[i].getVector3fMap() - cloud_B->points[j].getVector3fMap()).squaredNorm();
                if (dist < min_dist)
                {
                    min_dist = dist;
                    min_index = j;
                }
            }
            correspondences.push_back(min_index);
        }
        return 0;
    }
    else
    {
        std::cout << "ICP did not converge." << std::endl;
        return 1;
    }
}

vector<int> ICPAlgorithm::getCorrespondences()
{
    return correspondences;
}

// int ICPAlgorithm::icp(vector<pair<double, double>> &origin, const vector<pair<double, double>> &A,
//                       const vector<pair<double, double>> &B, vector<int> &correspondences)
// {
//     correspondences.resize(A.size());

//     for (size_t i = 0; i < A.size(); ++i)
//     {
//         double minDist = distance(A[i], B[0]);
//         int minIndex = 0;
//         for (size_t j = 1; j < B.size(); ++j)
//         {
//             double dist = distance(A[i], B[j]);
//             if (dist < minDist)
//             {
//                 bool alreadySelected = false;
//                 for (size_t k = 0; k < i; ++k)
//                 {
//                     if (correspondences[k] == static_cast<int>(j))
//                     {
//                         alreadySelected = true;
//                         break;
//                     }
//                 }
//                 if (!alreadySelected)
//                 {
//                     minDist = dist;
//                     minIndex = static_cast<int>(j);
//                 }
//             }
//         }
//         correspondences[i] = minIndex;
//     }

//     const int maxIterations = 10;
//     double prevAvgDist = numeric_limits<double>::max();

//     vector<pair<double, double>> transformedA = A;

//     for (int iter = 0; iter < maxIterations; ++iter)
//     {
//         Vector2d centroidA = computeCentroid(transformedA);
//         Vector2d centroidB = computeCentroid(B);

//         Matrix2d H = Matrix2d::Zero();
//         for (size_t i = 0; i < transformedA.size(); ++i)
//         {
//             Vector2d p1(transformedA[i].first, transformedA[i].second);
//             Vector2d p2(B[correspondences[i]].first, B[correspondences[i]].second);
//             Vector2d q1 = p1 - centroidA;
//             Vector2d q2 = p2 - centroidB;
//             H += q1 * q2.transpose();
//         }

//         JacobiSVD<Matrix2d> svd(H, ComputeFullU | ComputeFullV);
//         Matrix2d R = svd.matrixV() * svd.matrixU().transpose();

//         for (size_t i = 0; i < transformedA.size(); ++i)
//         {
//             Vector2d p(transformedA[i].first, transformedA[i].second);
//             Vector2d rotatedP = R * (p - centroidA) + centroidB;
//             transformedA[i].first = rotatedP.x();
//             transformedA[i].second = rotatedP.y();
//         }

//         for (size_t i = 0; i < transformedA.size(); ++i)
//         {
//             double minDist = numeric_limits<double>::max();
//             int minIndex = -1;
//             for (size_t j = 0; j < B.size(); ++j)
//             {
//                 bool alreadySelected = false;
//                 for (size_t k = 0; k < i; ++k)
//                 {
//                     if (correspondences[k] == static_cast<int>(j))
//                     {
//                         alreadySelected = true;
//                         break;
//                     }
//                 }
//                 if (!alreadySelected)
//                 {
//                     double dist = distance(transformedA[i], B[j]);
//                     if (dist < minDist)
//                     {
//                         minDist = dist;
//                         minIndex = static_cast<int>(j);
//                     }
//                 }
//             }
//             correspondences[i] = minIndex;
//         }

//         double avgDist = averageDistance(transformedA, B, correspondences);
//         // cout << avgDist << endl;

//         if (abs(prevAvgDist - avgDist) < 0.001)
//         {
//             avgDist = averageDistance(transformedA, B, correspondences);

//             cout << "average distance:\n"
//                  << avgDist << endl;
//             if (avgDist > 200 || correspondences.size() < 4)
//                 return 1;
//             else
//                 return 0;
//         }
//         prevAvgDist = avgDist;
//     }
//     cout << "icp can't find a stable result" << endl;
//     return 1;
// }

// Vector2d ICPAlgorithm::computeCentroid(const vector<pair<double, double>> &points)
// {
//     Vector2d centroid(0.0, 0.0);
//     for (const auto &p : points)
//     {
//         centroid[0] += p.first;
//         centroid[1] += p.second;
//     }
//     centroid /= points.size();
//     return centroid;
// }

// double ICPAlgorithm::averageDistance(const vector<pair<double, double>> &A, const vector<pair<double, double>> &B,
//                                      const vector<int> &correspondences)
// {
//     double totalDistance = 0.0;
//     for (size_t i = 0; i < A.size(); ++i)
//     {
//         totalDistance += distance(A[i], B[correspondences[i]]);
//     }
//     return totalDistance / A.size();
// }