#include "icp_algorithm.h"
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
    // icp.setMaxCorrespondenceDistance(180);
    icp.setMaximumIterations(50);

    pcl::PointCloud<pcl::PointXYZ> Final;
    icp.align(Final);
    if (icp.hasConverged())
    {
        std::cout << "ICP converged." << std::endl;
        std::cout << "The score is " << icp.getFitnessScore() << std::endl;
        Eigen::Matrix4f transformation = icp.getFinalTransformation();
        // std::cout << "ICP final transformation matrix: \n"
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
        std::cout << "ERROR: ICP did not converge." << std::endl;
        return 1;
    }
}

vector<int> ICPAlgorithm::getCorrespondences()
{
    return correspondences;
}