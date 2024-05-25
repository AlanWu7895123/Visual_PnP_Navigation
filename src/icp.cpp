#include "icp.h"

ICPAlgorithm::ICPAlgorithm() {}

Vector2d ICPAlgorithm::computeCentroid(const vector<pair<double, double>> &points)
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

double ICPAlgorithm::distance(const pair<double, double> &p1, const pair<double, double> &p2)
{
    return sqrt(pow(p1.first - p2.first, 2) + pow(p1.second - p2.second, 2));
}

double ICPAlgorithm::averageDistance(const vector<pair<double, double>> &A, const vector<pair<double, double>> &B, const vector<int> &correspondences)
{
    double totalDistance = 0.0;
    for (size_t i = 0; i < A.size(); ++i)
    {
        totalDistance += distance(A[i], B[correspondences[i]]);
    }
    return totalDistance / A.size();
}

int ICPAlgorithm::icp(const vector<pair<double, double>> &A, const vector<pair<double, double>> &B, vector<int> &correspondences)
{
    correspondences.resize(A.size());

    for (size_t i = 0; i < A.size(); ++i)
    {
        double minDist = distance(A[i], B[0]);
        int minIndex = 0;
        for (size_t j = 1; j < B.size(); ++j)
        {
            double dist = distance(A[i], B[j]);
            if (dist < minDist)
            {
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

    const int maxIterations = 50;
    double prevAvgDist = numeric_limits<double>::max();

    vector<pair<double, double>> transformedA = A;

    for (int iter = 0; iter < maxIterations; ++iter)
    {
        Vector2d centroidA = computeCentroid(transformedA);
        Vector2d centroidB = computeCentroid(B);

        Matrix2d H = Matrix2d::Zero();
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

        for (size_t i = 0; i < transformedA.size(); ++i)
        {
            Vector2d p(transformedA[i].first, transformedA[i].second);
            Vector2d rotatedP = R * (p - centroidA) + centroidB;
            transformedA[i].first = rotatedP.x();
            transformedA[i].second = rotatedP.y();
        }

        for (size_t i = 0; i < transformedA.size(); ++i)
        {
            double minDist = numeric_limits<double>::max();
            int minIndex = -1;
            for (size_t j = 0; j < B.size(); ++j)
            {
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

        // cout << "迭代 " << iter + 1 << " 匹配结果：" << endl;
        // for (size_t i = 0; i < transformedA.size(); ++i)
        // {
        //     cout << "A[" << i << "] -> B[" << correspondences[i] + 1 << "]" << endl;
        // }

        double avgDist = averageDistance(transformedA, B, correspondences);

        if (abs(prevAvgDist - avgDist) < 0.001)
        {
            cout << "平均距离：" << avgDist << endl;
            if (avgDist > 200)
                return 1;
            else
                return 0;
        }
        prevAvgDist = avgDist;
    }
    return 1;
}

void ICPAlgorithm::calTransformed(const vector<pair<double, double>> &sourceCloud,
                                  const vector<pair<double, double>> &targetCloud,
                                  const vector<int> &correspondences,
                                  Matrix3d &rotationMatrix,
                                  Vector2d &translationVector)
{
    MatrixXd srcMat(2, correspondences.size());
    MatrixXd tgtMat(2, correspondences.size());

    for (size_t i = 0; i < correspondences.size(); ++i)
    {
        srcMat.col(i) << sourceCloud[i].first, sourceCloud[i].second;
        tgtMat.col(i) << targetCloud[correspondences[i]].first, targetCloud[correspondences[i]].second;
    }

    Vector2d srcCentroid = srcMat.rowwise().mean();
    Vector2d tgtCentroid = tgtMat.rowwise().mean();

    MatrixXd srcCentered = srcMat.colwise() - srcCentroid;
    MatrixXd tgtCentered = tgtMat.colwise() - tgtCentroid;

    MatrixXd covariance = srcCentered * tgtCentered.transpose();

    JacobiSVD<MatrixXd> svd(covariance, ComputeFullU | ComputeFullV);
    MatrixXd U = svd.matrixU();
    MatrixXd V = svd.matrixV();

    MatrixXd R = V * U.transpose();

    Vector2d t = tgtCentroid - R * srcCentroid;

    rotationMatrix.setIdentity();
    rotationMatrix.block<2, 2>(0, 0) = R;

    translationVector = t;
}

void ICPAlgorithm::transformPointCloud(const vector<pair<double, double>> &points,
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

void ICPAlgorithm::convertToPointCloud(const vector<pair<double, double>> &points, PointCloud::Ptr cloud, float z)
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

cv::Mat ICPAlgorithm::estimateCameraPose(const vector<pair<double, double>> &pointsN, const vector<pair<double, double>> &points37,
                                         const vector<int> &correspondences)
{
    PointCloud::Ptr cloudA(new PointCloud);
    PointCloud::Ptr cloudB(new PointCloud);
    convertToPointCloud(pointsN, cloudA, 0);
    convertToPointCloud(points37, cloudB, 1800);

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

    // K.at<double>(0, 0) = 469; // f_x
    // K.at<double>(1, 1) = 469; // f_y
    // K.at<double>(0, 2) = 963; // c_x
    // K.at<double>(1, 2) = 594; // c_y

    cv::Mat rvec, tvec;
    cv::solvePnP(objectPoints, imagePoints, K, cv::Mat(), rvec, tvec);

    cv::Mat R;
    cv::Rodrigues(rvec, R);

    cv::Mat pose = cv::Mat::eye(4, 4, CV_64F);
    R.copyTo(pose(cv::Rect(0, 0, 3, 3)));
    tvec.copyTo(pose(cv::Rect(3, 0, 1, 3)));

    return pose.inv();
}
