#ifndef HUNGARIAN_H
#define HUNGARIAN_H

#include <vector>
#include <eigen3/Eigen/Dense>

class HungarianAlgorithm
{
public:
    // HungarianAlgorithm(const std::vector<std::vector<double>> &costMatrix);
    // double solve(std::vector<int> &result);
    double Solve(const Eigen::MatrixXd &costMatrix, std::vector<int> &assignment);

private:
    // const double INF = 1e9;
    // const std::vector<std::vector<double>> &costMatrix;
    // int n, m;
    // std::vector<double> u, v;
    // std::vector<int> p, way;
};

#endif // HUNGARIAN_H
