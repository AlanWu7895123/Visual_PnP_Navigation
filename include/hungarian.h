#ifndef HUNGARIAN_H
#define HUNGARIAN_H

#include <vector>
#include <eigen3/Eigen/Dense>

class HungarianAlgorithm
{
public:
    double Solve(const Eigen::MatrixXd &costMatrix, std::vector<int> &assignment);
};

#endif // HUNGARIAN_H
