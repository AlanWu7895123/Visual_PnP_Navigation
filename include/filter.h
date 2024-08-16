#ifndef FILTER_H
#define FILTER_H

#include <deque>
#include <vector>
#include <iostream>
#include <utility>
#include "utils.h"

class CameraPositionFilter
{
public:
    CameraPositionFilter(size_t windowSize, std::vector<double> weights);

    int addPosition(const std::pair<double, double> &camera_position);

    std::pair<double, double> getPose();

private:
    std::pair<double, double> weightedMovingAverageFilter(
        const std::deque<std::pair<double, double>> &window,
        const std::vector<double> &weights);

    std::deque<std::pair<double, double>> window;
    const size_t windowSize;
    std::vector<double> weights;

    std::pair<double, double> avg_position;
};

#endif // FILTER_H
