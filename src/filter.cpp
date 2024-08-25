#include <cmath>
#include "filter.h"

CameraPositionFilter::CameraPositionFilter(size_t windowSize, std::vector<double> weights)
    : windowSize(windowSize), weights(weights)
{
}

int CameraPositionFilter::addPosition(const std::pair<double, double> &camera_position)
{
    pair<double, double> position = camera_position;
    cout << "camera position = (" << camera_position.first << "," << camera_position.second << ")" << endl;
    if (abs(camera_position.first) > 10000 || abs(camera_position.second) > 10000)
    {
        cout << "illegal position, skipped" << endl;
        return 1;
    }

    if (abs(camera_position.first) < 0.01)
        position.first = 0;
    if (abs(camera_position.second) < 0.01)
        position.second = 0;

    window.push_back(position);
    if (window.size() > windowSize)
    {
        // if (distance(avg_position, camera_position) > 1000)
        // {
        //     std::cout << "ERROR: The new point is so far..." << std::endl;
        //     window.pop_back();
        //     return 1;
        // }
        // else
        // {
        window.pop_front();
        // }
    }

    if (window.size() == windowSize)
    {
        std::pair<double, double> filteredPoint = weightedMovingAverageFilter(window, weights);
        avg_position = filteredPoint;
        std::cout << "Filtered coordinates: (" << filteredPoint.first << ", " << filteredPoint.second << ")" << std::endl;
        return 0;
    }
    else
    {
        std::cout << "Insufficient data for filtering. Waiting for more points..." << std::endl;
        return 1;
    }
}

std::pair<double, double> CameraPositionFilter::weightedMovingAverageFilter(
    const std::deque<std::pair<double, double>> &window,
    const std::vector<double> &weights)
{
    double sum_weights = 0;
    double weighted_sum_x = 0;
    double weighted_sum_y = 0;

    for (size_t i = 0; i < window.size(); ++i)
    {
        weighted_sum_x += window[i].first * weights[i];
        weighted_sum_y += window[i].second * weights[i];
        sum_weights += weights[i];
    }

    return {weighted_sum_x / sum_weights, weighted_sum_y / sum_weights};
}

std::pair<double, double> CameraPositionFilter::getPose()
{
    return avg_position;
}
