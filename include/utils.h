#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include "struct.h"

using namespace std;

vector<pair<double, double>> read37Points(string filename)
{
    cout << "read 37 points" << endl;
    vector<pair<double, double>> points37;
    std::ifstream file(filename);

    if (!file.is_open())
    {
        std::cerr << "Error: Unable to open the file." << std::endl;
        return points37;
    }

    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        std::vector<double> numbers;
        double number;
        while (iss >> number)
        {
            numbers.push_back(number);
        }

        if (numbers.size() != 3)
        {
            std::cerr << "Error: Invalid number of values in line." << std::endl;
            continue;
        }

        std::cout << "Read numbers: " << numbers[0] << ", " << numbers[1] << ", " << numbers[2] << std::endl;
        pair<double, double> tmp = {numbers[1], numbers[2]};
        points37.push_back(tmp);
    }

    file.close();
    return points37;
}

vector<pair<double, double>> readNPoints(string filename)
{
    cout << "read N points" << endl;
    vector<pair<double, double>> pointsN;
    std::ifstream file(filename);

    // 检查文件是否成功打开
    if (!file.is_open())
    {
        std::cerr << "Error: Unable to open the file." << std::endl;
        return pointsN;
    }

    std::string line;
    int i = 1;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        std::vector<double> numbers;
        double number;
        while (iss >> number)
        {
            numbers.push_back(number);
        }

        if (numbers.size() != 2)
        {
            std::cerr << "Error: Invalid number of values in line." << std::endl;
            continue;
        }

        // std::cout << "Read numbers: " << i << ", " << numbers[0] << ", " << numbers[1] << std::endl;
        i++;
        pair<double, double> tmp = {numbers[0], numbers[1]};
        pointsN.push_back(tmp);
    }

    // 关闭文件
    file.close();
    return pointsN;
}

int kbhit()
{
    struct termios oldt, newt;
    int ch;
    int oldf;

    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

    ch = getchar();

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);

    if (ch != EOF)
    {
        ungetc(ch, stdin);
        return 1;
    }

    return 0;
}

std::pair<double, double> weightedMovingAverageFilter(
    const std::deque<std::pair<double, double>> &window,
    const std::vector<double> &weights)
{
    double sumWeights = 0;
    double weightedSumX = 0;
    double weightedSumY = 0;

    for (size_t i = 0; i < window.size(); ++i)
    {
        weightedSumX += window[i].first * weights[i];
        weightedSumY += window[i].second * weights[i];
        sumWeights += weights[i];
    }

    return {weightedSumX / sumWeights, weightedSumY / sumWeights};
}

std::ostream &operator<<(std::ostream &os, State state)
{
    switch (state)
    {
    case State::INIT:
        os << "INIT";
        break;
    case State::TEST:
        os << "TEST";
        break;
    case State::TEST_RUNNING:
        os << "TEST_RUNNING";
        break;
    case State::CAPUTRE_IMAGE:
        os << "CAPUTRE_IMAGE";
        break;
    case State::FEATURE_DETECT:
        os << "FEATURE_DETECT";
        break;
    case State::COORDINATE_CALCULATION:
        os << "COORDINATE_CALCULATION";
        break;
    case State::CAPUTRE_IMAGE_RUNNING:
        os << "CAPUTRE_IMAGE_RUNNING";
        break;
    case State::FEATURE_DETECT_RUNNING:
        os << "FEATURE_DETECT_RUNNING";
        break;
    case State::COORDINATE_CALCULATION_RUNNING:
        os << "COORDINATE_CALCULATION_RUNNING";
        break;
    case State::FINISHED:
        os << "FINISHED";
        break;
    default:
        os << "UNKNOWN";
        break;
    }
    return os;
}

#endif // UTILS_H