#ifndef READ_FILE_H
#define READ_FILE_H

#include <vector>
#include <fstream>
#include <sstream>
#include <string>

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

        std::cout << "Read numbers: " << i << ", " << numbers[0] << ", " << numbers[1] << std::endl;
        i++;
        pair<double, double> tmp = {numbers[0], numbers[1]};
        pointsN.push_back(tmp);
    }

    // 关闭文件
    file.close();
    return pointsN;
}

#endif // READ_FILE_H