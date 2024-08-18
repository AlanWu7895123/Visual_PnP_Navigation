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
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <arpa/inet.h>
#include <netinet/in.h>

#include "struct.h"

using namespace Eigen;
using namespace std;
using namespace cv;

vector<pair<double, double>> read37Points(string filename);

vector<pair<double, double>> readNPoints(string filename);

int kbhit();

std::pair<double, double> weightedMovingAverageFilter(
    const std::deque<std::pair<double, double>> &window,
    const std::vector<double> &weights);

std::ostream &operator<<(std::ostream &os, State state);

Matrix3d getRotationMatrix(cv::Mat pose_inv);

Vector2d getTranslationVector(cv::Mat pose_inv);

double distance(const pair<double, double> &p1, const pair<double, double> &p2);

void calTransformed(const vector<pair<double, double>> &sourceCloud,
                    const vector<pair<double, double>> &targetCloud,
                    const vector<int> &correspondences,
                    Matrix3d &rotationMatrix,
                    Vector2d &translationVector);

std::vector<unsigned char> hexArrayToBytes(const char *hexArray);

std::string bytesToHex(const char *data, size_t len);

ProtocolHeader deSerializeProtocolHeader(const char *data);

std::string serializeProtocolHeader(const ProtocolHeader &header);

char *stringToHex(const std::string &input);

char *concatStrings(const char *str1, const char *str2);

#endif // UTILS_H