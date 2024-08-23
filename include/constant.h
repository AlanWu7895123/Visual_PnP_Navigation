// constant.h
#ifndef CONSTANT_H
#define CONSTANT_H

#include <mutex>
#include <condition_variable>
#include <cmath>
#include <eigen3/Eigen/Core>
// #include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <vector>
#include <thread>

#include "camera.h"
#include "filter.h"
#include "detect.h"
// #include "icp.h"
// #include "icp_algorithm.h"
// #include "pnp.h"
#include "utils.h"
#include "network_camera.h"
#include "json.hpp"
#include "controller.h"
#include "agv.h"

extern std::unordered_map<std::string, std::string> config;

extern std::atomic<State> currentState;
extern std::mutex stateMutex;
extern queue<std::atomic<State>> qState;
extern std::atomic<AGVMoveStep> agvMoveStep;
extern double currentAngle;
extern Eigen::Vector3d relative;
extern std::vector<std::thread> threads;

extern Camera camera;
extern NetworkCamera networkCamera;
extern cv::Mat myCameraMatrix, myDistCoeffs;
extern std::string cameraUrl;
extern AGV agv;

extern cv::Mat buffer;
extern pair<double, double> targetPose;
extern pair<double, double> currentPose;
extern pair<double, double> agvPose;

extern const size_t windowSize;
extern std::vector<double> weights;
extern CameraPositionFilter filter;

extern vector<pair<double, double>> sourcePoints;
extern std::vector<std::pair<double, double>> targetPoints;
extern vector<int> correspondences;
extern Matrix3d rotationMatrix;
extern Vector2d translationVector;
extern ofstream out;

#endif // CONSTANT_H
