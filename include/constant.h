// constant.h
#ifndef CONSTANT_H
#define CONSTANT_H

#include <mutex>
#include <condition_variable>
#include <cmath>
#include <eigen3/Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <vector>
#include <thread>

#include "camera.h"
#include "filter.h"
#include "detect.h"
#include "icp.h"
#include "pnp.h"
#include "utils.h"
#include "network_camera.h"
#include "json.hpp"
#include "controller.h"
#include "agv.h"

extern std::atomic<State> currentState;
extern std::atomic<AGVToward> currentAGVToward;
extern std::vector<std::thread> threads;

extern Camera camera;
extern NetworkCamera networkCamera;
extern std::string url;
extern Controller configController; // 配置API
extern Controller stateController;  // 状态API
extern Controller moveController;   // 控制API
extern AGV agv;

extern cv::Mat buffer;
extern pair<double, double> targetPose;
extern pair<double, double> currentPose;

extern const size_t windowSize;
extern std::vector<double> weights;
extern CameraPositionFilter filter;

extern vector<pair<double, double>> pointsN;
extern std::vector<std::pair<double, double>> points37;
extern vector<int> correspondences;
extern Matrix3d rotationMatrix;
extern Vector2d translationVector;
extern ofstream out;

#endif // CONSTANT_H
