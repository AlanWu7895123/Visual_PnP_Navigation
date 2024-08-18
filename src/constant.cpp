#include "constant.h"

std::atomic<State> currentState(State::INIT);
std::atomic<AGVToward> currentAGVToward(AGVToward::VERTICAL);
std::vector<std::thread> threads;

Camera camera;
NetworkCamera networkCamera;
std::string url = "";
Controller configController("192.168.192.5", "19207"); // 配置API
Controller stateController("192.168.192.5", "19204");  // 状态API
Controller moveController("192.168.192.5", "19206");   // 控制API
AGV agv("192.168.192.5");

cv::Mat buffer;
pair<double, double> targetPose;
pair<double, double> currentPose{9999, 9999};

const size_t windowSize = 10;
std::vector<double> weights = {1, 1, 1, 1, 1, 2, 2, 2, 2, 3};
CameraPositionFilter filter(windowSize, weights);

vector<pair<double, double>> pointsN;
std::vector<std::pair<double, double>> points37;
vector<int> correspondences;
Matrix3d rotationMatrix = Matrix3d::Identity();
Vector2d translationVector = Vector2d::Zero();
ofstream out("../data/trajectory.txt", ios::out);