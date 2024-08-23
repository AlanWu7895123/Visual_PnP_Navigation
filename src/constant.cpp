#include "constant.h"

std::unordered_map<std::string, std::string> config = readConfig("../config/base.config");

std::atomic<State> currentState(State::INIT);
std::mutex stateMutex;
queue<std::atomic<State>> qState;
std::atomic<AGVMoveStep> agvMoveStep(AGVMoveStep::ROTATE);
double currentAngle;
Eigen::Vector3d relative(-115.0, -763.5, 1.0);
std::vector<std::thread> threads;

Camera camera;
NetworkCamera networkCamera;
cv::Mat myCameraMatrix, myDistCoeffs;
string cameraUrl = config["cameraUrl"];
AGV agv(config["agvIp"]);

cv::Mat buffer;
pair<double, double> targetPose;
pair<double, double> currentPose;
pair<double, double> agvPose;

const size_t windowSize = 10;
std::vector<double> weights = {1, 1, 1, 1, 1, 2, 2, 2, 2, 3};
CameraPositionFilter filter(windowSize, weights);

vector<pair<double, double>> sourcePoints;
std::vector<std::pair<double, double>> targetPoints;
vector<int> correspondences;
Matrix3d rotationMatrix = Matrix3d::Identity();
Vector2d translationVector = Vector2d::Zero();
ofstream out(config["outputPath"], ios::out);