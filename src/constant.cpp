#include "constant.h"

std::unordered_map<std::string, std::string> config = readConfig("../config/base.config");

std::atomic<State> currentState(State::INIT);
std::mutex stateMutex;
queue<std::atomic<State>> qState;
std::atomic<AGVMoveStep> agvMoveStep(AGVMoveStep::ROTATE);
double currentAngle;
Eigen::Vector3d relative(115.0, 763.5, 1.0);
std::vector<std::thread> threads;

Camera camera;
NetworkCamera networkCamera;
cv::VideoCapture cap;
cv::Mat myCameraMatrix, myDistCoeffs;
string cameraUrl = config["cameraUrl"];
AGV agv(config["agvIp"]);

cv::Mat buffer;
std::mutex bufferMutex;
pair<double, double> targetPose;
pair<double, double> currentPose;
pair<double, double> agvPose;

// const size_t windowSize = 2;
// std::vector<double> weights = {1, 1};
const size_t windowSize = 10;
std::vector<double> weights = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
// std::vector<double> weights = {0.0003, 0.0022, 0.0150, 0.0650, 0.1480, 0.2220, 0.2220, 0.1480, 0.0650, 0.0150};
CameraPositionFilter filter(windowSize, weights);
CameraPositionFilter tFilter(windowSize, weights);

vector<pair<double, double>> sourcePoints;
std::vector<std::pair<double, double>> targetPoints;
vector<int> correspondences;
Matrix3d rotationMatrix = Matrix3d::Identity();
Vector2d translationVector = Vector2d::Zero();
Vector3d basic_t(-stoi(config["imageLength"]) / 2 * stod(config["scale"]), -stoi(config["imageWidth"]) / 2 * stod(config["scale"]), 1.0);
ofstream out(config["outputPath"], ios::out);