#include <mutex>
#include <condition_variable>
#include <cmath>
#include <eigen3/Eigen/Core>
#include <opencv2/core/eigen.hpp>

#include "camera.h"
#include "filter.h"
#include "detect.h"
#include "icp_algorithm.h"
#include "pnp.h"
#include "utils.h"
#include "network_camera.h"
#include "json.hpp"
#include "controller.h"
#include "agv.h"
#include "constant.h"
#include "agv_system.h"

using namespace pcl;
using namespace cv;
using namespace std;
using namespace efsm;

ofstream _angle_out("../data/angle.txt", ios::out);
ofstream _agvPose_out("../data/agvPose.txt", ios::out);
ofstream _t_out("../data/t.txt", ios::out);

void initThread()
{
    // currentState = State::INIT_RUNNING;
    // {
    //     std::lock_guard<std::mutex> stateLock(stateMutex);
    //     if (currentState != State::FINISHED)
    //         currentState = State::CAPUTRE_IMAGE;
    // }
    // namedWindow("origin camera image", WINDOW_NORMAL);
    // return;

    // networkCamera.setUrl(cameraUrl);
    // if (!networkCamera.open())
    // {
    //     {
    //         std::lock_guard<std::mutex> stateLock(stateMutex);
    //         currentState = State::FINISHED;
    //     }
    // }
    // else
    // {
    int imageLength = stoi(config["imageLength"]);
    int imageWidth = stoi(config["imageWidth"]);
    targetPoints = readMapPoinits("../data/points.txt");
    translationVector << -stoi(config["imageLength"]) / 2 * stod(config["scale"]),
        -stoi(config["imageWidth"]) / 2 * stod(config["scale"]);
    cout << translationVector << endl;
    {
        std::lock_guard<std::mutex> stateLock(stateMutex);
        if (currentState != State::FINISHED)
            currentState = State::CAPUTRE_IMAGE;
    }
    // namedWindow("origin camera image", WINDOW_NORMAL);
    namedWindow("final match result", WINDOW_AUTOSIZE);
    agv.setPermission();
    targetPose.first = stod(config["targetPoseX"]);
    targetPose.second = stod(config["targetPoseY"]);
    // }

    return;
}

void finishThread()
{
    networkCamera.close();
    return;
}

void captureImageThread()
{
    // currentState = State::CAPUTRE_IMAGE_RUNNING;
    // {
    //     std::lock_guard<std::mutex> stateLock(stateMutex);
    //     if (currentState != State::FINISHED)
    //         currentState = State::CAPUTRE_IMAGE_RUNNING;
    // }
    // cout << "capture image" << endl;
    {
        std::lock_guard<std::mutex> stateLock(stateMutex);
        currentState = State::CAPUTRE_IMAGE_RUNNING;
    }
    cap.open(cameraUrl);
    {
        std::lock_guard<std::mutex> stateLock(stateMutex);
        currentState = State::CAPUTRE_IMAGE;
    }
    cv::Mat frame;
    while (true)
    {
        // cout << "while true" << endl;
        // std::lock_guard<std::mutex> bufferLock(bufferMutex);
        // frame.copyTo(buffer);

        cap.read(frame);
        // imshow("origin camera image", frame);
        waitKey(1);
        {
            std::lock_guard<std::mutex> stateLock(stateMutex);

            if (currentState == State::CAPUTRE_IMAGE)
            {
                std::lock_guard<std::mutex> bufferLock(bufferMutex);
                frame.copyTo(buffer);
                if (currentState != State::FINISHED)
                    currentState = State::FEATURE_DETECT;
            }
            else if (currentState == State::FINISHED)
            {
                cap.release();
                break;
            }
        }

        // currentState = State::CAPUTRE_IMAGE;
    }
    return;
    // cv::namedWindow("Camera", cv::WINDOW_AUTOSIZE);
    // cv::Mat frame;
    // if (!networkCamera.getFrame(buffer))
    // {
    //     cout << "get camera image failed" << endl;
    //     {
    //         std::lock_guard<std::mutex> stateLock(stateMutex);
    //         if (currentState != State::FINISHED)
    //             currentState = State::CAPUTRE_IMAGE;
    //     }
    // }
    // else
    // {
    //     {
    //         std::lock_guard<std::mutex> bufferLock(bufferMutex);
    //         // frame.copyTo(buffer);
    //         imshow("origin camera image", buffer);
    //         waitKey(1);
    //     }
    //     {
    //         std::lock_guard<std::mutex> stateLock(stateMutex);
    //         if (currentState != State::FINISHED)
    //             // currentState = State::FEATURE_DETECT;
    //             currentState = State::CAPUTRE_IMAGE;
    //     }
    // }
    // return;
}

void featureDetectThread()
{
    // currentState = State::FEATURE_DETECT_RUNNING;
    cv::Mat frame;
    {
        std::lock_guard<std::mutex> bufferLock(bufferMutex);
        frame = buffer.clone();
    }

    ImageProcessor processor(frame);
    processor.convertToGray();
    processor.convertGrayToBinary();
    processor.findContours();
    processor.detectCircles();
    {
        std::lock_guard<std::mutex> stateLock(stateMutex);
        if (currentState != State::FINISHED)
            currentState = State::MATCHING;
    }
    return;
}

void matchingThread()
{
    // currentState = State::MATCHING_RUNNING;
    sourcePoints = readNPoints("../data/circles.txt");
    if (sourcePoints.size() > targetPoints.size())
    {
        std::cout << "ERROR: There are so many detect points..." << std::endl;
        {
            std::lock_guard<std::mutex> stateLock(stateMutex);
            if (currentState != State::FINISHED)
                currentState = State::CAPUTRE_IMAGE;
        }
        return;
    }
    cout << "icp translationVector = " << translationVector << endl;

    ICPAlgorithm icpAlg(sourcePoints, targetPoints, rotationMatrix, translationVector);
    icpAlg.calTransformed();
    int icpFlag = icpAlg.pclIcp();

    if (icpFlag == 0)
    {
        correspondences = icpAlg.getCorrespondences();
        cv::Mat img;
        {
            std::lock_guard<std::mutex> bufferLock(bufferMutex);
            img = buffer.clone();
        }
        for (int i = 0; i < sourcePoints.size(); i++)
        {
            // Point center(cvRound(sourcePoints[i].first), cvRound(sourcePoints[i].second));
            Point center(cvRound(sourcePoints[i].first / stod(config["scale"])),
                         cvRound(sourcePoints[i].second) / stod(config["scale"]));
            circle(img, center, 3, Scalar(0, 255, 0), -1, LINE_AA);
            string center_text = "No." + to_string(correspondences[i] + 1) + " " + to_string(center.x) + "," +
                                 to_string(center.y);
            Point text_position(center.x - 55, center.y - 55);
            putText(img, center_text, text_position, FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 0), 1, LINE_AA);
            circle(img, center, 45, Scalar(0, 0, 255), 3, LINE_AA);
        }
        imshow("final match result", img);
        waitKey(1);
        {
            std::lock_guard<std::mutex> stateLock(stateMutex);
            if (currentState != State::FINISHED)
                currentState = State::MAPPING;
        }
    }
    else
    {
        {
            std::lock_guard<std::mutex> stateLock(stateMutex);
            if (currentState != State::FINISHED)
                currentState = State::CAPUTRE_IMAGE;
        }
    }
    return;
}

void mappingThread()
{
    // currentState = State::MAPPING_RUNNING;
    PNPAlgorithm pnpAlg(sourcePoints, targetPoints, correspondences);
    pnpAlg.setSolveParams(stoi(config["iterationsCount"]), stof(config["reprojectionError"]),
                          stod(config["confidence"]));
    bool pnpFlag = pnpAlg.estimateCameraPose();
    if (pnpFlag)
    {
        cv::Mat pose_inv = pnpAlg.getPose();
        // cout << "----pose_inv----" << endl;
        // cout << pose_inv << endl;

        cv::Mat camera_position = pose_inv(cv::Rect(3, 0, 1, 3));

        cout << "----current rotationMatrix----" << endl;
        cout << rotationMatrix << endl;
        cout << "----current translationMatrix----" << endl;
        cout << translationVector << endl;
        Eigen::Matrix<double, 3, 1> transform = rotationMatrix * basic_t;
        // currentPose.first -= transform(0, 0);
        // currentPose.second -= transform(1, 0);
        double _x = camera_position.at<double>(0) - transform(0, 0);
        double _y = camera_position.at<double>(1) - transform(1, 0);

        int filterFlag = filter.addPosition({_x, _y});
        if (filterFlag == 0)
        {
            currentPose = filter.getPose();

            rotationMatrix = (rotationMatrix + getRotationMatrix(pose_inv)) / 2;
            translationVector = (translationVector + getTranslationVector(pose_inv)) / 2;
            cout << "----new rotationMatrix----" << endl;
            cout << rotationMatrix << endl;
            cout << "----new translationMatrix----" << endl;
            cout << translationVector << endl;
            // translationVector += basic_t;

            out << currentPose.first << " " << currentPose.second << endl;
            // double theta = std::atan2(rotationMatrix(1, 0), rotationMatrix(0, 0));

            // // 将角度转换为度数（如果需要）
            // double theta_degrees = theta * 180.0 / M_PI;

            Eigen::AngleAxisd angleAxis(rotationMatrix);
            double angle = angleAxis.angle(); // 提取旋转角度（弧度）

            // 将角度转换为度数
            double angle_degrees = angle * 180.0 / M_PI;

            _t_out << translationVector(0, 0) << "," << translationVector(1, 0) << endl;
            _angle_out << angle_degrees << endl;

            {
                std::lock_guard<std::mutex> stateLock(stateMutex);
                if (currentState != State::FINISHED)
                    currentState = State::MOVING;
                // currentState = State::CAPUTRE_IMAGE;
            }
            return;
        }
    }
    {
        std::lock_guard<std::mutex> stateLock(stateMutex);
        if (currentState != State::FINISHED)
            currentState = State::CAPUTRE_IMAGE;
    }
    return;
}

void movingThread()
{
    // currentState = State::MOVING_RUNNING;
    nlohmann::json responseJson = agv.getSpeed();
    Eigen::Matrix<double, 3, 1> tmp = rotationMatrix * relative;

    agvPose = {currentPose.first + tmp(0, 0), currentPose.second + tmp(1, 0)};
    cout << "agvPose = (" << agvPose.first << "," << agvPose.second << ")" << endl;
    _agvPose_out << "(" << agvPose.first << "," << agvPose.second << ")" << endl;

    if (config["mode"] != "test")
    {
        {
            std::lock_guard<std::mutex> stateLock(stateMutex);
            if (currentState != State::FINISHED)
                currentState = State::CAPUTRE_IMAGE;
        }
        return;
    }

    if (responseJson["is_stop"] == true)
    {
        if (agvMoveStep == AGVMoveStep::ROTATE)
        {
            double angle = calculateAngle(currentPose, agvPose, targetPose);
            // 将角度从弧度转换为度数
            _angle_out << "rotate angle = " << angle * 180.0 / M_PI << endl;
            int direction = 1;
            if (abs(angle) < 0.03)
            {
                cout << "finish rotate" << endl;
                if (config["testMode"] != "only rotate")
                    agvMoveStep = AGVMoveStep::FORWARD;
            }
            else
            {
                if (angle < 0)
                {
                    direction = -1;
                }
                cout << "request to agv rotate" << endl;
                // std::cout << "waiting before agv rotate\n";
                // std::this_thread::sleep_for(std::chrono::milliseconds(10000));
                _angle_out << "starte rotate, rotate params = " << abs(angle) << ", " << direction * 0.0015 << endl;
                agv.rotate(abs(angle), direction * 0.0015);
            }
        }
        if (agvMoveStep == AGVMoveStep::FORWARD)
        {
            double dist = distance(agvPose, targetPose);
            int direction = 1;
            if (dist < 30)
            {
                _agvPose_out << "arrive to the target pose" << endl;
                cout << "waiting back" << endl;
                sleep(100);
                cout << "finish working" << endl;
                {
                    std::lock_guard<std::mutex> stateLock(stateMutex);
                    currentState = State::FINISHED;
                }
                agvMoveStep = AGVMoveStep::BACK;
            }
            else
            {
                if (distance({0, 0}, agvPose) < distance({0, 0}, targetPose))
                {
                    direction = -1;
                }
                _agvPose_out << "start move, move params = " << dist / 1000 << "," << direction * 0.002 << endl;
                agv.move(dist / 1000, direction * 0.01, 0);
                agvMoveStep = AGVMoveStep::ROTATE;
            }
        }
        if (agvMoveStep == AGVMoveStep::BACK)
        {
            double dist = distance({0, 0}, currentPose);
            int direction = 1;
            if (abs(dist) < 150)
            {
                cout << "back to the zero pose" << endl;
                sleep(100);
            }
            else
            {
                if (distance(currentPose, agvPose) > distance({0, 0}, agvPose))
                {
                    direction = -1;
                }
                agv.move(dist / 1000, direction * 0.01, 0);
            }
        }
    }
    {
        std::lock_guard<std::mutex> stateLock(stateMutex);
        if (currentState != State::FINISHED)
            currentState = State::CAPUTRE_IMAGE;
    }
    return;
}

void testThread()
{
    // std::string configFileName = "../config/config.txt";
    // std::unordered_map<std::string, std::string> config = readConfig(configFileName);

    // currentState = State::TEST_RUNNING;
    string mapFilePath;
    int imageLength;
    int imageWidth;
    if (config.find("mapFilePath") != config.end())
    {
        mapFilePath = config["mapFilePath"];
        std::cout << "mapFilePath: " << mapFilePath << std::endl;
    }
    else
    {
        cout << "find config mapFilePath fail" << endl;
    }

    if (config.find("imageLength") != config.end())
    {
        imageLength = stoi(config["imageLength"]);
        std::cout << "imageLenth: " << imageLength << std::endl;
    }
    else
    {
        cout << "find config imageLength fail" << endl;
    }

    if (config.find("imageWidth") != config.end())
    {
        imageWidth = stoi(config["imageWidth"]);
        std::cout << "imageWidth: " << imageWidth << std::endl;
    }
    else
    {
        cout << "find config imageWidth fail" << endl;
    }

    string filename;
    cout << "Please enter the image name" << endl;
    cin >> filename;
    std::vector<std::pair<double, double>> targetPoints;
    targetPoints = readMapPoinits(mapFilePath);
    Matrix3d rotationMatrix = Matrix3d::Identity();
    Vector2d translationVector = Vector2d::Zero();

    translationVector << -imageLength / 2 * stod(config["scale"]),
        -imageWidth / 2 * stod(config["scale"]);

    // rotationMatrix << 0.981309, -0.192439, 2.36582e-05,
    //     0.192439, 0.981309, 1.11927e-06,
    //     -2.34314e-05, 3.45441e-06, 1;

    // translationVector << -1589.35,
    //     -1493.34;

    ImageProcessor processor(filename);
    processor.readImage();
    processor.convertToGray();
    processor.convertGrayToBinary();
    processor.findContours();
    processor.detectCircles();
    processor.saveResults();

    vector<pair<double, double>> sourcePoints;

    sourcePoints = readNPoints("../data/circles.txt");

    // cout << translationVector << endl;

    ICPAlgorithm icpAlg(sourcePoints, targetPoints, rotationMatrix, translationVector);

    icpAlg.calTransformed();
    icpAlg.pclIcp();

    vector<int> correspondences = icpAlg.getCorrespondences();

    cv::Mat img = imread("../data/" + filename + ".png");
    if (img.empty())
    {
        img = imread("../data/" + filename + ".jpg");
        if (img.empty())
        {
            cerr << "ERROR: Could not read the image." << endl;
            return;
        }
    }
    for (int i = 0; i < sourcePoints.size(); i++)
    {
        Point center(cvRound(sourcePoints[i].first / stod(config["scale"])),
                     cvRound(sourcePoints[i].second) / stod(config["scale"]));
        circle(img, center, 3, Scalar(0, 255, 0), -1, LINE_AA);
        string center_text = "No." + to_string(correspondences[i] + 1) + " " + to_string(center.x) + "," +
                             to_string(center.y);
        Point text_position(center.x - 55, center.y - 55);
        putText(img, center_text, text_position, FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 0), 1, LINE_AA);
        circle(img, center, 45, Scalar(0, 0, 255), 3, LINE_AA);
    }
    imwrite("../data/" + filename + "_match_result.jpg", img);

    if (sourcePoints.size() < 4)
    {
        cerr << "ERROR: Not enough points in the source point cloud." << endl;
        return;
    }

    // calTransformed(sourcePoints, targetCloud, correspondences, rotationMatrix, translationVector);

    PNPAlgorithm pnpAlg(sourcePoints, targetPoints, correspondences);

    // if (!readCameraParameters("../config/camera_config.yaml", myCameraMatrix, myDistCoeffs))
    // {
    //     cout << "get camera config fail" << endl;
    //     return;
    // }
    // pnpAlg.setCameraConfig(myCameraMatrix, myDistCoeffs);
    pnpAlg.estimateCameraPose();

    cv::Mat pose_inv = pnpAlg.getPose();
    rotationMatrix = getRotationMatrix(pose_inv);
    translationVector = getTranslationVector(pose_inv);

    double theta = std::atan2(rotationMatrix(1, 0), rotationMatrix(0, 0));

    // 将角度转换为度数（如果需要）
    double theta_degrees = theta * 180.0 / M_PI;

    cout << theta_degrees << endl;

    Eigen::Vector3d t(translationVector(0), translationVector(1), 0.0);

    Eigen::Matrix<double, 3, 1> y_new = rotationMatrix * relative;

    cout << "----new rotationMatrix----" << endl;
    cout << rotationMatrix << endl;
    cout << "----new translationMatrix----" << endl;
    cout << translationVector << endl;

    cv::Mat camera_position = pose_inv(cv::Rect(3, 0, 1, 3));

    cout << "Camera Position in World Coordinates: " << endl
         << "X: " << camera_position.at<double>(0) + stoi(config["imageLength"]) / 2 * stod(config["scale"]) << ", "
         << "Y: " << camera_position.at<double>(1) + stoi(config["imageWidth"]) / 2 * stod(config["scale"]) << ", "
         << "Z: " << camera_position.at<double>(2) << endl;

    std::cout << "x: " << y_new(0, 0) + camera_position.at<double>(0) + stoi(config["imageLength"]) / 2 * stod(config["scale"]) << std::endl;
    std::cout << "y: " << y_new(1, 0) + camera_position.at<double>(1) + stoi(config["imageWidth"]) / 2 * stod(config["scale"]) << std::endl;
}

// int main()
// {
//     int mode;
//     cout << "choose mode, 0 is test, 1 is normal\n";
//     cin >> mode;
//     if (mode == 0)
//     {
//         // qState.push(State::TEST);
//         currentState = State::TEST;
//     }
//     else if (mode == 1)
//     {
//         // qState.push(State::INIT);
//         currentState = State::INIT;
//     }
//     else
//     {
//         cout << "error mode, please run again" << endl;
//         return 0;
//     }

//     while (currentState != State::FINISHED)
//     {
//         // if (!qState.empty())
//         // {
//         //     currentState = qState.front();
//         //     qState.pop();
//         // }
//         if (kbhit())
//         {
//             char ch = getchar();
//             if (ch == 'q' || ch == 'Q')
//             {
//                 std::cout << "\nQ key pressed. Stopping the system..." << std::endl;
//                 {
//                     std::lock_guard<std::mutex> stateLock(stateMutex);
//                     currentState = State::FINISHED;
//                 }
//                 threads.emplace_back(finishThread);
//                 // qState.push(State::FINISHED);
//                 break;
//             }
//         }

//         // cout << threads.size() << "----";
//         for (auto it = threads.begin(); it != threads.end();)
//         {
//             if (it->joinable() && it->get_id() == std::this_thread::get_id())
//             // if (it->joinable())
//             {
//                 it->join();
//                 it = threads.erase(it);
//             }
//             else
//             {
//                 ++it;
//             }
//         }
//         // cout << threads.size() << endl;

//         cout << "currentState=" << currentState.load() << endl;
//         switch (currentState.load())
//         {
//         case State::INIT:
//         {
//             std::lock_guard<std::mutex> stateLock(stateMutex);
//             if (currentState != State::FINISHED)
//                 currentState = State::INIT_RUNNING;
//         }
//             threads.emplace_back(initThread);
//             std::this_thread::sleep_for(std::chrono::milliseconds(1000));
//             threads.emplace_back(captureImageThread);
//             std::this_thread::sleep_for(std::chrono::milliseconds(5000));
//             break;
//         case State::CAPUTRE_IMAGE:
//             std::this_thread::sleep_for(std::chrono::milliseconds(100));
//             // {
//             //     std::lock_guard<std::mutex> stateLock(stateMutex);
//             //     if (currentState != State::FINISHED)
//             //         currentState = State::CAPUTRE_IMAGE_RUNNING;

//             //     std::lock_guard<std::mutex> bufferLock(bufferMutex);
//             //     // frame.copyTo(buffer);
//             //     // cap.read(buffer);
//             //     cap >> buffer;
//             //     if (buffer.empty())
//             //     {
//             //         cerr << "Error: 无法读取帧" << endl;
//             //         if (currentState != State::FINISHED)
//             //             currentState = State::CAPUTRE_IMAGE;
//             //     }
//             //     else
//             //     {
//             //         imshow("origin camera image", buffer);
//             //         waitKey(1);
//             //         if (currentState != State::FINISHED)
//             //             // currentState = State::FEATURE_DETECT;
//             //             currentState = State::CAPUTRE_IMAGE;
//             //     }
//             // }
//             // cout << "start a thread for image captture" << endl;
//             // threads.emplace_back(captureImageThread);
//             break;
//         case State::FEATURE_DETECT:
//         {
//             std::lock_guard<std::mutex> stateLock(stateMutex);
//             if (currentState != State::FINISHED)
//                 currentState = State::FEATURE_DETECT_RUNNING;
//         }
//             threads.emplace_back(featureDetectThread);
//             break;
//         case State::MATCHING:
//         {
//             std::lock_guard<std::mutex> stateLock(stateMutex);
//             if (currentState != State::FINISHED)
//                 currentState = State::MATCHING_RUNNING;
//         }
//             threads.emplace_back(matchingThread);
//             break;
//         case State::MAPPING:
//         {
//             std::lock_guard<std::mutex> stateLock(stateMutex);
//             if (currentState != State::FINISHED)
//                 currentState = State::MAPPING_RUNNING;
//         }
//             threads.emplace_back(mappingThread);
//             break;
//         case State::MOVING:
//         {
//             std::lock_guard<std::mutex> stateLock(stateMutex);
//             if (currentState != State::FINISHED)
//                 currentState = State::MOVING_RUNNING;
//         }
//             threads.emplace_back(movingThread);
//             break;
//         case State::TEST:
//         {
//             std::lock_guard<std::mutex> stateLock(stateMutex);
//             currentState = State::FINISHED;
//         }
//             threads.emplace_back(testThread);
//             break;
//         default:
//             break;
//         }
//         // cout << threads.size() << endl;
//         std::this_thread::sleep_for(std::chrono::milliseconds(200));
//     }

//     for (auto &t : threads)
//     {
//         if (t.joinable())
//         {
//             t.join();
//         }
//     }

//     std::cout << "System finished." << std::endl;

//     return 0;
// }

int main()
{
    int mode;
    cout << "choose mode, 0 is test, 1 is normal\n";
    cin >> mode;

    AGVSystem agvSystem(mode);
    agvSystem.setTarget(1, 1);
    agvSystem.run();
    return 0;
}