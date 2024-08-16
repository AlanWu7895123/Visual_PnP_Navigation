#include <mutex>
#include <condition_variable>
#include <cmath>
#include <eigen3/Eigen/Core>
#include <opencv2/core/eigen.hpp>

#include "camera.h"
#include "filter.h"
#include "detect.h"
#include "icp.h"
#include "pnp.h"
#include "utils.h"

using namespace pcl;
using namespace cv;

std::atomic<State> currentState(State::INIT);
std::vector<std::thread> threads;
Camera camera;
cv::Mat buffer;

const size_t windowSize = 10;
std::vector<double> weights = {1, 1, 1, 1, 1, 2, 2, 2, 2, 3};
CameraPositionFilter filter(windowSize, weights);

vector<pair<double, double>> pointsN;
std::vector<std::pair<double, double>> points37;
vector<int> correspondences;
Matrix3d rotationMatrix = Matrix3d::Identity();
Vector2d translationVector = Vector2d::Zero();
ofstream out("../data/trajectory.txt", ios::out);

void initThread()
{
    if (!camera.open())
    {
        currentState = State::FINISHED;
    }
    else
    {
        points37 = read37Points("../data/points.txt");
        translationVector << -1024,
            -768;
        currentState = State::CAPUTRE_IMAGE;
    }
}

void finishThread()
{
    camera.close();
}

void captureImageThread()
{
    cv::namedWindow("Camera", cv::WINDOW_AUTOSIZE);
    cv::Mat frame;
    if (!camera.getFrame(frame))
    {
        cout << "get camera image failed" << endl;
        currentState = State::CAPUTRE_IMAGE;
    }
    else
    {
        frame.copyTo(buffer);
        currentState = State::FEATURE_DETECT;
    }
}

void featureDetectThread()
{
    cv::Mat frame = buffer.clone();
    ImageProcessor processor(frame);
    processor.convertToGray();
    processor.convertGrayToBinary();
    processor.findContours();
    processor.detectCircles();
    currentState = State::MATCHING;
}

void matchingThread()
{

    pointsN = readNPoints("../data/circles.txt");
    if (pointsN.size() > points37.size())
    {
        std::cout << "ERROR: There are so many detect points..." << std::endl;
        currentState = State::CAPUTRE_IMAGE;
        return;
    }

    ICPAlgorithm icpAlg(pointsN, points37, rotationMatrix, translationVector);
    icpAlg.calTransformed();
    int icpFlag = icpAlg.pclIcp();

    if (icpFlag == 0)
    {
        correspondences = icpAlg.getCorrespondences();
        cv::Mat img = buffer.clone();
        for (int i = 0; i < pointsN.size(); i++)
        {
            Point center(cvRound(pointsN[i].first), cvRound(pointsN[i].second));
            circle(img, center, 3, Scalar(0, 255, 0), -1, LINE_AA);
            string center_text = "No." + to_string(correspondences[i] + 1) + " " + to_string(center.x) + "," + to_string(center.y);
            Point text_position(center.x - 55, center.y - 55);
            putText(img, center_text, text_position, FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 0), 1, LINE_AA);
            circle(img, center, 45, Scalar(0, 0, 255), 3, LINE_AA);
        }
        imshow("final match result", img);
        waitKey(1);
        currentState = State::MAPPING;
    }
    else
    {
        currentState = State::CAPUTRE_IMAGE;
    }
}

void mappingThread()
{
    PNPAlgorithm pnpAlg(pointsN, points37, correspondences);
    pnpAlg.estimateCameraPose();
    cv::Mat pose_inv = pnpAlg.getPose();

    cv::Mat camera_position = pose_inv(cv::Rect(3, 0, 1, 3));

    int pnpFlag = filter.addPosition({camera_position.at<double>(0), camera_position.at<double>(1)});
    if (pnpFlag == 0)
    {
        pair<double, double> pose = filter.getPose();

        rotationMatrix = getRotationMatrix(pose_inv);
        translationVector = getTranslationVector(pose_inv);
        cout << "----new rotationMatrix----" << endl;
        cout << rotationMatrix << endl;
        cout << "----new translationMatrix----" << endl;
        cout << translationVector << endl;

        out << pose.first + 1024 << " " << pose.second + 768 << endl;
    }
    currentState = State::CAPUTRE_IMAGE;
}

void testThread()
{
    string filename;
    cout << "Please enter the image name" << endl;
    cin >> filename;
    std::vector<std::pair<double, double>> points37;
    points37 = read37Points("../data/points.txt");
    Matrix3d rotationMatrix = Matrix3d::Identity();
    Vector2d translationVector = Vector2d::Zero();

    translationVector << -1024,
        -768;

    // rotationMatrix << 0.993109, 0.117198, -2.50351e-06,
    //     -0.117198, 0.993109, -1.39689e-05,
    //     8.49121e-07, 1.41661e-05, 1;

    // translationVector << -944.02,
    //     -798.208;

    ImageProcessor processor(filename);
    processor.readImage();
    processor.convertToGray();
    processor.convertGrayToBinary();
    processor.findContours();
    processor.detectCircles();
    processor.saveResults();

    vector<pair<double, double>> pointsN;

    pointsN = readNPoints("../data/circles.txt");

    ICPAlgorithm icpAlg(pointsN, points37, rotationMatrix, translationVector);

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
    for (int i = 0; i < pointsN.size(); i++)
    {
        Point center(cvRound(pointsN[i].first), cvRound(pointsN[i].second));
        circle(img, center, 3, Scalar(0, 255, 0), -1, LINE_AA);
        string center_text = "No." + to_string(correspondences[i] + 1) + " " + to_string(center.x) + "," + to_string(center.y);
        Point text_position(center.x - 55, center.y - 55);
        putText(img, center_text, text_position, FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 0), 1, LINE_AA);
        circle(img, center, 45, Scalar(0, 0, 255), 3, LINE_AA);
    }
    imwrite("../data/" + filename + "_match_result.jpg", img);

    if (pointsN.size() < 4)
    {
        cerr << "ERROR: Not enough points in the source point cloud." << endl;
        return;
    }

    // calTransformed(pointsN, targetCloud, correspondences, rotationMatrix, translationVector);

    PNPAlgorithm pnpAlg(pointsN, points37, correspondences);

    pnpAlg.estimateCameraPose();
    cv::Mat pose_inv = pnpAlg.getPose();
    rotationMatrix = getRotationMatrix(pose_inv);
    translationVector = getTranslationVector(pose_inv);

    cout << "----new rotationMatrix----" << endl;
    cout << rotationMatrix << endl;
    cout << "----new translationMatrix----" << endl;
    cout << translationVector << endl;

    cv::Mat camera_position = pose_inv(cv::Rect(3, 0, 1, 3));

    cout << "Camera Position in World Coordinates: " << endl
         << "X: " << camera_position.at<double>(0) << ", "
         << "Y: " << camera_position.at<double>(1) << ", "
         << "Z: " << camera_position.at<double>(2) << endl;
}

int main()
{
    int mode;
    cout << "choose mode, 0 is test, 1 is normal\n";
    cin >> mode;
    if (mode == 0)
    {
        currentState = State::TEST;
    }
    else if (mode == 1)
    {
        currentState = State::INIT;
    }
    else
    {
        cout << "error mode, please run again" << endl;
        return 0;
    }

    while (currentState != State::FINISHED)
    {
        if (kbhit())
        {
            char ch = getchar();
            if (ch == 'q' || ch == 'Q')
            {
                std::cout << "\nQ key pressed. Stopping the system..." << std::endl;
                threads.emplace_back(finishThread);
                currentState = State::FINISHED;
                break;
            }
        }

        for (auto it = threads.begin(); it != threads.end();)
        {
            if (it->joinable() && it->get_id() == std::this_thread::get_id())
            {
                it->join();
                it = threads.erase(it);
            }
            else
            {
                ++it;
            }
        }

        cout << "currentState=" << currentState.load() << endl;
        switch (currentState.load())
        {
        case State::INIT:
            threads.emplace_back(initThread);
            currentState = State::CAPUTRE_IMAGE;
            break;
        case State::CAPUTRE_IMAGE:
            threads.emplace_back(captureImageThread);
            currentState = State::CAPUTRE_IMAGE_RUNNING;
            break;
        case State::FEATURE_DETECT:
            threads.emplace_back(featureDetectThread);
            currentState = State::FEATURE_DETECT_RUNNING;
            break;
        case State::MATCHING:
            threads.emplace_back(matchingThread);
            currentState = State::MATCHING_RUNNING;
            break;
        case State::MAPPING:
            threads.emplace_back(mappingThread);
            currentState = State::MAPPING_RUNNING;
            break;
        case State::TEST:
            threads.emplace_back(testThread);
            currentState = State::FINISHED;
            break;
        default:
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    for (auto &t : threads)
    {
        if (t.joinable())
        {
            t.join();
        }
    }

    std::cout << "System finished." << std::endl;

    return 0;
}