#include "agv_system.h"

using namespace efsm;

AGVSystem::AGVSystem(int type)
    : _angle_out("../data/angle.txt", ios::out),
      _agvPose_out("../data/agvPose.txt", ios::out),
      _t_out("../data/t.txt", ios::out)
{
    if (type == 0)
    {
        currentState = State::TEST;
    }
    else if (type == 1)
    {
        targetPose.first = stod(config["targetPoseX"]);
        targetPose.second = stod(config["targetPoseY"]);
        currentState = State::INIT;
    }
}

bool AGVSystem::restart(start_param_base::ptr start_param)
{
    start_param_ = std::dynamic_pointer_cast<AGVSystem_start_param>(start_param);

    if (!start_param_)
    {
        return false;
    }
    setTarget(start_param_->x, start_param_->y);
    run();
    return true;
}

bool AGVSystem::pause() { return false; }

bool AGVSystem::resume() { return false; }

bool AGVSystem::stop()
{
    std::lock_guard<std::mutex> stateLock(stateMutex);
    currentState = State::FINISHED;
    return true;
}

void AGVSystem::run_once() {}

void AGVSystem::setTarget(double x, double y)
{
    targetPose.first = x;
    targetPose.second = y;
}

State AGVSystem::getCurrentState()
{
    std::lock_guard<std::mutex> stateLock(stateMutex);
    return currentState;
}
void AGVSystem::run()
{
    cout << "target pose is (" << targetPose.first << "," << targetPose.second << ")" << endl;
    while (currentState != State::FINISHED)
    {
        if (kbhit())
        {
            char ch = getchar();
            if (ch == 'q' || ch == 'Q')
            {
                std::cout << "\nQ key pressed. Stopping the system..." << std::endl;
                {
                    std::lock_guard<std::mutex> stateLock(stateMutex);
                    currentState = State::FINISHED;
                }
                threads.emplace_back(std::bind(&AGVSystem::finishThread, this));
                break;
            }
        }
        for (auto it = threads.begin(); it != threads.end();)
        {
            if (it->joinable() && it->get_id() == std::this_thread::get_id())
            // if (it->joinable())
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
        {
            std::lock_guard<std::mutex> stateLock(stateMutex);
            if (currentState != State::FINISHED)
                currentState = State::INIT_RUNNING;
        }
            threads.emplace_back(std::bind(&AGVSystem::initThread, this));
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            threads.emplace_back(std::bind(&AGVSystem::captureImageThread, this));
            std::this_thread::sleep_for(std::chrono::milliseconds(5000));
            break;
        case State::CAPUTRE_IMAGE:
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            break;
        case State::FEATURE_DETECT:
        {
            std::lock_guard<std::mutex> stateLock(stateMutex);
            if (currentState != State::FINISHED)
                currentState = State::FEATURE_DETECT_RUNNING;
        }
            threads.emplace_back(std::bind(&AGVSystem::featureDetectThread, this));
            break;
        case State::MATCHING:
        {
            std::lock_guard<std::mutex> stateLock(stateMutex);
            if (currentState != State::FINISHED)
                currentState = State::MATCHING_RUNNING;
        }
            threads.emplace_back(std::bind(&AGVSystem::matchingThread, this));
            break;
        case State::MAPPING:
        {
            std::lock_guard<std::mutex> stateLock(stateMutex);
            if (currentState != State::FINISHED)
                currentState = State::MAPPING_RUNNING;
        }
            threads.emplace_back(std::bind(&AGVSystem::mappingThread, this));
            break;
        case State::MOVING:
        {
            std::lock_guard<std::mutex> stateLock(stateMutex);
            if (currentState != State::FINISHED)
                currentState = State::MOVING_RUNNING;
        }
            threads.emplace_back(std::bind(&AGVSystem::movingThread, this));
            break;
        case State::TEST:
        {
            std::lock_guard<std::mutex> stateLock(stateMutex);
            currentState = State::FINISHED;
        }
            threads.emplace_back(std::bind(&AGVSystem::testThread, this));
            break;
        default:
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    for (auto &t : threads)
    {
        if (t.joinable())
        {
            t.join();
        }
    }
    std::cout << "System finished." << std::endl;
}

void AGVSystem::initThread()
{
    targetPoints = readMapPoinits(config["mapFilePath"]);
    translationVector << -stoi(config["imageLength"]) / 2 * stod(config["scale"]),
        -stoi(config["imageWidth"]) / 2 * stod(config["scale"]);
    cout << "init translationVector = " << translationVector << endl;
    {
        std::lock_guard<std::mutex> stateLock(stateMutex);
        if (currentState != State::FINISHED)
            currentState = State::CAPUTRE_IMAGE;
    }
    namedWindow("final match result", WINDOW_AUTOSIZE);
    agv.setPermission();

    return;
}

void AGVSystem::finishThread()
{
    networkCamera.close();
    return;
}

void AGVSystem::captureImageThread()
{
    {
        std::lock_guard<std::mutex> stateLock(stateMutex);
        currentState = State::CAPUTRE_IMAGE_RUNNING;
    }
    if (!cap.open(cameraUrl))
    {
        {
            std::lock_guard<std::mutex> stateLock(stateMutex);
            currentState = State::FINISHED;
        }
        return;
    }
    {
        std::lock_guard<std::mutex> stateLock(stateMutex);
        currentState = State::CAPUTRE_IMAGE;
    }
    cv::Mat frame;
    while (true)
    {
        if (!cap.read(frame))
        {
            cout << "get image failed" << endl;
            continue;
        }
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
    }
    return;
}

void AGVSystem::featureDetectThread()
{
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

void AGVSystem::matchingThread()
{
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

void AGVSystem::mappingThread()
{
    PNPAlgorithm pnpAlg(sourcePoints, targetPoints, correspondences);
    pnpAlg.setSolveParams(stoi(config["iterationsCount"]), stof(config["reprojectionError"]),
                          stod(config["confidence"]));
    bool pnpFlag = pnpAlg.estimateCameraPose();
    if (pnpFlag)
    {
        cv::Mat pose_inv = pnpAlg.getPose();
        cv::Mat camera_position = pose_inv(cv::Rect(3, 0, 1, 3));
        cout << "----current rotationMatrix----" << endl;
        cout << rotationMatrix << endl;
        cout << "----current translationMatrix----" << endl;
        cout << translationVector << endl;
        Eigen::Matrix<double, 3, 1> transform = rotationMatrix * basic_t;
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
            out << currentPose.first << " " << currentPose.second << endl;
            Eigen::AngleAxisd angleAxis(rotationMatrix);
            double angle = angleAxis.angle();
            double angle_degrees = angle * 180.0 / M_PI;
            _t_out << translationVector(0, 0) << "," << translationVector(1, 0) << endl;
            _angle_out << angle_degrees << endl;

            {
                std::lock_guard<std::mutex> stateLock(stateMutex);
                if (currentState != State::FINISHED)
                    currentState = State::MOVING;
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

void AGVSystem::movingThread()
{
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

void AGVSystem::testThread()
{
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

    ImageProcessor processor(filename);
    processor.readImage();
    processor.convertToGray();
    processor.convertGrayToBinary();
    processor.findContours();
    processor.detectCircles();
    processor.saveResults();

    vector<pair<double, double>> sourcePoints;

    sourcePoints = readNPoints("../data/circles.txt");

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

    PNPAlgorithm pnpAlg(sourcePoints, targetPoints, correspondences);

    pnpAlg.estimateCameraPose();

    cv::Mat pose_inv = pnpAlg.getPose();
    rotationMatrix = getRotationMatrix(pose_inv);
    translationVector = getTranslationVector(pose_inv);

    double theta = std::atan2(rotationMatrix(1, 0), rotationMatrix(0, 0));

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
