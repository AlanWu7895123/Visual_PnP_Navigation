#include "utils.h"

vector<pair<double, double>> read37Points(string filename)
{
    cout << "read 37 points" << endl;
    vector<pair<double, double>> points37;
    ifstream file(filename);

    if (!file.is_open())
    {
        cerr << "Error: Unable to open the file." << endl;
        return points37;
    }

    string line;
    while (getline(file, line))
    {
        istringstream iss(line);
        vector<double> numbers;
        double number;
        while (iss >> number)
        {
            numbers.push_back(number);
        }

        if (numbers.size() != 3)
        {
            cerr << "Error: Invalid number of values in line." << endl;
            continue;
        }

        // cout << "Read numbers: " << numbers[0] << ", " << numbers[1] << ", " << numbers[2] << endl;
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
    ifstream file(filename);

    // 检查文件是否成功打开
    if (!file.is_open())
    {
        cerr << "Error: Unable to open the file." << endl;
        return pointsN;
    }

    string line;
    int i = 1;
    while (getline(file, line))
    {
        istringstream iss(line);
        vector<double> numbers;
        double number;
        while (iss >> number)
        {
            numbers.push_back(number);
        }

        if (numbers.size() != 2)
        {
            cerr << "Error: Invalid number of values in line." << endl;
            continue;
        }

        // cout << "Read numbers: " << i << ", " << numbers[0] << ", " << numbers[1] << endl;
        i++;
        pair<double, double> tmp = {numbers[0], numbers[1]};
        pointsN.push_back(tmp);
    }

    // 关闭文件
    file.close();
    return pointsN;
}

int kbhit()
{
    struct termios oldt, newt;
    int ch;
    int oldf;

    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

    ch = getchar();

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);

    if (ch != EOF)
    {
        ungetc(ch, stdin);
        return 1;
    }

    return 0;
}

pair<double, double> weightedMovingAverageFilter(const deque<pair<double, double>> &window, const vector<double> &weights)
{
    double sumWeights = 0;
    double weightedSumX = 0;
    double weightedSumY = 0;

    for (size_t i = 0; i < window.size(); ++i)
    {
        weightedSumX += window[i].first * weights[i];
        weightedSumY += window[i].second * weights[i];
        sumWeights += weights[i];
    }

    return {weightedSumX / sumWeights, weightedSumY / sumWeights};
}

ostream &operator<<(ostream &os, State state)
{
    switch (state)
    {
    case State::INIT:
        os << "INIT";
        break;
    case State::TEST:
        os << "TEST";
        break;
    case State::TEST_RUNNING:
        os << "TEST_RUNNING";
        break;
    case State::CAPUTRE_IMAGE:
        os << "CAPUTRE_IMAGE";
        break;
    case State::FEATURE_DETECT:
        os << "FEATURE_DETECT";
        break;
    case State::MATCHING:
        os << "MATCHING";
        break;
    case State::MAPPING:
        os << "MAPPING";
        break;
    case State::CAPUTRE_IMAGE_RUNNING:
        os << "CAPUTRE_IMAGE_RUNNING";
        break;
    case State::FEATURE_DETECT_RUNNING:
        os << "FEATURE_DETECT_RUNNING";
        break;
    case State::MATCHING_RUNNING:
        os << "MATCHING_RUNNING";
        break;
    case State::MAPPING_RUNNING:
        os << "MAPPING_RUNNING";
        break;
    case State::FINISHED:
        os << "FINISHED";
        break;
    default:
        os << "UNKNOWN";
        break;
    }
    return os;
}

Matrix3d getRotationMatrix(Mat pose_inv)
{
    Matrix3d rotationMatrix = Matrix3d::Identity();
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            rotationMatrix(i, j) = pose_inv.at<double>(i, j);
        }
    }
    return rotationMatrix;
}

Vector2d getTranslationVector(Mat pose_inv)
{
    Vector2d translationVector = Vector2d::Zero();
    for (int i = 0; i < 2; ++i)
    {
        translationVector(i) = pose_inv.at<double>(i, 3);
    }
    return translationVector;
}

double distance(const pair<double, double> &p1, const pair<double, double> &p2)
{
    return sqrt(pow(p1.first - p2.first, 2) + pow(p1.second - p2.second, 2));
}

void calTransformed(const vector<pair<double, double>> &sourceCloud,
                    const vector<pair<double, double>> &targetCloud,
                    const vector<int> &correspondences,
                    Matrix3d &rotationMatrix,
                    Vector2d &translationVector)
{
    MatrixXd srcMat(2, correspondences.size());
    MatrixXd tgtMat(2, correspondences.size());

    for (size_t i = 0; i < correspondences.size(); ++i)
    {
        srcMat.col(i) << sourceCloud[i].first, sourceCloud[i].second;
        tgtMat.col(i) << targetCloud[correspondences[i]].first, targetCloud[correspondences[i]].second;
    }

    Vector2d srcCentroid = srcMat.rowwise().mean();
    Vector2d tgtCentroid = tgtMat.rowwise().mean();

    MatrixXd srcCentered = srcMat.colwise() - srcCentroid;
    MatrixXd tgtCentered = tgtMat.colwise() - tgtCentroid;

    MatrixXd covariance = srcCentered * tgtCentered.transpose();

    JacobiSVD<MatrixXd> svd(covariance, ComputeFullU | ComputeFullV);
    MatrixXd U = svd.matrixU();
    MatrixXd V = svd.matrixV();

    MatrixXd R = V * U.transpose();

    Vector2d t = tgtCentroid - R * srcCentroid;

    rotationMatrix.setIdentity();
    rotationMatrix.block<2, 2>(0, 0) = R;

    translationVector = t;
}