#include "utils.h"

vector<pair<double, double>> readMapPoinits(string filename)
{
    cout << "read 37 points" << endl;
    vector<pair<double, double>> targetPoints;
    ifstream file(filename);

    if (!file.is_open())
    {
        cerr << "Error: Unable to open the file." << endl;
        return targetPoints;
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
        targetPoints.push_back(tmp);
    }

    file.close();
    return targetPoints;
}

vector<pair<double, double>> readNPoints(string filename)
{
    cout << "read N points" << endl;
    vector<pair<double, double>> sourcePoints;
    ifstream file(filename);

    // 检查文件是否成功打开
    if (!file.is_open())
    {
        cerr << "Error: Unable to open the file." << endl;
        return sourcePoints;
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
        sourcePoints.push_back(tmp);
    }
    cout << "close file success" << endl;

    // 关闭文件
    file.close();
    return sourcePoints;
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

std::vector<unsigned char> hexArrayToBytes(const char *hexArray)
{
    std::vector<unsigned char> bytes;

    size_t len = strlen(hexArray);
    if (len % 2 != 0)
    {
        throw std::invalid_argument("Hex array length must be even");
    }

    for (size_t i = 0; i < len; i += 2)
    {
        unsigned char byte = 0;
        if (hexArray[i] >= '0' && hexArray[i] <= '9')
        {
            byte = (hexArray[i] - '0') << 4;
        }
        else if (hexArray[i] >= 'A' && hexArray[i] <= 'F')
        {
            byte = (hexArray[i] - 'A' + 10) << 4;
        }
        else if (hexArray[i] >= 'a' && hexArray[i] <= 'f')
        {
            byte = (hexArray[i] - 'a' + 10) << 4;
        }
        else
        {
            throw std::invalid_argument("Invalid hex character");
        }

        if (hexArray[i + 1] >= '0' && hexArray[i + 1] <= '9')
        {
            byte |= (hexArray[i + 1] - '0');
        }
        else if (hexArray[i + 1] >= 'A' && hexArray[i + 1] <= 'F')
        {
            byte |= (hexArray[i + 1] - 'A' + 10);
        }
        else if (hexArray[i + 1] >= 'a' && hexArray[i + 1] <= 'f')
        {
            byte |= (hexArray[i + 1] - 'a' + 10);
        }
        else
        {
            throw std::invalid_argument("Invalid hex character");
        }

        bytes.push_back(byte);
    }

    return bytes;
}

std::string bytesToHex(const char *data, size_t len)
{
    std::ostringstream oss;
    for (size_t i = 0; i < len; ++i)
    {
        oss << std::hex << std::setw(2) << std::setfill('0') << (static_cast<int>(data[i]) & 0xff);
    }
    return oss.str();
}

ProtocolHeader deSerializeProtocolHeader(const char *data)
{
    ProtocolHeader header;
    size_t offset = 0;

    // Extract m_sync (1 byte)
    header.m_sync = static_cast<uint8_t>(data[offset]);
    offset += 1;

    // Extract m_version (1 byte)
    header.m_version = static_cast<uint8_t>(data[offset]);
    offset += 1;

    // Extract and convert m_number (2 bytes, big-endian to host-endian)
    uint16_t number_be;
    memcpy(&number_be, data + offset, sizeof(number_be));
    header.m_number = ntohs(number_be);
    offset += sizeof(number_be);

    // Extract and convert m_length (4 bytes, big-endian to host-endian)
    uint32_t length_be;
    memcpy(&length_be, data + offset, sizeof(length_be));
    header.m_length = ntohl(length_be);
    offset += sizeof(length_be);

    // Extract and convert m_type (2 bytes, big-endian to host-endian)
    uint16_t type_be;
    memcpy(&type_be, data + offset, sizeof(type_be));
    header.m_type = ntohs(type_be);
    offset += sizeof(type_be);

    // Extract m_reserved (6 bytes)
    memcpy(header.m_reserved, data + offset, sizeof(header.m_reserved));
    offset += sizeof(header.m_reserved);

    return header;
}

std::string serializeProtocolHeader(const ProtocolHeader &header)
{
    std::string serialized;

    // Add m_sync and m_version (1 byte each, no conversion needed)
    serialized.append(1, header.m_sync);
    serialized.append(1, header.m_version);

    // Convert m_number to big-endian and add to serialized string
    uint16_t number_be = htons(header.m_number);
    serialized.append(reinterpret_cast<const char *>(&number_be), sizeof(number_be));

    // Convert m_length to big-endian and add to serialized string
    uint32_t length_be = htonl(header.m_length);
    serialized.append(reinterpret_cast<const char *>(&length_be), sizeof(length_be));

    // Convert m_type to big-endian and add to serialized string
    uint16_t type_be = htons(header.m_type);
    serialized.append(reinterpret_cast<const char *>(&type_be), sizeof(type_be));

    // Add m_reserved (10 bytes, no conversion needed)
    serialized.append(reinterpret_cast<const char *>(header.m_reserved), sizeof(header.m_reserved));

    return serialized;
}

char *protocolHeaderToHexChar(const ProtocolHeader &header)
{
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    ss << std::setw(2) << static_cast<int>(header.m_sync);
    ss << std::setw(2) << static_cast<int>(header.m_version);
    ss << std::setw(2) << static_cast<int>((header.m_number >> 8) & 0xFF);
    ss << std::setw(2) << static_cast<int>(header.m_number & 0xFF);
    ss << std::setw(2) << static_cast<int>((header.m_length >> 24) & 0xFF);
    ss << std::setw(2) << static_cast<int>((header.m_length >> 16) & 0xFF);
    ss << std::setw(2) << static_cast<int>((header.m_length >> 8) & 0xFF);
    ss << std::setw(2) << static_cast<int>(header.m_length & 0xFF);
    ss << std::setw(2) << static_cast<int>((header.m_type >> 8) & 0xFF);
    ss << std::setw(2) << static_cast<int>(header.m_type & 0xFF);
    for (size_t i = 0; i < sizeof(header.m_reserved); ++i)
    {
        ss << std::setw(2) << static_cast<int>(header.m_reserved[i]);
    }
    std::string hexStr = ss.str();
    char *hexChar = new char[hexStr.length() + 1];
    std::strcpy(hexChar, hexStr.c_str());
    return hexChar;
}

char *stringToHex(const std::string &input)
{
    static const char *const lut = "0123456789ABCDEF";
    size_t len = input.length();
    char *output = new char[2 * len + 1];
    for (size_t i = 0; i < len; ++i)
    {
        const unsigned char c = input[i];
        output[2 * i] = lut[c >> 4];
        output[2 * i + 1] = lut[c & 15];
    }
    output[2 * len] = '\0';
    return output;
}

char *concatStrings(const char *str1, const char *str2)
{
    size_t len1 = strlen(str1);
    size_t len2 = strlen(str2);
    char *result = new char[len1 + len2 + 1];
    strcpy(result, str1);
    strcat(result, str2);
    return result;
}

std::string trim(const std::string &str)
{
    size_t first = str.find_first_not_of(' ');
    size_t last = str.find_last_not_of(' ');
    return (first == std::string::npos || last == std::string::npos) ? "" : str.substr(first, last - first + 1);
}

std::unordered_map<std::string, std::string> readConfig(const std::string &filename)
{
    std::unordered_map<std::string, std::string> config;
    std::ifstream file(filename);

    if (!file.is_open())
    {
        std::cerr << "Unable to open config file: " << filename << std::endl;
        return config;
    }

    std::string line;
    while (std::getline(file, line))
    {
        // 忽略注释行和空行
        if (line.empty() || line[0] == '#')
        {
            continue;
        }

        size_t delimiterPos = line.find('=');
        if (delimiterPos != std::string::npos)
        {
            std::string key = trim(line.substr(0, delimiterPos));
            std::string value = trim(line.substr(delimiterPos + 1));
            config[key] = value;
        }
    }

    file.close();
    return config;
}

bool readCameraParameters(const std::string &filename, cv::Mat &cameraMatrix, cv::Mat &distCoeffs)
{
    // 打开配置文件
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cerr << "Failed to open " << filename << std::endl;
        return false;
    }

    // 读取内参和畸变参数
    fs["camera_matrix"] >> cameraMatrix;
    fs["dist_coeffs"] >> distCoeffs;

    // 关闭文件
    fs.release();

    return true;
}

double angleToYAxis(double x, double y)
{
    return std::atan2(y, x) * 180.0 / M_PI;
}

double calculateAngle(pair<double, double> x, pair<double, double> y, pair<double, double> z)
{
    // 向量XY
    double xy_x = y.first - x.first;
    double xy_y = y.second - x.second;

    // 向量XZ
    double xz_x = z.first - x.first;
    double xz_y = z.second - x.second;

    // 点积
    double dotProduct = xy_x * xz_x + xy_y * xz_y;

    // 向量XY和XZ的模长
    double xy_magnitude = std::sqrt(xy_x * xy_x + xy_y * xy_y);
    double xz_magnitude = std::sqrt(xz_x * xz_x + xz_y * xz_y);

    // 计算夹角的cos值
    double cosTheta = dotProduct / (xy_magnitude * xz_magnitude);

    // 使用反余弦函数计算角度
    double angle = std::acos(cosTheta);

    // 计算向量的叉积来判断旋转方向
    double crossProduct = xy_x * xz_y - xy_y * xz_x;

    // 如果叉积为负，说明角度为负
    if (crossProduct < 0)
    {
        angle = -angle;
    }

    return angle;
}