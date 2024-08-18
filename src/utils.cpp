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