#include "detect.h"

using namespace cv;
using namespace std;

ImageProcessor::ImageProcessor() {}

ImageProcessor::ImageProcessor(const std::string &filename) : filename(filename) {}

ImageProcessor::ImageProcessor(const cv::Mat &image) : image(image) {}

bool ImageProcessor::readImage()
{
    image = imread("../data/" + filename + ".png");
    if (image.empty())
    {
        image = imread("../data/" + filename + ".jpg");
        if (image.empty())
        {
            cerr << "Error: Could not read the image." << endl;
            return false;
        }
    }
    return true;
}

void ImageProcessor::convertToGray()
{
    cvtColor(image, grayImg, COLOR_BGR2GRAY);
}

void ImageProcessor::convertGrayToBinary()
{
    threshold(grayImg, binaryImg, stoi(config["binaryThreshold"]), 255, THRESH_BINARY);
}

void ImageProcessor::findContours()
{
    cv::findContours(binaryImg, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
    contoursImg = image.clone();
    drawContours(contoursImg, contours, -1, Scalar(0, 255, 0), 2);
}

void ImageProcessor::detectCircles()
{
    cv::Mat bw_image;
    cvtColor(contoursImg, bw_image, COLOR_BGR2GRAY);
    HoughCircles(bw_image, circles, HOUGH_GRADIENT, stoi(config["houghDp"]), stoi(config["houghMinDist"]),
                 stoi(config["houghParam1"]),
                 stoi(config["houghParam2"]), stoi(config["houghMinRadius"]), stoi(config["houghMaxRadius"]));

    cvtColor(bw_image, detectImg, COLOR_GRAY2BGR);

    ofstream out("../data/circles.txt", ios::out);
    for (size_t i = 0; i < circles.size(); i++)
    {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        out << center.x * stod(config["scale"]) << " " << center.y * stod(config["scale"]) << endl;
        int radius = cvRound(circles[i][2]);

        circle(detectImg, center, 3, Scalar(0, 255, 0), -1, LINE_AA);
        string center_text = "No." + to_string(i) + " " +
                             to_string(int((center.x - stoi(config["imageLength"]) / 2) * stod(config["scale"]))) + "," +
                             to_string(int((center.y - stoi(config["imageWidth"]) / 2) * stod(config["scale"])));
        Point text_position(center.x - 55, center.y - 55);
        putText(detectImg, center_text, text_position, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 1, LINE_AA);
        circle(detectImg, center, radius, Scalar(0, 0, 255), 3, LINE_AA);
    }
}

void ImageProcessor::saveResults()
{
    imwrite("../data/" + filename + "_binary_result.jpg", binaryImg);
    imwrite("../data/" + filename + "_contours_result.jpg", contoursImg);
    imwrite("../data/" + filename + "_hough_result.jpg", detectImg);
}

cv::Mat ImageProcessor::getContoursImg()
{
    return contoursImg;
}

cv::Mat ImageProcessor::getDetectImg()
{
    return detectImg;
}
