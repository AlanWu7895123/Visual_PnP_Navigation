#include "image_processor.h"

using namespace cv;
using namespace std;

ImageProcessor::ImageProcessor(const std::string &filename) : filename(filename) {}

ImageProcessor::ImageProcessor(const cv::Mat &image) : image(image) {}

bool ImageProcessor::readImage()
{
    image = imread("../data/" + filename + ".png");
    // image = imread("../data/" + filename + ".jpg");
    if (image.empty())
    {
        cerr << "Error: Could not read the image." << endl;
        return false;
    }
    return true;
}

void ImageProcessor::convertToGray()
{
    cvtColor(image, gray_image, COLOR_BGR2GRAY);
}

void ImageProcessor::applyThreshold()
{
    threshold(gray_image, binary_image, 140, 255, THRESH_BINARY);
}

void ImageProcessor::findContours()
{
    cv::findContours(binary_image, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
    result = image.clone();
    drawContours(result, contours, -1, Scalar(0, 255, 0), 2);
}

void ImageProcessor::detectCircles()
{
    // cv::Mat bw_image = binary_image;
    // double rate = 430 / 150;
    cv::Mat bw_image;
    cvtColor(result, bw_image, COLOR_BGR2GRAY);
    HoughCircles(bw_image, circles, HOUGH_GRADIENT, 3, bw_image.rows / 12, 200, 100, 40, 50);

    cvtColor(bw_image, output, COLOR_GRAY2BGR);

    ofstream out("../data/circles.txt", ios::out);
    // ofstream _out("../data/circles0.txt", ios::out);
    for (size_t i = 0; i < circles.size(); i++)
    {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        out << center.x << " " << center.y << endl;
        // _out << center.x * 430 / 150 << " " << center.y * 430 / 150 << endl;
        // out << (center.x - image.cols / 2) * 430 / 150 << " " << -(center.y - image.rows / 2) * 430 / 150 << endl;
        int radius = cvRound(circles[i][2]);

        circle(output, center, 3, Scalar(0, 255, 0), -1, LINE_AA);
        // string center_text = "No." + to_string(i) + " " + to_string((center.x - image.cols / 2) * 430 / 150) + "," + to_string(-(center.y - image.rows / 2) * 430 / 150);
        string center_text = "No." + to_string(i) + " " + to_string(center.x) + "," + to_string(center.y);
        Point text_position(center.x - 55, center.y - 55);
        putText(output, center_text, text_position, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 1, LINE_AA);
        circle(output, center, radius, Scalar(0, 0, 255), 3, LINE_AA);
    }
}

void ImageProcessor::saveResults()
{
    // imwrite("../data/" + filename + "_contours_result.jpg", result);
    imwrite("../data/" + filename + "_hough_result.jpg", output);
}
