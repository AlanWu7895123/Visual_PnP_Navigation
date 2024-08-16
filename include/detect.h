#ifndef DETECT_H
#define DETECT_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

class ImageProcessor
{
public:
    // Constructor
    ImageProcessor();
    ImageProcessor(const std::string &filename);
    ImageProcessor(const cv::Mat &image);

    // Methods
    bool readImage();
    void convertToGray();
    void convertGrayToBinary();
    void findContours();
    void detectCircles();
    void saveResults();

    cv::Mat getDetectImg();
    cv::Mat getContoursImg();

private:
    std::string filename;
    cv::Mat image;
    cv::Mat grayImg;
    cv::Mat binaryImg;

    cv::Mat contoursImg;
    cv::Mat detectImg;

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec3f> circles;
};

#endif // DETECT_H
