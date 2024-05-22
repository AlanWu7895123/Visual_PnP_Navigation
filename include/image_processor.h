#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

class ImageProcessor
{
public:
    // Constructor
    ImageProcessor(const std::string &filename);
    ImageProcessor(const cv::Mat &image);

    // Methods
    bool readImage();
    void convertToGray();
    void applyThreshold();
    void findContours();
    void detectCircles();
    void saveResults();

    cv::Mat output;

private:
    std::string filename;
    cv::Mat image;
    cv::Mat gray_image;
    cv::Mat binary_image;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec3f> circles;
    cv::Mat result;
};

#endif // IMAGE_PROCESSOR_H
