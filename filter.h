//
// Created by Chu-Hsuan Lin on 2022/2/7.
//
#include <opencv2/opencv.hpp>

#ifndef CONTENT_BASED_IMAGE_RETRIEVAL_FILTER_H
#define CONTENT_BASED_IMAGE_RETRIEVAL_FILTER_H

int sobel_filter(cv::Mat &src, cv::Mat &mag, cv::Mat &angle, int size);
int gabor_filter(cv::Mat &src, std::vector<cv::Mat> &dst, int kernel_size, bool gabor_show);
cv::Mat law_kernel(int type);
int law_filter(cv::Mat &src, cv::Mat &dst, int kernel_type);
#endif //CONTENT_BASED_IMAGE_RETRIEVAL_FILTER_H
