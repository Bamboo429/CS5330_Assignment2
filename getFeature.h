//
// Created by Chu-Hsuan Lin on 2022/2/5.
//

#ifndef CONTENT_BASED_IMAGE_RETRIEVAL_GETFEATURE_H
#define CONTENT_BASED_IMAGE_RETRIEVAL_GETFEATURE_H

using namespace std;
using namespace cv;

struct parameter{
    int hist_bin;
    int hist2d_bin;
    int hist2d_type;
    int hist3d_bin;
    int sobel_size;
    int sobel_bin;

};
vector<float> baseline(cv::Mat &src);
cv::Mat histogram3d(cv::Mat &src, int bins, int* range_arr);
cv::Mat histogram2d(cv::Mat &src, int bins, int* range_arr);
vector<cv::Mat> histogram(cv::Mat &src, int bins, int* range_arr);
vector<float> mat2vector(cv::Mat &mat);
int getFeatures(cv::Mat &src, vector<float> &features, vector<char*> &feature_name, char* feature_type, vector<float> &weight);
int sobel_hist(cv::Mat &src, int sobel_size, int bins, cv::Mat &hist_mag, cv::Mat &hist_ang, cv::Mat &hist2d);
int bgr2chromaticity(cv::Mat &src,cv::Mat &dst);

#endif //CONTENT_BASED_IMAGE_RETRIEVAL_GETFEATURE_H
