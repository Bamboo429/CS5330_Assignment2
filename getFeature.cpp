//
// Created by Chu-Hsuan Lin on 2022/2/5.
//

// This file is for collecting features. Including each feature function and combine them for project tasks

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "filter.h"
#include <numeric>
#include "getFeature.h"

using namespace cv;
using namespace std;

// find the center 9*9 pixel as feature vector
vector<float> baseline(cv::Mat &src){

    int channel = src.channels();    // number of channels
    int loc_c = src.cols/2;
    int loc_r = src.rows/2;
    vector<float> base;

    for(int i = loc_r-4; i< loc_r+5; i++){
        for (int j = loc_c-4; j < loc_c+5; j++){
            for (int k = 0; k < channel; k++) {
                int value = src.at<Vec3b>(i, j)[k];
                base.push_back(value);
            }
        }
    }
    return base;
}

// histogram for RGB 3D
cv::Mat histogram3d(cv::Mat &src, int bins, int* range_arr) {

    // define range through bins
    int r1 = range_arr[1]-range_arr[0];
    int r2 = range_arr[3]-range_arr[2];
    int r3 = range_arr[5]-range_arr[4];

    int channel = src.channels();    // number of channels
    int range1 = r1 / bins;
    int range2 = r2 / bins;
    int range3 = r3 / bins;

    int z_1,z_2,z_3;
    float N = src.rows*src.cols;

    //initial output cv::Mat
    int size[3] = { bins, bins, bins };
    cv::Mat hist3d(3, size, CV_32F, cv::Scalar(0));

    if (channel != 3){
        std::cout << " Error : Image channel less than 3" << endl;
        return hist3d;
    }

    // Calculate the histogram of the image
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {

            uchar val_1 = src.at<Vec3b>(i, j)[0];
            uchar val_2 = src.at<Vec3b>(i, j)[1];
            uchar val_3 = src.at<Vec3b>(i, j)[2];

            for (int z = 0; z < bins; z++) {
                //check both upper and lower limits
                if (val_1 >= z * range1 && val_1 < (z + 1) * range1)
                    z_1 = z;
                if (val_2 >= z * range2 && val_2 < (z + 1) * range2)
                    z_2 = z;
                if (val_3 >= z * range3 && val_3 < (z + 1) * range3)
                    z_3 = z;
            }
            //increment the index that contains the point
            hist3d.at<float>(z_1,z_2,z_3) +=1;
        }
    }

    //Normalize
    for (int i=0; i< hist3d.total();i++){
        hist3d.at<float>(i) = hist3d.at<float>(i)/N;
    }

    return hist3d;
}

cv::Mat histogram2d(cv::Mat &src, int bins, int *range_arr) {

    // define the range
    float r1 = range_arr[1]-range_arr[0];
    float r2 = range_arr[3]-range_arr[2];

    int channel = src.channels();    // number of channels
    float range1 = r1 / bins;
    float range2 = r2 / bins;

    int size[2] = { bins, bins};
    cv::Mat hist2d(2, size, CV_32F, cv::Scalar(0));

    int z_1,z_2;

    float N = src.rows * src.cols;
    if (channel != 2){
        std::cout << " Error : Image channel less than 2" << endl;
        exit(-1);
    }

    // Calculate the histogram of the image
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {

            uchar val_1 = src.at<Vec2b>(i, j)[0];
            uchar val_2 = src.at<Vec2b>(i, j)[1];

            for (int z = 0; z < bins; z++) {

                //check both upper and lower limits
                if (val_1 >= z * range1 && val_1 < (z + 1) * range1)
                    z_1 = z;
                if (val_2 >= z * range2 && val_2 < (z + 1) * range2)
                    z_2 = z;
            }
            //increment the index that contains the point
            hist2d.at<float>(z_1,z_2) +=1;
        }
    }

    for (int i=0; i< hist2d.total();i++){
        hist2d.at<float>(i) = hist2d.at<float>(i)/N;
    }

    return hist2d;
}

vector<cv::Mat> histogram(cv::Mat &src, int bins, int* range_arr) {

    int r = range_arr[1]-range_arr[0];
    int channel = src.channels();// number of channels
    int range = r / bins;

    vector<cv::Mat> hist(channel);  // initial histogram arrays

    float N = src.rows*src.cols;

    // Initalize histogram arrays
    for (int i = 0; i < hist.size(); i++)
        hist[i] = cv::Mat::zeros(1, bins, CV_32FC1);

    // Calculate the histogram of the image
    for (int i = 0; i < src.rows; i++){
        for (int j = 0; j < src.cols; j++){
            for (int k = 0; k < channel; k++){
                for (int z = 0; z < bins; z++) {
                    //check both upper and lower limits
                    uchar val = channel == 1 ? src.at<uchar>(i, j) : src.at<Vec3b>(i, j)[k];
                    if (val >= z * range && val < (z + 1) * range) {
                        hist[k].at<float>(z) += 1;
                    }
                }
            }
        }
    }

    //Normalize
    for (int c=0; c<channel; c++){
        for (int i=0; i< hist[c].total();i++){
            hist[c].at<float>(i) = hist[c].at<float>(i)/(N);
        }
    }

    return hist;
}


int sobel_hist(cv::Mat &src, int sobel_size, int bins, cv::Mat &hist_mag, cv::Mat &hist_ang, cv::Mat &hist2d){

    cv::Mat mag,angle,merge_sobel;

    // get image pass sobel filter
    sobel_filter(src,mag,angle,sobel_size);

    //combine together for histogram 2d
    cv::Mat channel[2] = {mag, angle};
    merge(channel, 2, merge_sobel);

    // cal histogram of mag and orientation
    int range_arr1[2] = {0,256};
    hist_mag = histogram(mag, 16, range_arr1)[0];
    int range_arr2[2] = {0,360};
    hist_ang = histogram(angle, 10,range_arr2)[0];

    // cal hist 2d
    int range_arr[4] = {0,256, 0,360};
    hist2d = histogram2d(merge_sobel,bins,range_arr);

    return 0;
}

int contrast(cv::Mat &src, int size, cv::Mat &dst){

    cv::Mat grayImage;

    int range = size/2; //define contrast range
    int n=0;

    if (src.channels()>1){
        cv::cvtColor(src, grayImage, cv::COLOR_BGR2GRAY);
    }
    else{
        src.copyTo(grayImage);
    }

    dst.create(src.size(), CV_16FC1);
    dst=0;

    for (int i=0; i<src.rows; i++){
        for(int j=0; j<src.cols; j++){
            n=0;
            for (int r=i-range; r <= i+range; r++){
                for(int c = j-range; c <= j+range; c++){
                    if (r < 0 or r > src.rows or c < 0 or c > src.cols){
                        dst.at<float>(i,j) += 0;
                    }
                    else {
                        n+=1;
                        dst.at<float>(i, j) += (src.at<uchar>(i, j) - src.at<uchar>(r, c)) ^ 2;
                    }
                }
            }
        }
    }

    //normalize
    for (int i=0; i< dst.total();i++){
        dst.data[i] = dst.data[i]/n;
    }
    return 0;
}

float entropy(vector<float> input){

    float etp=0;
    int size = input.size();

    for(int i=0; i<size; i++){
        etp += input[i]/size*log(input[i]/size);
    }
    return etp;
}

//https://stackoverflow.com/questions/10167534/how-to-find-out-what-type-of-a-mat-object-is-with-mattype-in-opencv
string type2str(int type) {
    string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
}

// Conver mat to vector for cal distance metric
vector<float> mat2vector(cv::Mat &mat){

    vector<float> array;
    string type = type2str(mat.type());

    if (mat.isContinuous()) {
        array.assign((float*)mat.data, (float*)mat.data + mat.total()*mat.channels());
    }
    else {
        for (int i = 0; i < mat.rows; ++i) {
            array.insert(array.end(), mat.ptr<float>(i), mat.ptr<float>(i)+mat.cols*mat.channels());
        }
    }
    return array;
}

int bgr2chromaticity(cv::Mat &src,cv::Mat &tmp){

    //cv::Mat tmp;
    tmp.create(src.size(),CV_32FC3);

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {

            // get each b,g,r value
            Vec3b intensity = src.at<Vec3b>(i, j);
            float b = intensity.val[0];
            float g = intensity.val[1];
            float r = intensity.val[2];

            // cal sum of b,g,r
            float rgbSum = (float) (r + g + b);
            if (rgbSum ==0){
                float redNorm = 0;
                float greenNorm = 0;
                float blueNorm = 0;
            }
            // cal rg chromaticity
            float redNorm = (float) (r / rgbSum);
            float greenNorm = (float) (g / rgbSum);
            float blueNorm = (float) (b / rgbSum);

            // back to scale 0-255
            tmp.at<Vec3f>(i, j)[0] = (blueNorm*255);
            tmp.at<Vec3f>(i, j)[1] = (greenNorm*255);
            tmp.at<Vec3f>(i, j)[2] = (redNorm*255);
        }
    }

    return 0;

}
int getFeatures(cv::Mat &src, vector<float> &features, vector<char*> &feature_name, char* feature_type, vector<float> &weight){

    // clear vectors
    features.clear();
    feature_name.clear();

    // gain feature baseline for task 1
    if( strcmp(feature_type, "baseline")==0 ) {

        vector<float> feature_baseline = baseline(src);

        // save feature and feature name for appending csv file
        feature_name.insert(feature_name.end(), feature_baseline.size(), "baseline");
        features.insert(features.end(), feature_baseline.begin(), feature_baseline.end());

        // set weight
        weight.push_back(1);

    }

    // gain feature histogram
    else if( strcmp(feature_type, "histogram") ==0 ) {

        vector<float> tmp, hist;
        int range_arr[2] = {0,256};
        vector<cv::Mat> hist_mat = histogram(src, 16, range_arr);
        for (int i = 0; i < hist_mat.size(); i++) {
            tmp = mat2vector(hist_mat[i]);
            hist.insert(hist.end(), tmp.begin(), tmp.end());
        }

        feature_name.insert(feature_name.end(), hist.size(), "histogram");
        features.insert(features.end(), hist.begin(), hist.end());

        weight.push_back(1);
    }

    // gain feature histogram 2d for task 2
    else if( strcmp(feature_type,"hist_2d")==0 ) {

        cv::Mat ch_src,gr_src;
        gr_src.create(src.size(), CV_8UC3);

        // convert to rg chromaticity
        bgr2chromaticity(src,ch_src);
        ch_src.convertTo(gr_src, CV_8UC3);

        cv::Mat bgr[3];
        split(gr_src,bgr);

        cv::Mat merge_src;
        cv::Mat channels[2] = {bgr[1], bgr[2]};
        merge(channels, 2, merge_src);

        // cal histogram 2d
        int range_arr[4] = {0,256,0,256};
        cv::Mat hist2d_mat = histogram2d(merge_src, 16, range_arr);

        // convert to vector
        vector<float> hist_2d = mat2vector(hist2d_mat);

        // save feature and feature name for appending csv file
        feature_name.insert(feature_name.end(), hist_2d.size(), "histogram2d");
        features.insert(features.end(), hist_2d.begin(), hist_2d.end());

        //set weight
        weight.push_back(1);

    }

    // gain feature 3d histogram
    else if( strcmp(feature_type, "hist_3d")==0 ){

        int range_arr[6] = {0,256,0,256,0,256};
        cv::Mat hist3d_mat = histogram3d(src, 8, range_arr);
        vector<float> hist_3d = mat2vector(hist3d_mat);

        feature_name.insert(feature_name.end(), hist_3d.size(), "histogram3d");
        features.insert(features.end(), hist_3d.begin(), hist_3d.end());

        weight.push_back(1);
    }

    // gain multiple histogram for task 3
    else if( strcmp(feature_type, "multi_hist") ==0 ){

        int range_arr[6] = {0,256,0,256,0,256};
        // set range of top and btn image
        cv::Rect r_top( 0, 0, src.cols, src.rows/2);
        cv::Rect r_btn( 0, src.rows/2, src.cols, src.rows/2);

        // copy top and btn img to new mat
        cv::Mat img_top = src(r_top).clone();
        cv::Mat img_btn = src(r_btn).clone();

        // cal 3d histogram
        cv::Mat hist_mat_top = histogram3d(img_top,8, range_arr);
        cv::Mat hist_mat_btn = histogram3d(img_btn,8, range_arr);

        vector<float> hist_top = mat2vector(hist_mat_top);
        vector<float> hist_btn = mat2vector(hist_mat_btn);

        // concate top and btn histogram
        vector<float> hist;
        hist.clear();
        hist.insert(hist.end(),hist_top.begin(),hist_top.end());
        hist.insert(hist.end(),hist_btn.begin(),hist_btn.end());

        feature_name.insert(feature_name.end(), hist.size(), "multi_hist");
        features.insert(features.end(), hist.begin(), hist.end());

        // set weight
        weight.push_back(1);

    }

    // gain texture_color feature for task 4
    else if( strcmp( feature_type, "texture_color") ==0 ){

        vector<cv::Mat> hist_sobel_mat;
        cv::Mat hist2d, sobel_mag, sobel_ang;
        vector<float> hist_sobel;
        hist_sobel.clear();

        // gain sobel histogram, including mag, orientation, and 2d
        sobel_hist(src, 3, 16, sobel_mag, sobel_ang, hist2d);

        vector<float> tmp = mat2vector(sobel_mag);
        hist_sobel.insert(hist_sobel.end(), tmp.begin(), tmp.end());

        vector<float> tmp_ang = mat2vector(sobel_ang);
        hist_sobel.insert(hist_sobel.end(), tmp_ang.begin(), tmp_ang.end());

        // not using this feature
        //vector<float> tmp_2d = mat2vector(hist2d);
        //hist_sobel.insert(hist_sobel.end(), tmp_2d.begin(), tmp_2d.end());

        feature_name.insert(feature_name.end(), hist_sobel.size(), "sobel_hist");
        features.insert(features.end(), hist_sobel.begin(), hist_sobel.end());

        // gain color feature
        int range_arr[6] = {0,256,0,256,0,256};
        cv::Mat hist_mat3d = histogram3d(src, 8, range_arr);
        vector<float> hist = mat2vector(hist_mat3d); //  histogram

        feature_name.insert(feature_name.end(), hist.size(), "hist3d");
        features.insert(features.end(), hist.begin(), hist.end());

        //set weight
        weight.push_back(0.5);
        weight.push_back(0.5);
    }

    // extension - blue can
    else if( strcmp(feature_type, "blue_can") == 0 ) {


        cv::Mat frame_HSV, frame_threshold;

        // conver to HSV and capture blue color
        cv::cvtColor(src, frame_HSV, COLOR_BGR2HSV);
        inRange(frame_HSV, Scalar(100, 150, 50), Scalar(140, 255, 255), frame_threshold);

        // cal contours
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        findContours(frame_threshold, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);


        cv::Mat submat;
        src.copyTo(submat);
        if (contours.size() > 0) {
            int idx = 0, largestComp = 0;
            double maxArea = 0;
            for (; idx >= 0; idx = hierarchy[idx][0]) {
                const vector<Point> &c = contours[idx];
                double area = fabs(contourArea(Mat(c)));
                if (area > maxArea) {
                    maxArea = area;
                    largestComp = idx;
                }
            }

            // draw rect bounding box
            //Scalar color(255,0,0);
            cv::Rect boundRect = boundingRect(contours[largestComp]);


            //set area threshold and substrast the area
            if (maxArea > 5000) {
                submat = src(boundRect).clone();
            }
        }

        int range_arr[6] = {0,256,0,256,0,256};
        cv::Mat  hist_mat = histogram3d(submat, 8, range_arr);
        vector<float> hist_blue = mat2vector(hist_mat);

        feature_name.insert(feature_name.end(), hist_blue.size(), "blue_hist");
        features.insert(features.end(), hist_blue.begin(), hist_blue.end());

        weight.push_back(1);

    }

    // gain feature for task 5
    else if( strcmp(feature_type, "custom_design") ==0 ){

        // img pass laws filter L5E5 and L5R5
        cv::Mat L5E5, L5R5;
        law_filter(src, L5E5, 1); //kernel type 1 for L5E5
        law_filter(src, L5R5, 4); //kernel type 4 for L5R5

        int range_arr[6] = {0,256, 0,256, 0,256};
        cv::Mat L5E5_hist_mat = histogram3d(L5E5, 8, range_arr );
        cv::Mat L5R5_hist_mat = histogram3d(L5R5, 8, range_arr );
        cv::Mat src_hist_mat = histogram3d(src, 8, range_arr );

        vector<float> L5E5_hist = mat2vector(L5E5_hist_mat);
        vector<float> L5R5_hist = mat2vector(L5R5_hist_mat);
        vector<float> src_hist = mat2vector(src_hist_mat);

        feature_name.insert(feature_name.end(), L5E5_hist.size(), "L5E5");
        features.insert(features.end(), L5E5_hist.begin(), L5E5_hist.end());

        feature_name.insert(feature_name.end(), L5R5_hist.size(), "L5R5");
        features.insert(features.end(), L5R5_hist.begin(), L5R5_hist.end());

        feature_name.insert(feature_name.end(), src_hist.size(), "RGB");
        features.insert(features.end(), src_hist.begin(), src_hist.end());


        //set weight
        weight.push_back(1);
        weight.push_back(1);
        weight.push_back(1);
    }

    // extension - gabor filter
    else if( strcmp(feature_type, "gabor_feature") ==0 ){

        // img to Gray
        cv::Mat frame_gray;
        cv::cvtColor(src, frame_gray, COLOR_BGR2GRAY);

        // gabor filter size 31, 8 different thetas
        vector<cv::Mat> garbor_vector;
        gabor_filter(frame_gray, garbor_vector, 31, 0);

        for(int i=0; i< garbor_vector.size(); i++){
            int range_arr[2] = {0,256};
            vector<cv::Mat> hist_gabor_mat = histogram(garbor_vector[i], 8, range_arr);
            for(int j=0; j< hist_gabor_mat.size(); j++){
                vector<float> hist_gabor = mat2vector(hist_gabor_mat[j]);
                feature_name.insert(feature_name.end(), hist_gabor.size(), "gabor_hist");
                features.insert(features.end(), hist_gabor.begin(), hist_gabor.end());
            }
        }

        /*
        cv::Mat E5R5;
        law_filter(frame_gray, E5R5, 9 );
        int range_arr[2] = {0,255};
        vector<cv::Mat> law_mat = histogram(E5R5, 16, range_arr);
        vector<float> law_hist = mat2vector(law_mat[0]);

        feature_name.insert(feature_name.end(), law_hist.size(), "law");
        features.insert(features.end(), law_hist.begin(), law_hist.end());
        */

        int range_arr[6] = {0,256, 0,256, 0,256};
        cv::Mat src_hist_mat = histogram3d(src, 8, range_arr );

        vector<float> src_hist = mat2vector(src_hist_mat);

        feature_name.insert(feature_name.end(), src_hist.size(), "RGB");
        features.insert(features.end(), src_hist.begin(), src_hist.end());


        weight.push_back(0.7);
        weight.push_back(0.3);

    }

    else if ( strcmp(feature_type, "banana") ==0 ){

        bool AREA = true;

        // convert to HSC scale and find yellow area
        cv::Mat frame_HSV, frame_threshold;
        cv::cvtColor(src, frame_HSV, COLOR_BGR2HSV);

        inRange(frame_HSV, Scalar(20, 50, 150), Scalar(30, 255, 255), frame_threshold);
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        findContours(frame_threshold, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

        cv::Mat submat;
        src.copyTo(submat);

        // find the biggest area in contours
        if (contours.size() > 0) {
            int idx = 0, largestComp = 0;
            double maxArea = 0;
            for (; idx >= 0; idx = hierarchy[idx][0]) {
                const vector<Point> &c = contours[idx];
                double area = fabs(contourArea(Mat(c)));
                if (area > maxArea) {
                    maxArea = area;
                    largestComp = idx;
                }
            }


            Scalar color(255,0,0);
            cv::Rect boundRect = boundingRect(contours[largestComp]);
            rectangle( src, boundRect.tl(), boundRect.br(), color, 2 );

            //imshow("r", src);
            //waitKey(0);
            // threshold for selecting analyze image
            // if area too small do not cal feature
            if (boundRect.area() > 10000) {
                submat = src(boundRect).clone();
            }
            else{
                AREA = false;
            }

        }

        cv::Mat graysub;
        cv::cvtColor(submat, graysub, COLOR_BGR2GRAY);
        cv::Mat L5E5_mat,E5R5_mat;
        law_filter(graysub, L5E5_mat, 1);
        law_filter(graysub, E5R5_mat, 9);

        int range_arr[2] = {0, 256};
        vector<cv::Mat> hist_mat1 = histogram(L5E5_mat, 16, range_arr);
        vector<cv::Mat> hist_mat2 = histogram(E5R5_mat, 16, range_arr);
        vector<float> hist_3d1 = mat2vector(hist_mat1[0]);
        vector<float> hist_3d2 = mat2vector(hist_mat2[0]);

        int range_arr2[6] = {0, 256, 0, 256, 0, 256};
        cv::Mat hist_mat = histogram3d(submat, 8, range_arr);
        vector<float> hist_3d = mat2vector(hist_mat);


        if (AREA == true) {

            feature_name.insert(feature_name.end(), hist_3d1.size(), "banana_l5e5");
            features.insert(features.end(), hist_3d1.begin(), hist_3d1.end());
            feature_name.insert(feature_name.end(), hist_3d2.size(), "banana_e5r5");
            features.insert(features.end(), hist_3d2.begin(), hist_3d2.end());
            feature_name.insert(feature_name.end(), hist_3d.size(), "banana_color");
            features.insert(features.end(), hist_3d.begin(), hist_3d.end());
        }

        //if area too small set feature differ from regular features
        else{

            feature_name.insert(feature_name.end(), hist_3d1.size(), "banana_l5e5");
            features.insert(features.end(), hist_3d1.size(), 100);
            feature_name.insert(feature_name.end(), hist_3d2.size(), "banana_e5r5");
            features.insert(features.end(), hist_3d2.size(), 100);
            feature_name.insert(feature_name.end(), hist_3d.size(), "banana_color");
            features.insert(features.end(), hist_3d.size(), 100);

        }


        // set weight
        weight.push_back(1);
        weight.push_back(1);
        weight.push_back(1);

        // test for banana
        /*
        cv::Mat frame_gray;
        cv::cvtColor(submat, frame_gray, COLOR_BGR2GRAY);

        vector<cv::Mat> garbor_vector;
        gabor_filter(frame_gray, garbor_vector, 31, 0);

        for(int i=0; i< garbor_vector.size(); i++){
            int range_arr[2] = {0,256};
            vector<cv::Mat> hist_gabor_mat = histogram(garbor_vector[i], 8, range_arr);
            for(int j=0; j< hist_gabor_mat.size(); j++){
                vector<float> hist_gabor = mat2vector(hist_gabor_mat[j]);
                feature_name.insert(feature_name.end(), hist_gabor.size(), "gabor_hist");
                features.insert(features.end(), hist_gabor.begin(), hist_gabor.end());
            }
        }
        */
        /*
        cv::Mat frame_gray,law_filtered;
        cv::cvtColor(submat, frame_gray, COLOR_BGR2GRAY);

        law_filter(frame_gray, law_filtered, 1);
        int range_arr[2] = {0, 256};
        vector<cv::Mat> hist_mat = histogram(law_filtered, 16, range_arr);

        for(int i=0; i< hist_mat.size(); i++) {
            vector<float> hist_banana = mat2vector(hist_mat[i]);
            feature_name.insert(feature_name.end(), hist_banana.size(), "banana_texture");
            features.insert(features.end(), hist_banana.begin(), hist_banana.end());
        }
        */

        /*
        cv::Mat frame_gray,law_filtered;
        cv::cvtColor(submat, frame_gray, COLOR_BGR2GRAY);

        vector<cv::Mat> hist_sobel_mat;
        cv::Mat hist2d, sobel_mag, sobel_ang;
        vector<float> hist_sobel;
        hist_sobel.clear();

        sobel_hist(frame_gray, 3, 16, sobel_mag, sobel_ang, hist2d);
        //sobel_hist(src, para.sobel_size, para.sobel_bin, hist_sobel_mat, hist2d);

        vector<float> tmp = mat2vector(sobel_mag);
        hist_sobel.insert(hist_sobel.end(), tmp.begin(), tmp.end());

        //std::cout << 3 << endl;
        vector<float> tmp_ang = mat2vector(sobel_ang);
        hist_sobel.insert(hist_sobel.end(), tmp_ang.begin(), tmp_ang.end());

        //std::cout << 4 << endl;
        vector<float> tmp_2d = mat2vector(hist2d);
        hist_sobel.insert(hist_sobel.end(), tmp_2d.begin(), tmp_2d.end());

        feature_name.insert(feature_name.end(), hist_sobel.size(), "banana_sobel");
        features.insert(features.end(), hist_sobel.begin(), hist_sobel.end());


        //std::cout << 1 << endl;
        cv::Mat ch_src,gr_src;
        bgr2chromaticity(submat,ch_src);
        ch_src.convertTo(gr_src, CV_8UC3);
        //std::cout << 2 << endl;
        cv::Mat bgr[3];
        split(gr_src,bgr);
        //std::cout << 3 << endl;

        cv::Mat merge_src;
        cv::Mat channels[2] = {bgr[1], bgr[2]};
        merge(channels, 2, merge_src);

        int range_arr2[4] = {0,256,0,256};
        cv::Mat hist2d_mat = histogram2d(merge_src, 8, range_arr2);
        //std::cout << 4 << endl;
        vector<float> hist_rg = mat2vector(hist2d_mat);

        feature_name.insert(feature_name.end(), hist_rg.size(), "banana_color");
        features.insert(features.end(), hist_rg.begin(), hist_rg.end());

        */


    }

    // extension - green can
    else if (strcmp(feature_type, "green_can") ==0 ){

        //std::cout << "greem" << endl;
        cv::Mat ch_src,gr_src;
        //
        //cv::Mat frame_gray,law_filtered;
        //cv::cvtColor(src, frame_gray, COLOR_BGR2GRAY);

        cv::Mat L5E5_mat;
        law_filter(src, L5E5_mat, 1);

        //imshow("L5",L5E5_mat);
        //waitKey(0);

        //std::cout << L5E5_mat.channels() << endl;
        //gr_src.create(src.size(), CV_8UC3);

        // rg chromaticity
        bgr2chromaticity(L5E5_mat,ch_src);
        ch_src.convertTo(gr_src, CV_8UC3);

        cv::Mat bgr[3];
        split(gr_src,bgr);

        cv::Mat merge_src;
        cv::Mat channels[2] = {bgr[1], bgr[2]};
        merge(channels, 2, merge_src);

        // hist 2d for rg chromaticity
        int range_arr[4] = {0,256,0,256};
        cv::Mat hist2d_mat = histogram2d(merge_src, 16, range_arr);
        vector<float> hist_2d = mat2vector(hist2d_mat);

        feature_name.insert(feature_name.end(), hist_2d.size(), "L5E5_rg");
        features.insert(features.end(), hist_2d.begin(), hist_2d.end());

        int range_arr2[6] = {0,256,0,256,0,256};
        cv::Mat hist3d_mat = histogram3d(src,8, range_arr2);
        vector<float> hist3d = mat2vector(hist3d_mat);

        feature_name.insert(feature_name.end(), hist3d.size(), "hist3d");
        features.insert(features.end(), hist3d.begin(), hist3d.end());

        // set weight
        weight.push_back(0.7);
        weight.push_back(0.3);


    }

    else{
        std::cout << "Error feature type, please input again!" << endl;
    }
    return 0;


}





