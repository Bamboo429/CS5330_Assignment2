//
// Created by Chu-Hsuan Lin on 2022/2/7.
//

//The file is all filter using in the project. Including sobel filter, gabor filter, laws filter

#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;

int sobel_filter(cv::Mat &src, cv::Mat &mag, cv::Mat &angle, int size){

    cv::Mat grayImage;
    cv::Mat sx, sy, abs_sobel_x, abs_sobel_y;

    if (src.channels()>1){
        cv::cvtColor(src, grayImage, cv::COLOR_BGR2GRAY);
    }
    else{
        src.copyTo(grayImage);
    }

    cv::Sobel(grayImage, sx, CV_64F, 1, 0, size);
    cv::Sobel(grayImage, sy, CV_64F, 0, 1, size);

    cv::Mat output_x,output_y,output;
    output_x.create(sx.size(),CV_16UC1);
    output_y.create(sx.size(),CV_16UC1);

    //magnitude of sobel image
    //x^2 and y^2
    output_x = sx.mul(sx);
    output_x.convertTo(output_x,CV_32FC1);
    output_y = sy.mul(sy);
    output_y.convertTo(output_y,CV_32FC1);

    //sqrt(x^2+y^2)
    sqrt(output_x+output_y,output);
    output.convertTo(mag,CV_16UC1);

    //phase of sobel image
    cv::phase(sx,sy, angle,true);
    angle.convertTo(angle,CV_16UC1);

    return 0;
}

int gabor_filter(cv::Mat &src, std::vector<cv::Mat> &dst, int kernel_size, bool gabor_show){

    // setting for gabor filter
    double sig = 4, th = 4, lm = 5, gm = 0.5, ps = 0;

    dst.clear();

    // gain different gabor filters kernel
    for (int t=1; t<=th ; t++){
        for (int l=5; l<=15; l+=lm) {
            cv::Mat gaborKernel,gaborSrc;
            gaborKernel = getGaborKernel(cv::Size(kernel_size, kernel_size), sig, CV_PI / t, l, gm, ps, CV_32F);
            filter2D(src, gaborSrc, src.depth(), gaborKernel);
            //imshow(std::to_string(t)+ std::to_string(l), gaborSrc);
            dst.push_back(gaborSrc);

            // show gabor kernel
            if(gabor_show==1){
                imshow(to_string(t)+ to_string(l), gaborKernel);
            }
        }
    }

    return 0;
}


cv::Mat law_kernel(int type){

    // set kernel vector for each element
    float l[5] = {1,4,6,4,1};
    cv::Mat L5_1 = cv::Mat(5, 1, CV_32FC1, l);
    cv::Mat L5_2 = cv::Mat(1, 5, CV_32FC1, l);

    float e[5] = {-1,-2,0,2,1};
    cv::Mat E5_1 = cv::Mat(5, 1, CV_32FC1, e);
    cv::Mat E5_2 = cv::Mat(1, 5, CV_32FC1, e);

    float s[5] = {-1,0,2,0,-1};
    cv::Mat S5_1 = cv::Mat(5, 1, CV_32FC1, s);
    cv::Mat S5_2 = cv::Mat(1, 5, CV_32FC1, s);

    float w[5] = {-1,2,0,-2,1};
    cv::Mat W5_1 = cv::Mat(5, 1, CV_32FC1, w);
    cv::Mat W5_2 = cv::Mat(1, 5, CV_32FC1, w);

    float r[5] = {1,-4,6,-4,1};
    cv::Mat R5_1 = cv::Mat(5, 1, CV_32FC1, r);
    cv::Mat R5_2 = cv::Mat(1, 5, CV_32FC1, r);

    // compute the kernel according the input type
    cv::Mat kernel;
    switch (type){
        case 1 :
            kernel = L5_1 * E5_2;
            break;
        case 2:
            kernel = L5_1 * S5_2;
            break;
        case 3:
            kernel = L5_1 * W5_2;
            break;
        case 4:
            kernel = L5_1 * R5_2;
            break;
        case 5:
            kernel = L5_1 * L5_2;
            break;
        case 6:
            kernel = E5_1 * E5_2;
            break;
        case 7:
            kernel = E5_1 * S5_2;
            break;
        case 8:
            kernel = E5_1 * W5_2;
            break;
        case 9:
            kernel = E5_1 * R5_2;
            break;
        case 10:
            kernel = S5_1 * W5_2;
            break;
        case 11:
            kernel = S5_1 * R5_2;
            break;
        case 12:
            kernel = S5_1 * S5_2;
            break;
        case 13:
            kernel = W5_1 * W5_2;
            break;
        case 14:
            kernel = W5_1 * R5_2;
            break;
        case 15:
            kernel = R5_1 * R5_2;
            break;
    }

    return kernel;
}

int law_filter(cv::Mat &src, cv::Mat &dst, int kernel_type){

    // cal law filter
    cv::Mat kernel = law_kernel(kernel_type);
    filter2D(src,dst, -1, kernel);

    //imshow("dst",dst);
    //cv::waitKey(0);

    return 0;

}