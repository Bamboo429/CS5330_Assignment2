//
// Created by Chu-Hsuan Lin on 2022/2/5.
//

#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <vector>

#include "getFeature.h"
#include "csv_util.h"
#include "matching.h"
#include "filter.h"

using namespace std;


int main456(int argc, char *argv[]) {
    char dirname[1024];
    char buffer[256];
    FILE *fp;
    DIR *dirp;
    struct dirent *dp;
    int i;
    cv::Mat img,mag,angle,img_top,img_btn;
    vector<float> feature_baseline;
    std::vector<char*> filenames;
    std::vector<std::vector<float>> features;
    // check for sufficient arguments
    /*if( argc < 2) {
        printf("usage: %s <directory path>\n", argv[0]);
        exit(-1);
    }
     */

    parameter p;
    p.hist_bin = 8;

    //dirname = "/Users/chuhsuanlin/Documents/NEU/Course/Spring 2022/CS 5330 Pattern Reginition & Computer Vission/Project/Project 2/olympus/";
    // get the directory path
    strcpy(dirname, "/Users/chuhsuanlin/Documents/NEU/Course/Spring 2022/CS 5330 Pattern Reginition & Computer Vission/Project/Project 2/test_database");
    printf("Processing directory %s\n", dirname );


    // open the directory
    dirp = opendir( dirname );
    if( dirp == NULL) {
        printf("Cannot open directory %s\n", dirname);
        exit(-1);
    }
    vector<float> hist,tmp,hist_top,hist_btn;
    vector<char*> feature_name;
    //read_image_data_csv("feature_baseline", filenames, features, 1 );
    // loop over all the files in the image file listing
    while( (dp = readdir(dirp)) != NULL ) {

        // check if the file is an image
        if( strstr(dp->d_name, ".jpg") ||
            strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".tif") ) {

            printf("processing image file: %s\n", dp->d_name);

            // build the overall filename
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);

            img = cv::imread(buffer);

            int ORIENTATION_COUNT = 3; // number of orientations (theta values)
            vector<double> thetas; //the orientation of the filter

            for (size_t i = 0; i <= ORIENTATION_COUNT; i++)
                thetas.push_back(i * M_PI / 4); //M_PI is pi

            int kernel_size = 9; // size of filter
            double sigma = 1; // standard deviation of gaussian envelope
            double gamma = 1; // spatial aspect ratio
            double psi = 0; // phase offset
            double lambda = 10;
            cv::Mat gaborKernel = getGaborKernel( Size(31,31), 4.0, CV_PI/4, 10.0, 0.5, 0, CV_32F );

            Mat dstImage;
            filter2D(img,dstImage,img.depth(),gaborKernel);


            //mag.convertTo(img_top,CV_8U,1.0/255.0);


            //std::cout << img.rows << "   " << img.cols << endl;
            cv::imshow("Image",img);
            //sobel_filter(img,mag,angle,3);
            cv::imshow("Mag",dstImage);
            //cv::imshow("Angle",angle);
            //string type = type2str(img.type());
            //

            //getFeatures(img,hist, feature_name, p );

            cv::Rect r_top( 0, 0, img.cols, img.rows/2);
            cv::Rect r_btn( 0, img.rows/2, img.cols, img.rows/2);

            img_top = img(r_top).clone(); // now B has a seperate *copy* of the pixels
            img_btn = img(r_btn).clone(); // now B has a seperate *copy* of the pixels

            //imshow("ori",img);
            //imshow("top",img_top);
            //imshow("btn",img_btn);
            //waitKey(0);
            //break;
            //cv::Mat hist_mat_top = histogram3d(img_top,8);
            //cv::Mat hist_mat_btn = histogram3d(img_btn,8);

            //hist_top = mat2vector(hist_mat_top);
            //hist_btn = mat2vector(hist_mat_btn);

            //hist.clear();
            //hist.insert(hist.end(),hist_top.begin(),hist_top.end());
            //hist.insert(hist.end(),hist_btn.begin(),hist_btn.end());
            //std::cout << hist[1] << endl;


            //for(float a : aaa){
            //    std::cout << a << endl;
            //}
            //feature_baseline = baseline(img);
            //append_image_data_csv("feature_hist_multi.csv", dp->d_name, hist, 0);

            //read_image_data_csv("feature_hist2d.csv", filenames, features, 0 );

            //append_header_csv("header_test.cvs","img_name",filenames,0);
            cv::waitKey(0);
            break;
        }
    }


    printf("Terminating\n");

    return(0);
}
