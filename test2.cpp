//
// Created by Chu-Hsuan Lin on 2022/2/9.
//



#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric>
#include "getFeature.h"
#include "csv_util.h"
#include "matching.h"
#include "filter.h"
#include <algorithm>

using namespace std;

template <typename T>
vector<size_t> sort_indexes(const vector<T> &v) {

    // initialize original index locations
    vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    // using std::stable_sort instead of std::sort
    // to avoid unnecessary index re-orderings
    // when v contains elements of equal values
    stable_sort(idx.begin(), idx.end(),
                [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

    return idx;
}


int main123(){

    char dirname[1024];
    char feature_filename[64];
    char feature_loc[1024];

    FILE *fp;
    DIR *dirp;
    struct dirent *dp;
    char buffer[256];
    cv::Mat img;

    parameter p;
    p.hist_bin = 2;
    p.hist2d_bin = 16;
    p.hist2d_type = 2;
    p.hist3d_bin = 8;
    p.sobel_bin = 16;
    p.sobel_size = 3;

    char src_name[128] = "pic.0164.jpg";
    char* feature_type = "hist_2d";
    char* distance_metric = "inter_hist";

    bool flag_hearder = false;
    vector<float> write_feature;
    vector<char*> feature_name, filenames,src_filename;
    std::vector<std::vector<float>> features;
    vector<float> feature, src_feature, diff;

    //string dirname = "/Users/chuhsuanlin/Documents/NEU/Course/Spring 2022/CS 5330 Pattern Reginition & Computer Vission/Project/Project 2/olympus/";
    //string filename = "features.csv";

    strcpy(dirname, "/Users/chuhsuanlin/Documents/NEU/Course/Spring 2022/CS 5330 Pattern Reginition & Computer Vission/Project/Project 2/olympus/");
    //strcpy(dirname, "/Users/chuhsuanlin/Documents/NEU/Course/Spring 2022/CS 5330 Pattern Reginition & Computer Vission/Project/Project 2/test_bluecan/");
    //std::cout << "cvs_exit1" << endl;
    strcpy(feature_filename,feature_type);
    strcat(feature_filename, ".csv");
    //std::cout << "cvs_exit2" << endl

    strcpy(feature_loc, (string(dirname)+string(feature_filename)).c_str());

    if (csv_check(feature_loc) == -1){
        // open the directory
        dirp = opendir( dirname );
        if( dirp == NULL) {
            printf("Cannot open directory %s\n", dirname);
            exit(-1);
        }
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

                //cout << img.size << endl;
                vector<float> weight;
                getFeatures(img,write_feature, filenames, feature_type,weight);
                //cout << 2 << endl;
                if (flag_hearder == false){
                    //int append_header_csv(char *filename, char *col_name, std::vector<char*> &feature_name, int reset_file);
                    append_header_csv(feature_loc,"img_name", filenames,0);
                    flag_hearder = true;
                }

                append_image_data_csv(feature_loc,dp->d_name, write_feature, 0);
            }
            //break;
        }
        //waitKey(0);
        //append_header_csv(feature_loc,"img_name",filenames,0);
        //getFeatures();
    }
    else{

        vector<char*> header;
        read_image_data_csv(feature_loc, filenames, features, header, 0 );

        vector<int> featureindex;
        vector<char*> featurename;
        findFeatureloc(header, featurename, featureindex );


        // build the overall filename
        strcpy(buffer, dirname);
        strcat(buffer, "/");
        strcat(buffer, src_name);
        std::cout << buffer << endl;

        cv::Mat src = imread(buffer);
        if (!src.data) {
            std::cout << "Image not loaded";
            return -1;
        }

        vector<float> weight;
        getFeatures(src,src_feature, src_filename, feature_type, weight);
        std::cout << "size  " << src_feature.size() << endl;

        vector<vector<float>> all_diff;
        std::cout << featurename.size() << endl;
        for (int i = 0; i < featurename.size() ; i++){
            //std::cout << featurename[0] << endl;
            int in_first = featureindex[i];
            int in_last = featureindex[i + 1];

            std::cout << in_first << "   " << in_last << endl;
            //vector<float> sub_src_feature;
            auto start = src_feature.begin() + in_first;
            auto end = src_feature.begin() + in_last ;

            vector<float> sub_src_feature(start, end);

            vector<vector<float>> sub_feature;
            for(int f=0; f<features.size(); f++) {

                auto start = features[f].begin() + in_first;
                auto end = features[f].begin() + in_last;

                vector<float> tmp(start,end);
                sub_feature.push_back(tmp);

            }

            vector<float> diff = calDistance(sub_src_feature, sub_feature, distance_metric );
            //std::cout << "diff  " << diff.size() << endl;
            all_diff.push_back(diff);

        }

        //vector<float> weight={0.5,0.5};
        vector<float> weight_diff;



        for (int i=0; i<features.size(); i++){
            float weighted = 0;
            for (int j=0; j< all_diff.size(); j++){
                //cout << "weight  " << weight[j] << endl;
                weighted += weight[j]*all_diff[j][i];
            }
            weight_diff.push_back(weighted);
        }
        //std::cout << "3  " << weight_diff.size() <<  endl;


        vector<char*> matching_files = findMatching(weight_diff,filenames, 4);
        std::cout << "weight  " << weight.size() << endl;
        std::cout << "features  " << features.size() << endl;
        std::cout << "all_diff  " << all_diff.size() << endl;

        for (int i=0; i<matching_files.size(); i++){
            std::cout << matching_files[i] << ' ';
            char file_loc[1024];

            strcpy(file_loc, (string(dirname)+string(matching_files[i])).c_str());
            //std::cout << file_loc << endl;
            cv::Mat img = imread(file_loc);
            cv::imshow(to_string(i),img);

        }
        imshow("src",src);
        waitKey(0);
    }


    return 0;

}

