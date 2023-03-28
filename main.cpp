
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
using namespace cv;

int main(int argc, char *argv[])
{

    // check for sufficient arguments
    if( argc < 6) {
        printf("usage: %s <directory path>\n", argv[0]);
        //exit(-1);
    }
    printf("exe name=%s\n", argv[0]);

    // test for argv
    /*
    for (int i=0; i<argc; i++){
        printf("\nargv=%s", argv[i]);
    }


    argv[1] = "pic.0343.jpg";
    argv[2] = "/Users/chuhsuanlin/Desktop/olympus";//
    argv[3] = "banana";//"green_can";
    argv[4] = "ssd";//"inter_hist";
    argv[5] = "10";
    */

    // set compared image and distance metric
    char src_name[256];//"pic.0001.jpg"; //
    strcpy(src_name,argv[1]);
    char dirname[256];
    strcpy(dirname,argv[2]);
    char feature_type[256];// = argv[3];
    strcpy(feature_type,argv[3]);
    char distance_metric[256];// = argv[4];
    strcpy(distance_metric,argv[4]);
    int n_best_match = atoi(argv[5]);//5;//10;


    // database location
    //strcpy(dirname, "/Users/chuhsuanlin/Documents/NEU/Course/Spring 2022/CS 5330 Pattern Reginition & Computer Vission/Project/Project 2/olympus/");

    // csv filename according to feature_type
    char csv_filename[64];
    char csv_loc[1024];
    strcpy(csv_filename,feature_type);
    strcat(csv_filename, ".csv");

    strcat(dirname,"/");
    // csv file location
    strcpy(csv_loc, (string(dirname)+string(csv_filename)).c_str());

    FILE *fp;
    DIR *dirp;
    struct dirent *dp;
    char buffer[1024];

    bool flag_hearder = false;

    // check if csv file exist
    if (csv_check(csv_loc) == -1){ // not exist
        // open the directory
        dirp = opendir( dirname );
        if( dirp == NULL) {
            printf("Cannot open directory %s\n", dirname);
            exit(-1);
        }

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

                cv::Mat img = cv::imread(buffer);

                vector<float> write_feature, weight;
                vector<char*> feature_names;
                // get features
                getFeatures(img,write_feature, feature_names, feature_type, weight);
                    //cout << 2 << endl;
                if (flag_hearder == false) {
                    //int append_header_csv(char *filename, char *col_name, std::vector<char*> &feature_name, int reset_file);
                    append_header_csv(csv_loc, "img_name", feature_names, 0);
                    flag_hearder = true;
                }
                append_image_data_csv(csv_loc, dp->d_name, write_feature, 0);

            }

        }
    }

    // cal the similarity
    // read the features in the csv file
    vector<char*> header,img_filenames;
    vector<vector<float>> features;
    read_image_data_csv(csv_loc, img_filenames, features, header, 0 );

    // find how many types of data in the csv file
    vector<int> feature_index;
    vector<char*> feature_names;
    findFeatureloc(header, feature_names, feature_index );

    // build the overall filename
    strcpy(buffer, dirname);
    strcat(buffer, src_name);


    // read the target image
    cv::Mat src = imread(buffer);
    if (!src.data) {
        std::cout << "Image not loaded";
        return -1;
    }
    //imshow("src", src);
    //waitKey(0);

    vector<char*> src_filename;
    vector<float> src_feature,weight;
    // gain features of target image
    getFeatures(src,src_feature, src_filename, feature_type, weight);


    vector<vector<float>> all_diff;

    // get different features
    for (int i = 0; i < feature_names.size() ; i++){
        //std::cout << feature_names[0] << endl;
        int in_first = feature_index[i];
        int in_last = feature_index[i + 1];

        //std::cout << in_first << "   " << in_last << endl;

        //get different feature
        auto start = src_feature.begin() + in_first;
        auto end = src_feature.begin() + in_last ;

        vector<float> sub_src_feature(start, end);
        vector<vector<float>> sub_features;
        for(int f=0; f<features.size(); f++) {

            auto start = features[f].begin() + in_first;
            auto end = features[f].begin() + in_last;

            vector<float> tmp(start,end);
            sub_features.push_back(tmp);
        }

        //std::cout << features.size() << endl;
        // distance metirc
        vector<float> diff = calDistance(sub_src_feature, sub_features, distance_metric );
        all_diff.push_back(diff);

    }

    //std::cout << features.size() << endl;
    vector<float> weight_diff;
    for (int i=0; i<features.size(); i++){

        float weighted = 0;
        for (int j=0; j< all_diff.size(); j++){
            weighted += weight[j]*all_diff[j][i];

        }
        weight_diff.push_back(weighted);
    }

    vector<char*> matching_files = findMatching(weight_diff,img_filenames, n_best_match+1);

    // show the matched image and their name
    std::cout << "Matching_files: " << ' ';
    for (int i=0; i<matching_files.size(); i++){
        std::cout << matching_files[i] << ' ';

        //show matching images
        char file_loc[1024];
        strcpy(file_loc, (string(dirname)+string(matching_files[i])).c_str());
        cv::Mat img = imread(file_loc);
        cv::imshow(to_string(i),img);

    }

    // show target image
    //imshow("src",src);
    waitKey(0);

    return 0;
}




