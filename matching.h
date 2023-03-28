//
// Created by Chu-Hsuan Lin on 2022/2/7.
//
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;

#ifndef CONTENT_BASED_IMAGE_RETRIEVAL_MATCHING_H
#define CONTENT_BASED_IMAGE_RETRIEVAL_MATCHING_H

float sum_square_diff(vector<float> a, vector<float> b);
float sum_abs_diff(vector<float> a, vector<float> b);
vector<char*> baseline_matching( string filename, vector<vector<float>> features, int n_bestmatch);
float histogram_intersection_diff(vector<float> a, vector<float> b);
int csv_check(char *filename);
vector<char*> findMatching(vector<float> src_distance, vector<char*> filenames, int n_bestmatch);
int findFeatureloc(vector<char*> header, vector<char*> &featurename, vector<int> &featureindex);
vector<float> calDistance(vector<float> src_feature, vector<vector<float>> features, char* distance_metric);


#endif //CONTENT_BASED_IMAGE_RETRIEVAL_MATCHING_H
