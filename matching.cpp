//
// Created by Chu-Hsuan Lin on 2022/2/5.
//

// The file is for matching method.
// Including Distance metrics method:  sum square difference, sum absolute difference and intersection histogram
// and sorting the difference and finding the matching file.

#include <iostream>
#include <vector>
#include <numeric>
#include <cstdio>
#include <cstdlib>


using namespace std;

// sum_square_diff
float sum_square_diff(vector<float> a, vector<float> b){

    if(a.size()!=b.size()){
        std::cout << "vector size mismatch" << endl;
        return -1;
    }
    int size = a.size();
    float diff =0;
    for(int i=0;i<size;i++){
        diff += (a.at(i)-b.at(i))*(a.at(i)-b.at(i));
    }

    //normalize
    diff/=size;
    return diff;
}

//sum_abs_diff
float sum_abs_diff(vector<float> a, vector<float> b){

    if(a.size()!=b.size()){
        std::cout << "vector size mismatch" << endl;
        return -1;
    }
    int size = a.size();
    float diff =0;
    for(int i=0;i<size;i++){
        diff += abs(a.at(i)-b.at(i));
    }
    diff/=size;
    return diff;
}

//
float histogram_intersection_diff(vector<float> a, vector<float> b){

    if(a.size()!=b.size()){
        std::cout << "vector size mismatch" << endl;
        return -1;
    }
    int size = a.size();
    float diff;
    float sum = 0;

    // find the min between 2 histogram
    for (int i=0; i<size; i++){
        sum += min(a.at(i),b.at(i));
    }
    diff = 1-sum;

    return diff;

}



// find index after sorting
//https://www.codegrepper.com/code-examples/cpp/c%2B%2B+sorting+and+keeping+track+of+indexes
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

// check csv file is exist or not
int csv_check(char *filename){

    FILE *fp;
    //DIR *dirp;

    fp = fopen(filename, "r");
    if( !fp ) {
        printf("CSV file not exist\n");
        return(-1);
    }
    else{
        return 0;
    }

}

// for unique function in findFeatureloc
bool string_compare(char *a, char *b)
{
    if (strcmp(a,b) == 0)
        return 1;
    else
        return 0;
}

// find feature index in csv file to set different weight for different features
int findFeatureloc(vector<char*> header, vector<char*> &featurename, vector<int> &featureindex){

    vector<char*> h_copy;
    copy(header.begin(), header.end(), back_inserter(h_copy));

    // find how many features in the csv file
    int uniqueCount = std::unique(header.begin(), header.end(),string_compare) - header.begin();

    // define index
    for(int i=1;i<uniqueCount;i++){
        auto it = find(h_copy.begin(), h_copy.end(), header.data()[i]);
        if (it != h_copy.end())
        {
            int index = it - h_copy.begin()-1;
            featureindex.push_back(index);
        }
        featurename.push_back(header[i]);
    }
    featureindex.push_back(header.size()-1);

    return 0;

}

// cal similarity  different distance metrics
vector<float> calDistance(vector<float> src_feature, vector<vector<float>> features, char* distance_metric){

    vector<float> feature,diff;

    // ssd
    if (strcmp(distance_metric, "ssd")==0 ){
        for (int i=0; i<features.size(); i++){
            feature = features[i];
            diff.push_back(sum_square_diff(feature, src_feature));
        }
    }

    //sad
    else if(strcmp(distance_metric,"sad")==0){
        for (int i=0; i<features.size(); i++){
            feature = features[i];
            diff.push_back(sum_abs_diff(feature, src_feature));
        }
    }

    // intersection hist
    else if(strcmp(distance_metric, "inter_hist")==0){
        for (int i=0; i<features.size(); i++){
            feature = features[i];
            diff.push_back(histogram_intersection_diff(feature, src_feature));
        }
    }

    // return diff vector
    return diff;
}

// find matching filename
vector<char*> findMatching(vector<float> src_distance, vector<char*> filenames, int n_bestmatch){

    vector<char*> match_files;

    // sort difference value
    vector<size_t> sort_index = sort_indexes(src_distance);

    //find matching filename according to sort index
    for (int i=0; i<n_bestmatch; i++){
        match_files.push_back(filenames[sort_index.at(i)]);
        //std::cout << src_distance[sort_index.at(i)] << endl;
    }
    return match_files;
}


