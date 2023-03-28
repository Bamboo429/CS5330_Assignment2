//
// Created by Chu-Hsuan Lin on 2022/2/15.
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

int mainxx(){

    std::cout << "Hello world" << endl;

}

