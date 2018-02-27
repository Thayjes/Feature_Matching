//
//  types.h
//  Feature_Matching
//
//  Created by Thayjes Srivas on 2/18/18.
//  Copyright © 2018 Caltech. All rights reserved.
//

#ifndef types_h
#define types_h
#include <stdio.h>
#include<iostream>
#include <string.h>
#include <algorithm>
#include "opencv2/core/core.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include "feature.hpp"

typedef struct {
    bool matched = false;
    cv::Mat matched_img;
    std::vector<cv::Point2f> matched_points;
    std::vector<int> matched_indices;
    std::vector<cv::KeyPoint> matched_keypoints, matched_img_keypoints;
    std::vector<cv::DMatch> good_matches;

} matched_features;

typedef struct {
    int id; // This corresponds to the feature number
    int query_id; // This corresponds to the keypoint number when the feature was added.
    cv::KeyPoint feature_point; // The actual feature keypoint
    cv::Mat img; // The original img in which the feature was located and added
    std::string img_path; // The path to the current img being processed (will be used for displaying purposes)
    cv::Mat descriptor; // The descriptor vector for the feature, it is used for matching.
    bool last_seen = false; // As long as we find matches, it will be true
} feature;

// A SmartVector class, identical to vector but with the added matlab indexing functionality
template<typename T>
class SmartVector : public std::vector<T>{
public:
    // act like operator[]
    T operator()(size_t _Pos){
        return (*this)[_Pos];
    }
    // act like matlab operator()
    SmartVector<T> operator()(std::vector<size_t>& positions){
        SmartVector<T> sub;
        sub.resize(positions.size());
        size_t sub_i = 0;
        for(
            std::vector<size_t>::iterator pit = positions.begin();
            pit != positions.end();
            pit++,sub_i++){
            sub[sub_i] = (*this)[*pit];
        }
        return sub;
    }
};

struct find_queryId
{
    int queryId;
    find_queryId(int queryId) : queryId(queryId) {}
    bool operator () ( const cv::DMatch& m) const
    {
        return m.queryIdx == queryId;
    }
};



struct check_duplicate
{
    cv::Point2f n;
    check_duplicate(cv::Point2f i) : n(i)
    { }
    inline bool operator()(const feature& m) const { return m.feature_point.pt == n; }
};




#endif /* types_h */
