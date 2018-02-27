//
//  feature.hpp
//  Feature_Matching
//
//  Created by Thayjes Srivas on 2/8/18.
//  Copyright © 2018 Caltech. All rights reserved.
//

#ifndef feature_hpp
#define feature_hpp

#include <stdio.h>
#include<iostream>
#include <string.h>
#include <algorithm>
#include "types.h"
#include "opencv2/core/core.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>

#define NUM_FTS 5;




/*
 * =====================================================================================
 *        Class:  Feature
 *  Description:
 * =====================================================================================
 */
class Feature
{
public:
    /* ====================  LIFECYCLE     ======================================= */
    Feature(std::string img_path, cv::Ptr<cv::Feature2D> fd_in);                             /* constructor */
    
    /* ====================  ACCESSORS     ======================================= */
    
    void drawFeatures();
    void displayMatches();
    void display_image();
    
    /* ====================  MUTATORS      ======================================= */
    
    void match(std::string imgpath);
    void updateFeatures(std::vector<cv::KeyPoint> kp2, cv::Mat img2, std::string img_path, std::vector<cv::DMatch> good_matches);
    void initializeFeatures(std::vector<cv::KeyPoint> keypoints, cv::Mat img, std::string img_path);
    void addFeatures(std::vector<cv::KeyPoint> &keypoints, cv::Mat img, std::string img_path, feature& curr_feature);
    
    /* ====================  OPERATORS     ======================================= */
    
protected:
    /* ====================  METHODS       ======================================= */
    
    /* ====================  DATA MEMBERS  ======================================= */
    cv::Mat descriptor;
    std::vector<cv::KeyPoint> keypoints, matched_keypoints;
    matched_features m_f;


private:
    /* ====================  METHODS       ======================================= */
    
    /* ====================  DATA MEMBERS  ======================================= */
    cv::Mat gray;
    cv::Mat curr_img, matched_img; // The current image being processed
    cv::Ptr<cv::Feature2D> fd;
    SmartVector<feature> features;
    std::vector<cv::DMatch> curr_matches;
    bool matched = false;

    
}; /* -----  end of class feature  ----- */



#endif /* feature_hpp */
