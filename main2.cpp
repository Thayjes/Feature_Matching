//
//  main2.cpp
//  Feature_Matching
//
//  Created by Thayjes Srivas on 2/9/18.
//  Copyright © 2018 Caltech. All rights reserved.
//

#include <stdio.h>
#include "feature.hpp"

using namespace std;
using namespace cv;

int main()
{
    std::stringstream oss;
    std::string pathcurr, pathnext;
    pathcurr = "/Users/thayjessrivas/Documents/AV_SLAMwFeatures/AV_images/ir_002510.jpg";
    cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SIFT::create(500 /*nfeatures*/, 3 /*nOctaveLayers*/, 0.01 /*contrastThreshold*/);
    cv::Ptr<cv::Feature2D> f2d2 = cv::BRISK::create();
    Feature* feat = new Feature(pathcurr, f2d);
    feat->drawFeatures();
    
    for(int k = 2511; k < 3000; k++){
        oss << "/Users/thayjessrivas/Documents/AV_SLAMwFeatures/AV_images/ir_00" << k << ".jpg";
        pathnext = oss.str();
        feat->match(pathnext);
        feat->displayMatches();
        feat->drawFeatures();
        cv::destroyAllWindows();
        oss.str(std::string());
        oss.clear();
        
        // Now we have a framework setup. Let us try to track the (x, y) pixel co-ordinates of
        // one of the keypoints across several images.
        
    }
    
    return 0;
}
