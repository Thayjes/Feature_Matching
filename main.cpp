//
//  main.cpp
//  Feature_Matching
//
//  Created by Thayjes Srivas on 2/8/18.
//  Copyright © 2018 Caltech. All rights reserved.
//

#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;

static void help()
{
    printf("\n This program demonstrates using features2d detector, descriptor extractor and simple matcher \n"
           "Using the SIFT descriptor: \n"
           "\n"
           );
}

int main(int argc, const char * argv[]) {
    // insert code here...
    std::cout << argc << std::endl;
    if(argc != 3)
    {
        return -1;
    }
    
    cv::Ptr<Feature2D> f2d = xfeatures2d::SIFT::create(150 /*nfeatures*/, 3 /*nOctaveLayers*/, 0.01 /*contrastThreshold*/);
    
        cv::Mat img_1 = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat img_2 = cv::imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
        if(img_1.empty() || img_2.empty())
        {
            printf("Cant read one of the images \n");
            return -1;
        }
        

        //-- Step 1: Detect the keypoints:
        std::vector<KeyPoint> keypoints_1, keypoints_2;
        f2d->detect( img_1, keypoints_1 );
        f2d->detect( img_2, keypoints_2 );
        
        //-- Step 2: Calculate descriptors (feature vectors)
        Mat descriptors_1, descriptors_2;
        f2d->compute( img_1, keypoints_1, descriptors_1 );
        f2d->compute( img_2, keypoints_2, descriptors_2 );
        
        //-- Step 3: Matching descriptor vectors using FlannBasedMatcher :
        FlannBasedMatcher matcher;
        std::vector< DMatch > matches;
        matcher.match( descriptors_1, descriptors_2, matches );
        
        double max_dist = 0; double min_dist = 100;
        //-- Quick calculation of max and min distances between keypoints
        for( int i = 0; i < descriptors_1.rows; i++ )
        { double dist = matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }
        printf("-- Max dist : %f \n", max_dist );
        printf("-- Min dist : %f \n", min_dist );
        //-- Draw only "good" matches (i.e. whose distance is less than 2.5*min_dist,
        //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
        //-- small)
        //-- PS.- radiusMatch can also be used here.
        std::vector< DMatch > good_matches;
        for( int i = 0; i < descriptors_1.rows; i++ )
        { if( matches[i].distance <= max(2.5*min_dist, 0.02) )
        { good_matches.push_back( matches[i]); }
        }
        //-- Draw only "good" matches
        Mat img_matches;
        drawMatches( img_1, keypoints_1, img_2, keypoints_2,
                    good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                    std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        //-- Show detected matches
        imshow( "Good Matches", img_matches );
        for( int i = 0; i < min(20, (int)good_matches.size()); i++ )
        { printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n --Pixel Co-ordinate of KeyPoint 1: (%f, %f) -- Pixel Co-ordinate of KeyPoint 2: (%f, %f) \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx, keypoints_1[good_matches[i].queryIdx].pt.x, keypoints_1[good_matches[i].queryIdx].pt.y, keypoints_2[good_matches[i].trainIdx].pt.x, keypoints_2[good_matches[i].trainIdx].pt.y); }
        waitKey(-1);
    
    

    /*
    namedWindow("matches", 1);
    Mat img_matches;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_matches);
    imshow("matches", img_matches);
    waitKey(-1);
     */




    
    return 0;
}
