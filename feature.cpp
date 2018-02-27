//
//  feature.cpp
//  Feature_Matching
//
//  Created by Thayjes Srivas on 2/8/18.
//  Copyright © 2018 Caltech. All rights reserved.
//

#include "feature.hpp"
using namespace cv;
using namespace std;

bool compareByDistance(const cv::DMatch &a, const cv::DMatch &b)
{
    return a.distance < b.distance;
}

/*
 *--------------------------------------------------------------------------------------
 *       Class:  Feature
 *      Method:  Feature
 * Description:  Initializes a feature class, given an input img path and 2d feature descriptor.
 In this example we use SIFT. The constructor initializes keypoints and a descriptor.
 *--------------------------------------------------------------------------------------
 */
Feature::Feature(std::string img_path, cv::Ptr<cv::Feature2D> fd_in)
{
    // Set the gray image based on the path
    curr_img = cv::imread(img_path, CV_LOAD_IMAGE_GRAYSCALE);
    //Detects the sift features of gray and stores it in std::vector<Keypoint> private member
    //of the feature class.
    fd = fd_in;
    fd->detect(curr_img, keypoints);
    initializeFeatures(keypoints, curr_img, img_path);
    keypoints.clear();
    for(int i = 0; i < features.size(); i++)
        keypoints.push_back(features[i].feature_point);
    
}  /* -----  end of method Feature::Feature  (constructor)  ----- */

/*
 *--------------------------------------------------------------------------------------
 *       Class:  Feature
 *      Method:  drawFeatures
 * Description: Displays the current keypoint features in the image.
 *--------------------------------------------------------------------------------------
 */
void
Feature::drawFeatures()
{
    if(features.size() < 1)
    {
        printf("There are no features to draw!");
        return;
    }
    cv::Mat keypoint_img;
    cv::Mat draw_image = cv::imread(features[0].img_path, CV_LOAD_IMAGE_GRAYSCALE);
    cout << "Drawing on img path = " << features[0].img_path << endl;
    cv::cvtColor(draw_image, keypoint_img, CV_GRAY2BGR);
    std::vector<cv::KeyPoint> kp;
    //keypoint_img.create(gray.rows, gray.cols, CV_8UC3);
    /*
    if(!keypoint_img.empty())
        cv::drawKeypoints(gray, keypoints, keypoint_img, Scalar(255, 0, 0), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::namedWindow("Keypoints found");
        cv::imshow("Keypoints found", keypoint_img);
        cv::waitKey(0);
     */
    
    // An alternative, loop through the features and draw a circle around each one. Along with the id.
    
    std::size_t pos = features[0].img_path.find("ir");
    std::string img_path = features[0].img_path.substr(pos);
    cv::namedWindow(img_path);
    for(int i = 0; i < (int)features.size(); i++)
    {
        
        cv::String text_id = std::to_string(features[i].id);
        cv::Point2f curr_feat_pos = features[i].feature_point.pt;
        cv::Point center((int)curr_feat_pos.x, (int)curr_feat_pos.y);
        printf("Feature %d = [%f, %f]\n", features[i].id, curr_feat_pos.x, curr_feat_pos.y);
        if(features[i].last_seen)
            cv::circle(keypoint_img, center, 2, cv::Scalar(255, 0, 0));
        else
            cv::circle(keypoint_img, center, 2, cv::Scalar(0, 0, 255));
        cv::Point text_pos((int)curr_feat_pos.x - 5, (int)curr_feat_pos.y - 5);
        cv::putText(keypoint_img, text_id, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar::all(255), 0);
        
        //kp.push_back(features[i].feature_point);
        
    }
    //cv::drawKeypoints(draw_image, kp, keypoint_img);
    cv::imshow(img_path, keypoint_img);
    cv::waitKey(0);
    
}

/*
 *--------------------------------------------------------------------------------------
 *       Class:  Feature
 *      Method:  match
 * Description:
 *--------------------------------------------------------------------------------------
 */
void
Feature::match(std::string imgpath)
{
    int num_fts = NUM_FTS;
    std::cout << "Looking for matches in: " << imgpath << std::endl;
    // Read in the new image. Find its keypoints and corresponding descriptor.
    // We will search for our features in these keypoints.
    cv::Mat img2 = cv::imread(imgpath, CV_LOAD_IMAGE_GRAYSCALE);
    std::vector<cv::KeyPoint> kp2, kp1; cv::Mat d1;
    cv::Mat d2;
    fd->detect(img2, kp2); fd->compute(img2, kp2, d2);
    
    // Create a matcher object and a vector to store the matches.
    cv::FlannBasedMatcher matcher; std::vector<cv::DMatch> matches;
    
    // If this is the very first time we are matching
    if(!matched)
    {
        std::cout << "Not yet matched!" << std::endl;
        // Loop through the features and create the descriptor matrix
        for(int i = 0; i < features.size(); i++)
        {
            d1.push_back(features[i].descriptor);
            //features[i].descriptor.copyTo(d1.row(i));
        }
        // Find matches for the features
        matcher.match(d1, d2, matches);
        double max_dist = 0; double min_dist = 100;
        //-- Quick calculation of max and min distances between keypoints
        for( int i = 0; i < d1.rows; i++ )
        { double dist = matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }
        printf("-- Max dist : %f \n", max_dist );
        printf("-- Min dist : %f \n", min_dist );
        
        // Create vectors to store the good matches and good indices of the feature points.
        std::vector< DMatch > good_matches(matches.size());
        std::vector<std::size_t> good_indices; int good_count = 0;
        printf("The total number of matches is %lu\n", matches.size());
        // Clear the matches and keypoints which will be used for displaying the matched features.
        curr_matches.clear(); keypoints.clear();
        // Filter matches based on distance
        for( int i = 0; i < d1.rows; i++ )
        {
            if( matches[i].distance <= 200)
            {
                good_count++;
                good_matches[i] = matches[i];
                good_indices.push_back(good_matches[i].trainIdx);
                // Here we assign the correct query id for the feature. This will allow us to know if
                // we were unable to track the feature and need to find a new one.
                // Update curr_matches and keypoints based on distance
                curr_matches.push_back(matches[i]);
                keypoints.push_back(features[i].feature_point);
                good_matches[i].queryIdx = features[i].query_id;
            }
        }
        // Reassign query IDs, required for displaying the matches.
        for(int i = 0; i < curr_matches.size(); i++)
            curr_matches[i].queryIdx = i;
        printf("The number of good matches is %d\n", good_count);
        
        // Now we update the features based on the matches. If a match was found, we update
        // the feature to the new point else we initialize a new feature from the matched image.
        // Ideally we would like to match all the features.
        matched = true; matched_img = img2;
        updateFeatures(kp2, img2, imgpath, good_matches);
        // Now the initial match is done, update the matched img.
        
    }
    
    else{
        std::cout << "Searching for matching features in " << imgpath << std::endl;
        // Repeat the same procedure
        // Update the current and matched images
        curr_img = matched_img;
        matched_img = img2;
        // Fill the descriptor matrix based on the current features
        for(int i = 0; i < features.size(); i++)
        {
            //cout << features[i].descriptor.size << endl;
            d1.push_back(features[i].descriptor);
        }
        // Look for matches
        if ( d1.empty() )
            cvError(0,"MatchFinder","1st descriptor empty",__FILE__,__LINE__);
        if ( d2.empty() )
            cvError(0,"MatchFinder","2nd descriptor empty",__FILE__,__LINE__);
        printf("The size of the descriptor matrix is %dx%d", d1.rows, d1.cols);
        matcher.match(d1, d2, matches);
        double max_dist = 0; double min_dist = 100;
        //-- Quick calculation of max and min distances between keypoints
        for( int i = 0; i < d1.rows; i++ )
        { double dist = matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }
        printf("-- Max dist : %f \n", max_dist );
        printf("-- Min dist : %f \n", min_dist );
        std::vector< DMatch > good_matches(matches.size());
        std::vector<std::size_t> good_indices;
        int good_count = 0;
        printf("The total number of matches is %lu\n", matches.size());
        curr_matches.clear();keypoints.clear();
        for( int i = 0; i < d1.rows; i++ )
        {   if( matches[i].distance <= 200)
                { good_matches[i] = matches[i];
                    good_indices.push_back(good_matches[i].trainIdx);
                    good_count++;
                    // Here we assign the correct query id for the feature. This will allow us to know if
                    // we were unable to track the feature and need to find a new one.
                    curr_matches.push_back(matches[i]);
                    good_matches[i].queryIdx = features[i].query_id;
                    keypoints.push_back(features[i].feature_point);
                }
        }
        for(int i = 0; i < curr_matches.size(); i++)
            curr_matches[i].queryIdx = i;
        printf("The number of good matches is %d\n", good_count);
        updateFeatures(kp2, img2, imgpath, good_matches);
    }
    
}


/*
 *--------------------------------------------------------------------------------------
 *       Class:  Feature
 *      Method:  displayMatches
 * Description:  Draws the matched features on the current and matches image side by side.
 *--------------------------------------------------------------------------------------
 */

void
Feature::displayMatches()
{
    Mat img_matches; std::vector<cv::KeyPoint> kp1, kp2;
    fd->detect(matched_img, kp2);
    fd->detect(curr_img, kp1);
    for(int i = 0; i < keypoints.size(); i++)
        //cout << "Keypoint query ID = " << keypoints[i]
    cv::drawMatches(curr_img, keypoints, matched_img, kp2,
                curr_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    
    //-- Show detected matches
    /*
    cv::drawMatches(curr_img, curr_keypoints, matched_img, kp2, curr_matches, img_matches, Scalar::all(-1),
                    Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
     */
    imshow( "Good Matches", img_matches);
    waitKey(-1);
}

/*
 *--------------------------------------------------------------------------------------
 *       Class:  Feature
 *      Method:  updateFeatures
 * Description:
 *--------------------------------------------------------------------------------------
 */
void
Feature::updateFeatures(/*The Keypoints in the new image */std::vector<cv::KeyPoint> kp2, /*The new image itself*/cv::Mat img2, std::string img_path, /*The vector of good matches */std::vector<cv::DMatch> good_matches)
{
    // In this function, we loop through each feature and check if a match was found.
    int count_failed = 0; std::vector<cv::KeyPoint> temp_kp2(kp2); std::vector<int> failed_features;
    cout << "Temp Keypoints Size Before = " << temp_kp2.size() << endl;
    for(int i = 0; i < features.size(); i++)
        cout << "Feature's Query ID = " << features[i].query_id << endl;
    for(int i = 0; i < good_matches.size(); i++)
        cout << "Matches' Query ID = " << good_matches[i].queryIdx << endl;
    
    for(int i = 0;  i < (int)features.size(); i++)
    {
        // For each feature, we check if its query ID was found in the good matches.
        // If it is not, we assume that we were unable to locate the feature in the next image!!
        int curr_qid = features[i].query_id;
        
        std::vector<cv::DMatch>::iterator result = std::find_if(good_matches.begin(), good_matches.end(), [curr_qid] (const cv::DMatch& s) {return s.queryIdx == curr_qid;}
        );
        // If it was found
        if((*result).queryIdx == curr_qid)
        {
            printf("Feature Number %d, Query ID %d, Matched!\n", features[i].id, features[i].query_id);
            features[i].last_seen = true;
            // Update the feature point location, descriptor, queryid and imgpath.
            features[i].feature_point = kp2[(*result).trainIdx]; //kp2[good_matches[i].trainIdx];
            std::vector<cv::KeyPoint> curr_keypoint(1, features[i].feature_point);
            fd->compute(matched_img, curr_keypoint, features[i].descriptor);
            features[i].query_id = (*result).trainIdx;
            features[i].img_path = img_path;
            // Erase this keypoint from the temp vector, as it should not be used to initialize a new feature.
            //temp_kp2.erase(temp_kp2.begin() + features[i].query_id);
            /*
            feature curr_feature;
            curr_feature.img = img2;
            curr_feature.feature_point = kp2[good_matches[i].trainIdx];
            curr_feature.img_path = img_path;
            features.push_back(curr_feature);
             */
        }
        // If it was not found
        else
        {
            failed_features.push_back(i); // Store the failed features
            printf("Feature Number %d, Query ID %d, Not Matched!\n", features[i].id, features[i].query_id);
            count_failed++;
            features[i].last_seen = false;
            cout << "Searching for new feature..." << endl;
            // Procedure to acquire a new feature follows
            // Ideas
            // 1. Find a new feature very close the one that was lost?
            // Pros: Maybe we can initialize depth based on the previous feature.
            // Cons: It may be very close the edge of the image, the reason why it was lost in the first place.
            addFeatures(temp_kp2, img2, img_path, features[i]);
            printf("Feature Number %d was successfully updated!\n", features[i].id);
        }
    }
    printf("Number of Features Not Matched = %d\n", count_failed);
    cout << "Temp Keypoints Size After = " << temp_kp2.size() << endl;
}

/*
 *--------------------------------------------------------------------------------------
 *       Class:  Feature
 *      Method:  initializeFeatures
 * Description: Provided keypoints and an image. This function will look for keypoints in the middle region
                of the image and store feature points in the features vector. This can be used when the a feature
                we have been tracking goes out of the view and we cannot find it. We then replace that with a new feature.
 NOTE1: More filters/restrictions on the feature points can be added here.
 NOTE2: NUM_FTS is the number of features we are tracking. This is defined in feature.hpp.
 
 *--------------------------------------------------------------------------------------
 */
void
Feature::initializeFeatures(std::vector<cv::KeyPoint> keypoints, cv::Mat img, std::string img_path)
{
    // A temporary variable to store the position of the current keypoint
    cv::Point2f temp;
    // Initialize loop variables for keypoints and features
    int num = NUM_FTS; int i =0; int j =0;
    while(i < num)
    {
        temp = keypoints[j].pt;
        if(temp.x >= 0.25*img.cols & temp.x <=  0.75*img.cols & temp.y>=0.25*img.rows & temp.y<= 0.75*img.rows)
        {
            // Check if this point is already a feature
                feature curr_feature;
                curr_feature.img = img;
                curr_feature.feature_point = keypoints[j];
                curr_feature.id = i;
                curr_feature.img_path = img_path;
                curr_feature.query_id = j;
                std::vector<cv::KeyPoint> curr_keypoint(1, keypoints[j]);
                fd->compute(img, curr_keypoint, curr_feature.descriptor);
                features.push_back(curr_feature);
                i++;
        }
        j++;
    }
}

/*
 *--------------------------------------------------------------------------------------
 *       Class:  Feature
 *      Method:  addFeatures
 * Description: Provided keypoints and an image. This function will look for keypoints in the middle region
 of the image and reset a feature. This can be used when a feature
 we have been tracking goes out of the view and we cannot find it. We then replace that with a new feature.
 NOTE1: More filters/restrictions on the feature points can be added here.
 NOTE2: NUM_FTS is the number of features we are tracking. This is defined in feature.hpp.
 
 *--------------------------------------------------------------------------------------
 */
void Feature::addFeatures(std::vector<cv::KeyPoint> &keypoints, cv::Mat img, std::string img_path, feature &curr_feature)
{
    cv::Point2f temp;
    for(int i = 0; i < keypoints.size(); i++)
    {
        temp = keypoints[i].pt;
        if(temp.x >= 0.25*img.cols & temp.x <=  0.75*img.cols & temp.y>=0.25*img.rows & temp.y<= 0.75*img.rows)
        {
            // Update the feature
            std::vector<feature>::iterator it = std::find_if(features.begin(), features.end(), check_duplicate(temp));
            if(it == features.end())
            {
                if(matched)
                    cout << "New keypoint = " << keypoints[i].pt << endl;
                curr_feature.img = img;
                curr_feature.feature_point = keypoints[i];
                curr_feature.img_path = img_path;
                curr_feature.query_id = i;
                std::vector<cv::KeyPoint> curr_keypoint(1, keypoints[i]);
                fd->compute(img, curr_keypoint, curr_feature.descriptor);
                // Erase the keypoint from the vector, so its not used to intialize another feature!
                //keypoints.erase(keypoints.begin() + i);
                // Exit the loop
                break;
            }
        }
    }
}
/*
 *--------------------------------------------------------------------------------------
 *       Class:  Feature
 *      Method:  displayImage
 * Description:
 *--------------------------------------------------------------------------------------
 */
void
Feature::display_image()
{
    cv::imshow("Gray", gray);
    waitKey(-1);
}
