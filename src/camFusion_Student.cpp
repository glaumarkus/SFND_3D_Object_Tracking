#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"
#include "tree.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{   

    // check division by 0
    int kp_size = kptMatches.size();
    if (kp_size == 0)
        return;

    // for clustering several points are adressed here
    // 1. we need to check if keypoint match lies within region of interest
    // 2. to identify and remove outliers we will use keypoint distances and remove those that are further than 1 std_dev away
    double kp_mean = 0.0;
    double std_dev = 0.0;
    std::vector<double> distances;
    std::vector<cv::DMatch> temp_kptMatches;

    for (cv::DMatch match: kptMatches) 
        if (boundingBox.roi.contains(kptsCurr[match.trainIdx].pt)) {
            kp_mean += match.distance;
            distances.push_back(match.distance);
            temp_kptMatches.push_back(match);
        }
    
    kp_mean /= kp_size;

    for (auto& d: distances)
        std_dev += (d - kp_mean)*(d - kp_mean);
    
    std_dev /= kp_size;
    std_dev = sqrt(std_dev);

    double lower_limit, upper_limit;

    lower_limit = kp_mean - std_dev;
    upper_limit = kp_mean + std_dev;

    for (const cv::DMatch t: temp_kptMatches)
        if (t.distance >= lower_limit && t.distance <= upper_limit)
            boundingBox.kptMatches.push_back(t);
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    double min_dist_between_distances = 80.0;

    // collect kpt distances in both pictures
    std::vector<double> distances;

    std::for_each (kptMatches.begin(), kptMatches.end()-1, [&](cv::DMatch kpt1){
    

        auto kptRef_prev = kptsPrev.at(kpt1.queryIdx).pt;
        auto kptRef_curr = kptsCurr.at(kpt1.trainIdx).pt;

        std::for_each (kptMatches.begin()+1, kptMatches.end(), [&](cv::DMatch kpt2){

            auto kptTemp_prev = kptsPrev.at(kpt2.queryIdx).pt;
            auto kptTemp_curr = kptsCurr.at(kpt2.trainIdx).pt;

            double distance_prev = cv::norm(kptRef_prev - kptTemp_prev);
            double distance_curr = cv::norm(kptRef_curr - kptTemp_curr);

            if (
                distance_curr > min_dist_between_distances && 
                distance_prev > 1e-10
                ){
                distances.push_back(distance_curr / distance_prev);
            }
        });
    });

    if (distances.size() == 0)
    {
        TTC = -1000;
    }
    else {
       
        /* mean calc - median proved to be more robust

        double ratio_mean = 0.0;
        int ratio_size = distances.size();

        for (const auto& d: distances){
            ratio_mean += d;
        }
        ratio_mean /= ratio_size;
        
        */
        std::sort(distances.begin(), distances.end());
        double ratio_median = 
          distances.size() % 2 == 0 ? distances[distances.size() / 2] : (distances[distances.size()/2] + distances[distances.size()/2+1])/2;
  
        double delta_t = 1 / frameRate;
        TTC = -delta_t / (1 - ratio_median);

    }
}

static void clusterHelper(
    int index, 
    const std::vector<std::vector<float>>& points, 
    std::vector<int>& cluster, 
    std::vector<bool>& processed, 
    KdTree* tree,
    float& CLUSTER_DISTANCE) {

    processed[index] = true;
    cluster.push_back(index);
    std::vector<int> nearest = tree->search(points[index], CLUSTER_DISTANCE);

    for (int idx : nearest) {
        if (!processed[idx]) {
            clusterHelper(idx, points, cluster, processed, tree, CLUSTER_DISTANCE);
        }
    }
}

void get_min_distance_from_lidar(std::vector<LidarPoint> &l, double& closest_obs){

    KdTree* tree = new KdTree;
    std::vector<std::vector<float>> pts_filtered;
    int i = 0;

    // fill tree and keep ego filtered pts
    for (const auto& pt: l){
        if (abs(pt.y) <= 2.0){
            std::vector<float> pts = std::vector<float> {(float) pt.x, (float) pt.y, (float) pt.z};
            pts_filtered.push_back(pts);
            tree->insert(pts, i++);
        }
    }

    /*
    Cluster params
    */
    int cluster_min = 150;
    float cluster_dist = 0.15;

    std::vector<bool> processed(pts_filtered.size(), false);

    i = 0;
    while (i < pts_filtered.size()){
        if (processed[i]) {
            i++;
            continue;
        }
        std::vector<int> temp_cluster;
        clusterHelper(i, pts_filtered, temp_cluster, processed, tree, cluster_dist);
        if (temp_cluster.size() < cluster_min){
            processed[i] = true;
            i++;
            continue;
        }

        // smaller x value than target cluster?
        for (const auto& idx: temp_cluster)
            if (pts_filtered[idx][0] < closest_obs && pts_filtered[idx][0] > 0.0)
                closest_obs = pts_filtered[idx][0];
        if (closest_obs == 10000)
          closest_obs = NAN;
    }
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    double delta_t = 1 / frameRate;

    double obs_prev = 10000;
    double obs_curr = 10000;

    get_min_distance_from_lidar(lidarPointsPrev, obs_prev);
    get_min_distance_from_lidar(lidarPointsCurr, obs_curr);
  
    //std::cout << "Closest Distance: " << obs_curr << std::endl;
    
    TTC = obs_curr * delta_t / (obs_prev - obs_curr);
}


void matchBoundingBoxes(
    std::vector<cv::DMatch> &matches, 
    std::map<int, int> &bbBestMatches, 
    DataFrame &prevFrame, 
    DataFrame &currFrame)
{

    for (const auto& last_box: prevFrame.boundingBoxes){

        std::map<int,int> box_overlaps;

        for (const auto& current_box: currFrame.boundingBoxes){
            box_overlaps[current_box.boxID] = 0;

            for (const auto &match: matches){

                auto &last_keypoint = prevFrame.keypoints[match.queryIdx].pt;
                auto &current_keypoint = currFrame.keypoints[match.trainIdx].pt;

                if (last_box.roi.contains(last_keypoint) && current_box.roi.contains(current_keypoint))
                    box_overlaps[current_box.boxID]++;

            }
        }
      
        std::map<int, int>::iterator it = box_overlaps.begin();
      
        int max_pts = 0;
        int max_idx = 0;
        
        while (it != box_overlaps.end()){
          
          if (it->second > max_pts){
            max_pts = it->second;
            max_idx = it->first;
          }
          it++;
        }
      
        bbBestMatches[last_box.boxID] = currFrame.boundingBoxes[max_idx].boxID;
    }
}
