#include <numeric>
#include <stdexcept>
#include "matching2D.hpp"

using namespace std;

/**
 * Find the match for keypoints in two camera images using the descriptors.
 * 
 * @param kPtsSource <std::vector<cv::KeyPoint>> Source keypoints.
 * @param kPtsRef <std::vector<cv::KeyPoint>> Reference keypoints.
 * @param descSource <cv::Mat> Descriptor source.
 * @param descRef <cv::Mat> Descriptor reference.
 * @param matches <std::vector<cv::DMatch>> Matches.
 * @param descriptorTypeCategory <std::string> Category of the descriptor (either DES_HOG or DES_BINARY).
 * @param matcherType <std::string> Type of the matcher (MAT_BF or MAT_FLANN).
 * @param selectorType <std::string> Type of the selector (SEL_NN or SEL_KNN).
 */
void matchDescriptors(
    std::vector<cv::KeyPoint> &kPtsSource, 
    std::vector<cv::KeyPoint> &kPtsRef, 
    cv::Mat &descSource, 
    cv::Mat &descRef,
    std::vector<cv::DMatch> &matches, 
    std::string descriptorType, 
    std::string matcherType, 
    std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
        if (descriptorType.compare("DES_HOG") == 0)
          matcher = cv::BFMatcher::create(cv::NORM_L2, crossCheck);
        else
          matcher = cv::BFMatcher::create(cv::NORM_HAMMING, crossCheck);
    else if (matcherType.compare("MAT_FLANN") == 0)
        if (descriptorType.compare("DES_HOG") == 0)
          matcher = cv::FlannBasedMatcher::create();
        else {
          cv::Ptr<cv::flann::IndexParams> params = cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2);
          matcher = cv::makePtr<cv::FlannBasedMatcher>(params);
        }


    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        int k=2;

        std::vector<std::vector<cv::DMatch>>  matches_temp;
        matcher->knnMatch(descSource, descRef, matches_temp,k);
        const double dist_threshold = 0.8;
        for (auto m: matches_temp)
            if (m.size()>2)
                if (m[1].distance != 0 && (m[0].distance / m[1].distance ) > dist_threshold)
                    matches.push_back(m[0]);
    }
}
// Use one of several types of state-of-art descriptors to uniquely identify keypoints
// Possible descriptors: 
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {
        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(32, false);
    else if (descriptorType.compare("ORB") == 0)
        extractor = cv::ORB::create(500, 1.2, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
    else if (descriptorType.compare("FREAK") == 0)
        extractor = cv::xfeatures2d::FREAK::create(true, true, 22.0, 4,std::vector<int>());
    else if (descriptorType.compare("AKAZE") == 0)
        extractor = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0,3,0.001, 4,4,cv::KAZE::DIFF_PM_G2);
    else if (descriptorType.compare("SIFT") == 0)
        extractor = cv::xfeatures2d::SIFT::create(0,3,0.04,10,1.6);

    double t = (double)cv::getTickCount();
    // perform feature description
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {
        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    int block_size = 2;         
    int aperture_size = 3;      
    int threshold = 100;     
    double k = 0.04;                  

    cv::Mat dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::Mat dst_norm;
    cv::Mat dst_norm_scaled;

    cv::cornerHarris(img, dst, block_size, aperture_size, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    for (int j = 0; j < dst_norm.rows; ++j)
    {
        for (int i = 0; i < dst_norm.cols; ++i)
        {
            int dist = static_cast<int>(dst_norm.at<float>(j, i));

            if (dist > threshold)
            {
                cv::KeyPoint new_kp;
                
                new_kp.pt = cv::Point2f(i, j);
                new_kp.size = 2 * aperture_size;
                new_kp.response = dist;

                bool check = false;

                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    double kp_check = cv::KeyPoint::overlap(new_kp, *it);

                    if (kp_check > 0.0)
                    {
                        check = true;
                        if (new_kp.response > it->response)
                        {
                            *it = new_kp; 
                            break; 
                        }
                    }
                }
                if (!check)
                    keypoints.push_back(new_kp);
            }
        }
    }
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    cv::Ptr<cv::FeatureDetector> detector;

    if (detectorType == "FAST")
        detector = cv::FastFeatureDetector::create();
    else if (detectorType == "BRISK")
        detector = cv::BRISK::create();
    else if (detectorType == "ORB")
        detector = cv::ORB::create();
    else if (detectorType == "AKAZE")
        detector = cv::AKAZE::create();
    else if (detectorType == "SIFT")
        detector = cv::xfeatures2d::SIFT::create();
    else
        throw std::runtime_error("Detector " + detectorType + " not known to this program.");

    detector->detect(img, keypoints);
}