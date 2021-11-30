#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

Ptr<FastFeatureDetector> detector = FastFeatureDetector::create();
Ptr<AgastFeatureDetector> detector2 = AgastFeatureDetector::create();
Ptr<SIFT> sift_detector = SIFT::create();

Ptr<FREAK> desc_comp = FREAK::create();

Ptr<BFMatcher> matcher = BFMatcher::create();

ofstream outfile;

// finds matches based on the descriptors of two images.
void match(Mat desc1, Mat desc2, vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2, int window_size, vector<DMatch> &matches_out) {
    vector<vector<DMatch>> knn_matches;
    matcher->knnMatch(desc1, desc2, knn_matches, 2);

    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.7f;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            int left_ind = knn_matches[i][0].queryIdx;
            int right_ind = knn_matches[i][0].trainIdx;
            KeyPoint left = keypoints1[left_ind];
            KeyPoint right = keypoints2[right_ind];

            if (abs(left.pt.x - right.pt.x) < window_size && abs(left.pt.y - right.pt.y < window_size)) {
                matches_out.push_back(knn_matches[i][0]);
            }
        }
    }
}

// calculates disparity output as well as the RMSE and percentage of bad matches
void calculateDisp(vector<DMatch> &matches, Mat &disp_out, Mat &actual_disp, vector<KeyPoint> &keypoints_1,
                    vector<KeyPoint> &keypoints_2, long double &running_mse, long double &bad_matches) {

    float delta = 10.0f; // delta for bad matches %

    // calculate disparity for each keypoint, and calculates RMSE and percentage of bad matches
    for (int i=0; i<matches.size(); i++) {
        DMatch m = matches[i];

        int index_query = m.queryIdx;
        int train_query = m.trainIdx;
        KeyPoint left = keypoints_1[index_query];
        KeyPoint right = keypoints_2[train_query];


        float distance = (left.pt.x - right.pt.x);
        float disp = distance*8;

        disp_out.at<uchar>((int) left.pt.y, (int) left.pt.x) = disp;

        int truth_at_pt = (int) actual_disp.at<uchar>((int) left.pt.y, (int) left.pt.x);

        running_mse += pow((distance - (truth_at_pt/8.0)),2.0f);
        if (abs(distance - (truth_at_pt/8)) > delta) {
            bad_matches++;
        }
    }

    // calculate errors
    running_mse = running_mse / matches.size();
    bad_matches = bad_matches / matches.size();
    running_mse = sqrt(running_mse);
}

// performs all analysis (calls above functions) on the images in a folder of our dataset
void analyzeImagePair(string folderName) {
    Mat img_1 = imread("dataset/" + folderName + "/im2.ppm");
   Mat img_2 = imread("dataset/" + folderName + "/im6.ppm");
   Mat actual_disp_unscaled = imread("dataset/" + folderName + "/disp2.pgm", IMREAD_UNCHANGED);
   Mat actual_disp;
   normalize(actual_disp_unscaled, actual_disp, 0.0, 255.0, NORM_MINMAX, CV_16UC1);

   std::vector<KeyPoint> keypoints_1, keypoints_2, keypoints_3, keypoints_4, keypoints_5, keypoints_6;

   // FAST detector
   detector->detect(img_1, keypoints_1);
   detector->detect(img_2, keypoints_2);

   // AGAST detector
   detector2->detect(img_1, keypoints_3);
   detector2->detect(img_2, keypoints_4);

   // SIFT detector
    sift_detector->detect(img_1, keypoints_5);
    sift_detector->detect(img_2, keypoints_6);

    Mat desc1, desc2, desc3, desc4, desc5, desc6;

    std::vector<KeyPoint> &keypoints_1_ref = keypoints_1;
    std::vector<KeyPoint> &keypoints_2_ref = keypoints_2;
    std::vector<KeyPoint> &keypoints_3_ref = keypoints_3;
    std::vector<KeyPoint> &keypoints_4_ref = keypoints_4;
    std::vector<KeyPoint> &keypoints_5_ref = keypoints_5;
    std::vector<KeyPoint> &keypoints_6_ref = keypoints_6;

    // FAST + FREAK descriptors
    desc_comp->compute(img_1, keypoints_1_ref,desc1);
    desc_comp->compute(img_2, keypoints_2_ref, desc2);

    // AGAST + FREAK descriptors
    desc_comp->compute(img_1, keypoints_3_ref,desc3);
    desc_comp->compute(img_2, keypoints_4_ref, desc4);

    // SIFT descriptor
    sift_detector->compute(img_1, keypoints_5_ref,desc5);
    sift_detector->compute(img_2, keypoints_6_ref, desc6);

    // FAST + FREAK matching and disparity calculation
    outfile << endl << "Computing disparity for " << folderName << endl;
    vector<DMatch> good_matches_fast_freak;
    match(desc1, desc2, keypoints_1, keypoints_2, 20, good_matches_fast_freak);
    Mat out_matches1;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches_fast_freak, out_matches1);
    imwrite("output/" + folderName + "_matches_fast_freak.png", out_matches1);

    long double running_mse_fast_freak = 0;
    long double bad_matches_fast_freak = 0;

    Mat disparity_out_fast_freak = cv::Mat::zeros(img_1.size(), CV_16UC1);
    calculateDisp(good_matches_fast_freak, disparity_out_fast_freak, actual_disp, keypoints_1, keypoints_2, running_mse_fast_freak, bad_matches_fast_freak);
    imwrite("output/" + folderName + "_fast_freak_disp.png", disparity_out_fast_freak);
    outfile << "FAST + FREAK RMSE: " << running_mse_fast_freak << endl;
    outfile << "FAST + FREAK Bad matches: " << bad_matches_fast_freak*100 << "%" << endl;

    // AGAST + FREAK matching and disparity calculation
    vector<DMatch> good_matches_agast_freak;
    match(desc3, desc4, keypoints_3, keypoints_4, 20, good_matches_agast_freak);
    Mat out_matches2;
    drawMatches(img_1, keypoints_3, img_2, keypoints_4, good_matches_agast_freak, out_matches2);
    imwrite("output/" + folderName + "_matches_agast_freak.png", out_matches2);

    long double running_mse_agast_freak = 0;
    long double bad_matches_agast_freak = 0;

    Mat disparity_out_agast_freak = cv::Mat::zeros(img_1.size(), CV_16UC1);
    calculateDisp(good_matches_agast_freak, disparity_out_agast_freak, actual_disp, keypoints_3, keypoints_4,
                   running_mse_agast_freak, bad_matches_agast_freak);
    imwrite("output/" + folderName + "_agast_freak_disp.png", disparity_out_agast_freak);
    outfile << "AGAST + FREAK RMSE: " << running_mse_agast_freak << endl;
    outfile << "AGAST + FREAK Bad matches: " << bad_matches_agast_freak*100 << "%" << endl;

    // SIFT matching and disparity calculation
    vector<DMatch> good_matches_sift;
    match(desc5, desc6, keypoints_5, keypoints_6, 20, good_matches_sift);
    Mat out_matches3;
    drawMatches(img_1, keypoints_5, img_2, keypoints_6, good_matches_sift, out_matches3);
    imwrite("output/" + folderName + "_matches_sift.png", out_matches3);

    long double running_mse_sift = 0;
    long double bad_matches_sift = 0;

    Mat disparity_out_sift = cv::Mat::zeros(img_1.size(), CV_16UC1);
    calculateDisp(good_matches_sift, disparity_out_sift, actual_disp, keypoints_5, keypoints_6, running_mse_sift, bad_matches_sift);
    imwrite("output/" + folderName + "_sift_disp.png", disparity_out_sift);
    outfile << "SIFT RMSE: " << running_mse_sift << endl;
    outfile << "SIFT Bad matches: " << bad_matches_sift*100 << "%" << endl;
}

/** @function main */
int main(int argc, char** argv)
{
    outfile.open("output/evaluation_results.txt", ios_base::trunc);
    analyzeImagePair("sawtooth");
    analyzeImagePair("barn1");
    analyzeImagePair("barn2");
    analyzeImagePair("bull");
    analyzeImagePair("venus");
    analyzeImagePair("poster");


    // visualize the matches
//    Mat out_matches;
//    drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, out_matches);
//
//    Mat resized;
//    resize(out_matches, resized, Size(1920,1080));
//    imshow("out", resized);
//    waitKey();
    waitKey();

    return 0;
}
