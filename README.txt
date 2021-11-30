The following project implements Stereo correspondence algorithms using the FAST and AGAST algorithms for feature detection and the FREAK algorithm with each obtained keypoints for feature description.
Then, SIFT feature detector and descriptor was used to evaluate the previously obtained results.

A disparity matrix was produced for each set of descriptors. 
The implemented feature detectors and descriptors were evaluated on the Middlebury 2001 Stereo Vision Dataset. 

To calculate the Root Mean Square Error and the percentage of Bad Matches, I referred to the following paper: 
“A Taxonomy and Evaluation of Dense Two-Frame Stereo Correspondence Algorithms” (International Journal of Computer Vision, 2002). 

Last, all the different images of the dataset were used to evaluate the results of the three different approaches.

To compile and run this project, make sure to open the project in Code-Blocks and then build it and run it. 
