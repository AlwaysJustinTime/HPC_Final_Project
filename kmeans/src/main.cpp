// referenced from openCV Github:
// https://github.com/opencv/opencv/blob/master/samples/cpp/kmeans.cpp

#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ml/ml.hpp"
#include <iostream>
#include <string>
#include <fstream>

#define DATA_CSV "../data/suite_160/cluster_2.csv"

using namespace cv;
using namespace std;


// Timer function
double CLOCK() {
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC,  &t);
	return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}


// to read the given file into a cv::Mat
Mat readCSV(char* filename, Mat* true_labels) {
    ifstream inputfile (filename, ifstream::in);
    string current_line;
    // vector allows you to add data without knowing the exact size beforehand
    vector< vector<double> > all_data;
    vector<int> all_labels;
    // Start reading lines as long as there are lines in the file
    while(getline(inputfile, current_line)){
        // Now inside each line we need to seperate the cols
        vector<double> values;
        int label;
        stringstream temp(current_line);
        string single_value;

        int count = 0;
        while(getline(temp,single_value,',')){
            if (count == 2) {
                label = atoi(single_value.c_str());
            }
            else {
                // convert the string element to a integer value
                values.push_back(atof(single_value.c_str()));
            }
            count += 1;
        }
        // add the row to the complete data vector
        all_data.push_back(values);
        all_labels.push_back(label);
    }

    // Now add all the data into a Mat element
    Mat vect = Mat::zeros((int)all_data.size(), (int)all_data[0].size(), CV_32FC1);
    true_labels = Mat::zeros((int)all_labels.size(), 1, CV_32S);
    // // Loop over vectors and add the data
    for(int rows = 0; rows < (int)all_data.size(); rows++){
        for(int cols= 0; cols< (int)all_data[0].size(); cols++){
            vect.at<float>(rows,cols) = all_data[rows][cols]; 
        }
        true_labels.at<int>(rows) = all_labels[rows];
    }

    // cout << "M = " << endl << " " << vect << endl;
    return vect;
}

int evaluate_labels(mat true_labels, mat guess_labels)
{
    int number_correct = 0;
    for (int i = 0; i < true_labels.size(); i++)
    {
        if (true_labels.at<int>(i) == guess_labels.at<int>(i))
            number_correct++;
    }
    return number_correct;
}

// static void help()
// {
//     cout << "\nThis program demonstrates kmeans clustering.\n"
//             "It generates an image with random points, then assigns a random number of cluster\n"
//             "centers and uses kmeans to move those cluster centers to their representitive location\n"
//             "Call\n"
//             "./kmeans\n" << endl;
// }

int main( int argc, char** argv )
{
    if(argc < 3) {
        cout << "Please enter more arguments" << endl;
        return -1;
    }

    double start_time, end_time;
    start_time = CLOCK();

    Mat guess_labels;
    std::vector<Point2f> centers;
    int i;
    int clusterCount = atoi(argv[1]);
    char* fname= argv[2];
    
    // get points and true  from CSV
    Mat *true_labels
    Mat points = readCSV(fname, true_labels);

    double compactness = kmeans(points, clusterCount, guess_labels,
        TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0),
            3, KMEANS_PP_CENTERS, centers);
    cout << "Compactness: " << compactness << endl;

    int number_correct = evaluate_labels(true_labels, guess_labels);
    cout << "Accuracy: " << number_correct << "/" << true_labels.size() << endl;

    end_time = CLOCK();
    cout << "OpemMP execution time: " << end_time - start_time << " ms" << endl;

    return 0;
}
