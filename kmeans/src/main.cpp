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

int number_samples;

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

    number_samples = (int)all_data.size();
    // Now add all the data into a Mat element
    Mat vect = Mat::zeros(number_samples, (int)all_data[0].size(), CV_32FC1);
    *true_labels = Mat::zeros(number_samples, 1, CV_32S);
    // // Loop over vectors and add the data
    for(int rows = 0; rows < number_samples; rows++){
        for(int cols= 0; cols< (int)all_data[0].size(); cols++){
            vect.at<float>(rows,cols) = all_data[rows][cols]; 
        }
        true_labels->at<int>(rows) = all_labels[rows];
    }

    // cout << "M = " << endl << " " << vect << endl;
    return vect;
}

int evaluate_labels(Mat true_labels, Mat guess_labels)
{
    int true_label_counts[10];
    int guess_label_counts[10];
    // intialize to zero
    for (int i = 0; i < 10; i++)
    {
        true_label_counts[i] = 0;
        guess_label_counts[i] = 0;
    }

    for (int i = 0; i < number_samples; i++)
    {
        true_label_counts[true_labels.at<int>(i)]++;
        guess_label_counts[guess_labels.at<int>(i)]++;
    }
    
    int number_incorrect = 0;
    int i = 0;
    while (true_label_counts[i] > 0)
    {
        number_incorrect += abs(true_label_counts[i] - guess_label_counts[i]);
        i++;
    }
    return (number_samples-(number_incorrect/2));
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
    Mat true_labels;
    Mat points = readCSV(fname, &true_labels);

    double compactness = kmeans(points, clusterCount, guess_labels,
        TermCriteria( TermCriteria::COUNT, 1000, 0),
            3, KMEANS_PP_CENTERS, centers);
    cout << "Compactness: " << compactness << endl;

    end_time = CLOCK();
    cout << "OpemMP execution time: " << end_time - start_time << " ms" << endl;

    int number_incorrect = evaluate_labels(true_labels, guess_labels);
    cout << "Accuracy: " << number_incorrect << "/" << number_samples << endl;

    return 0;
}
