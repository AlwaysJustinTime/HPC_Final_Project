#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <errno.h>
#include <mpi.h>
#include <limits.h>
#include <bits/stdc++.h>
#include <iostream>
#include <string>
#include <fstream>

#define MAX_ITERATIONS 1000

using namespace std;

int numOfClusters;
int numOfElements;
int num_of_processes;

char* fname = NULL;
vector< vector<double> > points;
vector<int> labels;


// Timer function
double CLOCK() {
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC,  &t);
	return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}

// to get the percentage of correct labels from the given 2 arrays
double percentCorrectLabels(int* origLabels, int* finalLabels, int count) {
    int nCorrect = 0;
    for (int i = 0; i < count; i++) {
        // printf("Orig Labels: %d, Final: %d\n", origLabels[i], finalLabels[i]);
        if (origLabels[i] == finalLabels[i]) {
            nCorrect += 1;
        }
    }
    return nCorrect / (count * 1.0);
}



// to read the given file into a cv::Mat
bool readCSV(vector< vector<double> >* data, vector<int>* labels, char* filename) {
    ifstream inputfile (filename, ifstream::in);
    if (!inputfile) {
        return false;
    }
    string current_line;
    // vector allows you to add data without knowing the exact size beforehand
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
        data->push_back(values);
        labels->push_back(label);
    }
    return true;
}

/* This function goes through that data points and assigns them to a cluster */
void assign2Cluster(double k_x[], double k_y[], double recv_x[], double recv_y[], int assign[])
{
	double min_dist = 10000000;
	double x=0, y=0, temp_dist=0;
	int k_min_index = 0;

	for(int i = 0; i < (numOfElements/num_of_processes) + 1; i++)
	{
		for(int j = 0; j < numOfClusters; j++)
		{
			x = abs(recv_x[i] - k_x[j]);
			y = abs(recv_y[i] - k_y[j]);
			temp_dist = sqrt((x*x) + (y*y));

			// new minimum distance found
			if(temp_dist < min_dist)
			{
				min_dist = temp_dist;
				k_min_index = j;
			}
		}

		// update the cluster assignment of this data points
		assign[i] = k_min_index;
	}

}

/* Recalcuate k-means of each cluster because each data point may have
   been reassigned to a new cluster for each iteration of the algorithm */
void calcKmeans(double k_means_x[], double k_means_y[], double data_x_points[], double data_y_points[], int k_assignment[])
{
	double total_x = 0;
	double total_y = 0;
	int numOfpoints = 0;

	for(int i = 0; i < numOfClusters; i++)
	{
		total_x = 0;
		total_y = 0;
		numOfpoints = 0;

		for(int j = 0; j < numOfElements; j++)
		{
			if(k_assignment[j] == i)
			{
				total_x += data_x_points[j];
				total_y += data_y_points[j];
				numOfpoints++;
			}
		}

		if(numOfpoints != 0)
		{
			k_means_x[i] = total_x / numOfpoints;
			k_means_y[i] = total_y / numOfpoints;
		}
	}

}

int main(int argc, char *argv[])
{
	// initialize the MPI environment
	MPI_Init(NULL, NULL);

	// get number of processes
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// get rank
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	// send buffers
	double *k_means_x = NULL;		// k means corresponding x values
	double *k_means_y = NULL;		// k means corresponding y values
	int *k_assignment = NULL;		// each data point is assigned to a cluster
    int *orig_labels = NULL;
	double *data_x_points = NULL;
	double *data_y_points = NULL;

	// receive buffer
	double *recv_x = NULL;
	double *recv_y = NULL;
	int *recv_assign = NULL;

    double start_time, end_time;

	if(world_rank == 0)
	{
		if(argc != 3)
		{
            cout << "Please enter more arguments" << endl;
            cout << "<cluster count> <filename>" << endl;
			exit(-1);
		}

        start_time = CLOCK();



        numOfClusters = atoi(argv[1]);
        fname= argv[2];
        num_of_processes = world_size;
        printf("Clusters: %d, FName: %s, Num Proc: %d\n", numOfClusters, fname, num_of_processes);


		// broadcast the number of clusters to all nodes
		MPI_Bcast(&numOfClusters, 1, MPI_INT, 0, MPI_COMM_WORLD);

		// allocate memory for arrays
		k_means_x = (double *)malloc(sizeof(double) * numOfClusters);
		k_means_y = (double *)malloc(sizeof(double) * numOfClusters);

		if(k_means_x == NULL || k_means_y == NULL)
		{
			perror("malloc");
			exit(-1);
		}

		printf("Reading input data from file...\n\n");
        if (readCSV(&points, &labels, fname)) {
            // printf("Points loaded\n");
        } else {
            // printf("NOT loaded!\n");
            return -1;
        }

		// count number of lines to find out how many elements
		numOfElements = points.size();

		printf("There are a total number of %d elements in the file.\n", numOfElements);

		// broadcast the number of elements to all nodes
		MPI_Bcast(&numOfElements, 1, MPI_INT, 0, MPI_COMM_WORLD);

		// allocate memory for an array of data points
		data_x_points = (double *)malloc(sizeof(double) * numOfElements);
		data_y_points = (double *)malloc(sizeof(double) * numOfElements);
		k_assignment = (int *)malloc(sizeof(int) * numOfElements);
        orig_labels = (int *)malloc(sizeof(int) * numOfElements);

		if(data_x_points == NULL || data_y_points == NULL || k_assignment == NULL)
		{
			perror("malloc");
			exit(-1);
		}

        for (int i = 0; i < numOfElements; i++) {
            data_x_points[i] = points.at(i).at(0);
            data_y_points[i] = points.at(i).at(1);
            k_assignment[i] = labels.at(i);
            orig_labels[i] = labels.at(i);
        }

		// randomly select initial k-means
		time_t t;
		srand((unsigned) time(&t));
		int random;
		for(int i = 0; i < numOfClusters; i++) {
			random = rand() % numOfElements;
			k_means_x[i] = data_x_points[random];
			k_means_y[i] = data_y_points[random];
		}

		printf("Running k-means algorithm for %d iterations...\n\n", MAX_ITERATIONS);
		for(int i = 0; i < numOfClusters; i++)
		{
			printf("Initial K-means: (%f, %f)\n", k_means_x[i], k_means_y[i]);
		}

		// allocate memory for receive buffers
		recv_x = (double *)malloc(sizeof(double) * ((numOfElements/num_of_processes) + 1));
		recv_y = (double *)malloc(sizeof(double) * ((numOfElements/num_of_processes) + 1));
		recv_assign = (int *)malloc(sizeof(int) * ((numOfElements/num_of_processes) + 1));

		if(recv_x == NULL || recv_y == NULL || recv_assign == NULL)
		{
			perror("malloc");
			exit(-1);
		}
	}
	else
	{	// I am a worker node

		num_of_processes = world_size;

		// receive broadcast of number of clusters
		MPI_Bcast(&numOfClusters, 1, MPI_INT, 0, MPI_COMM_WORLD);

		// receive broadcast of number of elements
		MPI_Bcast(&numOfElements, 1, MPI_INT, 0, MPI_COMM_WORLD);

		// allocate memory for arrays
		k_means_x = (double *)malloc(sizeof(double) * numOfClusters);
		k_means_y = (double *)malloc(sizeof(double) * numOfClusters);

		if(k_means_x == NULL || k_means_y == NULL)
		{
			perror("malloc");
			exit(-1);
		}

		// allocate memory for receive buffers
		recv_x = (double *)malloc(sizeof(double) * ((numOfElements/num_of_processes) + 1));
		recv_y = (double *)malloc(sizeof(double) * ((numOfElements/num_of_processes) + 1));
		recv_assign = (int *)malloc(sizeof(int) * ((numOfElements/num_of_processes) + 1));

		if(recv_x == NULL || recv_y == NULL || recv_assign == NULL)
		{
			perror("malloc");
			exit(-1);
		}
	}

	/* Distribute the work among all nodes. The data points itself will stay constant and
	   not change for the duration of the algorithm. */
	MPI_Scatter(data_x_points, (numOfElements/num_of_processes) + 1, MPI_DOUBLE,
		recv_x, (numOfElements/num_of_processes) + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	MPI_Scatter(data_y_points, (numOfElements/num_of_processes) + 1, MPI_DOUBLE,
		recv_y, (numOfElements/num_of_processes) + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	int count = 0;
	while(count < MAX_ITERATIONS)
	{
		// broadcast k-means arrays
		MPI_Bcast(k_means_x, numOfClusters, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(k_means_y, numOfClusters, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		// scatter k-cluster assignments array
		MPI_Scatter(k_assignment, (numOfElements/num_of_processes) + 1, MPI_INT,
			recv_assign, (numOfElements/num_of_processes) + 1, MPI_INT, 0, MPI_COMM_WORLD);

		// assign the data points to a cluster
		assign2Cluster(k_means_x, k_means_y, recv_x, recv_y, recv_assign);

		// gather back k-cluster assignments
		MPI_Gather(recv_assign, (numOfElements/num_of_processes)+1, MPI_INT,
			k_assignment, (numOfElements/num_of_processes)+1, MPI_INT, 0, MPI_COMM_WORLD);

		// let the root process recalculate k means
		if(world_rank == 0)
		{
			calcKmeans(k_means_x, k_means_y, data_x_points, data_y_points, k_assignment);
			//printf("Finished iteration %d\n",count);
		}

		count++;
	}

	if(world_rank == 0)
	{
		printf("--------------------------------------------------\n");
		printf("FINAL RESULTS:\n");
		for(int i = 0; i < numOfClusters; i++)
		{
			printf("Cluster #%d: (%f, %f)\n", i, k_means_x[i], k_means_y[i]);
		}

		end_time = CLOCK();
        cout << "MPI execution time: " << end_time - start_time << " ms" << endl;
        printf("--------------------------------------------------\n");
	}


	//MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

}