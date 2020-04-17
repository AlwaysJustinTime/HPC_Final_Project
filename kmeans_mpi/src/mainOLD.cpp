// mpic++ main.cpp -lm -o main

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <bits/stdc++.h>
#include <iostream>
#include <string>
#include <fstream>


#define MAX_ITER 100

using namespace std;


int DATA_BYTES, CLUSTER_BYTES;
int nCentroids, size;

int clusterCount;
char* fname;
vector< vector<double> > points;
vector<int> labels;
double* x;
double* y;
int* l;





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


double dis2d(double a, double b, double x, double y)
{
	return double(pow((a-x),2)+pow((b-y),2));
}


int assignCluster(int x,int y, double cluster_points_x[],double cluster_points_y[], int K)
{	

	double min=INT_MAX;
    int min_ind;
	for(int j=0;j<K;j++)
	{
		double dist = dis2d(x, y, cluster_points_x[j], cluster_points_y[j]);
		if(dist<min)
		{
			min=dist;
			min_ind = j;
		}
	}
	return min_ind;


}

double percentCorrectLabels(int* origLabels, int* finalLabels, int count) {
    int nCorrect = 0;
    for (int i = 0; i < count; i++) {
        printf("Orig Labels: %d, Final: %d\n", origLabels[i], finalLabels[i]);
        if (origLabels[i] == finalLabels[i]) {
            nCorrect += 1;
        }
    }
    return nCorrect / (count * 1.0);
}



int main(int argc, char** argv)
{
    if(argc < 3) {
        cout << "Please enter more arguments" << endl;
        cout << "<cluster count> <filename>" << endl;
        return -1;
    }
    clusterCount = atoi(argv[1]);
    fname= argv[2];


	MPI_Init(&argc, &argv);
	int num_proc, my_rank;
    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    // printf("%d %d\n", num_proc, my_rank);
    MPI_Status stat;

	// printf("Data loading...\n");
	if (readCSV(&points, &labels, fname)) {
		// printf("Points loaded\n");
	} else {
		// printf("NOT loaded!\n");
		return -1;
	}

	int tag=200;
	int K=clusterCount;
    size = (int)points.size();
    
	DATA_BYTES = size * sizeof(double);
	CLUSTER_BYTES = clusterCount * sizeof(int);

    x = (double*)(malloc(DATA_BYTES));
	y = (double*)(malloc(DATA_BYTES));
    l = (int*)(malloc(size * sizeof(int)));

    for (int i = 0; i < size; i++) {
        x[i] = points.at(i).at(0);
        y[i] = points.at(i).at(1);
        l[i] = labels.at(i);
    }


    if(my_rank==0)
    {
    	
    	// printf("Master node!\n");
        // printf("Size = %d\n", size);
    	int assignment[size];

    	int points_per_node = size/(num_proc-1);

    	double cluster_points_x[K];
        double cluster_points_y[K];
    	for(int i=0;i<K;i++)
    	{
    		int random_pt = rand()%size;
    		cluster_points_x[i] = points.at(random_pt).at(0);
            cluster_points_y[i] = points.at(random_pt).at(1);
    		// printf("%f %f\n", cluster_points_x[i], cluster_points_y[i]);
    		// printf("%d  ", cluster_points[i]);
    	}
    	// printf("\n");
    	for(int iter=0;iter<MAX_ITER;iter++)
    	{
            int start_index=0 , end_index=-1;
            for(int j=0;j<num_proc-2;j++)
            {
                start_index = j*points_per_node;
                end_index = (j+1)*points_per_node-1;
                int num_indices = end_index- start_index+1;
                MPI_Send(&start_index, 1, MPI_INT, j+1, tag, MPI_COMM_WORLD);
                MPI_Send(&end_index, 1, MPI_INT, j+1, tag, MPI_COMM_WORLD);

            }
            end_index+=1;	
            MPI_Send(&end_index, 1, MPI_INT, num_proc-1, tag, MPI_COMM_WORLD);
            // size=180;
            int s = size;
            MPI_Send(&(s), 1, MPI_INT, num_proc-1, tag, MPI_COMM_WORLD);

            
            // for(int i=0;i<K;i++)
            // 	printf("%d  ", cluster_points[i]);
            
            MPI_Bcast(cluster_points_x, K, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(cluster_points_y, K, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            int cluster_sum_x[K], cluster_sum_y[K];
            // int cluster_count_r[K], cluster_count_g[K], cluster_count_b[K];
            int cluster_count[K];
            memset(cluster_count, 0, sizeof(cluster_count));
            // memset(cluster_count_g, 0, sizeof(cluster_count_g));
            // memset(cluster_count_b, 0, sizeof(cluster_count_b));
            memset(cluster_sum_x, 0, sizeof(cluster_sum_x));
            memset(cluster_sum_y, 0, sizeof(cluster_sum_y));

            for (int proc = 1; proc < num_proc; proc++)
            {
                /* code */
                
                for(int i=0;i<K;i++)
                {
                    double sum_x, sum_y, count;  //
                    MPI_Recv(&sum_x, 1, MPI_DOUBLE, proc, tag, MPI_COMM_WORLD, &stat);
                    MPI_Recv(&sum_y, 1, MPI_DOUBLE, proc, tag, MPI_COMM_WORLD, &stat);
                    MPI_Recv(&count, 1, MPI_DOUBLE, proc, tag, MPI_COMM_WORLD, &stat);
                    // MPI_Recv(&count_g, 1, MPI_INT, proc, tag, MPI_COMM_WORLD, &stat);
                    // MPI_Recv(&count_b, 1, MPI_INT, proc, tag, MPI_COMM_WORLD, &stat);
                    cluster_sum_x[i]+=sum_x;
                    cluster_sum_y[i]+=sum_y;
                    cluster_count[i]+=count;
                    // cluster_count_g[i]+=count_g;
                    // cluster_count_b[i]+=count_b;
                    // printf("Cluster, Sum, Count for X: %d %f %f\n",i, cluster_sum_x[i], cluster_count[i]);
                    // printf("Cluster, Sum, Count for Y: %d %f %f\n",i, cluster_sum_y[i], cluster_count[i]);
                }

            }
            // printf("New points:\n");
            for(int i=0;i<K;i++)
            {
                if(cluster_count[i]==0)
    	 		{
    	 			int random_pt = rand()%size;
                    cluster_points_x[i] = x[random_pt];
                    cluster_points_y[i] = y[random_pt];
                    // printf("No points assigned... Assigning cluster randomly\n");
                    continue;
    	 		}
                cluster_points_x[i] = cluster_sum_x[i]/cluster_count[i];
                cluster_points_y[i] = cluster_sum_y[i]/cluster_count[i];
                // printf("%d : %f %f\n", i ,cluster_points_x[i], cluster_points_y[i]);
            }


            if(iter== (MAX_ITER - 1))
            {
                int sti, endi ;
                for(int i=1;i<num_proc;i++){
                    MPI_Recv(&sti, 1, MPI_INT, i, tag, MPI_COMM_WORLD, &stat);
                    MPI_Recv(&endi, 1, MPI_INT, i, tag, MPI_COMM_WORLD, &stat);
                    int assignment_k[endi-sti+1];
                    MPI_Recv(&assignment_k, endi-sti+1, MPI_INT, i, tag, MPI_COMM_WORLD, &stat);
                    for(int j=sti;j<=endi;j++)
                    {
                        assignment[j] = assignment_k[j-sti];
                        // printf("%d ---- %d\n", j, assignment[j]);
                    }

                }
                // writeRawImage(outputFile, assignment, cluster_points_r,  cluster_points_g, cluster_points_b, size);
            }



    	}
    	// printf("Answer:\n");
    	for(int i=0;i<K;i++)
    	{
    		// printf("%f %f\n", cluster_points_x[i], cluster_points_y[i]);
    	}
    	// MPI_Recv(&sum, sizeof(int *), MPI_INT, 1, 200, MPI_COMM_WORLD, &stat);
    	// printf("%d at Master\n", sum);

        double pCorrect = percentCorrectLabels(l, assignment, size);
        printf("PERCENT CORRECT = %f\n", pCorrect);
    }

    else
    {
    	// printf("In slave: %d\n", my_rank);
    	for(int iter=0;iter<MAX_ITER;iter++)
    	{
	    	int start, end;
	    	MPI_Recv(&start, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &stat);
	    	MPI_Recv(&end, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &stat);

	    	double cluster_points_x[K], cluster_points_y[K];
	    	MPI_Bcast(cluster_points_x, K, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	    	MPI_Bcast(cluster_points_y, K, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	    	// printf("Here already\n");
	    	map<int,  vector<int> > cluster;
	    	int assignment[end-start+1];
	    	for(int i=start;i<=end;i++)
	    	{
	    		// has to be changed
	    		// printf("Assigning cluster to %f %f...\n", x[i], y[i]);
	    		int c = assignCluster(x[i], y[i], cluster_points_x, cluster_points_y , K);
	    		// 
	    		// printf("%d---%d\n", i, c);
	    		if(cluster.find(c)==cluster.end())
	            {    vector<int> v;
	                 v.push_back(i);
	                 cluster.insert(make_pair(c,v));
	            }
	            else
	            cluster[c].push_back(i);

	        	assignment[i-start] = c;
	    	}

	    	 map<int,vector<int> >::iterator it;


	    	 for(int i=0;i<K;i++)
	    	 {
	    	 	double sum_x=0, sum_y=0, count=0;
	    	 	{
	    	 		for(int j=0;j<cluster[i].size();j++)
	    	 		{
	    	 			sum_x+=x[cluster[i][j]];
	    	 			sum_y+=y[cluster[i][j]];
	    	 		}
	    	 		count = cluster[i].size();
	    	 	}
	    	 	MPI_Send(&sum_x, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
	    	 	MPI_Send(&sum_y, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);

	    	 	MPI_Send(&count, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);


	    	 }


	    	if(iter== (MAX_ITER - 1))
	    	{
	    		MPI_Send(&start, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);
	    		MPI_Send(&end, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);
	    		MPI_Send(assignment, end-start+1, MPI_INT, 0, tag, MPI_COMM_WORLD);
	    	} 
	        // cout<<"Map key and Map value-size:"<<endl;

	        // for(it=cluster.begin();it!=cluster.end();it++)
	        // {  
	        //     cout<<it->first<<":"<<it->second.size() << endl;

	        //     // for(int j=0;j<it->second.size();j++)
	        //     //   cout<<it->second[j]<<" ";
	        //     // cout<<endl;   
	        // }




 		}
	}
    MPI_Finalize();

}