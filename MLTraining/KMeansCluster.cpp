// MLTraining.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include <cfloat>
#include <fstream>
#include <omp.h>
#include "Util.cpp"
#include "Data.cpp"


bool checkIfNewClustersAreDifferent(std::vector<std::vector<double>>& new_centroid_centres,
                                    std::vector<data<double>>& centroid_centres, int k, 
                                    int number_of_features) {

    for (uint32_t i = 0; i < k; i++) {

        for (int j = 0; j < number_of_features; j++) {

            if (new_centroid_centres[i][j] != centroid_centres[i].features[j]) {
                return true;
            }

        }

    }
    return false;
}

int main(int argc, char** argv)
{
    int32_t k = 0;
    uint64_t iter = 0;
    /*Read from a CSV file */
    std::string input_filename = "";
    std::string output_filename = "";
    std::vector<std::vector<double>> input;// = { {1.0, 1.0},{2.0, 1.0},{4.0, 3.0},{5.0, 4.0} };
    if(argc > 3) {
        input_filename = argv[1];
        std::ifstream csv_file;
        csv_file.open(input_filename);
        input = Util::parseCSVfile(csv_file);	
        k = std::stoi(argv[2]);
	output_filename = argv[3];
    }
    else {
        std::cerr<<"Please provide the correct path to the CSV file and the number of clusters "<<std::endl;
        return 0;
    }

    std::vector<data<double>> dataset;
    bool repeatClustering = true;

    int32_t number_of_features = input[0].size();
    for (int i = 0; i < input.size(); i++) {


        dataset.push_back(data<double>(input[i]));

    }

    /*Distance matrix will have k rows and n (size of dataset) columns */
    uint64_t n = static_cast<int64_t>(dataset.size());

    std::cout << "Size of the input dataset is " << n << std::endl;

    /*K centroid points for k clusters. Each centroid is has n dimensions which is equal to that of the features in a dataset
    So each centroid is of type data*/
    std::vector<data<double>> centroid_centres;
    /*Assign a label to each cluster*/
    /*Let the first k points of the dataset be the intital cluster centroid points */
    for (int32_t i = 0; i < k; i++) {

        centroid_centres.push_back(dataset[i]);

    }
    
    
    /* Compute distance of each datapoint with each of the cluster centroid and update it in the distance matrix*/

    /*For each of the K cluster centroids */

    while(repeatClustering) {
        
        /*Computing new clusters */
        std::vector<uint64_t> cluster_sizes(k, 0);

	iter++;

        for (uint32_t i = 0; i < n; i++) {

        /*Calculate distance between the ith dataset and jth cluster centroid and assign to the closest centroid*/
            std::vector<double>& data_vector = dataset[i].features;
	    int cluster_label = 0;
            double min = DBL_MAX;
 
	    for (uint32_t j = 0; j < k; j++) {
		double distance = Util::calculateEuclideanDist(centroid_centres[j].features, data_vector);
                if(distance < min) {
		    cluster_label = j;
		    min = distance;

		 }
            }
	    dataset[i].setClusterInfo(cluster_label);
            cluster_sizes[cluster_label]++;

        }
#if 0
        std::cout << " printing distance matrix " << std::endl;
        for (uint32_t i = 0; i < k; i++) {

            /*Calculate distance between the ith cluster centroid and the jth dataset*/

            for (uint32_t j = 0; j < n; j++) {
                std::cout<<distance_matrix[i][j]<<" ";
            }
            std::cout<<std::endl;

        }

    /*assign each dataset to the nearest centroid */

    for (uint32_t i = 0; i < n; i++) {

        int cluster_label = 0;
        double min = DBL_MAX;

        /*check distance matrix column wise and assign the point to the closest centroid*/
        //distance matrix has k rows and n columns
        for (int j = 0; j < k; j++) {

            if (distance_matrix[j][i] < min) {
                min = distance_matrix[j][i];
                cluster_label = j;
            }

        }
        dataset[i].setClusterInfo(cluster_label);
        cluster_sizes[cluster_label]++;

    }
#endif
    /*
    for (uint32_t j = 0; j < n; j++) {
        dataset[j].display();

    }
    */
    /*compute centres of clusters */
    std::vector<std::vector<double>> new_centroid_centres(k, std::vector<double>(number_of_features, 0));

    /*Compute centroid of clusters based on the datapoints assigned to them*/
    for (uint32_t i = 0; i < n; i++) {

        int32_t cluster_name = dataset[i].getClusterInfo();
        for (int j = 0; j < number_of_features; j++) {

            new_centroid_centres[cluster_name][j] += dataset[i].features[j];
        }

    }

    for (uint32_t i = 0; i < k; i++) {

        //std::cout << cluster_sizes[i] << " " << std::endl;

        for (int j = 0; j < number_of_features; j++) {

            new_centroid_centres[i][j] = new_centroid_centres[i][j] / cluster_sizes[i];

        }
    }

    /*check if the new centroid centres are the same as the exisitng centroid centres*/
    repeatClustering = checkIfNewClustersAreDifferent(new_centroid_centres, centroid_centres, k, number_of_features);

    /*Reassign cluster centres */

    for (int32_t i = 0; i < k; i++) {

        for (int j = 0; j < number_of_features; j++) {
            centroid_centres[i].features[j] = new_centroid_centres[i][j];
        }

    }

}

std::cout << " Clustering completed " << std::endl;
std::cout <<" Number of iterations : "<<iter<<std::endl;
std::cout << "Writing data and their cluster labels to output file "<<output_filename<<std::endl;

Util::writeToCSVfile(dataset, output_filename);

return 0;
    
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
