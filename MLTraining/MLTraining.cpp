// MLTraining.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include "Util.cpp"
#include "Data.cpp"


bool checkIfNewClustersAreDifferent(std::vector<std::vector<double>> new_centroid_centres,
                                    std::vector<data<double>> centroid_centres, int k, 
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

int main()
{
    int32_t k = 0;
    std::vector<std::vector<double>> dataset_array = { {1.0, 1.0},{2.0, 1.0},{4.0, 3.0},{5.0, 4.0} };
    std::vector<data<double>> dataset;
    bool repeatClustering = true;

    int32_t number_of_features = dataset_array[0].size();
    for (int i = 0; i < dataset_array.size(); i++) {


        dataset.push_back(data<double>(dataset_array[i]));

    }

    std::cout << "Enter the number of clusters" << std::endl;
    std::cin >> k;

    /*Distance matrix will have k rows and n (size of dataset) columns */
    uint64_t n = static_cast<int64_t>(dataset.size());

    std::cout << "Size of the input dataset is " << n << std::endl;

    std::vector<std::vector<double>> distance_matrix(k, std::vector<double>(n, 0.0));

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

        for (uint32_t i = 0; i < k; i++) {

        /*Calculate distance between the ith cluster centroid and the jth dataset*/

            for (uint32_t j = 0; j < n; j++) {
                distance_matrix[i][j] = Util::calculateEuclideanDist(centroid_centres[i].features, dataset[j].features);
            }

        }

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
        //std::cout << "Assigning cluster to "<<i<<"th data"<<std::endl;
        for (int j = 0; j < k; j++) {

            if (distance_matrix[j][i] < min) {
                min = distance_matrix[j][i];
                cluster_label = j;
            }

        }
        dataset[i].setClusterInfo(cluster_label);
        cluster_sizes[cluster_label]++;

    }
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

    
#if 0
    for (uint32_t i = 0; i < k; i++) {

        for (int j = 0; j < number_of_features; j++) {

            std::cout << new_centroid_centres[i][j] << " ";

        }
        std::cout << std::endl;
    }
#endif
    /*check if the new centroid centres are the same as the exisitng centroid centres*/
    repeatClustering = checkIfNewClustersAreDifferent(new_centroid_centres, centroid_centres, k, number_of_features);

    std::cout << " repeat cluster check done " << std::endl;
    /*Reassign cluster centres */

    for (int32_t i = 0; i < k; i++) {

        for (int j = 0; j < number_of_features; j++) {
            centroid_centres[i].features[j] = new_centroid_centres[i][j];
        }

    }

#if 0
    std::cout << " Printing new cluster centroids " << std::endl;
    for (uint32_t j = 0; j < k; j++) {
        centroid_centres[j].display();

    }
#endif

}
  
std::cout << " Printing data and their new centres" << std::endl;

for (uint32_t j = 0; j < n; j++) {
    dataset[j].display();

}
    


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
