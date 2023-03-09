// MLTraining.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#define BLOCK_SIZE 4
#define DATA_TYPE float

#include <iostream>
#include <vector>
#include <set>
#include <unordered_set>
#include <map>
#include <cfloat>
#include <fstream>
#include <omp.h>
#include "Util.cpp"
#include "Data.cpp"

#pragma omp declare reduction (merge : std::set<uint32_t> : omp_out.insert(omp_in.begin(), omp_in.end()))

//#pragma omp declare reduction (merge : std::vector<uint32_t> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
template<typename T>
static inline std::vector<uint32_t> getNeighbours(uint32_t index,  std::vector<std::vector<T>>& dataunordered_set, 
		                                  T epsilon, uint32_t n, uint32_t number_of_features) {

    std::vector<T> curr_point = dataunordered_set[index];

    std::vector<uint32_t> neighbours;

    std::vector<T> distance_vector(n , 0.0);

    #pragma omp parallel for 
    for (uint32_t i = 0; i < n; i++) {

        distance_vector[i] = Util::calculateEuclideanDist<T>(dataunordered_set[i], curr_point, number_of_features);

    }

    #pragma omp parallel for reduction(merge:neighbours)
    for (uint32_t i = 0; i < n; i++) {

        if(distance_vector[i] <= epsilon)
	    neighbours.push_back(i);

    }

    return neighbours;

}

static inline void mergeNeighbours(std::map<uint32_t, std::set<uint32_t>>& core_points) {

    for(auto& core_point:core_points) {
        
        auto& neighbours1 = core_point.second;
        
        for(auto pt:neighbours1) {

	    if((pt != core_point.first) && (core_points.find(pt) != core_points.end())) {
		
		/*merge */
	        auto& neighbours2 = core_points[pt];

		neighbours1.insert(neighbours2.begin(), neighbours2.end());

		core_points.erase(pt);

	    }

	}	

    }
}


#pragma omp declare reduction (unordered_map_add : std::map<uint32_t,  std::set<uint32_t>> : omp_out.insert(omp_in.begin(), omp_in.end()))
int main(int argc, char** argv)
{
    uint32_t min_points = 0;
    uint32_t num_of_clusters = 0;
    DATA_TYPE epsilon = 0.0;
    uint32_t number_of_features = 0;
    /*Read from a CSV file */
    std::string input_filename = "";
    std::string output_filename = "";
    std::vector<std::vector<DATA_TYPE>> input;
    double start_time, end_time;
  
    start_time = omp_get_wtime();
    try {
    if(argc > 4) {
        input_filename = argv[1];
        std::ifstream csv_file;
        csv_file.open(input_filename);
        input = Util::parseCSVfile<DATA_TYPE>(csv_file);	
        min_points = std::stoi(argv[2]);
	epsilon = std::stod(argv[3]);
	output_filename = argv[4];
    }
    else {
        std::cerr<<"Please provide the correct path to the CSV file, minimum point, epsilon and output file "<<
		    std::endl;
        return 0;
    }

    uint32_t n = static_cast<int32_t>(input.size());

    DATA_TYPE epsilon_square = epsilon * epsilon;

    std::cout << "Size of the input dataset is " << n << std::endl;

    end_time = omp_get_wtime();

    std::cout << "Time to load dataset " << (end_time - start_time) <<"s"<< std::endl;
    if(n > 0) {
     
        number_of_features = input[0].size();

    }


    std::map<uint32_t, std::set<uint32_t>> core_points;
    /*initialize all points as noise points */
    std::vector<uint32_t> cluster_info(n, 0);

    std::vector<uint32_t> neighbour_count(n, 0);
    /* For each point determine the number of neighbourhood points*/
    
    std::cout << "Evaluating core points "<<std::endl;     
   
    start_time = omp_get_wtime(); 

    for (uint32_t i = 0; i < n; i = i + BLOCK_SIZE) {

	/*compute neighbours for BLOCK_SIZE points */
        std::vector<std::vector<DATA_TYPE>> neighbours(BLOCK_SIZE, std::vector<DATA_TYPE>(n, FLT_MAX));

	#pragma omp parallel for
	for(uint32_t j = 0; j < n; j = j + BLOCK_SIZE) {

            for(uint32_t k = 0; (k < BLOCK_SIZE) & ((k + i) < n); k++) {

                for(uint32_t l = 0; (l < BLOCK_SIZE) & ((l + j) < n); l++) {

                    neighbours[k][l + j] = Util::calculateEuclideanDist(input[k + i], input[l + j], 
									number_of_features);


                }

            }
        }

	for(uint32_t k = 0; k < BLOCK_SIZE; k++) {

	    std::set<uint32_t> points;

	    #pragma omp parallel for reduction(merge:points)
	    for(uint32_t l = 0; l < n; l++) {

	       if(neighbours[k][l] <= epsilon_square) {

	            points.insert(l);

	       }

	    }

	    if(points.size() >= min_points)
	        core_points[k + i] = points;


	}

	if((i % 10000) == 0)
            mergeNeighbours(core_points);


    }

#if 0
     for(auto a:core_points) {

       std::cout<<a.first<<" - ";
       for(auto b:a.second) {

           std::cout<<b<<", ";
       }
       std::cout<<std::endl;

    }
#endif

    end_time = omp_get_wtime();

    std::cout << "Time to evaluate core points " << (end_time - start_time) <<"s"<< std::endl;

    start_time = omp_get_wtime();

    mergeNeighbours(core_points);

    end_time = omp_get_wtime();

    std::cout << "Time to merge clusters " << (end_time - start_time) <<"s"<< std::endl;

    std::cout << "Assigning clusters " << std::endl;

    start_time = omp_get_wtime();
    
    for(auto& points_in_cluster:core_points) {

	     num_of_clusters++;

	     for(auto p:points_in_cluster.second) {

                 cluster_info[p] = num_of_clusters; 

             }

    }

    end_time = omp_get_wtime();

    std::cout << "Time to assign clusters " << (end_time - start_time) <<"s"<< std::endl;

    std::cout << " Clustering completed " << std::endl;
    std::cout <<" Number of clusters : "<<num_of_clusters<<std::endl;
    std::cout << "Writing data and their cluster labels to output file "<<output_filename<<std::endl;
    std::cout << "Points with cluster label 0 are Noise points "<<std::endl;

    start_time = omp_get_wtime();
    
    Util::writeToCSVfile(input, cluster_info, output_filename);

    end_time = omp_get_wtime();

    std::cout << "Time to write output to file " << (end_time - start_time) <<"s"<< std::endl;

    
    }
    catch (const std::exception& e) {

            std::cout<<e.what()<<std::endl;

    }

    return 0;
    
}
