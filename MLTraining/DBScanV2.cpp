// MLTraining.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#define BLOCK_SIZE 4
#define DATA_TYPE float

#include <iostream>
#include <vector>
#include <numeric>
#include <set>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <queue>
#include <cfloat>
#include <fstream>
#include <omp.h>
#include "Util.cpp"
#include "Data.cpp"
#include <climits>

//#pragma omp declare reduction (merge : std::set<uint32_t> : omp_out.insert(omp_in.begin(), omp_in.end()))

#pragma omp declare reduction (merge : std::vector<uint32_t> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
template<typename T>
static inline std::vector<uint32_t> getNeighbours(uint32_t index,  std::vector<std::vector<T>>& dataunordered_set, 
		                                  uint32_t epsilon, uint32_t n, uint32_t number_of_features, 
						  std::vector<bool>& visited,
						  std::unordered_map<uint32_t, std::vector<uint32_t>>& nearest_neighbours) {

    std::vector<uint32_t> neighbours;
    
    /*Get all neighbours of that point so that you dont have to compute neighbours to that point again */

    #pragma omp parallel for reduction(merge:neighbours)
    for(auto& pt : nearest_neighbours[index]) {

        if(!visited[pt]) {
            neighbours.push_back(pt);
            visited[pt] = true;
        }

    }

    /*we no longer need to store the neighbours */
    nearest_neighbours.erase(index);

    std::vector<T> curr_point = dataunordered_set[index];

    #pragma omp parallel for reduction(merge:neighbours)
    for (uint32_t i = 0; i < n; i++) {

        if(!visited[i] && ( 
	    Util::asuint32(Util::calculateEuclideanDist<T>(dataunordered_set[i], curr_point, number_of_features))
	        <= epsilon)) {
	        neighbours.push_back(i);
	        visited[i] = true;

	}

    }

    return neighbours;

}

#pragma omp declare reduction (unordered_map_add : std::map<uint32_t,  std::set<uint32_t>> : omp_out.insert(omp_in.begin(), omp_in.end()))

int main(int argc, char** argv) {
    
    uint32_t min_points = 0;
    DATA_TYPE epsilon = 0.0;
    uint32_t number_of_features = 0;
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
    uint32_t total_cells = 1;

    DATA_TYPE epsilon_square = epsilon * epsilon;

    std::cout << "Size of the input dataset is " << n << std::endl;

    end_time = omp_get_wtime();

    std::cout << "Time to load dataset " << (end_time - start_time) <<"s"<< std::endl;
    if(n > 0) {
     
        number_of_features = input[0].size();

    }

    std::vector<DATA_TYPE> mins(number_of_features, FLT_MAX);
    std::vector<DATA_TYPE> maxs(number_of_features, 0.0);
    std::vector<DATA_TYPE> dimensions(number_of_features, 0.0);

    std::map<uint32_t, std::vector<DATA_TYPE>> spatial_index;

    for(uint32_t i = 0; i < n; i++) {

        for(uint32_t j = 0; j < number_of_features; j++) {

	    if(input[i][j] < mins[j])
	        mins[j] = input[i][j];

	    if(input[i][j] > maxs[j])
	        maxs[j] = input[i][j];

	}

    }

    for(uint32_t j = 0; j < number_of_features; j++) {

	dimensions[j] = std::ceil((maxs[j] - mins[j]) / (2.0 * epsilon)) + 1;
	total_cells *= dimensions[j];

    }

    std::cout<<"Total cells  "<<total_cells<< std::endl;

    std::vector<DATA_TYPE> m_swapped_dimensions(number_of_features, 0.0);
    /*SWAP dimensions */
    std::iota(m_swapped_dimensions.begin(), m_swapped_dimensions.end(), 0);
    
    // swap the dimensions descending by their cell sizes
    std::sort(m_swapped_dimensions.begin(), m_swapped_dimensions.end(), [&] (size_t a, size_t b) {
            return dimensions[a] < dimensions[b];
    });

     for(uint32_t i = 0; i < n; i++) {

	uint32_t key = 0;
	uint32_t accumulator = 1;

        for(uint32_t j = 0; j < number_of_features; j++) {

            size_t index = static_cast<size_t>(std::floor((input[i][j] - mins[j]) / (2.0 * epsilon)));
            key += index * accumulator;
            accumulator *= dimensions[j];
            
        }	    
	
	spatial_index[key].push_back(i);

    }

    std::unordered_map<uint32_t, std::vector<uint32_t>> nearest_neighbours;

    uint32_t epsilon_hex = Util::asuint32(epsilon_square);
    
    /* Reorder the cells */
    std::vector<std::vector<DATA_TYPE>> reordered_input;
    uint32_t reordered_indices = 0;
     std::unordered_map<uint32_t, uint32_t> map_to_original_index;

    for(auto index:spatial_index) {

        auto& vals = index.second;

	for(uint32_t i = 0; i < vals.size(); i++) {

	    reordered_input.push_back(input[vals[i]]);
	    map_to_original_index[reordered_indices] = vals[i];
	    reordered_indices++;

	}

    }

    input.swap(reordered_input);

    /*Find points within epsilon distance within a cell */
    for(auto index:spatial_index) {

        auto& vals = index.second;

        for(uint32_t i = 0; i < vals.size(); i++) {

	    std::vector<uint32_t>& neighbours =  nearest_neighbours[vals[i]];
	    #pragma omp parallel for reduction(merge:neighbours)
	    for(uint32_t k = i + 1; k < vals.size(); k++) {

                if( Util::asuint32(Util::calculateEuclideanDist<DATA_TYPE>(input[vals[i]], 
					                   input[vals[k]], number_of_features)) 
		                                            <= epsilon_hex) {
                   
		    neighbours.push_back(vals[k]);
		    #pragma omp critical
		    nearest_neighbours[vals[k]].push_back(vals[i]);
		}
	    
	    }
	}

    }

    std::unordered_map<uint32_t, std::vector<uint32_t>> core_points;
    /*initialize all points as noise points */
    std::vector<bool> visited(n, false);

    std::vector<uint32_t> cluster_info(n, 0);
    /* For each point determine the number of neighbourhood points*/
    
    std::cout << "Evaluating core points "<<std::endl;     
   
    start_time = omp_get_wtime(); 

    uint32_t num_clusters = 0;

    for (uint32_t i = 0; i < n; i++) {

	if(!visited[i]) { 
        
	    std::vector<uint32_t> compute;
            std::vector<uint32_t> cluster;

	    compute.push_back(i);

	    cluster.push_back(i);

	    visited[i] = true;

	    while(!compute.empty()) {
		   
		std::vector<uint32_t> neighbours;

		uint32_t index = compute[0];

		compute.erase(compute.begin());

		neighbours = getNeighbours(index, input, epsilon_hex, 
				            n, number_of_features, visited, nearest_neighbours);

		//#pragma omp parallel for reduction(merge:compute) reduction(merge:cluster)
		compute.insert(compute.end(), neighbours.begin(), neighbours.end());

		cluster.insert(cluster.end(), neighbours.begin(), neighbours.end());

	    }

	    if(cluster.size() >= min_points) {
	        
		core_points[++num_clusters] = cluster;
		
		std::cout<<"cluster "<<num_clusters<<" has "<<cluster.size()<<" points"<<std::endl;

	    }

	}

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

    std::cout << "Assigning clusters " << std::endl;

    start_time = omp_get_wtime();
    
    for(auto& points_in_cluster:core_points) {

	     for(auto p:points_in_cluster.second) {

                 cluster_info[map_to_original_index[p]] = points_in_cluster.first; 

             }

    }

    end_time = omp_get_wtime();

    std::cout << "Time to assign clusters " << (end_time - start_time) <<"s"<< std::endl;

    std::cout << " Clustering completed " << std::endl;
    std::cout <<" Number of clusters : "<<core_points.size()<<std::endl;
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
