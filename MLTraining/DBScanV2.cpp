// MLTraining.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

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

template<typename T>
static inline uint32_t computeKey(std::vector<T> point, std::vector<uint32_t> swapped_dimensions, std::vector<T> dimensions,
		                  std::vector<T> mins, T epsilon) {

    uint32_t key = 0;
    uint32_t accumulator = 1;

    for(auto d : swapped_dimensions) {

        size_t index = static_cast<size_t>(std::floor((point[d] - mins[d]) / (2.0 * epsilon)));
        key += index * accumulator;
        accumulator *= dimensions[d];

    }

    return key;

}


#pragma omp declare reduction (merge : std::vector<uint32_t> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
template<typename T>
static inline void getNeighbours(std::vector<uint32_t>& neighbours, 
		                 const uint32_t& index,  
				 std::vector<std::vector<T>>& dataunordered_set, 
		                 uint32_t& epsilon, uint32_t& number_of_features, 
				 std::vector<bool>& visited,
				 std::unordered_map<uint32_t, std::set<uint32_t>>& nearest_neighbours,
				 std::vector<uint32_t>& point_key_map,
				 std::map<uint32_t, std::vector<uint32_t>>& spatial_index) {

    std::vector<uint32_t> vals;

    uint32_t& cell_key = point_key_map[index];

    std::set<uint32_t>& neighbouring_keys = nearest_neighbours[cell_key];
    
    for(auto& pt : neighbouring_keys) {

        auto& points = spatial_index[pt];
	vals.insert(vals.end(), points.begin(), points.end());

    }

    const uint32_t& n = vals.size();

    std::vector<T>& curr_point = dataunordered_set[index];

    #pragma omp parallel for reduction(merge:neighbours)
    for (uint32_t i = 0; i < n; i++) {

        if(!visited[vals[i]] && ( 
	    Util::asuint32(Util::calculateEuclideanDist<T>(dataunordered_set[vals[i]], curr_point, number_of_features))
	        <= epsilon)) {
	        neighbours.push_back(vals[i]);
	        visited[vals[i]] = true;

	}

    }

}

#pragma omp declare reduction (unordered_map_add : std::map<uint32_t,  std::set<uint32_t>> : omp_out.insert(omp_in.begin(), omp_in.end()))

#pragma omp declare reduction (merge_set : std::set<uint32_t> : omp_out.insert(omp_in.begin(), omp_in.end()))

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

    std::map<uint32_t, std::vector<uint32_t>> spatial_index;

    for(uint32_t i = 0; i < n; i++) {

        for(uint32_t j = 0; j < number_of_features; j++) {

	    if(input[i][j] < mins[j])
	        mins[j] = input[i][j];

	    if(input[i][j] > maxs[j])
	        maxs[j] = input[i][j];

	}

    }

    /*Compute cell dimensions */
    for(uint32_t j = 0; j < number_of_features; j++) {

	dimensions[j] = std::ceil((maxs[j] - mins[j]) / (epsilon)) + 1;
	total_cells *= dimensions[j];

    }

    std::cout<<"Total cells  "<<total_cells<< std::endl;

    std::vector<uint32_t> m_swapped_dimensions(number_of_features, 0);
    /*SWAP dimensions */
    std::iota(m_swapped_dimensions.begin(), m_swapped_dimensions.end(), 0);
    
    // swap the dimensions descending by their cell sizes
    std::sort(m_swapped_dimensions.begin(), m_swapped_dimensions.end(), [&] (size_t a, size_t b) {
            return dimensions[a] < dimensions[b];
    });

    std::unordered_map<uint32_t, std::set<uint32_t>> neighbour_keys;

    std::vector<uint32_t> point_key_map(n, 0);

    for(uint32_t i = 0; i < n; i++) {

	auto key = computeKey(input[i], m_swapped_dimensions, dimensions, mins, epsilon);

	spatial_index[key].push_back(i);

	point_key_map[i] = key;

    }

    std::cout<<"Keys computed"<<std::endl;

     std::map<uint32_t,std::pair<uint32_t, uint32_t>> m_cell_index;

    /*Compute cell index */
    uint32_t accumulator = 0;

        // sum up the offset into the points array
    for (auto& cell : spatial_index) {
            auto& index  = m_cell_index[cell.first];
            index.first  = accumulator;
            index.second = cell.second.size();
            accumulator += cell.second.size();
    
    }

    // introduce an end dummy
    m_cell_index[total_cells].first  = spatial_index.size();
    m_cell_index[total_cells].second = 0;
    
    
    /*compute neighbouring cells */

    for(auto& cell : spatial_index) {
    
        std::vector<uint32_t> neighboring_cells;
        neighboring_cells.reserve(std::pow(3, number_of_features));
        neighboring_cells.push_back(cell.first);

        // cell accumulators
        uint32_t cells_in_lower_space = 1;
        uint32_t cells_in_current_space = 1;
        uint32_t number_of_points = m_cell_index.find(cell.first)->second.second;

        // fetch all existing neighboring cells
        for (size_t d : m_swapped_dimensions) {
             cells_in_current_space *= dimensions[d];

            for (size_t i = 0, end = neighboring_cells.size(); i < end; ++i) {
                const uint32_t current_cell = neighboring_cells[i];

                // check "left" neighbor - a.k.a the cell in the current dimension that has a lower number
                const uint32_t left = current_cell - cells_in_lower_space;
                const auto found_left = m_cell_index.find(left);
                if (current_cell % cells_in_current_space >= cells_in_lower_space) {
                    neighboring_cells.push_back(left);
                    number_of_points += found_left != m_cell_index.end() ? found_left->second.second : 0;
                }
                // check "right" neighbor - a.k.a the cell in the current dimension that has a higher number
                const uint32_t right = current_cell + cells_in_lower_space;
                const auto found_right = m_cell_index.find(right);
                if (current_cell % cells_in_current_space < cells_in_current_space - cells_in_lower_space) {
                    neighboring_cells.push_back(right);
                    number_of_points += found_right != m_cell_index.end() ? found_right->second.second : 0;
                }
            }
            cells_in_lower_space = cells_in_current_space;
       }

       for(auto pt:neighboring_cells) {

           neighbour_keys[cell.first].insert(pt);

       }

    }

    
    uint32_t epsilon_hex = Util::asuint32(epsilon_square);
    
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

		uint32_t& index = compute[0];

		compute.erase(compute.begin());

		getNeighbours(neighbours, index, input, epsilon_hex, number_of_features, visited, 
		              neighbour_keys, point_key_map, spatial_index);

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
    
    end_time = omp_get_wtime();

    std::cout << "Time to evaluate core points " << (end_time - start_time) <<"s"<< std::endl;

    std::cout << "Assigning clusters " << std::endl;

    start_time = omp_get_wtime();
    
    for(auto& points_in_cluster:core_points) {

        if(points_in_cluster.second.size() >= min_points) {

	    for(auto p:points_in_cluster.second) {

                cluster_info[p] = points_in_cluster.first; 

            }
	 }

    }

    end_time = omp_get_wtime();

    std::cout << "Time to assign clusters " << (end_time - start_time) <<"s"<< std::endl;

    std::cout << " Clustering completed " << std::endl;
    std::cout <<" Number of clusters : "<<num_clusters<<std::endl;
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
