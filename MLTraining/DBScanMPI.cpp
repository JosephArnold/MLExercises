// MLTraining.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#define BLOCK_SIZE 8
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
#include <mpi.h>

template<typename T>
static inline uint64_t computeKey(const std::vector<T>& point, const std::vector<uint64_t>& swapped_dimensions, 
		                  const std::vector<T>& dimensions,
		                  const std::vector<T>& mins, const T epsilon) {

    uint64_t key = 0;
    uint64_t accumulator = 1;

    for(auto d : swapped_dimensions) {

        size_t index = static_cast<size_t>(std::floor((point[d] - mins[d]) / (epsilon)));
        key += index * accumulator;
        accumulator *= dimensions[d];

    }

    return key;

}

template<typename T>
static inline void compute_keys(std::vector<Data<T>>& data_set, const uint64_t& n,
				std::map<uint64_t, std::set<uint64_t>>& spatial_index,
		                const std::vector<uint64_t>& swapped_dimensions, 
				const std::vector<T>& dimensions, 
		                const std::vector<T>& mins, const T epsilon) {

    for(uint64_t i = 0; i < n; i++) {

        auto key = computeKey(data_set[i].getFeatures(), swapped_dimensions, dimensions, mins, epsilon);

        spatial_index[key].insert(i);

        data_set[i].setCellNumber(key);

    }

       
}

template<typename T>
static inline std::vector<uint64_t> compute_neighbouring_keys(const uint64_t& key,
                                                              const std::vector<uint64_t>& m_swapped_dimensions,
					                      const std::vector<T>& dimensions,
					                      const std::map<uint64_t,std::pair<uint64_t, uint64_t>>& m_cell_index,
                                	                      const uint32_t& number_of_features
                                	     ) {

    std::vector<uint64_t> neighboring_cells;
    neighboring_cells.reserve(std::pow(3, number_of_features));
    neighboring_cells.push_back(key);

            // cell accumulators
    uint64_t cells_in_lower_space = 1;
    uint64_t cells_in_current_space = 1;
    uint64_t number_of_points = m_cell_index.find(key)->second.second;

            // fetch all existing neighboring cells
    for (size_t d : m_swapped_dimensions) {

    	cells_in_current_space *= dimensions[d];

        for (size_t i = 0, end = neighboring_cells.size(); i < end; ++i) {

            const uint64_t current_cell = neighboring_cells[i];

            // check "left" neighbor - a.k.a the cell in the current dimension that has a lower number
            const uint64_t left = current_cell - cells_in_lower_space;
            const auto found_left = m_cell_index.find(left);
            if (current_cell % cells_in_current_space >= cells_in_lower_space) {
                neighboring_cells.push_back(left);
                number_of_points += found_left != m_cell_index.end() ? found_left->second.second : 0;
            }
            
    	    // check "right" neighbor - a.k.a the cell in the current dimension that has a higher number
            const uint64_t right = current_cell + cells_in_lower_space;
            const auto found_right = m_cell_index.find(right);
            if (current_cell % cells_in_current_space < cells_in_current_space - cells_in_lower_space) {

                neighboring_cells.push_back(right);
                number_of_points += found_right != m_cell_index.end() ? found_right->second.second : 0;
            }
        }

        cells_in_lower_space = cells_in_current_space;

    }

    return neighboring_cells;


}

static inline void addClustersToVectors(std::vector<uint64_t>& cluster_lengths,  
		                        std::vector<uint64_t>& cluster_points,
					const std::vector<std::set<uint64_t>>& clusters) {

    for(auto& cluster:clusters) {

        cluster_lengths.push_back(cluster.size());

        for(auto& point_in_cluster:cluster) {

            cluster_points.push_back(point_in_cluster);

        }

    }

}

#pragma omp declare reduction (merge_set : std::set<uint64_t> : omp_out.insert(omp_in.begin(), omp_in.end()))
#pragma omp declare reduction (merge : std::vector<uint64_t> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
template<typename T>
static inline std::set<uint64_t> getNeighbours(const std::vector<uint64_t>& indices,  
				               std::vector<Data<T>>& dataset, 
		                               const T& epsilon, const uint64_t& number_of_features, 
				               const std::vector<uint64_t>& vals) {

    const uint64_t n = vals.size();
    const uint64_t indices_size = indices.size();

    std::set<uint64_t> neighbours;
    //#pragma omp parallel for schedule(dynamic, 32) private(neighboring_points) firstprivate(previous_cell) reduction(merge: rules)
    #pragma omp parallel for reduction(merge_set:neighbours)
    for(uint64_t index = 0; index < indices_size; index++) {
   
	auto& curr_point = dataset[indices[index]].getFeatures(); 

        for (uint64_t i = 0; i < n; i++) {

	    if(Util::calculateEuclideanDist(dataset[vals[i]].getFeatures(), 
	                                    curr_point, number_of_features) <= epsilon) {
	        neighbours.insert(vals[i]);

	    }

	}

    }

    return neighbours;

}

template <class I1, class I2>
static inline bool have_common_element(I1 first1, I1 last1, I2 first2, I2 last2) {
    while (first1 != last1 && first2 != last2) {
        if (*first1 < *first2)
            ++first1;
        else if (*first2 < *first1)
            ++first2;
        else
            return true;
    }
    return false;
}

static inline void mergeClustersWithCommonPoints( std::vector<std::set<uint64_t>>& clusters) {

    for(auto it = clusters.begin(); it != clusters.end(); it++) {

        for(auto jt = std::next(it); jt != clusters.end();) {

            //std::set<uint32_t> intersect;

            //set_intersection((*it).begin(), (*it).end(), (*jt).begin(), (*jt).end(),
            //                 std::inserter(intersect, intersect.begin()));
            
            if(have_common_element((*it).begin(), (*it).end(), (*jt).begin(), (*jt).end())) {

                (*it).insert((*jt).begin(), (*jt).end());

                jt = clusters.erase(jt);

            }
            else {

                ++jt;

            }

        } 

    }

}

#pragma omp declare reduction (map_add : std::unordered_map<uint64_t,  std::vector<uint64_t>> : omp_out.insert(omp_in.begin(), omp_in.end()))

int main(int argc, char** argv) {
    
    uint64_t min_points = 0;
    DATA_TYPE epsilon = 0.0;
    uint64_t number_of_features = 0;
    uint64_t n = 0;
    std::string input_filename = "";
    std::string output_filename = "";
    double start_time, end_time;
    DATA_TYPE epsilon_square = 0.0;
    std::vector<uint64_t> m_swapped_dimensions;
    std::vector<int32_t> num_of_points_to_cluster;
    std::vector<int32_t> displacements;
    std::vector<int32_t> indices_of_points;
    std::vector<DATA_TYPE> data_points;
    std::vector<DATA_TYPE> data_points_per_process;
    std::vector<Data<DATA_TYPE>> original_dataset;
    std::vector<Data<DATA_TYPE>> dataset;
    uint64_t num_of_points_distributed = 0;
    std::map<uint64_t, uint64_t> cell_size;
    std::map<uint64_t, std::set<uint64_t>> spatial_index;
    std::unordered_map<uint64_t, std::vector<uint64_t>> neighbour_keys;
    std::vector<int32_t> original_indices;
    std::vector<int32_t> count_points_per_process;
    std::vector<int32_t> count_of_clusters_per_process;
    std::vector<int32_t> displacements_of_dataPoints;
    std::vector<DATA_TYPE> mins;
    std::vector<DATA_TYPE> maxs;
    std::vector<DATA_TYPE> dimensions;
    std::map<uint64_t,std::pair<uint64_t, uint64_t>> m_cell_index;
    uint64_t total_cells = 1;

    std::vector<uint64_t> all_cluster_lens;
    MPI_Init(&argc, &argv);

    int32_t rank, num_of_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_of_procs);
 
    if(rank == 0) { 

	std::vector<std::vector<DATA_TYPE>> input;

        start_time = omp_get_wtime();
    
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
      
    	n = static_cast<uint64_t>(input.size());

    	epsilon_square = epsilon * epsilon;

    	std::cout << "Size of the input dataset is " << n << std::endl;

    	end_time = omp_get_wtime();

    	std::cout << "Time to load dataset " << (end_time - start_time) <<"s"<< std::endl;
    
	if(n > 0) {
     
            number_of_features = input[0].size();

    	}

    	mins.resize(number_of_features, FLT_MAX);
    	maxs.resize(number_of_features, 0.0);
    	dimensions.resize(number_of_features, 0.0);

    	for(uint64_t i = 0; i < n; i++) {

	    for(uint64_t j = 0; j < number_of_features; j++) {

	        if(input[i][j] < mins[j])
	            mins[j] = input[i][j];

	        if(input[i][j] > maxs[j])
	            maxs[j] = input[i][j];

	    }

        }

        /*Compute cell dimensions */
        for(uint64_t j = 0; j < number_of_features; j++) {

	    dimensions[j] = std::ceil((maxs[j] - mins[j]) / (epsilon)) + 1;
	    total_cells *= dimensions[j];

        }

        std::cout<<"Total cells  "<<total_cells<< std::endl;

        m_swapped_dimensions.resize(number_of_features, 0);
    
	/*SWAP dimensions */
        std::iota(m_swapped_dimensions.begin(), m_swapped_dimensions.end(), 0);
    
        // swap the dimensions descending by their cell sizes
        std::sort(m_swapped_dimensions.begin(), m_swapped_dimensions.end(), [&] (size_t a, size_t b) {
            return dimensions[a] < dimensions[b];
        });

        for(auto& ip:input) {

            dataset.push_back(Data(ip));

        }


    	/*compute keys for all cells */
    	compute_keys(dataset, n, spatial_index, m_swapped_dimensions, dimensions, mins, epsilon);

    	std::vector<std::vector<DATA_TYPE>> reordered_input;
    
	std::map<uint64_t, uint64_t> map_to_original;
    
    	/*Reorder the cells based on their spatial indexing */
   
    	uint64_t counter = 0;

    	for (auto& cell : spatial_index) {

            auto& index  = cell.second;
        
	    for(auto &pt:index) {

                reordered_input.push_back(input[pt]);

	        map_to_original[counter++] = pt; 

	    }

        }

    	//input.insert(input.begin(), std::make_move_iterator(reordered_input.begin()), 
        //            std::make_move_iterator(reordered_input.end()));;
	input.swap(reordered_input);

    	spatial_index.clear();

    	dataset.clear();

    	for(auto& ip:input) {

            dataset.push_back(Data(ip));

    	}

        /*compute keys again on the reordered input */
        compute_keys(dataset, n, spatial_index, m_swapped_dimensions, dimensions, mins, epsilon);
        
	/*Compute cell index */
        uint64_t accumulator = 0;

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
  

	std::cout << "Size of spatial index "<<spatial_index.size()<<std::endl;
    	/*compute neighbouring cells */
	//#pragma omp parallel for
    	for(auto& key : spatial_index) {

	    //auto it = spatial_index.begin();
            //advance(it, i);
    
           neighbour_keys[key.first] = compute_neighbouring_keys(key.first, m_swapped_dimensions,
                                             		          dimensions, m_cell_index, number_of_features);
	   
    	}

	uint32_t points_per_procs = n / num_of_procs;

	if(points_per_procs == 0)
	    points_per_procs = 1;

    	std::map<uint32_t, std::set<uint32_t>> points_procs;

    	uint32_t proc_count = 0;

	num_of_points_to_cluster.resize(num_of_procs);

	displacements.resize(num_of_procs);

	displacements_of_dataPoints.resize(num_of_procs);

	displacements_of_dataPoints[0] = 0;

	displacements[0] = 0;

	uint64_t total_points = 0;

	std::set<uint32_t> neighbours;
    	
	for(auto& k : spatial_index) {

	    std::vector<uint64_t>& neighbouring_keys = neighbour_keys[k.first];

            for(auto& pt : neighbouring_keys) {

                neighbours.insert(spatial_index[pt].begin(), spatial_index[pt].end());

            }

	    /*if the points are being assigned to the last process, assign all the remaining points to that process */
	    if(proc_count == num_of_procs - 1) {

                points_procs[proc_count].insert(k.second.begin(), k.second.end());

	    }		    
	    else if(points_procs[proc_count].size() < points_per_procs) {

                points_procs[proc_count].insert(k.second.begin(), k.second.end());
	    
	    }
	    else {
	    
	        points_procs[proc_count].insert(neighbours.begin(), neighbours.end()); //insert neighbours only for alternating points

	        neighbours.clear();

		proc_count++;

                points_procs[proc_count].insert(k.second.begin(), k.second.end());

	    }

	}

	 points_procs[proc_count].insert(neighbours.begin(), neighbours.end()); //insert neighbours of points in last process
	/*compute number of points allocated per process
	 */
	
	uint64_t i = 0;

        for(const auto& p : points_procs) {

            total_points += p.second.size();

	    num_of_points_to_cluster[p.first] = p.second.size();

	    num_of_points_distributed += p.second.size();

	    count_points_per_process.push_back(p.second.size() * number_of_features);

	    std::cout << "Number of points allotted to process "<<p.first<<" is "<<p.second.size()<<std::endl;

	    displacements[p.first + 1] = displacements[p.first] + p.second.size();

	    /*This will actually store the displacement values of the actual dataset. Since each data point is defined by 'number_of_features',
	     * the dispalcement values will be multilpied by the number of features */
	    displacements_of_dataPoints[p.first + 1] = displacements_of_dataPoints[p.first] + p.second.size() * number_of_features;

        }

	indices_of_points.resize(total_points);

	/*copy all the data points, their indices and also their cell numbers. ](Serializing the data structure) */
	for(const auto& p:points_procs) { /*for each process */

            for(auto& pt:p.second) { /*for each point alloted to the process */

                indices_of_points[i] = pt;

		/*copy the data point */
		for(auto feature:dataset[pt].getFeatures()) {

		    data_points.push_back(feature);

		}

	        i++;

            }

        }

	original_dataset = dataset;

	dataset.clear();

	spatial_index.clear();

	m_cell_index.clear();

	neighbour_keys.clear();


    }

    MPI_Bcast(&epsilon, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Bcast(&min_points, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(&number_of_features, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Scatter(num_of_points_to_cluster.data(), 1, MPI_INT, &n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    original_indices.resize(n);

    data_points_per_process.resize(n * number_of_features);

    MPI_Scatterv(indices_of_points.data(), num_of_points_to_cluster.data(), displacements.data(), MPI_INT, original_indices.data(),
                  n, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Scatterv(data_points.data(), count_points_per_process.data() , displacements_of_dataPoints.data(), MPI_FLOAT, data_points_per_process.data(),
                   n * number_of_features, MPI_FLOAT, 0, MPI_COMM_WORLD);

    /*copy the points alloted to the process of rank 'p' to dataset */
    int64_t k = 0;

    for(int64_t i = 0; i <  data_points_per_process.size(); i = i + number_of_features) {

	std::vector<DATA_TYPE> point;

	for(int64_t j = 0; j < number_of_features; j++) {

	    point.push_back(data_points_per_process[i + j]);


	}

	Data pt = Data(point);
	
	pt.setIndex(original_indices[k]); //k will iterate original_indices upto n
	
	dataset.push_back(pt);
	
	k++;

    }

    /*Compute spatial index, neighbouring keys for all the points assigned to processes other than root.
      They are already computed for the root process */
    mins.resize(number_of_features, FLT_MAX);
    maxs.resize(number_of_features, 0.0);
    dimensions.resize(number_of_features, 0.0);

    for(uint64_t i = 0; i < n; i++) {
	    
	auto& features = dataset[i].getFeatures();

        for(uint64_t j = 0; j < number_of_features; j++) {

	    if(features[j] < mins[j])
                mins[j] = features[j];

            if(features[j] > maxs[j])
                maxs[j] = features[j];

        }

    }

    /*Compute cell dimensions */
    for(uint64_t j = 0; j < number_of_features; j++) {

        dimensions[j] = std::ceil((maxs[j] - mins[j]) / (epsilon)) + 1;
        total_cells *= dimensions[j];

    }

    m_swapped_dimensions.resize(number_of_features, 0);

    /*SWAP dimensions */
    std::iota(m_swapped_dimensions.begin(), m_swapped_dimensions.end(), 0);

    // swap the dimensions descending by their cell sizes
    std::sort(m_swapped_dimensions.begin(), m_swapped_dimensions.end(), [&] (size_t a, size_t b) {
        return dimensions[a] < dimensions[b];
    });

    /*compute keys for all cells */
    compute_keys(dataset, n, spatial_index, m_swapped_dimensions, dimensions, mins, epsilon);

    for(auto& cell:spatial_index)
        cell_size[cell.first] = cell.second.size();

    /*Compute cell index */
    uint64_t accumulator = 0;

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

        neighbour_keys[cell.first] = compute_neighbouring_keys(cell.first, m_swapped_dimensions,
                                                               dimensions, m_cell_index, number_of_features);

    }

    epsilon_square = epsilon * epsilon;
    
    uint64_t num_clusters = 0;

    std::vector<std::set<uint64_t>> clusters;

    uint64_t num_of_points_clustered = 0;

    if(rank == 0) {

        start_time = omp_get_wtime();

    }

    for (uint64_t i = 0; i < n; i++) {

	if(!dataset[i].isVisited()) { 

            std::set<uint64_t> cluster;

	    std::vector<uint64_t> indices;
	    
	    indices.push_back(i);

	    while(!indices.empty()) {

		std::unordered_set<uint64_t> vals_set;

		const uint64_t indices_size = indices.size();

		uint64_t prev_key = INT_MAX;
		
		//#pragma omp parallel for reduction(merge_set:vals_set)
		for(uint64_t index = 0; index < indices_size; index++) {
		
		    uint64_t cell_key = dataset[indices[index]].getCellNumber();

		    if(cell_key != prev_key) {

                     /*only if a cell still contains points that are not added to a cluster
			     * Using set to prevent duplicate entry of points  */   

			std::vector<uint64_t>& neighbouring_keys = neighbour_keys[cell_key];

			for(auto& pt : neighbouring_keys) {
		        
			    if(cell_size[pt] > 0) {
                        
			        auto& points = spatial_index[pt];

			        for(auto& p:points) {

			            if(!dataset[p].isVisited())
				        vals_set.insert(p);

			        }

                             }

			 } 
		    
		    }

		    prev_key = cell_key;
		
	        }
                
		std::vector<uint64_t> vals(vals_set.begin(), vals_set.end());
		
		std::set<uint64_t> neighbours = getNeighbours(indices, dataset, epsilon_square, number_of_features, vals);
                
		indices.resize(neighbours.size());

                std::copy(neighbours.begin(), neighbours.end(), indices.begin());

                for(const auto& point: indices) {

                    cluster.insert(dataset[point].getIndex());

                    /*This is to keep a tab of the number of points that are not added yet to a cluster */
                    cell_size[dataset[point].getCellNumber()]--;

                    dataset[point].markVisited();

                }

	    }

	    /*if a point has no neighbours among in its viscinity then the cluster size will be 1 and it will be a noise point */
	    if(cluster.size() > 1) {
	    
	        num_clusters++;

	        clusters.push_back(cluster);

 	        num_of_points_clustered += cluster.size();

	    }

	}

    }
    
    /*Local Cluster computation done. Prepare to send the points to root process */

    std::vector<uint64_t> cluster_lengths;

    std::vector<uint64_t> cluster_points;

    if(num_of_procs > 2) {

        if((rank % 2)) { //odd process

            addClustersToVectors(cluster_lengths, cluster_points, clusters);

            /*send number of cluster, then cluster lengths and finally cluster points */
            MPI_Send(&num_clusters , 1 ,MPI_LONG, rank - 1, 1, MPI_COMM_WORLD);

            MPI_Send(cluster_lengths.data() , cluster_lengths.size() ,MPI_LONG, rank - 1, 2, MPI_COMM_WORLD);

            MPI_Send(cluster_points.data() , cluster_points.size() ,MPI_LONG, rank - 1, 3, MPI_COMM_WORLD);

            clusters.clear(); //No need to send the clusters any further

        }

	if(!(rank % 2) && (rank != (num_of_procs - 1))) { //even process but not the last one

            uint64_t clusters_from_next_rank;

            MPI_Status status;

            MPI_Recv(&clusters_from_next_rank ,1, MPI_LONG, rank + 1, 1, MPI_COMM_WORLD, &status);

            cluster_lengths.resize(clusters_from_next_rank);

            MPI_Recv(cluster_lengths.data() ,clusters_from_next_rank, MPI_LONG, rank + 1, 2, MPI_COMM_WORLD, &status);

            uint64_t points_to_receive = std::reduce(cluster_lengths.begin(), cluster_lengths.end());

            cluster_points.resize(points_to_receive);

            MPI_Recv(cluster_points.data() , points_to_receive, MPI_LONG, rank + 1, 3, MPI_COMM_WORLD, &status);

            /*Store the received clusters in map */
            uint32_t i = 0;

            uint32_t cluster_count = 0;

            std::cout << "Storing received clusters in map " <<std::endl;

            start_time = omp_get_wtime();

            while(i < cluster_points.size()) {

                std::set<uint64_t> current_cluster;

                for(uint64_t k = 0; k < cluster_lengths[cluster_count]; k++) { //k will index till each cluster length

                     current_cluster.insert(cluster_points[i + k]);

                }

                clusters.push_back(current_cluster);

                i += cluster_lengths[cluster_count];

                cluster_count++;

            }

            /*Now merge the clusters computed by the current process and that of the previous process */
            mergeClustersWithCommonPoints(clusters);

            /*Only even clusters that are not zero must send the result to root */
            cluster_lengths.clear();

            cluster_points.clear();


        }

    }

    /*Now that clusters are computed,*/ 
    /*send only from non root processes */
    if(rank != 0) {

      /*if number of process > 2
         * Only even processes other than 0 will have clusters to send. Odd processes would have cleared the clusters.
         *
         *if the number of processes = 2
         * only process 1 will have points to send.
         */

    
        addClustersToVectors(cluster_lengths, cluster_points, clusters);

    }

    uint64_t number_of_clusters_computed_root;
    
    /*First transmit the number of clusters in each process to each of the root process*/
    if(rank == 0) {

        number_of_clusters_computed_root = num_clusters;

	num_of_points_clustered = 0; /*do not send any points to the root process */

	num_clusters = 0; /*do not send any points to the root process */

        count_of_clusters_per_process.resize(num_of_procs);

	num_of_points_to_cluster.resize(num_of_procs);

    }

    std::vector<int32_t> displacements_cluster_len;

    MPI_Gather(&num_clusters, 1, MPI_INT, count_of_clusters_per_process.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Gather(&num_of_points_clustered, 1, MPI_INT, num_of_points_to_cluster.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(rank == 0) {

	end_time = omp_get_wtime();

        std::cout << "Time taken to cluster " << (end_time - start_time) <<"s"<< std::endl;
	
	displacements_cluster_len.resize(num_of_procs);

	/*stores the length of each of the clusters */
	all_cluster_lens.resize(std::reduce(count_of_clusters_per_process.begin(), count_of_clusters_per_process.end()));

	displacements_cluster_len[0] = 0;

	for (uint32_t i = 1; i < count_of_clusters_per_process.size(); i++) {
          
            displacements_cluster_len[i] = displacements_cluster_len[i-1] + count_of_clusters_per_process[i-1];
        
	}

    }
    
    MPI_Gatherv(cluster_lengths.data(), num_clusters, MPI_LONG,
                all_cluster_lens.data(), count_of_clusters_per_process.data(), displacements_cluster_len.data(), MPI_LONG,
                0, MPI_COMM_WORLD);

    /*Now we have the lenght of each of the clusters from each process 
     *
     * Let us now collect the points that have been clustered */
    std::vector<uint64_t> indices_of_all_points;
    
    if(rank == 0) {

	for(uint32_t i = 1; i < num_of_procs; i++) {

	      displacements_cluster_len[i] = displacements_cluster_len[i-1] + num_of_points_to_cluster[i - 1];


	}

	num_of_points_distributed = std::reduce(num_of_points_to_cluster.begin(), num_of_points_to_cluster.end());
	
	indices_of_all_points.resize(num_of_points_distributed);

    }

    MPI_Gatherv(cluster_points.data(), num_of_points_clustered, MPI_LONG,
                indices_of_all_points.data(), num_of_points_to_cluster.data(), displacements_cluster_len.data(), MPI_LONG,
                0, MPI_COMM_WORLD);


    if(rank == 0) {

	uint32_t i = 0;

	uint32_t cluster_count = 0;

	std::cout << "Storing received clusters in map " <<std::endl;

	start_time = omp_get_wtime();

        while(i < indices_of_all_points.size()) {

	    number_of_clusters_computed_root++;

	    std::set<uint64_t> current_cluster;

	    for(uint64_t k = 0; k < all_cluster_lens[cluster_count]; k++) { //k will index till each cluster length

                 current_cluster.insert(indices_of_all_points[i + k]);

	    }

	    clusters.push_back(current_cluster);

	    i += all_cluster_lens[cluster_count];

	    cluster_count++;

        }

	end_time = omp_get_wtime();

	std::cout << "Time taken to store clusters in map " << (end_time - start_time) <<"s"<< std::endl;

	std::cout << "Merging of clusters started " << std::endl;
	
	start_time = omp_get_wtime();

	mergeClustersWithCommonPoints(clusters);

        std::cout << "Merging of clusters completed " << std::endl;

	end_time = omp_get_wtime();

        std::cout << "Time taken to merge clusters " << (end_time - start_time) <<"s"<< std::endl;

        cluster_count = 0;

	uint64_t cluster_points = 0;

	for(auto& p: clusters) {

	    if(p.size() >= min_points) {

	        for(auto& pt:p) {

	            original_dataset[pt].setClusterInfo(cluster_count + 1);

	        }

	        cluster_count++;

		cluster_points += p.size();

	    }

	}

        std::cout <<" Number of clusters : "<<cluster_count<<std::endl;
	std::cout <<" Number of noise points : "<< (original_dataset.size() - cluster_points) << std::endl;

	std::cout << "Writing data and their cluster labels to output file "<<output_filename<<std::endl;
        std::cout << "Points with cluster label 0 are Noise points "<<std::endl;

        start_time = omp_get_wtime();

        Util::writeToCSVfile(original_dataset, output_filename);

        end_time = omp_get_wtime();

        std::cout << "Time to write output to file " << (end_time - start_time) <<"s"<< std::endl;

    }

    MPI_Finalize();

    return 0;
    
}
