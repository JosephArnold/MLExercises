// MLTraining.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <cfloat>
#include <fstream>
#include <omp.h>
#include "Util.cpp"
#include "Data.cpp"

#pragma omp declare reduction (merge : std::set<uint32_t> : omp_out.insert(omp_in.begin(), omp_in.end()))
static inline std::set<uint32_t> getNeighbours(uint32_t index,  std::vector<data<double>>& dataunordered_set, 
		                                           double epsilon, uint32_t n, uint32_t number_of_features) {

    std::vector<double> curr_point = dataunordered_set[index].features;

    std::set<uint32_t> neighbours;

    #pragma omp parallel for reduction(merge:neighbours)
    for (uint32_t i = 0; i < n; i++) {

        if( (Util::calculateEuclideanDist(dataunordered_set[i].features, curr_point, number_of_features) <= epsilon)) {
         
	    neighbours.insert(i);

        }

    }

    return neighbours;

}

static inline bool containsCommonElement(std::set<uint32_t>& a, 
		                         std::set<uint32_t>& b) {

	for(auto element:a) {

	        if(b.find(element) != b.end()) {
	         
		    return true;

		}
	    
	}

	return false;

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
    double epsilon = 0.0;
    uint32_t number_of_features = 0;
    /*Read from a CSV file */
    std::string input_filename = "";
    std::string output_filename = "";
    std::vector<std::vector<double>> input;

    try {
    if(argc > 4) {
        input_filename = argv[1];
        std::ifstream csv_file;
        csv_file.open(input_filename);
        input = Util::parseCSVfile(csv_file);	
        min_points = std::stoi(argv[2]);
	epsilon = std::stod(argv[3]);
	output_filename = argv[4];
    }
    else {
        std::cerr<<"Please provide the correct path to the CSV file, minimum point, epsilon and output file "<<
		    std::endl;
        return 0;
    }

    std::vector<data<double>> dataunordered_set;

    for (uint32_t i = 0; i < input.size(); i++) {

        dataunordered_set.push_back(data<double>(input[i]));

    }

    uint32_t n = static_cast<int32_t>(dataunordered_set.size());

    std::cout << "Size of the input dataset is " << n << std::endl;

    if(n > 0) {
     
        number_of_features = input[0].size();

    }


    std::map<uint32_t, std::set<uint32_t>> core_points;
    /*initialize all points as noise points */
    std::vector<int32_t> point_info(n, 0);

    /* For each point determine the number of neighbourhood points*/
    
    std::cout << "Evaluating core points "<<std::endl;     
    
    for (uint32_t i = 0; i < n; i++) {
        
	std::set<uint32_t> neighbours = getNeighbours(i, dataunordered_set, epsilon, n, number_of_features);

	point_info[i] = 1;

	if(neighbours.size() >= min_points) {
	    
	    core_points[i] = neighbours;

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

    mergeNeighbours(core_points);

    std::cout << "Assigning clusters " << std::endl;
    for(auto& points_in_cluster:core_points) {

	     num_of_clusters++;

	     for(auto p:points_in_cluster.second) {

                 dataunordered_set[p].setClusterInfo(num_of_clusters); 

                 point_info[p] = 2; /* boundary points */
                
             }

    }

    std::cout << " Clustering completed " << std::endl;
    std::cout <<" Number of clusters : "<<num_of_clusters<<std::endl;
    std::cout << "Writing data and their cluster labels to output file "<<output_filename<<std::endl;
    std::cout << "Points with cluster label 0 are Noise points "<<std::endl;

    Util::writeToCSVfile(dataunordered_set, output_filename);
    
    }
    catch (const std::exception& e) {

            std::cout<<e.what()<<std::endl;

    }

    return 0;
    
}
