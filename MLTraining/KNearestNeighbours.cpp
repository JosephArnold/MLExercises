// MLTraining.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <cfloat>
#include <fstream>
#include <omp.h>
#include <execution>
#include "Util.cpp"

#pragma omp declare reduction (merge : std::vector<double> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
static inline std::vector<double> computeDistance(uint32_t index,  std::vector<std::vector<double>>& dataunordered_set, 
		                                  uint32_t n, uint32_t number_of_features) {

    std::vector<double> curr_point = dataunordered_set[index];

    std::vector<double> neighbours;

    #pragma omp parallel for reduction(merge:neighbours)
    for (uint32_t i = 0; i < n; i++) {

        neighbours.push_back(Util::calculateEuclideanDist(dataunordered_set[i], curr_point, number_of_features)); 
         
    }

    return neighbours;

}


#pragma omp declare reduction (unordered_map_add : std::map<uint32_t,  std::set<uint32_t>> : omp_out.insert(omp_in.begin(), omp_in.end()))
int main(int argc, char** argv)
{
    uint32_t k = 0;
    uint32_t number_of_features = 0;
    /*Read from a CSV file */
    std::string input_filename = "";
    std::string output_filename = "";
    std::vector<std::vector<double>> input;

    try {
    if(argc > 3) {
        input_filename = argv[1];
        std::ifstream csv_file;
        csv_file.open(input_filename);
        input = Util::parseCSVfile(csv_file);	
        k = std::stoi(argv[2]);
	output_filename = argv[3];
    }
    else {
        std::cerr<<"Please provide the correct path to the CSV file, minimum point, epsilon and output file "<<
		    std::endl;
        return 0;
    }

    uint32_t n = static_cast<int32_t>(input.size());

    std::cout << "Size of the input dataset is " << n << std::endl;

    if(n > 0) {
     
        number_of_features = input[0].size();

    }

    /*initialize all points as noise points */
    std::vector<double> result;

    double dk = static_cast<double>(k);
    
    for (uint32_t i = 0; i < n; i++) {
        
	std::vector<double> neighbours = computeDistance(i, input, n, number_of_features);

	std::sort(neighbours.begin(), neighbours.end());
	    
	neighbours.resize(k);

	result.push_back(std::reduce(neighbours.begin(), neighbours.end()) / dk);

    }

	 
    std::cout << " Computing average of "<<k<<" nearest neighbours completed " << std::endl;

    std::sort(result.begin(), result.end());

    std::cout << "Sorting the distances " << std::endl;
    
    std::ofstream outfile;
    outfile.open (output_filename);

    for (uint32_t j = 0; j < n; j++) {

        outfile<<result[j]<<","<< std::endl;

    }

    outfile.close();
    
    }
    catch (const std::exception& e) {

            std::cout<<e.what()<<std::endl;

    }

    return 0;
    
}
