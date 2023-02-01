#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <cmath>
#include "Data.cpp"

class Util {

    public:
    template<typename T> static double calculateEuclideanDist(std::vector<T>& a, std::vector<T>& b) {

        double sum = 0.0;
        uint32_t i = 0;
        uint32_t n = a.size();
        while (i < n) {
            sum += (a[i] - b[i]) * (a[i] - b[i]);
            i++;
        }

	return sum;
        /* avoid sqrt(sum) for performance reasons */;


    }

    static std::vector<std::vector<double>> parseCSVfile(std::ifstream& str) {
        std::vector<std::vector<double>>   result;
        std::string                line;
        std::string                cell, row;

        while(std::getline(str,row)) {

            std::stringstream rowstream(row);
            std::vector<double> row_values;

            while(std::getline(rowstream,cell, ',')) {

                row_values.push_back(stod(cell));

            }

            result.push_back(row_values);
        }

        return result;

    }

    static int32_t writeToCSVfile(std::vector<data<double>>& dataset,  std::string output_filename) {
	
	std::ofstream outfile;
        outfile.open (output_filename);

        uint64_t n = dataset.size();

	if(n == 0){
	    std::cerr<<"No data to write "<<std::endl;
	    return -1;
	}

	int32_t features_size = dataset[0].features.size();

        for (uint64_t j = 0; j < n; j++) {
            
	    std::vector<double> features = dataset[j].features; 
	    
	    for(int32_t k = 0; k < features_size; k++) {
	        
	        outfile<<features[k]<<" , ";
		
	    }
	    
	    outfile << dataset[j].cluster_info << std::endl;

        }
        
	outfile.close();
	
	return 0;

    }

    static void displayInputMatrix(std::vector<std::vector<double>>& in) {

        int64_t numRows = in.size();
        if(numRows != 0) {
            int64_t col_size = in[0].size();
 
        for(int64_t i = 0; i < numRows; i++) {
            for(int64_t j = 0; j < col_size; j++) {

                std::cout<<" "<<in[i][j];

            }
            std::cout<<std::endl;
        }

     }
}

//template double calculateEuclideanDist(std::vector<double>, std::vector<double>);
};
