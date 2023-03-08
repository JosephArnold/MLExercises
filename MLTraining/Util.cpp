#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <cmath>
#include "Data.cpp"

class Util {

    public:
    template<typename T> static double calculateEuclideanDist(std::vector<T>& a, std::vector<T>& b,  uint32_t n) {

        double sum = 0.0;

	#pragma omp simd reduction(+:sum)
        for(uint32_t i = 0; i < n; i++) {

            sum += (a[i] - b[i]) * (a[i] - b[i]);
       
	}

	return sum;

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

    static inline double sigmoid(double x) {

        return 1.0 / (1.0 + exp(-x));

    }

    static inline std::vector<double> multiplyVectorwithMatrix(std::vector<double> vectorArray,
		                                                            std::vector<std::vector<double>> matrix,
									    uint32_t n,
									    uint32_t k
									    ) {

        std::vector<double> result(k, 0.0); 
        
	for(uint32_t i = 0; i < k; i++) {

	    for(uint32_t j = 0; j < n; j++) {

	        result[i] += vectorArray[j] * matrix[j][k];

	    }

	}

	return result;

    }

    static inline double dotProduct(std::vector<double> vectorArray1,
		                    std::vector<double> vectorArray2
                                    uint32_t n) {

        double result;

        for(uint32_t i = 0; i < n; i++) {

            result += vectorArray1[i] * vectorArray2[i];

        }

        return result;

    }

    std::vector<std::vector<double>> transpose(std::vector<std::vector<double>> matrix
		       		               uint32_t rows, uint32_t cols) {

	 std::vector<std::vector<double>> result(cols, std::vector<double>(rows, 0.0));

         for(uint32_t i = 0; i < cols; i++) {

	     for(uint32_t j = i + 1; j < rows; j++) {

                 result[i][j] = matrix[j][i];

        }

    }

    std::vector<double> addVectors(std::vector<double> vector1, 
		                   std::vector<double> vector2, uint32_t n) {

         std::vector<double> result;

         for(uint32_t i = 0; i < n; i++) {

             result.push_back(vector1[i] + vector2[i]);

        }

	return result;

    }



//template double calculateEuclideanDist(std::vector<double>, std::vector<double>);
};
