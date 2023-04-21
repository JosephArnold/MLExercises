#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <cmath>
#include "Data.cpp"

typedef union {
	float val;
	uint32_t hex;

} floatword;

class Util {

    public:
    template<typename T> static inline double calculateEuclideanDist(std::vector<T>& a, std::vector<T>& b,  uint32_t n) {

        T sum = 0.0;

	#pragma omp simd reduction(+:sum)
        for(uint32_t i = 0; i < n; i++) {

            sum += (a[i] - b[i]) * (a[i] - b[i]);
       
	}

	return sum;

    }

    template<typename T>
    static std::vector<std::vector<T>> parseCSVfile(std::ifstream& str) {
        std::vector<std::vector<T>>   result;
        std::string                line;
        std::string                cell, row;

        while(std::getline(str,row)) {

            std::stringstream rowstream(row);
            std::vector<T> row_values;

            while(std::getline(rowstream,cell, ',')) {

                row_values.push_back(stod(cell));

            }

            result.push_back(row_values);
        }

        return result;

    }

    template<typename T>
    static int32_t writeToCSVfile(std::vector<Data<T>>& dataset,  std::string output_filename) {
	
	std::ofstream outfile;
        outfile.open (output_filename);

        uint32_t n = dataset.size();

	if(n == 0){
	    std::cerr<<"No data to write "<<std::endl;
	    return -1;
	}

	uint32_t features_size = dataset[0].features.size();

        for (uint32_t j = 0; j < n; j++) {
            
	    std::vector<T> features = dataset[j].features; 
	    
	    for(uint32_t k = 0; k < features_size; k++) {
	        
	        outfile<<features[k]<<" , ";
		
	    }
	    
	    outfile << dataset[j].cluster_info << std::endl;

        }
        
	outfile.close();
	
	return 0;

    }

    template<typename T> static int32_t writeToCSVfile(std::vector<std::vector<T>>& dataset,
                                  std::vector<uint32_t> cluster_info,
                                  std::string output_filename) {

        std::ofstream outfile;
        outfile.open (output_filename);

        uint64_t n = dataset.size();

        if(n == 0){
            std::cerr<<"No data to write "<<std::endl;
            return -1;
        }

        int32_t features_size = dataset[0].size();

        for (uint64_t j = 0; j < n; j++) {

            std::vector<T> features = dataset[j];

            for(int32_t k = 0; k < features_size; k++) {

                outfile<<features[k]<<" , ";

            }

            outfile << cluster_info[j] << std::endl;

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
		                    std::vector<double> vectorArray2,
                                    uint32_t n) {

        double result;

        for(uint32_t i = 0; i < n; i++) {

            result += vectorArray1[i] * vectorArray2[i];

        }

        return result;

    }

    std::vector<std::vector<double>> transpose(std::vector<std::vector<double>> matrix,
		       		               uint32_t rows, uint32_t cols) {

	 std::vector<std::vector<double>> result(cols, std::vector<double>(rows, 0.0));

         for(uint32_t i = 0; i < cols; i++) {

	     for(uint32_t j = i + 1; j < rows; j++) {

                 result[i][j] = matrix[j][i];

             }

         }

	 return result;

    }

    template<typename T> static inline std::vector<T> addVectors(std::vector<T> vector1, 
		                                                 std::vector<T> vector2, 
							         uint32_t n) {

         std::vector<T> result;

         for(uint32_t i = 0; i < n; i++) {

             result.push_back(vector1[i] + vector2[i]);

        }

	return result;

    }

    template<typename T> static inline void displayVector(std::vector<T> vector1,
                                                          uint32_t n) {

         for(uint32_t i = 0; i < n; i++) {

	     std::cout<<vector1[i]<<" ";

         }
	 
	 std::cout<<std::endl;

    }


    template<typename T> static inline std::vector<T> subtractVectors(std::vector<T> vector1,
                                                        std::vector<T> vector2, 
							uint32_t n) {

         std::vector<T> result;

         for(uint32_t i = 0; i < n; i++) {

             result.push_back(vector1[i] - vector2[i]);

        }

        return result;

    }


    template<typename T> static inline std::vector<T> scaleVector(T alpha, 
		                                                  std::vector<T> vector1, 
								  uint32_t n) {

        for(uint32_t i = 0; i < n; i++) {

             vector1[i] = alpha * vector1[i];

        }

        return vector1;

    }

    static inline uint32_t asuint32(float x) {

        floatword num;
	num.val = x;
	return num.hex;

    }

//template double calculateEuclideanDist(std::vector<double>, std::vector<double>);
};
