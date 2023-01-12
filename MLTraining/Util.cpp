#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <cmath>

class Util {

    public:
    template<typename T> static double calculateEuclideanDist(std::vector<T>& a, std::vector<T>& b) {

        double sum = 0.0;
        uint64_t i = 0;

        while (i < a.size()) {
            sum += (a[i] - b[i]) * (a[i] - b[i]);
            i++;
        }

        return sqrt(sum);


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
