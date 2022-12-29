
#include <iostream>
#include <vector>

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

};

//template double calculateEuclideanDist(std::vector<double>, std::vector<double>);

