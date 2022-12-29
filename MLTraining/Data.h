#pragma once
#include<iostream>
#include<vector>

template<class T>
class data
{
public:
	std::vector<T> features;
	/*No cluster information initially*/
	int32_t cluster_info = 0;

	data(std::vector<T>& features_vector) {
		features = features_vector;
	}

	data(std::vector<T>& features_vector, int32_t cluster_val) {
		features = features_vector;
		cluster_info = cluster_val;
	}

	void setClusterInfo(int32_t value) {
		cluster_info = value;
	}

	int32_t getClusterInfo() {
		return cluster_info;
	}

	void display() {

		for (int32_t i = 0; i < features.size(); i++) {

			std::cout << features[i] << " ";
		}
		std::cout << cluster_info<<std::endl;


	}

};

