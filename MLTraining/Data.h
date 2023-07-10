#pragma once
#include<iostream>
#include<vector>

template<class T>
class Data
{
public:
	std::vector<T> features;

	std::vector<uint32_t> neighbours;
	
	/*No cluster information initially*/
	int32_t cluster_info = 0;

	/*Mark as non visited */
	bool visited = false;

	/*stores the cell to which the point belongs */
	uint32_t cell_num = 0;

	/*stores the index of the point */
        uint32_t index = 0;


	Data(std::vector<T>& features_vector) {
		features = features_vector;
	}

	Data(std::vector<T>& features_vector, int32_t cluster_val) {
		features = features_vector;
		cluster_info = cluster_val;
	}

	void setClusterInfo(int32_t value) {
		cluster_info = value;
	}

	int32_t getClusterInfo() {
		return cluster_info;
	}

	void setCellNumber(uint32_t value) {
                cell_num = value;
        }

        int32_t getIndex() {
                return index;
        }

	void setIndex(uint32_t value) {
                index = value;
        }

        int32_t getCellNumber() {
                return cell_num;
        }



	bool isVisited() {

	    return visited;

	}

	void markVisited() {

	    visited = true;

	}

	std::vector<T>& getFeatures() {

	    return this->features;

	}
	void display() {

	    for (int32_t i = 0; i < features.size(); i++) {

	        std::cout << features[i] << " ";
		
	    }
	
    	    std::cout << cluster_info<<std::endl;


	}

};

