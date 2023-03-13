// MLTraining.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#define BLOCK_SIZE 4
#define DATA_TYPE float

#include <iostream>
#include <vector>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <queue>
#include <cfloat>
#include <fstream>
#include <omp.h>
#include "Util.cpp"
#include "Data.cpp"

template <class T>
class ParticleSwarmOptimization {

    T c1, c2;
    T inertia_val;
    std::vector<std::vector<T>> particles;
    std::vector<std::vector<T>> velocities;
    vector<T> pbest;
    T gbest;
    std::vector<T> pbest_position;
    T gbest, gbest_position;
    uint32_t num_of_particles;
    uint32_t num_of_dimensions;

    public:
        ParticleSwarmOptimization(T param1, T param2, T inertia_val, 
			          std::vector<std::vector<T>> data)
	                          :c1(param1),c2(param2),inertia(inertia_val),
	                           particles(data){
			
	    num_of_particles = particles.size();

	    num_of_dimensions = particles[0].size();

	    std::vector<std::vector<T>> velocities_init(num_of_particles, 
			                                std::vector<T>(num_of_dimensions, 0.0));

	    velocities = velocities_init;
				   
	}
    

        T objectiveFunction(T x, T y) {
    
	    return pow((x - 3.14), 2.0) + pow((y - 2.72), 2.0) + 
		   sin(3.0 * x + 1.41) + sin(4.0 * y - 1.73);

	}

	void updateVelocity() {

	    T r1 = static_cast<T>(rand() / (RAND_MAX));

	    T r2 = r1;

	    for(uint32_t i = 0; i < num_of_particles; i++) {

		T obj = objectiveFunction(particles[i][0], particles[i][1]);

		if(obj < pbest) {

		    pbest_position = particles[i];

		}

		if(obj < gbest) {

                    gbest_position = particles[i];

                }

		auto pbest_diff = subtractVectors(pbest_position[i] - particles[i]);

		auto gbest_diff = subtractVectors(gbest_position[i] - particles[i]);

	        velocities[i] = scaleVector(inertia_val, velocities, num_of_dimensions) +
			        scaleVector(c1 * r1, pbest_diff, num_of_dimensions) +
				scaleVector(c2 * r2, gbest_diff, num_of_dimensions);  
	    
	    } 

	}

	void updateBestPositions() {

	    for(uint32_t i = 0; i < num_of_particles; i++) {

                T obj = objectiveFunction(particles[i][0], particles[i][1]);

                if(obj < pbest[i]) {

                    pbest_position = particles[i];

		    pbest[i] = obj;

		    if(pbest[i] < gbest) {
		        
		        gbest = pbest[i];

			gbest_position = particles[i];

		    }

                }

	    }

	}

        void updateNextPosition() {

	    for(uint32_t i = 0; i < num_of_particles; i++) {

	        particles[i] = addVectors( particles[i], velocities[i], num_of_dimensions); 

	    }

	}


};

int main(int argc, char** argv)
{
    
    uint32_t number_of_features = 0;
    uint32_t iterations = 0;
    /*Read from a CSV file */
    std::string input_filename = "";
    std::string output_filename = "";
    std::vector<std::vector<DATA_TYPE>> input;
    double start_time, end_time;
  
    start_time = omp_get_wtime();
    try {
    if(argc > 3) {
        input_filename = argv[1];
	iterations = stoi[2];
        std::ifstream csv_file;
        csv_file.open(input_filename);
        input = Util::parseCSVfile<DATA_TYPE>(csv_file);	
    }
    else {
        std::cerr<<"Please provide the correct path to the CSV file, minimum point, epsilon and output file "<<
		    std::endl;
        return 0;
    }

    uint32_t n = static_cast<int32_t>(input.size());

    std::cout << "Size of the input dataset is " << n << std::endl;

    end_time = omp_get_wtime();

    std::cout << "Time to load dataset " << (end_time - start_time) <<"s"<< std::endl;
    
    /*1. initialize velocites
     * 2. Move particles
     * 3. Update positions
     * 4. Update velocities
     * 5. Goto step2  */

    ParticleSwarmOptimization<DATA_TYPE> experiment(0.1, 0.1, 0.8, input);

    std::cout << "Evaluating core points "<<std::endl;     
   
    start_time = omp_get_wtime(); 

    for (uint32_t i = 0; i < iterations; i++) {

	if(!visited[i]) { 
        
	    std::queue<uint32_t> compute;

            std::vector<uint32_t> cluster;

	    compute.push(i);

	    cluster.push_back(i);

	    visited[i] = 1;

	    while(!compute.empty()) {
		    
	        uint32_t index = compute.front();

		compute.pop();

		std::vector<uint32_t> neighbours = getNeighbours(index, input, epsilon_hex, 
				                                 n, number_of_features, visited);

		for(auto pt:neighbours) {

		    compute.push(pt);

		    cluster.push_back(pt);

		    visited[pt] = 1;

		}

	    }

	    if(cluster.size() >= min_points) {
	        core_points[++num_clusters] = cluster;
		std::cout<<"cluster "<<num_clusters<<" has "<<cluster.size()<<" points"<<std::endl;
	    }

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
   
    end_time = omp_get_wtime();

    std::cout << "Time to evaluate core points " << (end_time - start_time) <<"s"<< std::endl;

#if 0
    start_time = omp_get_wtime();

    mergeNeighbours(core_points);

    end_time = omp_get_wtime();

    std::cout << "Time to merge clusters " << (end_time - start_time) <<"s"<< std::endl;
#endif
    std::cout << "Assigning clusters " << std::endl;

    start_time = omp_get_wtime();
    
    for(auto& points_in_cluster:core_points) {

	     for(auto p:points_in_cluster.second) {

                 cluster_info[p] = points_in_cluster.first; 

             }

    }

    end_time = omp_get_wtime();

    std::cout << "Time to assign clusters " << (end_time - start_time) <<"s"<< std::endl;

    std::cout << " Clustering completed " << std::endl;
    std::cout <<" Number of clusters : "<<core_points.size()<<std::endl;
    std::cout << "Writing data and their cluster labels to output file "<<output_filename<<std::endl;
    std::cout << "Points with cluster label 0 are Noise points "<<std::endl;

    start_time = omp_get_wtime();
    
    Util::writeToCSVfile(input, cluster_info, output_filename);

    end_time = omp_get_wtime();

    std::cout << "Time to write output to file " << (end_time - start_time) <<"s"<< std::endl;

    }
    catch (const std::exception& e) {

            std::cout<<e.what()<<std::endl;

    }

    return 0;
    
}
