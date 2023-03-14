// MLTraining.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#define DATA_TYPE float

#include <iostream>
#include <vector>
#include <cfloat>
#include <fstream>
#include <omp.h>
#include "Util.cpp"
#include "Data.cpp"

template <class T>
class ParticleSwarmOptimization {

    T c1, c2;
    T inertia;
    std::vector<std::vector<T>> particles;
    std::vector<std::vector<T>> velocities;
    std::vector<T> pbest;
    T gbest = FLT_MAX;
    std::vector<std::vector<T>> pbest_position;
    std::vector<T> gbest_position;
    uint32_t num_of_particles;
    uint32_t num_of_dimensions;

    public:
        ParticleSwarmOptimization(T param1, T param2, T inertia_val, 
			          std::vector<std::vector<T>> data)
	                          :c1(param1),c2(param2),inertia(inertia_val),
	                           particles(data){
			
	    num_of_particles = particles.size();

	    num_of_dimensions = particles[0].size();

	    pbest = std::vector<T>(num_of_particles, FLT_MAX);

	    pbest_position = std::vector<std::vector<T>>(num_of_particles, 
			                                 std::vector<T>(num_of_dimensions, 0.0));

	    gbest_position = std::vector<T>(num_of_dimensions, 0.0);

	}

	void displayGlobalbest_position() {

	    Util::displayVector(gbest_position, num_of_dimensions);

	}
   
        void displayParticlePositions() {

	    for(uint32_t i = 0; i < num_of_particles; i++) {

	        for(uint32_t j = 0; j < num_of_dimensions; j++) {

                    std::cout<<particles[i][j]<<" ";

		}

	        std::cout<<std::endl;

	    }

	}

        void displayParticleVelocities() {

            for(uint32_t i = 0; i < num_of_particles; i++) {

                for(uint32_t j = 0; j < num_of_dimensions; j++) {

                    std::cout<<velocities[i][j]<<" ";

                }

                std::cout<<std::endl;

            }

        }
	

	T getBestValueSoFar() {

	    return gbest;

	}

        T objectiveFunction(T x, T y) {
    
	    return pow(x, 2.0) + pow(y, 2.0);

	}

	void initiateVelocity(T alpha) {

	    for(uint32_t i = 0; i < num_of_particles; i++) {

	        velocities.push_back(Util::scaleVector(alpha, particles[i], num_of_dimensions));

	    }

	}

	void updateVelocity() {

	    T r1 = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);

	    T r2 = r1;

	    for(uint32_t i = 0; i < num_of_particles; i++) {

		auto pbest_diff = Util::subtractVectors<T>(pbest_position[i], particles[i], num_of_dimensions);

		auto gbest_diff = Util::subtractVectors<T>(gbest_position, particles[i], num_of_dimensions);

	        velocities[i] =  Util::addVectors<T>(Util::addVectors<T>(
					             Util::scaleVector<T>(inertia, velocities[i], num_of_dimensions),
			                             Util::scaleVector(c1 * r1, pbest_diff, num_of_dimensions), 
						     num_of_dimensions),
				                     Util::scaleVector(c2 * r2, gbest_diff, num_of_dimensions),
						     num_of_dimensions
						     );  
	    
	    } 

	}

	void updateBestPositions() {

	    for(uint32_t i = 0; i < num_of_particles; i++) {

                T obj = objectiveFunction(particles[i][0], particles[i][1]);

                if(obj < pbest[i]) {

                    pbest_position[i] = particles[i];

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

	        particles[i] = Util::addVectors<T>( particles[i], velocities[i], num_of_dimensions); 

	    }

	}


};

int main(int argc, char** argv)
{
    
    uint32_t iterations = 100;
    /*Read from a CSV file */
    std::string input_filename = "";
    std::string output_filename = "";
    std::vector<std::vector<DATA_TYPE>> input;
    double start_time, end_time;
  
    start_time = omp_get_wtime();
    try {
    if(argc > 2) {
        input_filename = argv[1];
	iterations = std::stoi(argv[2]);
        std::ifstream csv_file;
        csv_file.open(input_filename);
        input = Util::parseCSVfile<DATA_TYPE>(csv_file);	
    }
    else {
        std::cerr<<"Please provide the correct path to the CSV file "<<
		    std::endl;
        return 0;
    }

    uint32_t n = static_cast<int32_t>(input.size());

    std::cout << "Size of the input dataset is " << n << std::endl;

    end_time = omp_get_wtime();

    std::cout << "Time to load dataset " << (end_time - start_time) <<"s"<< std::endl;
    
    /*
     * 1. initialize velocites
     * 2. Move particles
     * 3. Update positions
     * 4. Update velocities
     * 5. Goto step2 if all iterations are not completed.  
     * 
     */

    ParticleSwarmOptimization<DATA_TYPE> experiment(2, 2, 0.3, input);

    std::cout << "Running Particle Swarm Optimization for "<<iterations<<" iterations."<<std::endl;     
   
    start_time = omp_get_wtime(); 

    DATA_TYPE initial_alpha = 0.0;

    experiment.initiateVelocity(initial_alpha);

    std::cout << "Velocities initiated "<<std::endl;

    for (uint32_t i = 0; i < iterations; i++) {

	experiment.updateNextPosition();

	experiment.updateBestPositions();

	experiment.updateVelocity();

    }

    std::cout << "Best cost "<<experiment.getBestValueSoFar()<<std::endl;

    std::cout << "Best position "<< std::endl;
    
    experiment.displayGlobalbest_position();

    end_time = omp_get_wtime();

    std::cout << "Run completed! Time taken: " << (end_time - start_time) <<"s"<< std::endl;

    }
    catch (const std::exception& e) {

            std::cout<<e.what()<<std::endl;

    }

    return 0;
    
}
