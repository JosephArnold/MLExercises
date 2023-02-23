#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <cfloat>
#include <fstream>
#include <omp.h>
#include <mpi.h>
#include "Util.cpp"
#include "Data.cpp"


int main(int argc, char **argv) {

	MPI_Init(&argc,&argv);
	int my_rank, nprocs;
	uint32_t n;
	uint32_t num_dimensions = 0;
	uint32_t* start_dist_matrix;
	uint32_t* end_dist_matrix;
	int* rcv_counts;
	int* displacements;
	double* distance_matrix;
	double* global_dist_matrix;
	uint32_t start, end;
	std::vector<std::vector<double>> input;
        MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
        MPI_Comm_size(MPI_COMM_WORLD,&nprocs);

	std::string input_filename = argv[1];
        std::ifstream csv_file;
        csv_file.open(input_filename);
        input = Util::parseCSVfile(csv_file);

 //	std::string output_filename = argv[2];

	n = input.size();

	if(n > 0) {
	    num_dimensions = input[0].size();
	}
	else {

	    std::cerr<<"Empty file "<<std::endl;

	    return 0;

	}

	if(my_rank == 0) {

		std::cout<<"size of the dataset is "<<n<<std::endl;

		std::cout<<"Number of MPI processes is "<<nprocs<<std::endl;

		start_dist_matrix = new uint32_t[nprocs];

		end_dist_matrix = new uint32_t[nprocs];

		std::cout<<"start and end dist allocated "<<std::endl;
		uint32_t num_rows = n / nprocs;

		for(uint32_t i = 0; i < nprocs; i++) {

		    start_dist_matrix[i] = i * num_rows;

		    end_dist_matrix[i] = start_dist_matrix[i] + num_rows;

		}

		std::cout<<"start and end dist computed "<<std::endl;

		uint32_t remaining_rows = n % nprocs;
		
		end_dist_matrix[nprocs - 1] += remaining_rows;

		 /*Allocate memory for global distance matrix */

        	global_dist_matrix = new double[n * n];

		rcv_counts = new int[nprocs];

		displacements = new int[nprocs];

		for(uint32_t i = 0 ; i < nprocs; i++) {

                    rcv_counts[i] = (end_dist_matrix[i] - start_dist_matrix[i]) * n;

                }

		displacements[0] = 0;

		for(uint32_t i = 1 ; i < nprocs; i++) {

                    displacements[i] = (displacements[i-1] + rcv_counts[i - 1]);

                }

		 std::cout<<"Receive count is calculated "<<my_rank<<std::endl;
	
	}
		
	MPI_Scatter(start_dist_matrix, 1, MPI_INT, &start, 1, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Scatter(end_dist_matrix, 1, MPI_INT, &end, 1, MPI_INT, 0, MPI_COMM_WORLD);

	std::cout<<"start is "<<start<<" and end is "<<end<<" in process "<<my_rank<<std::endl;

	uint32_t local_dist_matrix_size = (end - start);

	distance_matrix = (double*)malloc(sizeof(double) * local_dist_matrix_size * n);

	uint32_t k = 0;

	/*compute only from start row till end row in original input dataset */

	
	for(uint32_t i = start; i < end; i++) {

	    std::vector<double> curr_point = input[i];

	    /*update k = 0..local_dist_matrix_size rows */
	    for(uint32_t j = 0; j < n; j++) {

	        *(distance_matrix + (k * n) + j) = Util::calculateEuclideanDist(curr_point, input[j], n); 

	    }
	    
	    k++;

	}

	MPI_Barrier(MPI_COMM_WORLD);


	MPI_Gatherv(distance_matrix, local_dist_matrix_size * n, 
		    MPI_DOUBLE, global_dist_matrix, rcv_counts, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (my_rank == 0) {


	     std::cout<<"WORK DONE "<<std::endl;
 
	    std::cout<<"Printing global distance matrix "<<std::endl;

	    for (int i = 0; i < n; i++) {

	        for(uint32_t j = 0; j < n; j++) {

		    std::cout<<*(global_dist_matrix + i * n + j) << " ";

		}

		std::cout<<std::endl;

	    }

	}
        
	MPI_Finalize();

	return 0;

}
