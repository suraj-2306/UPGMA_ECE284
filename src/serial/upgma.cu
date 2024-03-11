#include "upgma.cuh"
#include <stdio.h>

/**
 * Prints information for each available GPU device on stdout
 */

void printGpuProperties () {
	int nDevices;

	// Store the number of available GPU device in nDevicess
	cudaError_t err = cudaGetDeviceCount(&nDevices);

	if (err != cudaSuccess) {
		fprintf(stderr, "GPU_ERROR: cudaGetDeviceCount failed!\n");
		exit(1);
	}

	// For each GPU device found, print the information (memory, bandwidth etc.)
	// about the device
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Device memory: %lu\n", prop.totalGlobalMem);
		printf("  Memory Clock Rate (KHz): %d\n",
				prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n",
				prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n",
				2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
	}
}

uint32_t getIndex(uint32_t numCols, uint32_t i, uint32_t j) {
	return i*numCols + j;
}

UPGMA::ReadDistMat::ReadDistMat(uint32_t size) {
	mat_dim = size;

	cudaError_t err;

	distMat    = new uint32_t [mat_dim*mat_dim];
	opMat      = new uint32_t [mat_dim*mat_dim];
	clusterLst = new int [mat_dim*mat_dim];

	err = cudaMalloc(&d_distMat, mat_dim*mat_dim*sizeof(uint32_t));
	if (err != cudaSuccess) {
		fprintf(stderr, "GPU_ERROR: cudaMalloc failed!\n");
		exit(1);
	}

	err = cudaMalloc(&d_opMat, mat_dim*mat_dim*sizeof(uint32_t));
	if (err != cudaSuccess) {
		fprintf(stderr, "GPU_ERROR: cudaMalloc failed!\n");
		exit(1);
	}

	err = cudaMalloc(&d_clusterLst, mat_dim*mat_dim*sizeof(uint32_t));
	if (err != cudaSuccess) {
		fprintf(stderr, "GPU_ERROR: cudaMalloc failed!\n");
		exit(1);
	}
}

void UPGMA::readFile(UPGMA::ReadDistMat* readDistMat) {

	printf("Reading from file\n");

	std::string filename = "./../src/serial/distMat.csv";
	std::ifstream file(filename);
	std::string line;    

	if (!file.is_open()) {
		std::cerr << "Error: Unable to open file " << filename << std::endl;
		return;
	}

	int i = 0;
	int j = 0;

	// Read each line of the CSV file
	while (std::getline(file, line)) {
		std::stringstream ss(line);
		int num;
		j = 0;
		// Parse each element in the line
		while (ss >> num) {
			readDistMat->distMat[getIndex(readDistMat->mat_dim, i, j)] = num;
			if (ss.peek() == ',') // Skip the comma
				ss.ignore();
			j++;
		}
		i++;
	}

	file.close();
}

void UPGMA::transferDistMat(UPGMA::ReadDistMat* readDistMat) {
	cudaError_t err;

	uint32_t numReads = readDistMat->mat_dim;

	err = cudaMemcpy(readDistMat->d_distMat, readDistMat->distMat, numReads*numReads*sizeof(uint32_t), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "GPU_ERROR: cudaMemCpy failed!\n");
		exit(1);
	}
}

void UPGMA::printDistMat(UPGMA::ReadDistMat* readDistMat) {

	uint32_t numReads = readDistMat->mat_dim;
	cudaError_t err;

	err = cudaMemcpy(readDistMat->distMat, readDistMat->d_distMat, numReads*numReads*sizeof(uint32_t), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		fprintf(stderr, "GPU_ERROR: cudaMemCpy failed!\n");
		exit(1);
	}

	printf("Distance Matrix:\n");

	for (uint64_t i=0; i < readDistMat->mat_dim; i++) {
		for (uint64_t j=0; j < readDistMat->mat_dim; j++) {
			printf("%u ", readDistMat->distMat[getIndex(readDistMat->mat_dim, i,j)]);
		}
		printf("\n");
	}
}

void UPGMA::printOpMat(UPGMA::ReadDistMat* readDistMat) {

	uint32_t numReads = readDistMat->mat_dim;
	cudaError_t err;

	err = cudaMemcpy(readDistMat->opMat, readDistMat->d_opMat, numReads*numReads*sizeof(uint32_t), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		fprintf(stderr, "GPU_ERROR: cudaMemCpy failed!\n");
		exit(1);
	}

	printf("Operation Matrix:\n");

	for (uint64_t i=0; i < readDistMat->mat_dim; i++) {
		for (uint64_t j=0; j < readDistMat->mat_dim; j++) {
			printf("%u ", readDistMat->opMat[getIndex(readDistMat->mat_dim, i,j)]);
		}
		printf("\n");
	}
}

void UPGMA::printClusterLst(UPGMA::ReadDistMat* readDistMat) {

	uint32_t numReads = readDistMat->mat_dim;
	cudaError_t err;

	err = cudaMemcpy(readDistMat->clusterLst, readDistMat->d_clusterLst, numReads*numReads*sizeof(uint32_t), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		fprintf(stderr, "GPU_ERROR: cudaMemCpy failed!\n");
		exit(1);
	}

	printf("Cluster List:\n");

	for (uint64_t i=0; i < readDistMat->mat_dim; i++) {
		for (uint64_t j=0; j < readDistMat->mat_dim; j++) {
			printf("%u ", readDistMat->clusterLst[getIndex(readDistMat->mat_dim, i,j)]);
		}
		printf("\n");
	}
}

void UPGMA::clearDistMat(UPGMA::ReadDistMat* readDistMat) {
	delete[] readDistMat->distMat;
	delete[] readDistMat->opMat;
	delete[] readDistMat->clusterLst;

	cudaFree(readDistMat->d_distMat);
	cudaFree(readDistMat->d_opMat);
	cudaFree(readDistMat->d_clusterLst);

	delete readDistMat;
}

__device__ uint32_t getIndexDev(uint32_t numCols, uint32_t i, uint32_t j) {
	return numCols*i + j;
}

__global__ void buildUpgma(
		uint32_t mat_dim,
		uint32_t* d_distMat,
		uint32_t* d_opMat,
		int* d_clusterLst) {

	int tx = threadIdx.x;
	int bx = blockIdx.x;
	// int gs = gridDim.x;
	// int bs = blockDim.x;	

	// printf("Starting buildUPGMA kernel\n");

	// for (int i = 0; i < 1000; i++){
	//     10+20;
	// }

	// d_distMat[getIndexDev(mat_dim,tx,bx)] += 100;

//	uint64_t end_condition = batchSize - 1;
//	uint64_t delta = end_condition / gs + 1; // Reads Per Block
//	uint64_t start = (bx)*delta;
//	uint64_t end =(start + delta - 1 > end_condition) ? end_condition : start + delta - 1;

	if(tx==0 && bx==0 ) {

		//1. mirrorDistMat(MATDIM);
		//void mirrorDistMat(int size) {
		printf("Starting buildUPGMA kernel\n");
		int *minLoc;
		int greaterCombLoc;
		for (int itr1_i = 0; itr1_i < mat_dim; itr1_i++) {
			for (int itr1_j = 0; itr1_j < itr1_i; itr1_j++)
				d_distMat[getIndexDev(mat_dim,itr1_j,itr1_i)] = d_distMat[getIndexDev(mat_dim,itr1_i,itr1_j)];
		}
		//}

		//2. initClusterLst(mat_dim);
		//void initClusterLst(int size) {

		//for (int itr2_i = 0; itr2_i < mat_dim; itr2_i++) {
		//	for (int itr2_j = 0; itr2_j < mat_dim; itr2_j++) {
		//		d_clusterLst[getIndexDev(mat_dim,itr2_i,itr2_j)] = (itr2_j) ? -1 : itr2_i;
		//	}
		//}

		for (int itr2_i = 0; itr2_i < mat_dim; itr2_i++) {
			for (int itr2_j = 0; itr2_j < mat_dim; itr2_j++) {
				d_clusterLst[getIndexDev(mat_dim,itr2_i,itr2_j)] = (itr2_j) ? -2 : itr2_i;
			}
			d_clusterLst[getIndexDev(mat_dim,itr2_i,1)] = -1;
		}
		//}

		for (int itr3_i = 0; itr3_i < mat_dim; itr3_i++) {
			for (int itr3_j = 0; itr3_j < mat_dim; itr3_j++)
				//d_opMat[getIndexDev(mat_dim,itr3_i,itr3_j)] = d_distMat[getIndexDev(mat_dim,itr3_i,itr3_j)]*1000;
				d_distMat[getIndexDev(mat_dim,itr3_i,itr3_j)] *=  1000 ;
		}

		//initd_opMat(mat_dim);
		//void initd_opMat(int size) {
		for (int itr3_i = 0; itr3_i < mat_dim; itr3_i++) {
			for (int itr3_j = 0; itr3_j < mat_dim; itr3_j++)
				//d_opMat[getIndexDev(mat_dim,itr3_i,itr3_j)] = d_distMat[getIndexDev(mat_dim,itr3_i,itr3_j)]*1000;
				d_opMat[getIndexDev(mat_dim,itr3_i,itr3_j)] = d_distMat[getIndexDev(mat_dim,itr3_i,itr3_j)];
		}
		//}

		// printd_opMat(mat_dim);

		for (int k = 0; k < mat_dim-2; k++) {
			//minLoc = matMinLoc(mat_dim);
			//int *matMinLoc(int size) {
			int itr4_min = __INT32_MAX__;
			int itr4_min_loc[2];
			for (int itr4_i = 0; itr4_i < mat_dim; itr4_i++) {
				for (int itr4_j = 0; itr4_j < mat_dim; itr4_j++) {
					if (d_opMat[getIndexDev(mat_dim,itr4_i,itr4_j)] < itr4_min && d_opMat[getIndexDev(mat_dim,itr4_i,itr4_j)] != 0) {
						itr4_min = d_opMat[getIndexDev(mat_dim,itr4_i,itr4_j)];
						itr4_min_loc[0] = itr4_i;
						itr4_min_loc[1] = itr4_j;
					}
				}
			}
			minLoc = itr4_min_loc;
			//}
			printf("Minx %d, Miny %d\n", minLoc[0], minLoc[1]);
			//greaterCombLoc = arrMax(minLoc, 2);
			//int arrMax(int d[mat_dim], int size) {
			int itr5_arr_max = 0;
			for (int itr5_i = 0; itr5_i < 2; itr5_i++)
				if (minLoc[itr5_i] > itr5_arr_max)
					itr5_arr_max = minLoc[itr5_i];
			//return itr5_arr_max;
			greaterCombLoc = itr5_arr_max;
			//}

			//grpLeaf(mat_dim, minLoc[0], minLoc[1]);
			//void grpLeaf(int size, int ele1, int ele2) {
			int *itr6_cluster_locs;
			//itr6_cluster_locs = findCluster(mat_dim, minLoc[0], minLoc[1]);
			//int *findCluster(int size, int ele1, int ele2) {
			int itr7_cluster_locs[2];
			//int itr7_find_count = 0;
			//int itr7_return_flag = 0;
			for (int itr7_i = 0; itr7_i < mat_dim; itr7_i++) {
				for (int itr7_j = 0; itr7_j < mat_dim; itr7_j++) {
					if (minLoc[0] == d_clusterLst[getIndexDev(mat_dim,itr7_i,itr7_j)]) {
						itr7_cluster_locs[0] = itr7_i;
						//itr7_find_count++;
					}
					//} else if (minLoc[1] == d_clusterLst[getIndexDev(mat_dim,itr7_i,itr7_j)]) {
					if (minLoc[1] == d_clusterLst[getIndexDev(mat_dim,itr7_i,itr7_j)]) {
						itr7_cluster_locs[1] = itr7_i;
						//itr7_find_count++;
					}

					//if (itr7_find_count == 2) {
					//	//return itr7_cluster_locs;
					//	itr6_cluster_locs = itr7_cluster_locs;
					//	//itr7_return_flag = 1;
					//	//break;
					//}
				}
				//if(itr7_return_flag == 1) {
				//	break;
				//}
			}

			//return itr7_cluster_locs;
			//if(itr7_return_flag == 0 ) {
			//	itr6_cluster_locs = itr7_cluster_locs;
			//}
			//}
			itr6_cluster_locs = itr7_cluster_locs;
			int itr6_j_save = 0;
			if (itr6_cluster_locs[0] != itr6_cluster_locs[1]) {
				int itr6_j;
				//for each thread
				for (itr6_j = 0; itr6_j < mat_dim; itr6_j++) {
					//if (d_clusterLst[getIndexDev(mat_dim,itr6_cluster_locs[0],itr6_j)] == -1)

					// Guard for last iteration? Branch divergence?

					if (d_clusterLst[getIndexDev(mat_dim,itr6_cluster_locs[0],itr6_j)] == -1) // && in the same row)
						//break;  
						itr6_j_save = itr6_j;  
				}
				// syncthread
				// for each thread
				for (int itr6_k = 0; itr6_k < mat_dim; itr6_k++) {
					int itr6_ele = d_clusterLst[getIndexDev(mat_dim,itr6_cluster_locs[1],itr6_k)];
					//if (itr6_ele == -1)
					//	break;
					if (itr6_ele >= -1) {
						d_clusterLst[getIndexDev(mat_dim,itr6_cluster_locs[0],itr6_k+itr6_j_save)] = itr6_ele;
						d_clusterLst[getIndexDev(mat_dim,itr6_cluster_locs[1],itr6_k)] = -2;
					//itr6_j++;
					}
				}
			} 
			//}

			for (int i = 0; i < mat_dim; i++) {
				for (int j = 0; j < mat_dim; j++) {
					if (d_opMat[getIndexDev(mat_dim,i,j)] != 0) {
						bool itr8_condition0 = false ;
						bool itr8_condition1 = false ;
						//int isEleInArr (int minLoc[mat_dim], int 2, int i) {
						for (int itr8_i = 0; itr8_i < 2; itr8_i++)
							if (i == minLoc[itr8_i])
								//return 1;
								itr8_condition0 = true;
						//}
						for (int itr8_i = 0; itr8_i < 2; itr8_i++)
							if (j == minLoc[itr8_i])
								//return 1;
								itr8_condition1 = true;
						//}
						//if (isEleInArr(minLoc, 2, i) || isEleInArr(minLoc, 2, j)) {
						if ( itr8_condition0 || itr8_condition1 ) {
							if (i == greaterCombLoc || j == greaterCombLoc)
								d_opMat[getIndexDev(mat_dim,i,j)] = 0;
							else
								if (i != j) {
									//matEntryUpdate(mat_dim, i, j);
									//void matEntryUpdate(int size, int ele1, int ele2) {
									int *itr9_cluster_loc;
									//itr9_cluster_loc = findCluster(mat_dim, i, j);
									//int *findCluster(int size, int ele1, int ele2) {
									int itr10_cluster_locs[2];
									int itr10_find_count = 0;
									int itr10_return_flag = 0;
									for (int itr10_i = 0; itr10_i < mat_dim; itr10_i++) {
										for (int itr10_j = 0; itr10_j < mat_dim; itr10_j++) {
											if (i == d_clusterLst[getIndexDev(mat_dim,itr10_i,itr10_j)]) {
												itr10_cluster_locs[0] = itr10_i;
												itr10_find_count++;
											} else if (j == d_clusterLst[getIndexDev(mat_dim,itr10_i,itr10_j)]) {
												itr10_cluster_locs[1] = itr10_i;
												itr10_find_count++;
											}

											if (itr10_find_count == 2) {
												//return itr10_cluster_locs;
												itr10_return_flag = 1;
												itr9_cluster_loc = itr10_cluster_locs;
												break;
											}
										}
										if( itr10_return_flag == 1 ) {
											break;
										}
									}

									//return itr10_cluster_locs;
									if( itr10_return_flag == 0 ) {
										itr9_cluster_loc = itr10_cluster_locs;
									}
									//}
									int itr9_distSum = 0;
									int itr9_distCount = 0;
									int itr9_e1_itr = 0;
									int itr9_e2_itr = 0;  

									while (d_clusterLst[getIndexDev(mat_dim,itr9_cluster_loc[0],itr9_e1_itr)] != -1) {
										itr9_e2_itr = 0;
										while (d_clusterLst[getIndexDev(mat_dim,itr9_cluster_loc[1],itr9_e2_itr)] != -1) {
											itr9_distSum += d_distMat[getIndexDev(mat_dim, d_clusterLst[getIndexDev(mat_dim, itr9_cluster_loc[0], itr9_e1_itr)], d_clusterLst[getIndexDev(mat_dim, itr9_cluster_loc[1], itr9_e2_itr)])];
											itr9_distCount++;
											itr9_e2_itr++;
										}
										itr9_e1_itr++;
									}

									d_opMat[getIndexDev(mat_dim,i,j)] = itr9_distSum/itr9_distCount;
									//}
								}
						}
				}
				}
			}
			// printClusterLst(mat_dim);
			// printOpMat(mat_dim);
		}
	}
}

void UPGMA::upgmaBuilder (UPGMA::ReadDistMat* readDistMat) {

	printf("upgmaBuilder invoked\n");
	int numBlocks = 1; // i.e. number of thread blocks on the GPU
	int blockSize = 8;  // i.e. number of GPU threads per thread block

	buildUpgma<<<numBlocks, blockSize>>>(readDistMat->mat_dim, readDistMat->d_distMat, readDistMat->d_opMat, readDistMat->d_clusterLst);    

	cudaError_t err;

	err = cudaMemcpy(readDistMat->distMat, readDistMat->d_distMat, readDistMat->mat_dim*readDistMat->mat_dim*sizeof(uint32_t), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		fprintf(stderr, "GPU_ERROR: cudaMemCpy failed!\n");
		exit(1);
	}

	cudaDeviceSynchronize();
}
