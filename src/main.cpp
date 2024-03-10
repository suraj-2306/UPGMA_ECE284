#include "upgmaSerial.h"
#include <stdio.h>

using namespace std;

int distMat[MATDIM][MATDIM] = {0,0,0,0,0,0,0,
	19,0,0,0,0,0,0,
	27,31,0,0,0,0,0,
	8,18,26,0,0,0,0,
	33,36,41,31,0,0,0,
	18,1,32,17,35,0,0,
	13,13,29,14,28,12,0};

int opMat[MATDIM][MATDIM];

int clusterLst[MATDIM][MATDIM];

void initClusterLst(int size) {
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			clusterLst[i][j] = (j) ? -1 : i;
		}
	}
}

void initOpMat(int size) {
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++)
			opMat[i][j] = distMat[i][j]*1000;
	}
}

void mirrorDistMat(int size) {
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < i; j++)
			distMat[j][i] = distMat[i][j];
	}
}

int *matMinLoc(int size) {
	int min = __INT32_MAX__;
	static int min_loc[2];
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			if (opMat[i][j] < min && opMat[i][j] != 0) {
				min = opMat[i][j];
				min_loc[0] = i;
				min_loc[1] = j;
			}
		}
	}
	return min_loc;
}

int arrMax(int d[MATDIM], int size) {
	int arr_max = 0;
	for (int i = 0; i < size; i++)
		if (d[i] > arr_max)
			arr_max = d[i];
	return arr_max;
}

int arrMin(int d[MATDIM], int size) {
	int arr_min = __INT_MAX__;
	for (int i = 0; i < size; i++)
		if (d[i] < arr_min)
			arr_min = d[i];
	return arr_min;
}

int isEleInArr (int d[MATDIM], int size, int ele) {
	for (int i = 0; i < size; i++)
		if (ele == d[i])
			return 1;
	return 0;
}

int *findCluster(int size, int ele1, int ele2) {
	static int cluster_locs[2];
	int find_count = 0;
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			if (ele1 == clusterLst[i][j]) {
				cluster_locs[0] = i;
				find_count++;
			} else if (ele2 == clusterLst[i][j]) {
				cluster_locs[1] = i;
				find_count++;
			}

			if (find_count == 2)
				return cluster_locs;
		}
	}

	return cluster_locs;
}

void grpLeaf(int size, int ele1, int ele2) {
	int *cluster_locs;
	cluster_locs = findCluster(size, ele1, ele2);

	if (cluster_locs[0] != cluster_locs[1]) {
		int j;
		for (j = 0; j < size; j++) {
			if (clusterLst[cluster_locs[0]][j] == -1)
				break;  
		}
		for (int k = 0; k < size; k++) {
			int ele = clusterLst[cluster_locs[1]][k];
			if (ele == -1)
				break;
			clusterLst[cluster_locs[0]][j] = ele;
			clusterLst[cluster_locs[1]][k] = -1;
			j++;
		}
	} 
}

void matEntryUpdate(int size, int ele1, int ele2) {
	int *cluster_loc;
	cluster_loc = findCluster(size, ele1, ele2);
	int distSum = 0;
	int distCount = 0;
	int e1_iter = 0;
	int e2_iter = 0;  

	while (clusterLst[cluster_loc[0]][e1_iter] != -1) {
		e2_iter = 0;
		while (clusterLst[cluster_loc[1]][e2_iter] != -1) {
			distSum += distMat[clusterLst[cluster_loc[0]][e1_iter]][clusterLst[cluster_loc[1]][e2_iter]];
			distCount++;
			e2_iter++;
		}
		e1_iter++;
	}

	opMat[ele1][ele2] = distSum/distCount;
}

void printClusterLst(int size) {
	printf("Cluster list: \n");
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			printf("%2d ", clusterLst[i][j]);      
		}
		printf("\n");
	}
}

void printMat(int size) {
	printf("Distance Matrix: \n");
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			printf("%0.1f ", distMat[i][j]);      
		}
		printf("\n");
	}
}

void printOpMat(int size) {
	printf("Operation Matrix: \n");
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			printf("%2d ", opMat[i][j]);      
		}
		printf("\n");
	}
}

int main() {

	int *minLoc;
	int greaterCombLoc;

	mirrorDistMat(MATDIM);

	initClusterLst(MATDIM);

	initOpMat(MATDIM);

	printOpMat(MATDIM);

	for (int k = 0; k < MATDIM-2; k++) {
		int itr_size = MATDIM;
		int min = __INT32_MAX__;
		int min_loc[2];
		for (int itr_i = 0; itr_i < itr_size; itr_i++) {
			for (int itr_j = 0; itr_j < itr_size; itr_j++) {
				if (opMat[itr_i][itr_j] < min && opMat[itr_i][itr_j] != 0) {
					min = opMat[itr_i][itr_j];
					min_loc[0] = itr_i;
					min_loc[1] = itr_j;
				}
			}
		}
		//return min_loc;
		minLoc = min_loc;
		//minLoc = matMinLoc(MATDIM);
		printf("Minx %d, Miny %d\n", minLoc[0], minLoc[1]);
		int itr_arr_max = 0;
		for (int itr_i = 0; itr_i < 2; itr_i++)
			if (minLoc[itr_i] > itr_arr_max)
				itr_arr_max = minLoc[itr_i];
		//return itr_arr_max;
		//greaterCombLoc = arrMax(minLoc, 2);
		greaterCombLoc = itr_arr_max;

		//grpLeaf(MATDIM, minLoc[0], minLoc[1]);
		//void grpLeaf(int size, int ele1, int ele2) {

		int ele1 =  minLoc[0];
		int ele2 =  minLoc[1];
		//int *cluster_locs;
		//cluster_locs = findCluster(itr_size, ele1, ele2);
		//int *findCluster(int size, int ele1, int ele2) {
		//static int cluster_locs[2];
		int cluster_locs[2];
		int itr3_find_count = 0;
		for (int itr3_i = 0; itr3_i < MATDIM; itr3_i++) {
			for (int itr3_j = 0; itr3_j < MATDIM; itr3_j++) {
				if (ele1 == clusterLst[itr3_i][itr3_j]) {
					cluster_locs[0] = itr3_i;
					itr3_find_count++;
				} else if (ele2 == clusterLst[itr3_i][itr3_j]) {
					cluster_locs[1] = itr3_i;
					itr3_find_count++;
				}

				if (itr3_find_count == 2)
					break;
				//return cluster_locs;
			}
		}

		//return cluster_locs;
		//}

		if (cluster_locs[0] != cluster_locs[1]) {
			int itr2_j;
			for (itr2_j = 0; itr2_j < itr_size; itr2_j++) {
				if (clusterLst[cluster_locs[0]][itr2_j] == -1)
					break;  
			}
			for (int itr_k = 0; itr_k < itr_size; itr_k++) {
				int ele = clusterLst[cluster_locs[1]][itr_k];
				if (ele == -1)
					break;
				clusterLst[cluster_locs[0]][itr2_j] = ele;
				clusterLst[cluster_locs[1]][itr_k] = -1;
				itr2_j++;
			}
		} 
		//}

		for (int i = 0; i < MATDIM; i++) {
			for (int j = 0; j < MATDIM; j++) {
				if (opMat[i][j] != 0) {
					bool condition1, condition2;
					condition1 = false;
					condition2 = false;
					for (int itr3_i = 0; itr3_i < 2; itr3_i++)
						if (i == minLoc[itr3_i])
							condition1 = true;
					for (int itr3_i = 0; itr3_i < 2; itr3_i++)
						if (j == minLoc[itr3_i])
							condition2 = true;
					//if (isEleInArr(minLoc, 2, i) || isEleInArr(minLoc, 2, j)) {
					if ( condition1 || condition2 ) {
						if (i == greaterCombLoc || j == greaterCombLoc)
							opMat[i][j] = 0;
						else
							if (i != j)
								matEntryUpdate(MATDIM, i, j);
						//void matEntryUpdate(int size, int i, int j) {
						int *itr4_cluster_loc;
						//int *findCluster(int size, int i, int j) {
						int cluster_locs[2];
						int itr5_find_count = 0;
						for (int itr5_i = 0; itr5_i < MATDIM; itr5_i++) {
							for (int itr5_j = 0; itr5_j < MATDIM; itr5_j++) {
								if (i == clusterLst[itr5_i][itr5_j]) {
									cluster_locs[0] = itr5_i;
									itr5_find_count++;
								} else if (j == clusterLst[itr5_i][itr5_j]) {
									cluster_locs[1] = itr5_i;
									itr5_find_count++;
								}

								if (itr5_find_count == 2)
									break;
								//return cluster_locs;
							}
						}

						//return cluster_locs;
						//}
						itr4_cluster_loc = cluster_locs;
						int distSum = 0;
						int distCount = 0;
						int e1_iter = 0;
						int e2_iter = 0;  

						while (clusterLst[itr4_cluster_loc[0]][e1_iter] != -1) {
							e2_iter = 0;
							while (clusterLst[itr4_cluster_loc[1]][e2_iter] != -1) {
								distSum += distMat[clusterLst[itr4_cluster_loc[0]][e1_iter]][clusterLst[itr4_cluster_loc[1]][e2_iter]];
								distCount++;
								e2_iter++;
							}
							e1_iter++;
						}

						opMat[i][j] = distSum/distCount;
						//}
					}
				}
				}
			}
			printClusterLst(MATDIM);
			printOpMat(MATDIM);
		}

		return 0;
	}
