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

	//1. mirrorDistMat(MATDIM);
	//void mirrorDistMat(int size) {
	for (int itr1_i = 0; itr1_i < MATDIM; itr1_i++) {
		for (int itr1_j = 0; itr1_j < itr1_i; itr1_j++)
			distMat[itr1_j][itr1_i] = distMat[itr1_i][itr1_j];
	}
	//}

	//2. initClusterLst(MATDIM);
	//void initClusterLst(int size) {
	for (int itr2_i = 0; itr2_i < MATDIM; itr2_i++) {
		for (int itr2_j = 0; itr2_j < MATDIM; itr2_j++) {
			clusterLst[itr2_i][itr2_j] = (itr2_j) ? -1 : itr2_i;
		}
	}
	//}

	//initOpMat(MATDIM);
	//void initOpMat(int size) {
	for (int itr3_i = 0; itr3_i < MATDIM; itr3_i++) {
		for (int itr3_j = 0; itr3_j < MATDIM; itr3_j++)
			opMat[itr3_i][itr3_j] = distMat[itr3_i][itr3_j]*1000;
	}
	//}

	printOpMat(MATDIM);

	for (int k = 0; k < MATDIM-2; k++) {
		//minLoc = matMinLoc(MATDIM);
		//int *matMinLoc(int size) {
		int itr4_min = __INT32_MAX__;
		int itr4_min_loc[2];
		for (int itr4_i = 0; itr4_i < MATDIM; itr4_i++) {
			for (int itr4_j = 0; itr4_j < MATDIM; itr4_j++) {
				if (opMat[itr4_i][itr4_j] < itr4_min && opMat[itr4_i][itr4_j] != 0) {
					itr4_min = opMat[itr4_i][itr4_j];
					itr4_min_loc[0] = itr4_i;
					itr4_min_loc[1] = itr4_j;
				}
			}
		}
		minLoc = itr4_min_loc;
		//}
		printf("Minx %d, Miny %d\n", minLoc[0], minLoc[1]);
		//greaterCombLoc = arrMax(minLoc, 2);
		//int arrMax(int d[MATDIM], int size) {
		int itr5_arr_max = 0;
		for (int itr5_i = 0; itr5_i < 2; itr5_i++)
			if (minLoc[itr5_i] > itr5_arr_max)
				itr5_arr_max = minLoc[itr5_i];
		//return itr5_arr_max;
		greaterCombLoc = itr5_arr_max;
		//}

		//grpLeaf(MATDIM, minLoc[0], minLoc[1]);
		//void grpLeaf(int size, int ele1, int ele2) {
		int *itr6_cluster_locs;
		//itr6_cluster_locs = findCluster(MATDIM, minLoc[0], minLoc[1]);
		//int *findCluster(int size, int ele1, int ele2) {
		int itr7_cluster_locs[2];
		int itr7_find_count = 0;
		int itr7_return_flag = 0;
		for (int itr7_i = 0; itr7_i < MATDIM; itr7_i++) {
			for (int itr7_j = 0; itr7_j < MATDIM; itr7_j++) {
				if (minLoc[0] == clusterLst[itr7_i][itr7_j]) {
					itr7_cluster_locs[0] = itr7_i;
					itr7_find_count++;
				} else if (minLoc[1] == clusterLst[itr7_i][itr7_j]) {
					itr7_cluster_locs[1] = itr7_i;
					itr7_find_count++;
				}

				if (itr7_find_count == 2) {
					//return itr7_cluster_locs;
					itr6_cluster_locs = itr7_cluster_locs;
					itr7_return_flag = 1;
					break;
				}
			}
			if(itr7_return_flag == 1) {
				break;
			}
		}

		//return itr7_cluster_locs;
		if(itr7_return_flag == 0 ) {
			itr6_cluster_locs = itr7_cluster_locs;
		}
		//}

		if (itr6_cluster_locs[0] != itr6_cluster_locs[1]) {
			int itr6_j;
			for (itr6_j = 0; itr6_j < MATDIM; itr6_j++) {
				if (clusterLst[itr6_cluster_locs[0]][itr6_j] == -1)
					break;  
			}
			for (int itr6_k = 0; itr6_k < MATDIM; itr6_k++) {
				int itr6_ele = clusterLst[itr6_cluster_locs[1]][itr6_k];
				if (itr6_ele == -1)
					break;
				clusterLst[itr6_cluster_locs[0]][itr6_j] = itr6_ele;
				clusterLst[itr6_cluster_locs[1]][itr6_k] = -1;
				itr6_j++;
			}
		} 
		//}

		for (int i = 0; i < MATDIM; i++) {
			for (int j = 0; j < MATDIM; j++) {
				if (opMat[i][j] != 0) {
					bool itr8_condition0 = false ;
					bool itr8_condition1 = false ;
					//int isEleInArr (int minLoc[MATDIM], int 2, int i) {
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
							opMat[i][j] = 0;
						else
							if (i != j) {
								//matEntryUpdate(MATDIM, i, j);
								//void matEntryUpdate(int size, int ele1, int ele2) {
								int *itr9_cluster_loc;
								//itr9_cluster_loc = findCluster(MATDIM, i, j);
								//int *findCluster(int size, int ele1, int ele2) {
								int itr10_cluster_locs[2];
								int itr10_find_count = 0;
								int itr10_return_flag = 0;
								for (int itr10_i = 0; itr10_i < MATDIM; itr10_i++) {
									for (int itr10_j = 0; itr10_j < MATDIM; itr10_j++) {
										if (i == clusterLst[itr10_i][itr10_j]) {
											itr10_cluster_locs[0] = itr10_i;
											itr10_find_count++;
										} else if (j == clusterLst[itr10_i][itr10_j]) {
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
								int itr9_e1_iter = 0;
								int itr9_e2_iter = 0;  

								while (clusterLst[itr9_cluster_loc[0]][itr9_e1_iter] != -1) {
									itr9_e2_iter = 0;
									while (clusterLst[itr9_cluster_loc[1]][itr9_e2_iter] != -1) {
										itr9_distSum += distMat[clusterLst[itr9_cluster_loc[0]][itr9_e1_iter]][clusterLst[itr9_cluster_loc[1]][itr9_e2_iter]];
										itr9_distCount++;
										itr9_e2_iter++;
									}
									itr9_e1_iter++;
								}

								opMat[i][j] = itr9_distSum/itr9_distCount;
								//}
							}
					}
			}
			}
		}
		printClusterLst(MATDIM);
		printOpMat(MATDIM);
	}

	return 0;
}
