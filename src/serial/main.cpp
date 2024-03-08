#include "upgmaSerial.h"

using namespace std;

int distMat[MATDIM][MATDIM] = {2,3,4,
                                9,2,5,
                                7,8,5};

int opMat[MATDIM][MATDIM];

int clusterLst[MATDIM][MATDIM];

int * matMinLoc (int d[MATDIM][MATDIM], int size) {
    int min = __INT32_MAX__;
    static int min_loc[2];
    for (int i = 0; i < size; i++) {
        for (int j = 0; j< size; j++) {
            if (d[i][j] < min) {
                min = d[i][j];
                min_loc[0] = i;
                min_loc[1] = j;
            }
        }
    }

    return min_loc;
}

int arrMax (int d[MATDIM], int size) {
    int arr_max = 0;
    for (int i = 0; i < size; i++)
        if (d[i] > arr_max)
            arr_max = d[i];
    return arr_max;
}

int arrMin (int d[MATDIM], int size) {
    int arr_min = __INT_MAX__;
    for (int i = 0; i < size; i++)
        if (d[i] < arr_min)
            arr_min = d[i];
    return arr_min;
}

int* findCluster (int clusterLst[MATDIM][MATDIM], int size, int ele1, int ele2) {
    static int cluster_locs[2];
    int find_count = 0;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (ele1 == clusterLst[i][j]) {
                cluster_locs[0] = i;
                find_count++;        
            }
            else if (ele2 == clusterLst[i][j]) {
                cluster_locs[1] = i;
                find_count++;
            }

            if (find_count == 2)
                return cluster_locs;
        }
    }
    
    return cluster_locs;
}

void grpLeaf (int clusterLst[MATDIM][MATDIM], int size, int ele1, int ele2) {
    for (int i = 0; i < size; i++) {
        int 
    }
}

int main() {

    int *loc;

    loc = matMinLoc(distMat, MATDIM);

    printf("%d, %d\n", loc[0], loc[1]);

    return 0;
}
