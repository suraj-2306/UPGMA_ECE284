#include <stdio.h>
// Accepts a 2D matrix and returns a matrix mirror along the left diagonal
void mirrorMat(int **d, int size);

// Returns a list of lists of length equal to the dimension of (square) distance
// matrix. Can implement as a fixed 2D matrix, or linked list. Update the
// prototype as required.
void initClusterLst(int size);

// Return the indexes corresponding to the smallest value in the matrix.
// Return smallest value if required.
int *matMinLoc(int **d, int size);

// Returns the minimum from an array. Use standard functions instead?
int arrMax(int *d, int size);

// Combine the 2 clusters in clusterList that contain ele1 and ele2, IF they
// belong to separate clusters
void grpLeaf(int **clusterLst, int size, int ele1, int ele2);

// Return an array of 2 elements that represent the location of ele1 and ele2 in
// the cluster list of lists. e.g. [ [0, 3], [1, 5], [2], [4], [6] ] if ele1 = 3
// and ele2 = 5, the result should be [0,1]
int *findCluster(int **clusterLst, int size, int ele1, int ele2);

// Recompute distances after cluster update
void matEntryUpdate(int **clusterLst, int size, int **d_ref, int **d_red,
                    int ele1, int ele2);

// Print the 2D distance matrix
void printMat(int **d, int size);

// Print the cluster list
void printClusterLst(int **clusterLst, int size);
