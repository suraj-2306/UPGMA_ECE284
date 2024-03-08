#include "upgmaSerial.h"

#define MATDIM 10

int distMat[MATDIM][MATDIM];

int opMat[MATDIM][MATDIM];

int clusterLst[MATDIM][MATDIM];

void initClusterLst(int size) {
  for (int i = 0; i < size; i++) {
    clusterLst[i][0] = i;
  }
}
int main() {
  initClusterLst(MATDIM);
  printf("UPGMA");
  return 0;
}
