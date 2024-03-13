#include "upgmaSerial.h"
#include <stdio.h>

using namespace std;

int distMat[MATDIM][MATDIM];

int opMat[MATDIM][MATDIM];

int clusterLst[MATDIM][MATDIM];

void readFile() {

	printf("Reading from file\n");

	std::string filename = "./distMat.csv";
	std::ifstream file(filename);
	std::string line;

	if (!file.is_open()) {
		std::cerr << "Error: Unable to open file " << filename << std::endl;
		return;
	}

	int i = 0;
	int j = 0;

	while (std::getline(file, line)) {
		std::stringstream ss(line);
		int num;
		j = 0;
		while (ss >> num) {
			distMat[i][j] = num;
			if (ss.peek() == ',')
				ss.ignore();
			j++;
		}
		i++;
	}

	file.close();
}

void scaleDistMat(int size) {
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      distMat[i][j] *= 1000;
    }
  }  
}

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
      opMat[i][j] = distMat[i][j];
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
      printf("%2d ", distMat[i][j]);      
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

  readFile();

  scaleDistMat(MATDIM);

  mirrorDistMat(MATDIM);

  initClusterLst(MATDIM);

  initOpMat(MATDIM);

  printOpMat(MATDIM);
  printClusterLst(MATDIM);

  for (int k = 0; k < MATDIM-2; k++) {
    minLoc = matMinLoc(MATDIM);
    printf("Minx %d, Miny %d\n", minLoc[0], minLoc[1]);
    greaterCombLoc = arrMax(minLoc, 2);

    grpLeaf(MATDIM, minLoc[0], minLoc[1]);

    for (int i = 0; i < MATDIM; i++) {
      for (int j = 0; j < MATDIM; j++) {
        if (opMat[i][j] != 0) {
          if (isEleInArr(minLoc, 2, i) || isEleInArr(minLoc, 2, j)) {
            if (i == greaterCombLoc || j == greaterCombLoc)
              opMat[i][j] = 0;
            else
              if (i != j)
                matEntryUpdate(MATDIM, i, j);
          }
        }
      }
    }
    // printClusterLst(MATDIM);
    // printOpMat(MATDIM);
  }

  printOpMat(MATDIM);

  return 0;
}