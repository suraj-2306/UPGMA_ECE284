#include <stdint.h>
#include <vector>
#include <string>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <sstream>
//#include "timer.hpp"

#define MATDIM 100

void printGpuProperties();
uint32_t getIndex(uint32_t numCols, uint32_t i, uint32_t j);

namespace UPGMA {

    struct ReadDistMat {
        uint32_t mat_dim;
        uint32_t* distMat;
        uint32_t* opMat;
        int* clusterLst;
        uint32_t* d_distMat;
        uint32_t* d_opMat;
        int* d_clusterLst;

        ReadDistMat(uint32_t size);
    };    

    void readFile(ReadDistMat* readDistMat);
    void transferDistMat(ReadDistMat* readDistMat);
    void upgmaBuilder(ReadDistMat* readDistMat);
    void printDistMat(ReadDistMat* readDistMat);
    void printOpMat(ReadDistMat* readDistMat);
    void printClusterLst(ReadDistMat* readDistMat);
    void clearDistMat(ReadDistMat* readDistMat);

}
