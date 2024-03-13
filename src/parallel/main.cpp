#include "upgma.cuh"
#include <stdio.h>

int main(int argc, char** argv) {
    // Timer below helps with the performance profiling (see timer.hpp for more
    // details)
    // Timer timer;
 
    // // Print GPU information
    // timer.Start();
    printGpuProperties();
    // fprintf(stdout, "Completed in %ld msec \n\n", timer.Stop());

    uint32_t mat_dim = MATDIM;

    UPGMA::ReadDistMat *readDistMat;

    readDistMat  = new UPGMA::ReadDistMat(mat_dim);

    // for (int i = 0; i < MATDIM*MATDIM; i++)
    //     readDistMat->distMat[i] = i;

    UPGMA::readFile(readDistMat);

    UPGMA::transferDistMat(readDistMat);

    UPGMA::upgmaBuilder(readDistMat);

    //UPGMA::printDistMat(readDistMat);

    //UPGMA::printOpMat(readDistMat);

    UPGMA::clearDistMat(readDistMat);

    return 0;
}
