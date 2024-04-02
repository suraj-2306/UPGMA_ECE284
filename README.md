# ECE284_UPGMA

Parallel Accelaration of UPGMA algorithm

UPGMA algorithm is used to create phylogenetic trees from raw base pair data of genetic material. The UPGMA algorithm has a cubical complexity of `n` in time and space, where `n` is the sequence length. A typical genetic material sequence runs into billions of base pairs. Hence, it is not feasible to deploy the algorithm serially for practical purposes. To accelerate the algorithm, we propose and implement several parallelization techniques using GPUs.

How to run?
1. Test data is available in src/parallel/ as csv files. Copy the desired configuration into distMat.csv.
2. Open src/parallel/upgma.cuh and updated MATDIM macro to match the desired matrix dimension.
3. Build and run the file src/parallel/main.cpp. This can be done using the available makelist and run sh script.
   Use the following command: ssh -i ~/.ssh/ssh_key  <dsmlp-login>.ucsd.edu /opt/launch-sh/bin/launch.sh -v 2080ti -c 8 -g 1 -m 8 -i yatisht/ece284-wi24:latest -f ./ECE284_UPGMA/run-commands.sh
