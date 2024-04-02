# ECE284_UPGMA

Parallel Accelaration of UPGMA algorithm

UPGMA algorithm is used to create phylogenetic trees from raw base pair data of genetic material. The UPGMA algorithm has a cubical complexity of `n` in time and space, where `n` is the sequence length. A typical genetic material sequence runs into billions of base pairs. Hence, it is not feasible to deploy the algorithm serially for practical purposes. To accelerate the algorithm, we propose and implement several parallelization techniques using GPUs.
