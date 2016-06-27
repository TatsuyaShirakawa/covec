# covec (COVECtorizer)

This code implements a higher order extension of Skipgram with Negative Sampling (SGNS), which is based on CP-decomposition.

Likewise SGNS, covec implicitly decomposes the multivariate pointwise mutual information (MPMI) matrix:

![Figure 1. decomposition of MPMI-matrix](img/covec_decomposition.png)

where < u,v,...,w > is a higher order inner product of same dimensional vectors u, v, ..., w.

## build

Run build.sh on the top of the repository, then covec is created in the build directory.

## sample

covec's input file is composed of space-separated lines as bellow:

```
the cat sat
cat sat on
sat on the
...
on the mat
```

In the above sample, each line indicates a cooccurrences of elements.


You can see the all options by covec -h.

```bash
usage: build/covec -i input_file [ options ]
Options and arguments:
--dim, -d DIM=128                        : the dimension of vectors
--batch_size, -b BATCH_SIZE=32           : BATCH_SIZE: the (mini)batch size
--num_epochs, -n NUM_EPOCHS=10           : NUM_EPOCHS: the number of epochs
--neg_size, -N NEGSIZE=1                 : the size of negative sampling
--sigma, -s SIGMA=0.1                    : initialize each element of vector with Normal(0, SIGMA)
--eta0, -e ETA0=0.005                    : initial learning rate for AdaGrad
--input_file, -i INPUT_FILE              : input file. supposed that each line is separated by SEP
--output_prefix, -o OUTPUT_PREFIX="covec": output file prefix
--sep, -s SEP=' '                        : separator of each line in INPUT_FILE
--help, -h                               : show this help message

```

## 