# covec (covectorizer: higher order extension of sgns)

This code implements a higher order extension of Skipgram with negative sampling (SGNS), which is based on CP-decomposition.

Likewise SGNS, covec implicitly decomposes the multivariate pointwise mutual information (MPMI) matrix:

![Figure 1. decomposition of MPMI-matrix](img/covec_decomposition.png)

where < u,v,...,w > is a higher order inner product of same dimensional vectors u, v, ..., w.

## build

Run build.sh on the top of the repository, then covec will be created in the build directory.

## sample

Each line of the covec's input file should be tab-separated and has same number of fields.
Sampele of this is like bellow:

```
the cat sat
cat sat on
sat on the
...
on the mat
```

In the above sample, each line indicates a cooccurrences of elements (eg. ("the", "cat", "sat") cooccur).

You can see the all options by putting help options "-h" to covec.

```bash
$ covec -h
usage: ./covec -i input_file [ options ]
Options and arguments:
--dim, -d DIM=128                       : the dimension of vectors
--batch_size, -b BATCH_SIZE=32          : the (mini)batch size
--num_epochs, -n NUM_EPOCHS=1           : the number of epochs
--neg_size, -N NEGSIZE=1                : the size of negative sampling
--sigma, -s SIGMA=0.1                   : initialize each element of vector with Normal(0, SIGMA)
--eta0, -e ETA0=0.005                   : initial learning rate for SGD
--eta1, -E ETA1=0.005                   : final learning rate for SGD
--input_file, -i INPUT_FILE             : input file. supposed that each line is separated by SEP
--output_prefix, -o OUTPUT_PREFIX="vec" : output file prefix
--sep, -S SEP='	'                       : separator of each line in INPUT_FILE
--help, -h                              : show this help message
```

