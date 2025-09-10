#!/bin/bash

echo "exporting number of openmp threads"
export OMP_NUM_THREADS=32

echo "launching the script"
sbatch --ntasks 8 --cpus-per-task 32 start_chiqw.sh
