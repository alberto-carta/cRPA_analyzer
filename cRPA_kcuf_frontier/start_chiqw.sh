#!/bin/bash

#SBATCH -n 256
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=1000
#SBATCH --job-name=test_donly
#SBATCH --output=std_euler_%j.out
#SBATCH --error=std_euler_%j.err

#QE_DIR=/path/to/qe-6.6/
#WAN90_DIR=/path/to/wannier90-3.1.0/
#RES=/cluster/home/pmlkvik/respack-master/RESPACK-20200113/build/PATH_TO_INSTALL/bin/
WAN2RES=/cluster/project/spaldin/wan2respack/1.0/bin/

source /cluster/project/spaldin/use_new_repository.sh
module load intel/2022.1.2
module load quantum_espresso/7.1
module load respack/1.0 wan2respack/1.0


seedname=kcuf
ncpu=256

echo "Number of OPENMP threads"
echo $OMP_NUM_THREADS



# RESPACK
echo "starting RESPACK"
mpirun  calc_chiqw < respack.in > LOG.chiqw
#
calc_w3d < respack.in > LOG.w3d
calc_j3d < respack.in > LOG.j3d


