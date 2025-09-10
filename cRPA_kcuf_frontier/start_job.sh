#!/bin/bash

#SBATCH -n 256
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=5000
#SBATCH --job-name=al-test
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

# scf
mpirun -n $ncpu pw.x -nk 4 < ${seedname}.scf.in > ${seedname}.scf.out

# nscf IBZ calc
mpirun -n $ncpu pw.x -nk 4 < ${seedname}.nscf.in > ${seedname}.nscf.out

# pre-process
python ${WAN2RES}wan2respack.py -pp conf.toml

# fix kpoint issue
sed -i -e 's/nscf/bands/g' ${seedname}.nscf_wannier.in

# wannier90 run
mpirun -n $ncpu pw.x -nk 4 < ${seedname}.nscf_wannier.in > ${seedname}.nscf_wannier.out
wannier90.x -pp ${seedname}
mpirun -n 36 pw2wannier90.x -pd .true. < ${seedname}.pw2wan.in > ${seedname}.pw2wan.out
wannier90.x ${seedname}

# wannier90 results to RESPACK inputs
python ${WAN2RES}wan2respack.py conf.toml

# RESPACK
mpirun -np $ncpu calc_chiqw < respack.in > LOG.chiqw

calc_w3d < respack.in > LOG.w3d
calc_j3d < respack.in > LOG.j3d


