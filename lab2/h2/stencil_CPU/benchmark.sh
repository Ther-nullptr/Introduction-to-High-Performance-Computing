#!/usr/bin/bash
#SBATCH -N 4
#SBATCH --nodes=4
#BATCH --ntasks-per-node=1

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <executable> <number of processes>" >&2
  exit 1
fi

export DAPL_DBG_TYPE=0

DATAPATH=/home/2023-spring/data/stencil_data

env OMP_NUM_THREADS=$2 ./$1 27 256 256 256 5 ${DATAPATH}/stencil_data_256x256x256
