#!/usr/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH -o result.txt

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <executable> <number of processes>" >&2
  exit 1
fi

export DAPL_DBG_TYPE=0

DATAPATH=/home/2023-spring/data/stencil_data

srun -N 1 -n $2 ./$1 27 384 384 384 16 ${DATAPATH}/stencil_data_384x384x384 ${DATAPATH}/stencil_answer_27_384x384x384_16steps
#srun -N 1 -n $2 ./$1 27 512 512 512 16 ${DATAPATH}/stencil_data_512x512x512 ${DATAPATH}/stencil_answer_27_512x512x512_16steps
