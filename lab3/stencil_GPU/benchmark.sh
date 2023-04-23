# !/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <executable>" >&2
  exit 1
fi

export DAPL_DBG_TYPE=0

DATAPATH='/mnt/c/Users/86181/Desktop/Introduction to High Performance Computing/lab/lab3/stencil_GPU'

$1 27 256 256 256 100 ${DATAPATH}/stencil_data_256x256x256

# srun -n 1 $1 27 256 256 256 100 ${DATAPATH}/stencil_data_256x256x256
#srun -n 1 $1 27 384 384 384 100 ${DATAPATH}/stencil_data_384x384x384
#srun -n 1 $1 27 512 512 512 100 ${DATAPATH}/stencil_data_512x512x512