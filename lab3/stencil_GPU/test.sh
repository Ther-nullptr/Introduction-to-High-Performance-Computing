# !/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <executable>" >&2
  exit 1
fi

export  DAPL_DBG_TYPE=0

DATAPATH='/mnt/c/Users/86181/Desktop/Introduction to High Performance Computing/lab/lab3/stencil_GPU/'

$1 27 256 256 256 16 ${DATAPATH}/stencil_data_256x256x256 ${DATAPATH}/stencil_answer_27_256x256x256_16steps
$1 27 384 384 384 16 ${DATAPATH}/stencil_data_384x384x384 ${DATAPATH}/stencil_answer_27_384x384x384_16steps
$1 27 512 512 512 16 ${DATAPATH}/stencil_data_512x512x512 ${DATAPATH}/stencil_answer_27_512x512x512_16steps

# srun -n 1