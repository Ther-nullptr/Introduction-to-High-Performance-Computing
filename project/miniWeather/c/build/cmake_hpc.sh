#!/bin/bash

export TEST_MPI_COMMAND="mpirun -n 1"

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=mpicxx                                                     \
      -DCXXFLAGS="-O3 -std=c++11 -I$(spack location -i parallel-netcdf)/include"   \
      -DLDFLAGS="-L$(spack location -i parallel-netcdf)/lib -lpnetcdf"                        \
      -DOPENACC_FLAGS="-fopenacc -foffload=\"-lm -latomic\""                          \
      -DOPENMP_FLAGS="-fopenmp"                                                       \
      -DNX=200                                                                        \
      -DNZ=100                                                                        \
      -DDATA_SPEC="DATA_SPEC_GRAVITY_WAVES"                                           \
      -DSIM_TIME=1000                                                                 \
      ..

