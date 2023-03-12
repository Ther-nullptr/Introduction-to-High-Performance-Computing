source /home/software/spack/share/spack/setup-env.sh
spack load gcc@10.4.0
spack compiler find
spack load openmpi
spack load intel-oneapi-compilers
spack load intel-oneapi-mkl@2022.2.1
spack load osu-micro-benchmarks ^openmpi