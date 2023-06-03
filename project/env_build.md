初始化spack环境后，请按以下步骤操作，完成依赖库的加载和环境变量设置

```bash
spack load gcc@10.4.0
spack load parallel-netcdf
spack load cmake@3.24.3

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(spack location -i parallel-netcdf)/lib
```