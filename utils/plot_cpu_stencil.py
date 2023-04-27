import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    x = np.array([1, 2, 4, 8, 16, 32, 48])
    y_naive_256 = np.array([1.600080, 3.177852, 6.085051, 12.603111, 20.932931, 26.761711, 25.030841])
    y_optimized_256 = np.array([8.798965, 17.335260, 34.816358, 65.009637, 89.015222, 99.487130, 105.418242])
    y_optimized_384 = np.array([8.812063, 17.080109, 33.657639, 64.623618, 95.309923, 98.680137, 102.156332])
    y_optimized_512 = np.array([8.785894, 16.297017, 33.671108, 65.635183, 96.239425, 94.114308, 95.000145])
    y_blocked_256 = np.array([8.671018, 16.774497, 34.025231, 64.457629, 98.816992, 88.222237, 89.801092])
    y_blocked_384 = np.array([8.121341, 16.987846, 30.519075, 65.628018, 103.380350, 95.634992, 101.326166])
    y_blocked_512 = np.array([8.228519, 17.080713, 33.097172, 65.617022, 95.104934, 95.287720, 101.214532])
    y_mpi_256 = np.array([9.352409, 18.593199, 36.912235, 72.368205, 139.159587, 222.522982, np.nan])
    y_mpi_384 = np.array([9.359576, 18.659575, 36.852993, 71.609775, 124.946529, 238.190426, np.nan])
    y_mpi_512 = np.array([9.044265, 18.183239, 35.940965, 71.477521, 138.643696, 230.143544, np.nan])

    # plot y_naive_256, y_optimized_256, y_optimized_384, y_optimized_512
    plt.figure()
    plt.plot(x, y_naive_256, 'r', label='naive', marker='x', linestyle='--')
    plt.plot(x, y_optimized_256, 'g', label='openmp optimized 256x256x256', marker='x', linestyle='--')
    plt.plot(x, y_optimized_384, 'b', label='openmp optimized 384x384x384', marker='x', linestyle='--')
    plt.plot(x, y_optimized_512, 'y', label='openmp optimized 512x512x512', marker='x', linestyle='--')

    plt.xlabel('Number of Threads')
    plt.ylabel('Gflops')
    plt.title('naive vs. compile time optimized')
    plt.legend()
    plt.savefig('naive_vs_compile_time_optimized.png')

    # plot y_optimized_256, y_optimized_384, y_optimized_512, y_blocked_256, y_blocked_384, y_blocked_512
    plt.figure()
    plt.plot(x, y_optimized_256, 'g', label='openmp 256x256x256', marker='x', linestyle='--')
    plt.plot(x, y_optimized_384, 'b', label='openmp 384x384x384', marker='x', linestyle='--')
    plt.plot(x, y_optimized_512, 'y', label='openmp 512x512x512', marker='x', linestyle='--')
    plt.plot(x, y_blocked_256, 'c', label='openmp blocked 256x256x256', marker='x', linestyle='--')
    plt.plot(x, y_blocked_384, 'm', label='openmp blocked 384x384x384', marker='x', linestyle='--')
    plt.plot(x, y_blocked_512, 'k', label='openmp blocked 512x512x512', marker='x', linestyle='--')

    plt.xlabel('Number of Threads')
    plt.ylabel('Gflops')
    plt.title('block optimized')
    plt.legend()
    plt.savefig('block_optimized.png')

    # plot y_optimized_256, y_optimized_384, y_optimized_512, y_mpi_256, y_mpi_384, y_mpi_512
    plt.figure()
    plt.plot(x, y_optimized_256, 'g', label='openmp optimized 256x256x256', marker='x', linestyle='--')
    plt.plot(x, y_optimized_384, 'b', label='openmp optimized 384x384x384', marker='x', linestyle='--')
    plt.plot(x, y_optimized_512, 'y', label='openmp optimized 512x512x512', marker='x', linestyle='--')
    plt.plot(x, y_mpi_256, 'c', label='mpi 256x256x256', marker='x', linestyle='--')
    plt.plot(x, y_mpi_384, 'm', label='mpi 384x384x384', marker='x', linestyle='--')
    plt.plot(x, y_mpi_512, 'k', label='mpi 512x512x512', marker='x', linestyle='--')

    plt.xlabel('Number of Threads')
    plt.ylabel('Gflops')
    plt.title('mpi vs. openmp')
    plt.legend()
    plt.savefig('mpi_vs_openmp.png')
