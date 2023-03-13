import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':
    naive_pd = pd.read_csv('naive.csv')
    index = naive_pd['size'].to_numpy()
    naive_data = naive_pd['Gflops'].to_numpy()

    naive_O3_pd = pd.read_csv('naive_O3.csv')
    naive_O3_data = naive_O3_pd['Gflops'].to_numpy()

    block_pd = pd.read_csv('block.csv')
    block_data = block_pd['Gflops'].to_numpy()

    unrolling_a_pd = pd.read_csv('unrolling_a.csv')
    unrolling_a_data = unrolling_a_pd['Gflops'].to_numpy()

    unrolling_b_pd = pd.read_csv('unrolling_b.csv')
    unrolling_b_data = unrolling_b_pd['Gflops'].to_numpy()

    register_pd = pd.read_csv('register.csv')
    register_data = register_pd['Gflops'].to_numpy()

    simd_pd = pd.read_csv('simd.csv')
    simd_data = simd_pd['Gflops'].to_numpy()

    plt.plot(index, naive_data, label='naive -O0', color='red', marker='x', linestyle='--')
    plt.plot(index, naive_O3_data, label='naive -O3', color='orange', marker='x', linestyle='--')
    plt.plot(index, block_data, label='block_size=64', color='yellow', marker='x', linestyle='--')
    plt.plot(index, unrolling_a_data, label='block_size=64, unrolling_a', color='green', marker='x', linestyle='--')
    plt.plot(index, unrolling_b_data, label='block_size=64, unrolling_b', color='blue', marker='x', linestyle='--')
    plt.plot(index, register_data, label='block_size=64, register', color='purple', marker='x', linestyle='--')
    plt.plot(index, simd_data, label='block_size=64, simd', color='black', marker='x', linestyle='--')

    plt.xlabel('Matrix size')
    plt.ylabel('Gflops')
    plt.title('Matrix Multiplication')
    plt.legend()
    plt.show()

