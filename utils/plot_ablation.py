import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':
    naive_pd = pd.read_csv('naive.csv')
    index = naive_pd['size'].to_numpy()
    # naive_data = naive_pd['Gflops'].to_numpy()

    # naive_O3_pd = pd.read_csv('naive_O3.csv')
    # naive_O3_data = naive_O3_pd['Gflops'].to_numpy()

    # block_pd = pd.read_csv('block.csv')
    # block_data = block_pd['Gflops'].to_numpy()

    # unrolling_a_pd = pd.read_csv('unrolling_a.csv')
    # unrolling_a_data = unrolling_a_pd['Gflops'].to_numpy()

    # unrolling_b_pd = pd.read_csv('unrolling_b.csv')
    # unrolling_b_data = unrolling_b_pd['Gflops'].to_numpy()

    # register_pd = pd.read_csv('register.csv')
    # register_data = register_pd['Gflops'].to_numpy()

    # simd_pd = pd.read_csv('simd.csv')
    # simd_data = simd_pd['Gflops'].to_numpy()

    alignment_pd = pd.read_csv('alignment.csv')
    alignment_data = alignment_pd['Gflops'].to_numpy()

    alignment_reset_pd_64 = pd.read_csv('alignment-reset-64.csv')
    alignment_reset_data_64 = alignment_reset_pd_64['Gflops'].to_numpy()

    alignment_reset_pd_68 = pd.read_csv('alignment-reset-68.csv')
    alignment_reset_data_68 = alignment_reset_pd_68['Gflops'].to_numpy()

    alignment_reset_pd_132 = pd.read_csv('alignment-reset-132.csv')
    alignment_reset_data_132 = alignment_reset_pd_132['Gflops'].to_numpy()

    alignment_reset_pd_260 = pd.read_csv('alignment-reset-260.csv')
    alignment_reset_data_260 = alignment_reset_pd_260['Gflops'].to_numpy()

    alignment_reset_pd_516 = pd.read_csv('alignment-reset-516.csv')
    alignment_reset_data_516 = alignment_reset_pd_516['Gflops'].to_numpy()

    # plt.plot(index, naive_data, label='naive -O0', color='red', marker='x', linestyle='--')
    # plt.plot(index, naive_O3_data, label='naive -O3', color='orange', marker='x', linestyle='--')
    # plt.plot(index, block_data, label='block_size=64', color='yellow', marker='x', linestyle='--')
    # plt.plot(index, unrolling_a_data, label='block_size=64, unrolling_a', color='green', marker='x', linestyle='--')
    # plt.plot(index, unrolling_b_data, label='block_size=64, unrolling_b', color='blue', marker='x', linestyle='--')
    # plt.plot(index, register_data, label='block_size=64, register', color='purple', marker='x', linestyle='--')
    plt.plot(index, alignment_data, label='block_size=64, alignment only', color='black', marker='x', linestyle='--')
    plt.plot(index, alignment_reset_data_64, label='block_size=64, alignment + reset', color='red', marker='x', linestyle='--')
    plt.plot(index, alignment_reset_data_68, label='block_size=68, alignment + reset', color='orange', marker='x', linestyle='--')
    plt.plot(index, alignment_reset_data_132, label='block_size=132, alignment + reset', color='yellow', marker='x', linestyle='--')
    plt.plot(index, alignment_reset_data_260, label='block_size=260, alignment + reset', color='green', marker='x', linestyle='--')
    plt.plot(index, alignment_reset_data_516, label='block_size=516, alignment + reset', color='blue', marker='x', linestyle='--')

    plt.xlabel('Matrix size')
    plt.ylabel('Gflops')
    plt.title('Matrix Multiplication')
    plt.legend()
    plt.show()

