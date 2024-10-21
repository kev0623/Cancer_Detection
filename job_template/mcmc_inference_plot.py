#!/usr/bin/env python

import os

# Restrict multithreading in numpy/scipy/scikit calls to 1 thread
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sys
import numpy as np
from matplotlib import pyplot as plt


def plot_mcmc_results(x_true_used, y_hat4d, y_x_true_used, x_map_scaled, y_x_map, x_mean, std, job_dir, N, iteration, val_min, val_max, num_px, noise_var):
    # some stats
    fig, axs = plt.subplots(3,2, figsize=(20,20))
    im1 = axs[0][0].imshow(x_true_used, vmin=val_min, vmax=val_max)
    fig.colorbar(im1, ax=axs[0][0])
    axs[0][0].set_title(r'$x_{{true}}$')
    #im2 = axs[0][1].imshow(np.flipud(y_hat4d.squeeze().reshape(29,29)))
    im2 = axs[0][1].imshow(y_x_true_used)
    fig.colorbar(im2, ax=axs[0][1])
    axs[0][1].set_title(r'$y_{{meas}}$')

    im3 = axs[1][0].imshow(x_map_scaled, vmin=val_min, vmax=val_max)
    fig.colorbar(im3, ax=axs[1][0])
    axs[1][0].set_title(r'$x_{{map}}$')

    im4 = axs[1][1].imshow(y_x_map)
    fig.colorbar(im4, ax=axs[1][1])
    axs[1][1].set_title(r'$y_{{MAP}}$')

    im5 = axs[2][0].imshow(x_mean[:,:,0], vmin=val_min, vmax=val_max)
    fig.colorbar(im5, ax=axs[2][0])
    axs[2][0].set_title(r'$x_{{Mean}}$')

    im6 = axs[2][1].imshow(std[:,:,0], vmax=val_max)
    fig.colorbar(im6, ax=axs[2][1])
    axs[2][1].set_title(r'$x_{{STD}}$')

    #plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.savefig(f'{job_dir}/stats_N{N}_i{iteration}_px{num_px}_noise{int(noise_var*100)}')

    # np.save('{}/x_true_N{}_i{}.npy'.format(job_dir, N, iteration), x_true)
    # np.save('{}/x_var_N{}_i{}.npy'.format(job_dir, N, iteration), var)
    # np.save('{}/x_mean_N{}_i{}.npy'.format(job_dir, N, iteration), x_mean)
    # np.save('{}/x_map_N{}_i{}.npy'.format(job_dir, N, iteration), x_map)
    plt.close()


def main():
    N = int(sys.argv[1])
    job_dir = sys.argv[3]
    matrix_id = int(sys.argv[2])
    iteration = sys.argv[4]
    nx = int(sys.argv[5])
    num_px = int(sys.argv[6])
    noise_var = float(sys.argv[7])

    print('Inference plotting')
    print(f'N = {N}')
    print(f'job_dir = {job_dir}')
    print(f'matrix_id = {matrix_id}')
    print(f'iteration = {iteration}')
    print(f'num_px = {num_px}')

    data = np.load(f'{job_dir}/statistics_N{N}_i{iteration}_px{num_px}_noise{int(noise_var*100)}.npz')
    x_true_used = data['x_true_used']
    y_hat4d = data['y_hat4d']
    y_x_true_used = data['y_x_true_used']
    x_map_scaled = data['x_map_scaled']
    y_x_map = data['y_x_map']
    var = data['var']
    std = data['std']
    x_mean = data['x_mean']
    val_min = data['val_min']
    val_max = data['val_max']


    '''
    x_true_used = np.load('{}/x_true_used_N{}_i{}.npy'.format(job_dir, N, iteration))
    y_hat4d = np.load('{}/y_hat4d_N{}_i{}.npy'.format(job_dir, N, iteration))
    y_x_true_used = np.load('{}/y_x_true_used_N{}_i{}.npy'.format(job_dir, N, iteration))
    # y_x_true_used = y_x_true_used.reshape((nx,nx))
    x_map_scaled = np.load('{}/x_map_scaled_N{}_i{}.npy'.format(job_dir, N, iteration))
    y_x_map = np.load('{}/y_x_map_N{}_i{}.npy'.format(job_dir, N, iteration))
    var = np.load('{}/var_N{}_i{}.npy'.format(job_dir, N, iteration))
    x_mean = np.load('{}/x_mean_N{}_i{}.npy'.format(job_dir, N, iteration))
    '''

    plot_mcmc_results(x_true_used, y_hat4d, y_x_true_used, x_map_scaled, y_x_map, x_mean, std, job_dir, N, iteration, val_min, val_max, num_px, noise_var)

if __name__ == '__main__':
    main()


