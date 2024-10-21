#!/usr/bin/env python

# Restrict multithreading in numpy/scipy/scikit calls to 1 thread
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from HT_2D_k import compute_multi_heat_boundaries, compute_heat_bulk
import matplotlib.pyplot as plt
import sys
import time




tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# CODES = os.environ['CODES']



tf.config.set_visible_devices([], 'GPU')

def get_pyplot_node_order(nx,ny):
    running_i = 0
    running_j = 0

    elem_i = 1
    n_elem_nx = nx - 1
    n_elem = n_elem_nx*n_elem_nx

    node_list = np.array([],dtype=np.int32)

    # Start with first element
    node_adds = [0, nx, 1, nx+1]
    node_list = np.append(node_list, node_adds)

    j = running_j + 1
    i = 0

    running_j = running_j + 1

    diagonal_length = 2
    diagonal_index = 1

    while (elem_i < n_elem):
        # Go down each diagonal
        while (diagonal_index <= diagonal_length):
            node_adds = np.array([j*nx+i, (j+1)*nx+i, j*nx+i+1, (j+1)*nx+i+1])

            not_in = np.isin(node_adds, node_list, invert=True)

            node_list = np.append(node_list, node_adds[not_in])

            j = j - 1
            i = i + 1

            diagonal_index = diagonal_index + 1
            elem_i += 1
        
        if (elem_i < (n_elem_nx*n_elem_nx - n_elem_nx)/2 + n_elem_nx):
            # If on lower diagonal, build up diagonal length, reset to upper-left
            diagonal_length += 1
            diagonal_index = 1
            i = 0
            j = running_j + 1
            running_j += 1
        else:
            # If on upper diagonal, reduce diagonal length, reset to upper-left
            diagonal_length -= 1
            diagonal_index = 1
            j = ny - 2
            i = running_i + 1
            running_i += 1
            
    node_list = np.argsort(node_list)
    
    return node_list

def main():
    N = int(sys.argv[1])
    matrix_id = int(sys.argv[2])
    iteration = int(sys.argv[4])

    if (len(sys.argv) < 4):
        print("No job directory specified. Defaulting to current directory")
        job_dir = os.getcwd()
    else:
        job_dir = sys.argv[3]

    with tf.device('/cpu:0'):

        t_start = time.time()
        nx = 28
        ny = 28
        pyplot_node_order = get_pyplot_node_order(nx,ny)
        gan_lr = '0.0002'
        gan_n_critic = '5'
        gan_bilinear = 'bilinear_'
        gan_gen_k = '5'

        gan_min = 0

        model_filename = '/home/kai_chun/Projects/Independent Study/Week 6/mnist/10_{}_{}x{}/Arch_B_zdim_100_lr_0.001_n_critic_5_gen_k_5_disc_k_5/wgan_model.h5'.format( gan_min, nx, ny)

        gan = load_model(model_filename, compile=False)

        z_dim = 100
        img_h = nx
        img_w = ny
        img_c = 1
        dim_like = img_h * img_w * img_c
        burn_in = 0.5
        batch_size = 1
        noise_var = 0.05

        val_min = 50
        val_max = 10
        num_cases = 9

        burn = int(burn_in * N)

        n_eff = N - burn
        n_iter = int(n_eff/batch_size)
        print('loading samples...')
        mcmc_samps = np.load('{}/samples_N{}_i{}.npy'.format(job_dir,N, iteration))
        y_hat4d = np.load('{}/y_hat4d_N{}_i{}.npy'.format(job_dir, N, iteration))

        image_filename = '{}/gan_inference/data_gen/Python/olson_data/{}_1_{}x{}/matrix_{}.png'.format(CODES, val_max, nx, ny, matrix_id)
        image = tf.keras.utils.load_img(image_filename, color_mode='grayscale', target_size=None,)
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = input_arr/127.5 - 1
        x_true = input_arr.squeeze()
        x_true_used = (val_max - 1.0) * (x_true - np.min(x_true)) / (np.max(x_true) - np.min(x_true)) + 1.0
        x_true_show = (1.0) * (x_true - np.min(x_true)) / (np.max(x_true) - np.min(x_true)) + 0.0

        eff_samps = np.squeeze(mcmc_samps[burn:,:,:])

        loss = np.zeros((n_eff))
        x_mean = np.zeros((img_w,img_h,1))
        x2_mean = np.zeros((img_w,img_h,1))

        print('n_iter = {}'.format(n_iter))
        for i in range(n_iter):
            g_z_0 = gan(eff_samps[i*batch_size:(i+1)*batch_size,:])

            g_z = g_z_0.numpy()[0].squeeze()

            g_z = (val_max - 1.0) * (g_z - np.min(g_z)) / (np.max(g_z) - np.min(g_z)) + 1.0


            gen_heat_bulk = compute_heat_bulk(g_z, nx, ny)

            diff = tf.constant(gen_heat_bulk) - tf.constant(y_hat4d)

            x_mean = x_mean + np.mean(g_z_0, axis = 0)
            x2_mean = x2_mean + np.mean(g_z_0**2, axis = 0)

            for k in range(batch_size):
                loss[(i*batch_size)+k] = 0.5*np.linalg.norm(diff[k,:,:,:])**2 + 0.5*noise_var*np.linalg.norm(eff_samps[(i*batch_size)+k,:])**2
                #loss[(i*batch_size)+k] = np.linalg.norm(diff[k,:,:,:])**2 


        x_mean = x_mean/n_iter
        x2_mean = x2_mean/n_iter
        var = x2_mean - (x_mean)**2

        map_ind = np.argmin(loss)

        x_map = gan(np.tile(np.expand_dims(eff_samps[map_ind,:], axis=0), (batch_size,1)))
        x_map = x_map.numpy()[0].squeeze()

        x_map_scaled = (val_max - 1.0) * (x_map - np.min(x_map)) / (np.max(x_map) - np.min(x_map)) + 1.0
        x_map_show = (1.0) * (x_map - np.min(x_map)) / (np.max(x_map) - np.min(x_map)) + 0.0
        print('g_z.min() = {}'.format(x_map_scaled.min()))
        print('g_z.max() = {}'.format(x_map_scaled.max()))

        y_x_map = compute_heat_bulk(x_map_scaled, img_h, img_w)
        y_x_map = y_x_map[pyplot_node_order]
        y_x_map = np.flipud(y_x_map.reshape(28,28))
        #x_map = x_map + np.min(x_map)
        #x_map = x_map / np.max(x_map) * val_max

        y_x_true_used = compute_heat_bulk(np.rot90(x_true_used,3), img_h, img_w)
        y_x_true_used = y_x_true_used[pyplot_node_order]


        np.save('{}/x_true_used_N{}_i{}.npy'.format(job_dir, N, iteration), x_true_used)
        np.save('{}/y_hat4d_N{}_i{}.npy'.format(job_dir, N, iteration), y_hat4d)
        np.save('{}/y_x_true_used_N{}_i{}.npy'.format(job_dir, N, iteration), y_x_true_used)
        np.save('{}/x_map_scaled_N{}_i{}.npy'.format(job_dir, N, iteration), x_map_scaled)
        np.save('{}/y_x_map_N{}_i{}.npy'.format(job_dir, N, iteration), y_x_map)
        np.save('{}/var_N{}_i{}.npy'.format(job_dir, N, iteration), var)
        np.save('{}/x_mean_N{}_i{}.npy'.format(job_dir, N, iteration), x_mean)



        # y_x_true_used = np.flipud(y_x_true_used.reshape(28,28))

        # some stats
        # fig, axs = plt.subplots(3,2, figsize=(20,20))
        # im1 = axs[0][0].imshow(x_true_used)
        # fig.colorbar(im1, ax=axs[0][0])
        # axs[0][0].set_title(r'$x_{{true}}$')
        # #im2 = axs[0][1].imshow(np.flipud(y_hat4d.squeeze().reshape(29,29)))
        # im2 = axs[0][1].imshow(y_x_true_used)
        # fig.colorbar(im2, ax=axs[0][1])
        # axs[0][1].set_title(r'$y_{{meas}}$')

        # im3 = axs[1][0].imshow(x_map_scaled)
        # fig.colorbar(im3, ax=axs[1][0])
        # axs[1][0].set_title(r'$x_{{map}}$')
        # im4 = axs[1][1].imshow(y_x_map)
        # fig.colorbar(im4, ax=axs[1][1])
        # axs[1][1].set_title(r'$y_{{map}}$')

        # im5 = axs[2][0].imshow(x_mean[:,:,0])
        # fig.colorbar(im5, ax=axs[2][0])
        # axs[2][0].set_title(r'$x_{{mean}}$')
        # im6 = axs[2][1].imshow(var[:,:,0])
        # fig.colorbar(im6, ax=axs[2][1])
        # axs[2][1].set_title(r'$x_{{var}}$')

        # #plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        # plt.savefig('{}/stats_N{}_i{}'.format(job_dir,N, iteration))

        # np.save('{}/x_true_N{}_i{}.npy'.format(job_dir, N, iteration), x_true)
        # np.save('{}/x_var_N{}_i{}.npy'.format(job_dir, N, iteration), var)
        # np.save('{}/x_mean_N{}_i{}.npy'.format(job_dir, N, iteration), x_mean)
        # np.save('{}/x_map_N{}_i{}.npy'.format(job_dir, N, iteration), x_map)
        # plt.close()

        # t_stop = time.time()
        # print('MCMC inference time = {} sec'.format(t_stop - t_start))
        # #plt.show()


