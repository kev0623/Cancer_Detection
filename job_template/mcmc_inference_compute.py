#!/usr/bin/env python
import os
# Restrict multithreading in numpy/scipy/scikit
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import time

# from heat_ss_occlusion_fn import compute_heat_bulk, compute_heat_bulk_preallocate
# from heat_ss_occlusion_fn_multi_temp import compute_multi_heat_boundaries

from HT_2D_k import compute_multi_heat_boundaries_with_flux, compute_heat_bulk
from mcmc_inference_bulk_compute import get_pyplot_node_order

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# CODES = os.environ['CODES']

N = int(sys.argv[1])
target_id = int(sys.argv[2])
iteration = int(sys.argv[4])
num_px = int(sys.argv[5])
noise_var = float(sys.argv[6])

if (len(sys.argv) < 4):
    print("No job directory specified. Defaulting to current directory")
    job_dir = os.getcwd()
else:
    job_dir = sys.argv[3]

tf.config.set_visible_devices([], 'GPU')

with tf.device('/cpu:0'):

    t_start = time.time()
    nx = 28
    ny = nx

    arch = 'B'
    lr = '0.001'
    n_critic = '5'
    gen_k = '5'
    disc_k = '5'
    zdim = 100
    val_max = 10
    val_min = 0

    model_suffix = f'{val_max}_{val_min}_{nx}x{ny}/Arch_{arch}_zdim_{zdim}_lr_{lr}_n_critic_{n_critic}_gen_k_{gen_k}_disc_k_{disc_k}/wgan_model.h5'
    model_filename = f'/home/kai_chun/Projects/Independent Study/Week 6/mnist/{model_suffix}'


    gan = load_model(model_filename, compile=False)

    z_dim = 5
    img_h = nx
    img_w = ny
    img_c = 1
    dim_like = img_h * img_w * img_c
    burn_in = 0.5
    batch_size = 1

    num_cases = 9

    burn = int(burn_in * N)

    n_eff = N - burn
    n_iter = int(n_eff/batch_size)

    # Loading MCMC results
    mcmc_samps = np.load(f'{job_dir}/samples_N{N}_i{iteration}_px{num_px}_noise{int(noise_var*100)}.npy')
    with open(f'{job_dir}/targets_N{N}_i{iteration}_px{num_px}_noise{int(noise_var*100)}.npy', 'rb') as f:
        a = np.load(f)
        b = np.load(f)
        y_hat4d = np.load(f)

    # Load true target
    image_filename = f'/home/kai_chun/Projects/Independent Study/Week 6/images/mnist_img_{target_id}.png'.format(CODES, val_max, nx, ny, target_id)
    image = tf.keras.utils.load_img(image_filename, color_mode='grayscale', target_size=None,)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = input_arr/127.5 - 1
    x_true = input_arr.squeeze()

    x_true = ((x_true * 127.5 + 127.5) / 255 * (val_max - 1.0)) + 1.0
    x_true_used = x_true
    x_true_show = x_true

    # Extract useful MCMC (post burnin) samples
    eff_samps = np.squeeze(mcmc_samps[burn:,:,:])

    # Generate statistic results
    loss = np.zeros((n_eff))
    x_mean = np.zeros((img_w,img_h,1))
    x2_mean = np.zeros((img_w,img_h,1))

    # Go through samples, generate statistics
    print('n_iter = {}'.format(n_iter))
    for i in range(n_iter):
        # Generate candidate solution (x, or thermal conductivity field) for each sample
        g_z_0 = gan(eff_samps[i*batch_size:(i+1)*batch_size,:])
        g_z = ((g_z_0 * 127.5 + 127.5) / 255 * (val_max - 1.0)) + 1.0


        # Compute the simulation result for that sample
        gen_heat_boundary = compute_heat_bulk(g_z.numpy().squeeze(), img_h, img_w)

        # Compute difference
        diff = tf.constant(gen_heat_boundary) - tf.constant(y_hat4d)

        # Add to mean, mean-squared arrays for statistics (will divide N_eff later)
        x_mean = x_mean + np.mean(g_z, axis = 0)
        x2_mean = x2_mean + np.mean(g_z**2, axis = 0)

        # Compute loss (minimized difference) for each sample, used to identify MAP solution (best solution)
        for k in range(batch_size):
            loss[(i*batch_size)+k] = 0.5*np.linalg.norm(diff[k,:,:,:])**2 + 0.5*noise_var*np.linalg.norm(eff_samps[(i*batch_size)+k,:])**2


    # Compute mean, variance images
    x_mean = x_mean/n_iter
    x2_mean = x2_mean/n_iter

    # Compute actual variance
    var = x2_mean - (x_mean)**2

    var += var.min() + 1e-6 # Adjust for rounding errors that brings variance below 0

    # Compute the standard deviation image
    std = np.sqrt(var)

    # Locate MAP sample (best sample)
    map_ind = np.argmin(loss)

    # Generate images to plot for MAP sample
    x_map = gan(np.tile(np.expand_dims(eff_samps[map_ind,:], axis=0), (batch_size,1)))
    x_map = x_map.numpy()[0].squeeze()
    x_map = ((x_map * 127.5 + 127.5) / 255 * (val_max - 1.0)) + 1.0
    x_map_scaled = x_map
    x_map_show = x_map

    # Get node ordering from FEniCSx to matplotlib
    node_order = get_pyplot_node_order(img_h,img_w)

    # Create images for plot
    x_true_used_plot = np.flipud(x_true_used)
    y_x_true_used = compute_heat_bulk(x_true_used_plot, img_h, img_w)
    y_x_true_used = y_x_true_used[node_order]
    y_x_true_used = np.flipud(y_x_true_used.reshape(nx,ny))

    x_map_scaled_plot = np.flipud(x_map_scaled)
    y_x_map = compute_heat_bulk(x_map_scaled_plot, img_h, img_w)
    y_x_map = y_x_map[node_order]
    y_x_map = np.flipud(y_x_map.reshape(nx,ny))

    np.savez(f'{job_dir}/statistics_N{N}_i{iteration}_px{num_px}_noise{int(noise_var*100)}.npz',
         x_true_used=x_true_used,
         y_hat4d=y_hat4d,
         y_x_true_used=y_x_true_used,
         x_map_scaled=x_map_scaled,
         y_x_map=y_x_map,
         var=var,
         std=std,
         x_mean=x_mean,
         x_map=x_map,
         val_min=val_min,
         val_max=val_max)


    t_stop = time.time()
    print('MCMC inference time = {} sec'.format(t_stop - t_start))
