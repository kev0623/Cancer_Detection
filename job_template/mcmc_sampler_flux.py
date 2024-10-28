#!/usr/bin/env python

import os


# Restrict multithreading in numpy/scipy/scikit calls to 1 thread
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from HT_2D_k import compute_multi_heat_boundaries_with_flux, compute_multi_heat_boundaries_downsample, compute_heat_bulk
from mcmc_inference_bulk_compute import get_pyplot_node_order
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow_probability as tfp
import tensorflow as tf
import sys
import time

from signal import signal, SIGINT
from sys import exit

tf.config.run_functions_eagerly(True)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)



def handler(signal_received, frame):
    # Handle any cleanup here
    print('SIGINT or CTRL-C detected. Exiting gracefully')
    exit(0)

signal(SIGINT, handler)

tf.config.set_visible_devices([], 'GPU')

with tf.device('/cpu:0'):
    t_start = time.time()

    # Create tensorflow_probability distribution
    tfd = tfp.distributions

    # CODES = os.environ['CODES']

    # Get run parameters

    # Number of samples in markov chain
    N = int(sys.argv[1])
    # Target image id
    target_id = int(sys.argv[2])
    # Directory to store results
    job_dir = sys.argv[3]
    # Iteration number for repeat tests
    iteration = int(sys.argv[4])
    # Number of "pixels" used to represent computed heat flux
    # (not used for standard analysis, set = 0)
    num_px = int(sys.argv[5])
    # Assumed level of noise in measurement
    noise_var = float(sys.argv[6])

    # Set random seed based on # of samples, iteration #
    np.random.seed(N*iteration)

    if (len(sys.argv) < 4):
        print("No job directory specified. Defaulting to current directory")
        job_dir = os.getcwd()
    else:
        job_dir = sys.argv[3]


    # Domain size for finite element computation
    nx = 28
    # Assume square domain
    ny = nx

    # Finite element domain node order is different than pyplot's,
    # makes figure creation difficult. get_pyplot_node_order detangles
    # FEniCSx's element node order into pyplot's node order.
    node_order = get_pyplot_node_order(nx, ny)
    
    # GAN parameters (hard coded :( )
    gan_lr = '0.0002'
    gan_n_critic = '5'
    gan_bilinear = 'bilinear_'
    gan_gen_k = '5'

    arch = 'B'
    lr = '0.001'
    n_critic = '5'
    gen_k = '5'
    disc_k = '5'
    zdim = 100

    val_min = 0
    val_max = 10


    model_suffix = f'{val_max}_{val_min}_{nx}x{ny}/Arch_{arch}_zdim_{zdim}_lr_{lr}_n_critic_{n_critic}_gen_k_{gen_k}_disc_k_{disc_k}/wgan_model.h5'
    model_filename = f'/home/kai_chun/Projects/Independent Study/Week 6/mnist/{model_suffix}'
    gan = load_model(model_filename, compile=False)

    num_cases = 9

    # Image dimensions
    img_h = nx
    img_w = ny
    img_c = 1
    dim_like = img_h * img_w * img_c
    # MCMC burn-in percentage
    burn_in = 0.5
    # Batch size for GAN (one sample at time...)
    batch_size = 1

    burn = int(burn_in * N)

    print(f'N = {N}')
    print(f'burn = {burn}')
    print(f'dim_like = {dim_like}')
    print(f'noise_var = {noise_var}')

    ## Create test input
    image_filename = f'/home/kai_chun/Projects/Independent Study/Week 6/images/mnist_img_{target_id}.png'
    try:
        image = tf.keras.utils.load_img(image_filename, color_mode='grayscale', target_size=None,)
    except:
        image = tf.keras.preprocessing.image.load_img(image_filename, color_mode='grayscale', target_size=None,)

    input_arr = tf.keras.preprocessing.image.img_to_array(image)

    # Normalize
    input_arr = input_arr/(127.5) - 1
    y_hat3d = input_arr.squeeze()
    y_hat3d = ((y_hat3d * 127.5 + 127.5) / 255 * (val_max - 1)) + 1.0
    #y_hat3d = (val_max - 1.0)*(y_hat3d - np.min(y_hat3d))/(np.max(y_hat3d) - np.min(y_hat3d)) + 1.0

    # Create a high-resolution target image image from forward model, and then downsample
    # to nominal resolution (nx, ny) to create an out-of-distribution target sample
    nx_high = nx * 2
    ny_high = ny * 2
    target_boundary = compute_heat_bulk(y_hat3d, img_h, img_w)

    target_storage = open(f'{job_dir}/targets_N{N}_i{iteration}_px{num_px}_noise{int(noise_var*100)}.npy', 'wb')
    np.save(target_storage, target_boundary)

    target_noise = np.random.normal(loc=0.0, scale=target_boundary.max()*noise_var, size=target_boundary.shape)

    target_boundary += target_noise
    np.save(target_storage, target_boundary)

    target_bulk = compute_heat_bulk(y_hat3d, img_h, img_w)

    y_hat4d = np.tile(target_boundary, (batch_size,1,1,1)).astype(np.float32)

    dim_like_boundary = target_boundary.shape[0]

    np.save(target_storage, y_hat4d)

    def joint_log_prob(z):
        gen_out = gan(z)
        gen_out = gen_out.numpy()[0].squeeze()
        gen_out = ((gen_out * 127.5 + 127.5) / 255 * (val_max - 1)) + 1.0

        # Use the output from the gan as thermal conductivity with the FEniCSx forward model
        # to generate a temperature distribution
        candidate_solution_simulation = compute_heat_bulk(gen_out, img_h, img_w)

        # Subtract target (measurement) from candidate solution to get difference
        diff_img = tf.reshape(tf.constant(candidate_solution_simulation) - tf.constant(target_boundary), [dim_like_boundary])

        # Compute prior probability
        prior = tfd.MultivariateNormalDiag(loc=np.zeros(zdim, dtype=np.float32),
                scale_diag=np.ones(zdim, dtype=np.float32))

        # Compute likelihood probability
        like = tfd.MultivariateNormalDiag(loc=np.zeros(dim_like_boundary, dtype=np.float32),
                scale_diag=np.sqrt(noise_var)*np.ones(dim_like_boundary, dtype=np.float32))

        # Return log( prior * like) = log(prior) + log(like)
        return (prior.log_prob(z) + like.log_prob(diff_img))

    def unnormalized_posterior(z):
        return joint_log_prob(z)



    # Set up and run MCMC through Tensorflow_probability
    @tf.function
    def run_chain():
        adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
                tfp.mcmc.HamiltonianMonteCarlo(target_log_prob_fn=unnormalized_posterior, step_size=np.float32(1.), num_leapfrog_steps=3),
                num_adaptation_steps=int(0.8*burn))

        initial_state = tf.constant(np.zeros((batch_size, zdim)).astype(np.float32))
        samples, [st_size, log_accept_ratio] = tfp.mcmc.sample_chain(
                num_results=N,
                num_burnin_steps=burn,
                current_state=initial_state,
                kernel=adaptive_hmc,
                trace_fn=lambda _, pkr: [pkr.inner_results.accepted_results.step_size,
                    pkr.inner_results.log_accept_ratio])

        p_accept = tf.reduce_mean(tf.exp(tf.minimum(log_accept_ratio, 0.)))
        return samples, st_size, log_accept_ratio, p_accept

    samples, st_size, log_accept_ratio, p_accept = run_chain()

    print(f'HMC acceptance ratio = {p_accept}')
    print(f'st_size = {st_size}')

    np.save(f'{job_dir}/samples_N{N}_i{iteration}_px{num_px}_noise{int(noise_var*100)}.npy', samples)

    t_stop = time.time()

    print(f'MCMC time = {t_stop - t_start} sec')




