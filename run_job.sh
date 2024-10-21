#!/bin/bash
# Run job for target image "matrix_1"

TARGET_ID=1
JOB_DIR="matrix_$TARGET_ID"
NUM_MCMC_SAMPLES=1000
IMG_WIDTH=28
NOISE=0.1
IT=1
NUM_PX=0 # Parameter for computing heat flux

echo "Running MCMC sampler"
./job_template/mcmc_sampler_flux.py $NUM_MCMC_SAMPLES $TARGET_ID $JOB_DIR $IT $NUM_PX $NOISE > $JOB_DIR/sampler_N${NUM_MCMC_SAMPLES}_$IT_px${NUM_PX}_noise$NOISE.log;

echo "Running inference compute"
./job_template/mcmc_inference_compute.py $NUM_MCMC_SAMPLES $TARGET_ID $JOB_DIR $IT $NUM_PX $NOISE > $JOB_DIR/inference_compute_N${NUM_MCMC_SAMPLES}_${IT}_px${NUM_PX}_noise$NOISE.log;

echo "Creating plots"
./job_template/mcmc_inference_plot.py $NUM_MCMC_SAMPLES $TARGET_ID $JOB_DIR $IT $IMG_WIDTH $NUM_PX $NOISE> $JOB_DIR/inference_plot_N${NUM_MCMC_SAMPLES}_${IT}_px${NUM_PX}_noise$NOISE.log

