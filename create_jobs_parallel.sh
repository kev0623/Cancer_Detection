#!/bin/bash

n_mcmc=150
nx=28

noise_var=0.01

cmd_file="commands_N${n_mcmc}_noise${noise_var}.txt"

if [ -f $cmd_file ]
then
    rm $cmd_file
fi

num_px=0


for n in `seq 50001 50002`;

do
    for i in `seq 1 6`;
    do
        if [ ! -d matrix_$n ]
        then
            mkdir -p matrix_$n
        fi



        echo -n "./job_template/mcmc_sampler_flux.py $n_mcmc $n matrix_$n $i $num_px $noise_var > matrix_${n}/sampler_N${n_mcmc}_${i}_px${num_px}_noise$noise_var.log;" >> $cmd_file
        echo -n "./job_template/mcmc_inference_compute.py $n_mcmc $n matrix_$n $i $num_px $noise_var > matrix_${n}/inference_compute_N${n_mcmc}_${i}_px${num_px}_noise$noise_var.log;" >> $cmd_file
        echo "./job_template/mcmc_inference_plot.py $n_mcmc $n matrix_$n $i $nx $num_px $noise_var> matrix_${n}/inference_plot_N${n_mcmc}_${i}_px${num_px}_noise$noise_var.log" >>  $cmd_file
    done
done
