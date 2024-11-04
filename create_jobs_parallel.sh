#!/bin/bash
echo "Start date and time: $(date)"
start_time=$(date +%s)
n_mcmc=15000
nx=28
noise_var=0.05
cmd_file="commands_N${n_mcmc}_noise${noise_var}.txt"
num_px=0

if [ -f $cmd_file ]
then
    rm $cmd_file
fi

for n in `seq 50200 50205`;


do
    for i in `seq 1 3`;
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

parallel --jobs 6 < $cmd_file

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
minutes=$((elapsed_time / 60))
seconds=$((elapsed_time % 60))
echo "Total runtime: ${minutes} minutes ${seconds} seconds"
