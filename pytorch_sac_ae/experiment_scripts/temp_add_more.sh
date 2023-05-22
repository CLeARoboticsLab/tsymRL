#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate aggregate_tb
wait
tb-reducer log_pixel/false_swing_pixel_2actionrep_0pt5supdfreq*/tb -o averaged/false_swing_pixel_2actionrep_0pt5supdfreq/ -r mean,std,min,max --handle-dup-steps 'mean' --lax-steps --lax-tags
wait
tb-reducer log_pixel/true_swing_pixel_2actionrep_0pt5supdfreq*/tb -o averaged/true_swing_pixel_2actionrep_0pt5supdfreq/ -r mean,std,min,max --handle-dup-steps 'mean' --lax-steps --lax-tags