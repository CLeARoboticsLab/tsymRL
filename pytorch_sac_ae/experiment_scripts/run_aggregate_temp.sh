conda init
eval "$(conda shell.bash hook)"
conda activate aggregate_tb
wait
tb-reducer lunar_log_fixedlogging/false_trial*/tb -o averaged/lunar_false_trials/ -r mean,std,min,max --handle-dup-steps 'mean' --lax-steps --lax-tags --overwrite
wait
tb-reducer lunar_log_fixedlogging/true_trial*/tb -o averaged/lunar_true_trials/ -r mean,std,min,max --handle-dup-steps 'mean' --lax-steps --lax-tags --overwrite
