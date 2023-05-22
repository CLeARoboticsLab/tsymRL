
# conda init
# eval "$(conda shell.bash hook)"
# conda activate aggregate_tb
# wait
# tb-reducer pendulum_log/half_varmasspend_4pend_log_true*/tb -o averaged/half_varmasspend_4pend_log_true/ -r mean,std,min,max --handle-dup-steps 'mean' --lax-steps --lax-tags --overwrite
# wait
# tb-reducer pendulum_log/half_varmasspend_4pend_log_false*/tb -o averaged/half_varmasspend_4pend_log_false/ -r mean,std,min,max --handle-dup-steps 'mean' --lax-steps --lax-tags --overwrite


# for i in {0,4}
# do
# python train.py --work_dir ./pendulum_log/100phased200kstep_half_varmasspend_4pend_log_true$i --gpu_choice 0 --replay_buffer_capacity 500000 --num_train_steps 500000 --task_name four_poles --eval_freq 4000 --seed $i --time_rev True --domain_name half_varmass_pend --percent_tsym 100 --percent_sampling phased&
# python train.py --work_dir ./pendulum_log/100phased200kstep_half_varmasspend_4pend_log_true$((i+1)) --gpu_choice 0 --replay_buffer_capacity 500000 --num_train_steps 500000 --task_name four_poles --eval_freq 4000 --seed $((i+1)) --time_rev True --domain_name half_varmass_pend --percent_tsym 100 --percent_sampling phased&
# python train.py --work_dir ./pendulum_log/100phased200kstep_half_varmasspend_4pend_log_true$((i+2)) --gpu_choice 1 --replay_buffer_capacity 500000 --num_train_steps 500000 --task_name four_poles --eval_freq 4000 --seed $((i+2)) --time_rev True --domain_name half_varmass_pend --percent_tsym 100 --percent_sampling phased&
# python train.py --work_dir ./pendulum_log/100phased200kstep_half_varmasspend_4pend_log_true$((i+3)) --gpu_choice 1 --replay_buffer_capacity 500000 --num_train_steps 500000 --task_name four_poles --eval_freq 4000 --seed $((i+3)) --time_rev True --domain_name half_varmass_pend --percent_tsym 100 --percent_sampling phased

# wait
# done

# conda init
# eval "$(conda shell.bash hook)"
# conda activate aggregate_tb
# wait
# tb-reducer pendulum_log/100phased200kstep_half_varmasspend_4pend_log_true*/tb -o averaged/100phased200kstep_half_varmasspend_4pend_log_true/ -r mean,std,min,max --handle-dup-steps 'mean' --lax-steps --lax-tags --overwrite

for i in {0,4}
do
python train.py --work_dir ./halfvarmass/100phasedin100kstep_half_varmasspend_4pend_log_true$i --gpu_choice 2 --replay_buffer_capacity 600000 --num_train_steps 600000 --task_name four_poles --eval_freq 4000 --seed $i --time_rev True --domain_name half_varmass_pend --percent_tsym 100 --percent_sampling phase_in --phase_percent 20&
python train.py --work_dir ./halfvarmass/100phasedin100kstep_half_varmasspend_4pend_log_true$((i+1)) --gpu_choice 2 --replay_buffer_capacity 600000 --num_train_steps 600000 --task_name four_poles --eval_freq 4000 --seed $((i+1)) --time_rev True --domain_name half_varmass_pend --percent_tsym 100 --percent_sampling phase_in --phase_percent 20&
python train.py --work_dir ./halfvarmass/100phasedin100kstep_half_varmasspend_4pend_log_true$((i+2)) --gpu_choice 3 --replay_buffer_capacity 600000 --num_train_steps 600000 --task_name four_poles --eval_freq 4000 --seed $((i+2)) --time_rev True --domain_name half_varmass_pend --percent_tsym 100 --percent_sampling phase_in --phase_percent 20&
python train.py --work_dir ./halfvarmass/100phasedin100kstep_half_varmasspend_4pend_log_true$((i+3)) --gpu_choice 3 --replay_buffer_capacity 600000 --num_train_steps 600000 --task_name four_poles --eval_freq 4000 --seed $((i+3)) --time_rev True --domain_name half_varmass_pend --percent_tsym 100 --percent_sampling phase_in --phase_percent 20

wait
done

conda init
eval "$(conda shell.bash hook)"
conda activate aggregate_tb
wait
tb-reducer halfvarmass/100phasedin100kstep_half_varmasspend_4pend_log_true*/tb -o averaged/100phasedin100kstep_half_varmasspend_4pend_log_true/ -r mean,std,min,max --handle-dup-steps 'mean' --lax-steps --lax-tags --overwrite