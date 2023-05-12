# python train.py --work_dir ./pendulum_log/log_true_0 --gpu_choice 2 --replay_buffer_capacity 100000 --num_train_steps 50000 --task_name swingup --eval_freq 4000 --seed 0 --time_rev True &
# python train.py --work_dir ./pendulum_log/log_false_0 --gpu_choice 2 --replay_buffer_capacity 50000 --num_train_steps 50000 --task_name swingup --eval_freq 4000 --seed 0  


# for i in {0,1,2,3,4,5,6,7,8,9}
# do
# python train.py --work_dir ./pendulum_log/log_true_$i --gpu_choice 2 --replay_buffer_capacity 100000 --num_train_steps 50000 --task_name swingup --eval_freq 4000 --seed $i --time_rev True &
# python train.py --work_dir ./pendulum_log/log_false_$i --gpu_choice 2 --replay_buffer_capacity 50000 --num_train_steps 50000 --task_name swingup --eval_freq 4000 --seed $i  
# wait
# done

# conda init
# eval "$(conda shell.bash hook)"
# conda activate aggregate_tb
# wait
# tb-reducer pendulum_log/log_true_*/tb -o averaged/pendulum_log_true/ -r mean,std,min,max --handle-dup-steps 'mean' --lax-steps --lax-tags --overwrite
# wait
# tb-reducer pendulum_log/log_false_*/tb -o averaged/pendulum_log_false/ -r mean,std,min,max --handle-dup-steps 'mean' --lax-steps --lax-tags --overwrite

# python train.py --work_dir ./pendulum_log/3pend_log_true_0 --gpu_choice 1 --replay_buffer_capacity 100000 --num_train_steps 50000 --task_name three_poles --eval_freq 4000 --seed 0 --time_rev True &
# python train.py --work_dir ./pendulum_log/3pend_log_false_0 --gpu_choice 1 --replay_buffer_capacity 50000 --num_train_steps 50000 --task_name three_poles --eval_freq 4000 --seed 0  



# for i in {0,2,4,6,8}
# do
# python train.py --work_dir ./pendulum_log/half_varmasspend_3pend_log_true$i --gpu_choice 2 --replay_buffer_capacity 400000 --num_train_steps 200000 --task_name three_poles --eval_freq 4000 --seed $i --time_rev True --domain_name half_varmass_pend&
# python train.py --work_dir ./pendulum_log/half_varmasspend_3pend_log_false$i --gpu_choice 2 --replay_buffer_capacity 200000 --num_train_steps 200000 --task_name three_poles --eval_freq 4000 --seed $i --domain_name half_varmass_pend&
# python train.py --work_dir ./pendulum_log/half_varmasspend_3pend_log_true$((i+1)) --gpu_choice 3 --replay_buffer_capacity 400000 --num_train_steps 200000 --task_name three_poles --eval_freq 4000 --seed $((i+1)) --time_rev True --domain_name half_varmass_pend&
# python train.py --work_dir ./pendulum_log/half_varmasspend_3pend_log_false$((i+1)) --gpu_choice 3 --replay_buffer_capacity 200000 --num_train_steps 200000 --task_name three_poles --eval_freq 4000 --seed $((i+1)) --domain_name half_varmass_pend  
# wait
# done

# conda init
# eval "$(conda shell.bash hook)"
# conda activate aggregate_tb
# wait
# tb-reducer pendulum_log/half_varmasspend_3pend_log_true*/tb -o averaged/half_varmasspend_3pend_log_true/ -r mean,std,min,max --handle-dup-steps 'mean' --lax-steps --lax-tags --overwrite
# wait
# tb-reducer pendulum_log/half_varmasspend_3pend_log_false*/tb -o averaged/half_varmasspend_3pend_log_false/ -r mean,std,min,max --handle-dup-steps 'mean' --lax-steps --lax-tags --overwrite

for i in {10,14}
do
python train.py --work_dir ./pendulum_log/sec_trials_half_varmasspend_3pend_log_true$i --gpu_choice 0 --replay_buffer_capacity 800000 --num_train_steps 400000 --task_name three_poles --eval_freq 4000 --seed $i --time_rev True --domain_name half_varmass_pend&
python train.py --work_dir ./pendulum_log/sec_trials_half_varmasspend_3pend_log_false$i --gpu_choice 0 --replay_buffer_capacity 400000 --num_train_steps 400000 --task_name three_poles --eval_freq 4000 --seed $i --domain_name half_varmass_pend&
python train.py --work_dir ./pendulum_log/sec_trials_half_varmasspend_3pend_log_true$((i+1)) --gpu_choice 1 --replay_buffer_capacity 800000 --num_train_steps 400000 --task_name three_poles --eval_freq 4000 --seed $((i+1)) --time_rev True --domain_name half_varmass_pend&
python train.py --work_dir ./pendulum_log/sec_trials_half_varmasspend_3pend_log_false$((i+1)) --gpu_choice 1 --replay_buffer_capacity 400000 --num_train_steps 400000 --task_name three_poles --eval_freq 4000 --seed $((i+1)) --domain_name half_varmass_pend&

python train.py --work_dir ./pendulum_log/sec_trials_half_varmasspend_3pend_log_true$((i+2)) --gpu_choice 0 --replay_buffer_capacity 800000 --num_train_steps 400000 --task_name three_poles --eval_freq 4000 --seed $((i+2)) --time_rev True --domain_name half_varmass_pend&
python train.py --work_dir ./pendulum_log/sec_trials_half_varmasspend_3pend_log_false$((i+2)) --gpu_choice 0 --replay_buffer_capacity 400000 --num_train_steps 400000 --task_name three_poles --eval_freq 4000 --seed $((i+2)) --domain_name half_varmass_pend&
python train.py --work_dir ./pendulum_log/sec_trials_half_varmasspend_3pend_log_true$((i+3)) --gpu_choice 1 --replay_buffer_capacity 800000 --num_train_steps 400000 --task_name three_poles --eval_freq 4000 --seed $((i+3)) --time_rev True --domain_name half_varmass_pend&
python train.py --work_dir ./pendulum_log/sec_trials_half_varmasspend_3pend_log_false$((i+3)) --gpu_choice 1 --replay_buffer_capacity 400000 --num_train_steps 400000 --task_name three_poles --eval_freq 4000 --seed $((i+3)) --domain_name half_varmass_pend
wait
done

conda init
eval "$(conda shell.bash hook)"
conda activate aggregate_tb
wait
tb-reducer pendulum_log/sec_trials_half_varmasspend_3pend_log_true*/tb -o averaged/sec_trials_half_varmasspend_3pend_log_true/ -r mean,std,min,max --handle-dup-steps 'mean' --lax-steps --lax-tags --overwrite
wait
tb-reducer pendulum_log/sec_trials_half_varmasspend_3pend_log_false*/tb -o averaged/sec_trials_half_varmasspend_3pend_log_false/ -r mean,std,min,max --handle-dup-steps 'mean' --lax-steps --lax-tags --overwrite
