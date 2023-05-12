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



for i in {0,1}
do
python train.py --work_dir ./pendulum_log/varmasspend_5pend_log_true$i --gpu_choice 3 --replay_buffer_capacity 1000000 --num_train_steps 500000 --task_name five_poles --eval_freq 4000 --seed $i --time_rev True --domain_name varmass_pend&
python train.py --work_dir ./pendulum_log/varmasspend_5pend_log_false$i --gpu_choice 3 --replay_buffer_capacity 500000 --num_train_steps 500000 --task_name five_poles --eval_freq 4000 --seed $i --domain_name varmass_pend  
wait
done

conda init
eval "$(conda shell.bash hook)"
conda activate aggregate_tb
wait
tb-reducer pendulum_log/varmasspend_5pend_log_true*/tb -o averaged/varmasspend_5pend_log_true/ -r mean,std,min,max --handle-dup-steps 'mean' --lax-steps --lax-tags --overwrite
wait
tb-reducer pendulum_log/varmasspend_5pend_log_false*/tb -o averaged/varmasspend_5pend_log_false/ -r mean,std,min,max --handle-dup-steps 'mean' --lax-steps --lax-tags --overwrite
