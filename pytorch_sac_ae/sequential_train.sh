#!/bin/bash
# python train.py --time_rev True --work_dir ./log_true --num_train_steps 100000 --seed 2
# wait
# python train.py --work_dir ./log_false --num_train_steps 50000 --seed 2 
# wait
# python train.py --time_rev True --work_dir ./log_true --num_train_steps 80000 --seed 2
# python train.py --time_rev True --work_dir ./log_true_256 --num_train_steps 80000 --seed 2 --batch_size 256 --critic_target_update_freq 4 --actor_update_freq 4
# wait
# python train.py --time_rev True --work_dir ./log_true --num_train_steps 80000 --seed 2

# python train.py --work_dir ./log_false_swing_sparse --num_train_steps 200000 --seed 1 --task_name swingup_sparse
# wait
# python train.py --work_dir ./log_true_swing_sparse --time_rev True --num_train_steps 200000 --seed 1 --task_name swingup_sparse

python train.py --work_dir ./log_true_swing_pixel --num_train_steps 1000000 --time_rev True --seed 1 --task_name swingup --encoder_type 'pixel' --decoder_type 'pixel' --eval_freq 10000
wait
python train.py --work_dir ./log_false_swing_pixel --num_train_steps 1000000 --seed 1 --task_name swingup --encoder_type 'pixel' --decoder_type 'pixel' --eval_freq 10000
