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


#python train.py --work_dir ./log/false_swing_pixel_2actionrep --replay_buffer_capacity 500000 --action_repeat 2 --num_train_steps 1000000 --seed 2 --task_name swingup --encoder_type 'pixel' --decoder_type 'pixel' --eval_freq 10000
#wait
# python train.py --work_dir ./log/true_swing_pixel_2actionrep --replay_buffer_capacity 1000000 --action_repeat 2 --num_train_steps 800000 --time_rev True --seed 2 --task_name swingup --encoder_type 'pixel' --decoder_type 'pixel' --eval_freq 10000
# wait
# python train.py --work_dir ./log/false_swing_pixel_2_2actionrep --replay_buffer_capacity 1000000 --action_repeat 2 --num_train_steps 500000 --seed 3 --task_name swingup --encoder_type 'pixel' --decoder_type 'pixel' --eval_freq 10000
# wait
# python train.py --work_dir ./log/true_swing_pixel_2_2actionrep --replay_buffer_capacity 1000000 --action_repeat 2 --num_train_steps 800000 --time_rev True --seed 3 --task_name swingup --encoder_type 'pixel' --decoder_type 'pixel' --eval_freq 10000


# Both these done
# python train.py --work_dir ./log_cp/cp_balance_false_0 --num_train_steps 100000 --seed 0 --task_name balance
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_balance_true_0 --num_train_steps 200000 --seed 0 --task_name balance
# wait

# python train.py --work_dir ./log_cp/cp_balance_false_1 --num_train_steps 100000 --seed 1 --task_name balance
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_balance_true_1 --num_train_steps 200000 --seed 1 --task_name balance
# wait

# python train.py --work_dir ./log_cp/cp_balance_false_2 --num_train_steps 100000 --seed 2 --task_name balance
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_balance_true_2 --num_train_steps 200000 --seed 2 --task_name balance
# wait

# python train.py --work_dir ./log_cp/cp_balance_false_3 --num_train_steps 100000 --seed 3 --task_name balance
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_balance_true_3 --num_train_steps 200000 --seed 3 --task_name balance
# wait

# python train.py --work_dir ./log_cp/cp_balance_false_4 --num_train_steps 100000 --seed 4 --task_name balance
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_balance_true_4 --num_train_steps 200000 --seed 4 --task_name balance
# wait

# python train.py --work_dir ./log_cp/cp_balance_false_5 --num_train_steps 100000 --seed 5 --task_name balance
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_balance_true_5 --num_train_steps 200000 --seed 5 --task_name balance
# wait

# python train.py --work_dir ./log_cp/cp_balance_false_6 --num_train_steps 100000 --seed 6 --task_name balance
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_balance_true_5 --num_train_steps 200000 --seed 6 --task_name balance
# wait

# python train.py --work_dir ./log_cp/cp_balance_false_7 --num_train_steps 100000 --seed 7 --task_name balance
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_balance_true_7 --num_train_steps 200000 --seed 7 --task_name balance
# wait

# python train.py --work_dir ./log_cp/cp_balance_false_8 --num_train_steps 100000 --seed 8 --task_name balance
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_balance_true_8 --num_train_steps 200000 --seed 8 --task_name balance
# wait

# python train.py --work_dir ./log_cp/cp_balance_false_9 --num_train_steps 100000 --seed 9 --task_name balance
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_balance_true_9 --num_train_steps 200000 --seed 9 --task_name balance
# wait

# Damping enabled in the cartpole xml
# python train.py --work_dir ./log_cp/cp_swingup_false_wstddamp_0 --num_train_steps 100000 --seed 0 --task_name swingup
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_swingup_true_wstddamp_0 --num_train_steps 200000 --seed 0 --task_name swingup
# wait

# python train.py --work_dir ./log_cp/cp_swingup_false_wstddamp_1 --num_train_steps 100000 --seed 1 --task_name swingup
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_swingup_true_wstddamp_1 --num_train_steps 200000 --seed 1 --task_name swingup
# wait

# python train.py --work_dir ./log_cp/cp_swingup_false_wstddamp_2 --num_train_steps 100000 --seed 2 --task_name swingup
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_swingup_true_wstddamp_2 --num_train_steps 200000 --seed 2 --task_name swingup
# wait

# python train.py --work_dir ./log_cp/cp_swingup_false_wstddamp_3 --num_train_steps 100000 --seed 3 --task_name swingup
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_swingup_true_wstddamp_3 --num_train_steps 200000 --seed 3 --task_name swingup
# wait

# python train.py --work_dir ./log_cp/cp_swingup_false_wstddamp_4 --num_train_steps 100000 --seed 4 --task_name swingup
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_swingup_true_wstddamp_4 --num_train_steps 200000 --seed 4 --task_name swingup
# wait

# python train.py --work_dir ./log_cp/cp_swingup_false_wstddamp_5 --num_train_steps 100000 --seed 5 --task_name swingup
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_swingup_true_wstddamp_5 --num_train_steps 200000 --seed 5 --task_name swingup
# wait

# python train.py --work_dir ./log_cp/cp_swingup_false_wstddamp_6 --num_train_steps 100000 --seed 6 --task_name swingup
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_swingup_true_wstddamp_5 --num_train_steps 200000 --seed 6 --task_name swingup
# wait

# python train.py --work_dir ./log_cp/cp_swingup_false_wstddamp_7 --num_train_steps 100000 --seed 7 --task_name swingup
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_swingup_true_wstddamp_7 --num_train_steps 200000 --seed 7 --task_name swingup
# wait

# python train.py --work_dir ./log_cp/cp_swingup_false_wstddamp_8 --num_train_steps 100000 --seed 8 --task_name swingup
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_swingup_true_wstddamp_8 --num_train_steps 200000 --seed 8 --task_name swingup
# wait

# python train.py --work_dir ./log_cp/cp_swingup_false_wstddamp_9 --num_train_steps 100000 --seed 9 --task_name swingup
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_swingup_true_wstddamp_9 --num_train_steps 200000 --seed 9 --task_name swingup
# wait




# python train.py --work_dir ./log_cp/cp_swingup_false_w2000xdamp_0 --num_train_steps 100000 --seed 0 --task_name swingup
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_swingup_true_w2000xdamp_0 --num_train_steps 200000 --seed 0 --task_name swingup
# wait

# python train.py --work_dir ./log_cp/cp_swingup_false_w100xdamp_1 --num_train_steps 100000 --seed 1 --task_name swingup
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_swingup_true_w100xdamp_1 --num_train_steps 200000 --seed 1 --task_name swingup
# wait

# python train.py --work_dir ./log_cp/cp_swingup_false_w100xdamp_2 --num_train_steps 100000 --seed 2 --task_name swingup
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_swingup_true_w100xdamp_2 --num_train_steps 200000 --seed 2 --task_name swingup
# wait

# python train.py --work_dir ./log_cp/cp_swingup_false_w100xdamp_3 --num_train_steps 100000 --seed 3 --task_name swingup
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_swingup_true_w100xdamp_3 --num_train_steps 200000 --seed 3 --task_name swingup
# wait

# python train.py --work_dir ./log_cp/cp_swingup_false_w100xdamp_4 --num_train_steps 100000 --seed 4 --task_name swingup
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_swingup_true_w100xdamp_4 --num_train_steps 200000 --seed 4 --task_name swingup
# wait

# python train.py --work_dir ./log_cp/cp_swingup_false_w100xdamp_5 --num_train_steps 100000 --seed 5 --task_name swingup
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_swingup_true_w100xdamp_5 --num_train_steps 200000 --seed 5 --task_name swingup
# wait

# python train.py --work_dir ./log_cp/cp_swingup_false_w100xdamp_6 --num_train_steps 100000 --seed 6 --task_name swingup
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_swingup_true_w100xdamp_5 --num_train_steps 200000 --seed 6 --task_name swingup
# wait

# python train.py --work_dir ./log_cp/cp_swingup_false_w100xdamp_7 --num_train_steps 100000 --seed 7 --task_name swingup
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_swingup_true_w100xdamp_7 --num_train_steps 200000 --seed 7 --task_name swingup
# wait

# python train.py --work_dir ./log_cp/cp_swingup_false_w100xdamp_8 --num_train_steps 100000 --seed 8 --task_name swingup
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_swingup_true_w100xdamp_8 --num_train_steps 200000 --seed 8 --task_name swingup
# wait

# python train.py --work_dir ./log_cp/cp_swingup_false_w100xdamp_9 --num_train_steps 100000 --seed 9 --task_name swingup
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_swingup_true_w100xdamp_9 --num_train_steps 200000 --seed 9 --task_name swingup
# wait


# python train.py --work_dir ./log_cp/cp_swingup_false_dblopt_0 --num_train_steps 100000 --seed 0 --task_name swingup --critic_target_update_freq 1 --actor_update_freq 1
# wait
# python train.py --work_dir ./log_cp/cp_swingup_false_dblopt_1 --num_train_steps 100000 --seed 1 --task_name swingup --critic_target_update_freq 1 --actor_update_freq 1
# wait
# python train.py --work_dir ./log_cp/cp_swingup_false_dblopt_2 --num_train_steps 100000 --seed 2 --task_name swingup --critic_target_update_freq 1 --actor_update_freq 1
# wait
# python train.py --work_dir ./log_cp/cp_swingup_false_dblopt_3 --num_train_steps 100000 --seed 3 --task_name swingup --critic_target_update_freq 1 --actor_update_freq 1
# wait

# # doubled actor critic and alpha learning rate to accompany doubled buffer size and doubled buffer adding per step
# python train.py --work_dir ./log_cp/cp_swingup_true_1opt42envstp_dblbtch_dblaaclr_0 --time_rev True --num_train_steps 100000 --seed 0 --task_name swingup --critic_target_update_freq 4 --actor_update_freq 4 --batch_size 256 --alpha_lr 0.0002 --actor_lr 0.002 --critic_lr 0.002
# wait
# python train.py --work_dir ./log_cp/cp_swingup_true_1opt42envstp_dblbtch_dblaaclr_1 --time_rev True --num_train_steps 100000 --seed 1 --task_name swingup --critic_target_update_freq 4 --actor_update_freq 4 --batch_size 256 --alpha_lr 0.0002 --actor_lr 0.002 --critic_lr 0.002
# wait
# python train.py --work_dir ./log_cp/cp_swingup_true_1opt42envstp_dblbtch_dblaaclr_2 --time_rev True --num_train_steps 100000 --seed 2 --task_name swingup --critic_target_update_freq 4 --actor_update_freq 4 --batch_size 256 --alpha_lr 0.0002 --actor_lr 0.002 --critic_lr 0.002
# wait
# python train.py --work_dir ./log_cp/cp_swingup_true_1opt42envstp_dblbtch_dblaaclr_3 --time_rev True --num_train_steps 100000 --seed 3 --task_name swingup --critic_target_update_freq 4 --actor_update_freq 4 --batch_size 256 --alpha_lr 0.0002 --actor_lr 0.002 --critic_lr 0.002
# wait
# python train.py --work_dir ./log_cp/cp_swingup_true_1opt42envstp_dblbtch_dblaaclr_4 --time_rev True --num_train_steps 100000 --seed 4 --task_name swingup --critic_target_update_freq 4 --actor_update_freq 4 --batch_size 256 --alpha_lr 0.0002 --actor_lr 0.002 --critic_lr 0.002
# wait
# python train.py --work_dir ./log_cp/cp_swingup_true_1opt42envstp_dblbtch_dblaaclr_5 --time_rev True --num_train_steps 100000 --seed 5 --task_name swingup --critic_target_update_freq 4 --actor_update_freq 4 --batch_size 256 --alpha_lr 0.0002 --actor_lr 0.002 --critic_lr 0.002
# wait



# python train.py --work_dir ./log_cp/cp_swingup_true_1opt42envstp_dblbtch_dblaclr_0 --time_rev True --num_train_steps 100000 --seed 0 --task_name swingup --critic_target_update_freq 4 --actor_update_freq 4 --batch_size 256 --actor_lr 0.002 --critic_lr 0.002
# wait
# python train.py --work_dir ./log_cp/cp_swingup_true_1opt42envstp_dblbtch_dblaclr_1 --time_rev True --num_train_steps 100000 --seed 1 --task_name swingup --critic_target_update_freq 4 --actor_update_freq 4 --batch_size 256 --actor_lr 0.002 --critic_lr 0.002
# wait
# python train.py --work_dir ./log_cp/cp_swingup_true_1opt42envstp_dblbtch_dblaclr_2 --time_rev True --num_train_steps 100000 --seed 2 --task_name swingup --critic_target_update_freq 4 --actor_update_freq 4 --batch_size 256 --actor_lr 0.002 --critic_lr 0.002
# wait
# python train.py --work_dir ./log_cp/cp_swingup_true_1opt42envstp_dblbtch_dblaclr_3 --time_rev True --num_train_steps 100000 --seed 3 --task_name swingup --critic_target_update_freq 4 --actor_update_freq 4 --batch_size 256 --actor_lr 0.002 --critic_lr 0.002
# wait
# python train.py --work_dir ./log_cp/cp_swingup_true_1opt42envstp_dblbtch_dblaclr_4 --time_rev True --num_train_steps 100000 --seed 4 --task_name swingup --critic_target_update_freq 4 --actor_update_freq 4 --batch_size 256 --actor_lr 0.002 --critic_lr 0.002
# wait
# python train.py --work_dir ./log_cp/cp_swingup_true_1opt42envstp_dblbtch_dblaclr_5 --time_rev True --num_train_steps 100000 --seed 5 --task_name swingup --critic_target_update_freq 4 --actor_update_freq 4 --batch_size 256 --actor_lr 0.002 --critic_lr 0.002
# wait


# Testing 1 update per environment step regardless of approach with large friction
# python train.py --work_dir ./log_cp/cp_swingup_false_w2000xdamp__dblopt_0 --num_train_steps 100000 --seed 0 --task_name swingup --critic_target_update_freq 1 --actor_update_freq 1
# wait
# # python train.py --time_rev True --work_dir ./log_cp/cp_swingup_true_w2000xdamp_0 --num_train_steps 200000 --seed 0 --task_name swingup
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_swingup_true_w2000xdamp_dblbtch_0 --num_train_steps 200000 --seed 0 --task_name swingup --batch_size 256
# wait

# python train.py --work_dir ./log_cp/cp_swingup_false_w2000xdamp__dblopt_1 --num_train_steps 100000 --seed 1 --task_name swingup --critic_target_update_freq 1 --actor_update_freq 1
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_swingup_true_w2000xdamp_1 --num_train_steps 200000 --seed 1 --task_name swingup
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_swingup_true_w2000xdamp_dblbtch_1 --num_train_steps 200000 --seed 1 --task_name swingup --batch_size 256
# wait

# python train.py --work_dir ./log_cp/cp_swingup_false_w2000xdamp__dblopt_2 --num_train_steps 100000 --seed 2 --task_name swingup --critic_target_update_freq 1 --actor_update_freq 1
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_swingup_true_w2000xdamp_2 --num_train_steps 200000 --seed 2 --task_name swingup
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_swingup_true_w2000xdamp_dblbtch_2 --num_train_steps 200000 --seed 2 --task_name swingup --batch_size 256
# wait

# python train.py --work_dir ./log_cp/cp_swingup_false_w2000xdamp__dblopt_3 --num_train_steps 100000 --seed 3 --task_name swingup --critic_target_update_freq 1 --actor_update_freq 1
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_swingup_true_w2000xdamp_3 --num_train_steps 200000 --seed 3 --task_name swingup
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_swingup_true_w2000xdamp_dblbtch_3 --num_train_steps 200000 --seed 3 --task_name swingup --batch_size 256
# wait

# python train.py --work_dir ./log_cp/cp_swingup_false_w2000xdamp__dblopt_4 --num_train_steps 100000 --seed 4 --task_name swingup --critic_target_update_freq 1 --actor_update_freq 1
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_swingup_true_w2000xdamp_4 --num_train_steps 200000 --seed 4 --task_name swingup
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_swingup_true_w2000xdamp_dblbtch_4 --num_train_steps 200000 --seed 4 --task_name swingup --batch_size 256
# wait

# python train.py --time_rev True --work_dir ./log_cp_fixes_to_updfreq/cp_swingup_true_0pt5supdfreq_doublebtchlrs_8 --num_train_steps 50000 --seed 8 --task_name swingup --critic_target_update_freq 1 --actor_update_freq 1 --critic_update_freq 1 --batch_size 256 --actor_lr 0.002 --critic_lr 0.002
# wait
# python train.py --work_dir ./log_cp_fixes_to_updfreq/cp_swingup_false_0pt5supdfreq_9 --num_train_steps 50000 --seed 9 --task_name swingup --critic_target_update_freq 1 --actor_update_freq 1 --critic_update_freq 1
# wait
# python train.py --time_rev True --work_dir ./log_cp_fixes_to_updfreq/cp_swingup_true_0pt5supdfreq_doublebtchlrs_9 --num_train_steps 50000 --seed 9 --task_name swingup --critic_target_update_freq 1 --actor_update_freq 1 --critic_update_freq 1 --batch_size 256 --actor_lr 0.002 --critic_lr 0.002
# wait
# python train.py --time_rev True --work_dir ./log_cp_fixes_to_updfreq/cp_swingup_true_0pt5supdfreq_w2000xdamp_0 --num_train_steps 50000 --seed 0 --task_name swingup --critic_target_update_freq 1 --actor_update_freq 1 --critic_update_freq 1
# wait
# for i in {0..9}
# do
# python train.py --work_dir ./log_cp_fixes_to_updfreq/cp_swingup_false_0pt3supdfreq$i --num_train_steps 50000 --seed $i --task_name swingup --critic_target_update_freq 1 --actor_update_freq 1 --critic_update_freq 1
# wait
# python train.py --time_rev True --work_dir ./log_cp_fixes_to_updfreq/cp_swingup_true_0pt3supdfreq$i --num_train_steps 50000 --seed $i --task_name swingup --critic_target_update_freq 1 --actor_update_freq 1 --critic_update_freq 1
# wait
# done
# python train.py --work_dir ./log_pixel/true_swing_pixel_2actionrep_0pt5supdfreq2 --replay_buffer_capacity 1000000 --action_repeat 2 --num_train_steps 200000 --time_rev True --seed 2 --task_name swingup --encoder_type 'pixel' --decoder_type 'pixel' --eval_freq 10000 --critic_target_update_freq 1 --actor_update_freq 1 --critic_update_freq 1

# for i in {2..8}
# do
# python train.py --gpu_choice 1 --work_dir ./log_pixel/false_swing_pixel_2actionrep_1pt0supdfreq$i --replay_buffer_capacity 1000000 --action_repeat 2 --num_train_steps 200000 --seed $i --task_name swingup --encoder_type 'pixel' --decoder_type 'pixel' --eval_freq 10000 --critic_target_update_freq 1 --actor_update_freq 1 --critic_update_freq 1 &
# python train.py --gpu_choice 2 --work_dir ./log_pixel/true_swing_pixel_2actionrep_1pt0supdfreq$i --replay_buffer_capacity 1000000 --action_repeat 2 --num_train_steps 200000 --time_rev True --seed $i --task_name swingup --encoder_type 'pixel' --decoder_type 'pixel' --eval_freq 10000 --critic_target_update_freq 1 --actor_update_freq 1 --critic_update_freq 1
# wait
# done
# python train.py --gpu_choice 2 --work_dir ./log_pixel/true_swing_pixel_2actionrep_1pt0supdfreq1 --replay_buffer_capacity 1000000 --action_repeat 2 --num_train_steps 200000 --time_rev True --seed 1 --task_name swingup --encoder_type 'pixel' --decoder_type 'pixel' --eval_freq 10000 --critic_target_update_freq 1 --actor_update_freq 1 --critic_update_freq 1

# python train.py --gpu_choice 1 --work_dir ./log_pixel/false_swing_pixel_2actionrep_1pt0supdfreq9 --replay_buffer_capacity 1000000 --action_repeat 2 --num_train_steps 200000 --seed 9 --task_name swingup --encoder_type 'pixel' --decoder_type 'pixel' --eval_freq 10000 --critic_target_update_freq 1 --actor_update_freq 1 --critic_update_freq 1 &
# python train.py --gpu_choice 2 --work_dir ./log_pixel/true_swing_pixel_2actionrep_1pt0supdfreq9 --replay_buffer_capacity 1000000 --action_repeat 2 --num_train_steps 200000 --time_rev True --seed 9 --task_name swingup --encoder_type 'pixel' --decoder_type 'pixel' --eval_freq 10000 --critic_target_update_freq 1 --actor_update_freq 1 --critic_update_freq 1
# wait

# conda init
# eval "$(conda shell.bash hook)"
# conda activate aggregate_tb
# wait
# tb-reducer log_pixel/false_swing_pixel_2actionrep_1pt0supdfreq*/tb -o averaged/false_swing_pixel_2actionrep_1pt0supdfreq/ -r mean,std,min,max --handle-dup-steps 'mean' --lax-steps --lax-tags --overwrite
# wait
# tb-reducer log_pixel/true_swing_pixel_2actionrep_1pt0supdfreq*/tb -o averaged/true_swing_pixel_2actionrep_1pt0supdfreq/ -r mean,std,min,max --handle-dup-steps 'mean' --lax-steps --lax-tags --overwrite

# python train.py --work_dir ./log_cp/cp_swingup_false_w2000xdamp__dblopt_6 --num_train_steps 100000 --seed 6 --task_name swingup --critic_target_update_freq 1 --actor_update_freq 1
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_swingup_true_w2000xdamp_6 --num_train_steps 200000 --seed 6 --task_name swingup
# wait

# python train.py --work_dir ./log_cp/cp_swingup_false_w2000xdamp__dblopt_7 --num_train_steps 100000 --seed 7 --task_name swingup --critic_target_update_freq 1 --actor_update_freq 1
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_swingup_true_w2000xdamp_7 --num_train_steps 200000 --seed 7 --task_name swingup
# wait

# python train.py --work_dir ./log_cp/cp_swingup_false_w2000xdamp__dblopt_8 --num_train_steps 100000 --seed 8 --task_name swingup --critic_target_update_freq 1 --actor_update_freq 1
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_swingup_true_w2000xdamp_8 --num_train_steps 200000 --seed 8 --task_name swingup
# wait

# python train.py --work_dir ./log_cp/cp_swingup_false_w2000xdamp__dblopt_9 --num_train_steps 100000 --seed 9 --task_name swingup --critic_target_update_freq 1 --actor_update_freq 1
# wait
# python train.py --time_rev True --work_dir ./log_cp/cp_swingup_true_w2000xdamp_9 --num_train_steps 200000 --seed 9 --task_name swingup
# wait


# # Humanoid testing
# python train.py --gpu_choice 1 --work_dir ./log_human/false_human_run_matchtorchsac_0pt5supdfreq1 --replay_buffer_capacity 5000000 --action_repeat 1 --num_train_steps 5000000 --seed 1 --domain_name humanoid --task_name run --eval_freq 10000 --critic_target_update_freq 1 --actor_update_freq 1 --critic_update_freq 1 &
# python train.py --gpu_choice 2 --work_dir ./log_human/true_human_run_matchtorchsac_0pt5supdfreq1 --replay_buffer_capacity 5000000 --action_repeat 1 --num_train_steps 5000000 --time_rev True --seed 1 --domain_name humanoid --task_name run --eval_freq 10000 --critic_target_update_freq 1 --actor_update_freq 1 --critic_update_freq 1
# wait

# Walker testing
# python train.py --gpu_choice 1 --work_dir ./log_human/false_walker_run_matchtorchsac_0pt5supdfreq5 --replay_buffer_capacity 5000000 --action_repeat 1 --num_train_steps 200000 --seed 5 --domain_name walker --task_name run --eval_freq 10000 --critic_target_update_freq 1 --actor_update_freq 1 --critic_update_freq 1 &
# python train.py --gpu_choice 2 --work_dir ./log_human/true_walker_run_matchtorchsac_0pt5supdfreq5 --replay_buffer_capacity 5000000 --action_repeat 1 --num_train_steps 200000 --time_rev True --seed 5 --domain_name walker --task_name run --eval_freq 10000 --critic_target_update_freq 1 --actor_update_freq 1 --critic_update_freq 1
# wait


# for i in {0..9}
# do
# python train.py --gpu_choice 3 --work_dir ./log_pixel/false_swing_pixel_2actionrep_w2000xdamp_0pt5supdfreq$i --replay_buffer_capacity 1000000 --action_repeat 2 --num_train_steps 200000 --seed $i --task_name swingup --encoder_type 'pixel' --decoder_type 'pixel' --eval_freq 10000 --critic_target_update_freq 1 --actor_update_freq 1 --critic_update_freq 1 &
# python train.py --gpu_choice 4 --work_dir ./log_pixel/true_swing_pixel_2actionrep_w2000xdamp_0pt5supdfreq$i --replay_buffer_capacity 1000000 --action_repeat 2 --num_train_steps 200000 --time_rev True --seed $i --task_name swingup --encoder_type 'pixel' --decoder_type 'pixel' --eval_freq 10000 --critic_target_update_freq 1 --actor_update_freq 1 --critic_update_freq 1
# wait
# done

# conda init
# eval "$(conda shell.bash hook)"
# conda activate aggregate_tb
# wait
# tb-reducer log_pixel/false_swing_pixel_2actionrep_w2000xdamp_0pt5supdfreq*/tb -o averaged/false_swing_pixel_2actionrep_w2000xdamp_0pt5supdfreq/ -r mean,std,min,max --handle-dup-steps 'mean' --lax-steps --lax-tags --overwrite
# wait
# tb-reducer log_pixel/true_swing_pixel_2actionrep_w2000xdamp_0pt5supdfreq*/tb -o averaged/true_swing_pixel_2actionrep_w2000xdamp_0pt5supdfreq/ -r mean,std,min,max --handle-dup-steps 'mean' --lax-steps --lax-tags --overwrite


# for i in {4..5}
# do
# python train.py --gpu_choice 2 --work_dir ./log_twopole/false_twopole_swing_2actionrep_0pt5supdfreq$i --replay_buffer_capacity 2000000 --action_repeat 2 --num_train_steps 1000000 --seed $i --task_name two_poles --eval_freq 10000 --critic_target_update_freq 1 --actor_update_freq 1 --critic_update_freq 1 &
# python train.py --gpu_choice 3 --work_dir ./log_twopole/true_twopole_swing_2actionrep_0pt5supdfreq$i --replay_buffer_capacity 2000000 --action_repeat 2 --num_train_steps 1000000 --time_rev True --seed $i --task_name two_poles --eval_freq 10000 --critic_target_update_freq 1 --actor_update_freq 1 --critic_update_freq 1
# wait
# done


# python train.py --gpu_choice 2 --work_dir ./log_twopole/false_twopole_swing_2actionrep_0pt5supdfreq0 --replay_buffer_capacity 2000000 --action_repeat 2 --num_train_steps 1000000 --seed 0 --task_name two_poles --eval_freq 10000 --critic_target_update_freq 1 --actor_update_freq 1 --critic_update_freq 1 &
# python train.py --gpu_choice 3 --work_dir ./log_twopole/true_twopole_swing_2actionrep_0pt5supdfreq0 --replay_buffer_capacity 2000000 --action_repeat 2 --num_train_steps 1000000 --time_rev True --seed 0 --task_name two_poles --eval_freq 10000 --critic_target_update_freq 1 --actor_update_freq 1 --critic_update_freq 1
# wait

# python train.py --gpu_choice 2 --work_dir ./log_twopole/false_twopole_swing_2actionrep_0pt5supdfreq1 --replay_buffer_capacity 2000000 --action_repeat 2 --num_train_steps 1000000 --seed 1 --task_name two_poles --eval_freq 10000 --critic_target_update_freq 1 --actor_update_freq 1 --critic_update_freq 1 &
# python train.py --gpu_choice 3 --work_dir ./log_twopole/true_twopole_swing_2actionrep_0pt5supdfreq1 --replay_buffer_capacity 2000000 --action_repeat 2 --num_train_steps 1000000 --time_rev True --seed 1 --task_name two_poles --eval_freq 10000 --critic_target_update_freq 1 --actor_update_freq 1 --critic_update_freq 1
# wait

for i in {1,3,5,7,9}
do
python train.py --gpu_choice 0 --work_dir ./lunar_log_fixedlogging/false_trial$i --replay_buffer_capacity 300000 --num_train_steps 150000 --seed $i --eval_freq 2000 &
python train.py --gpu_choice 1 --work_dir ./lunar_log_fixedlogging/true_trial$i --replay_buffer_capacity 300000 --num_train_steps 150000 --time_rev True --seed $i --eval_freq 2000 
# python train.py --gpu_choice 2 --work_dir ./lunar_log_fixedlogging/false_trial$((i+1)) --replay_buffer_capacity 300000 --num_train_steps 150000 --seed $((i+1)) --eval_freq 2000 &
# python train.py --gpu_choice 3 --work_dir ./lunar_log_fixedlogging/true_trial$((i+1)) --replay_buffer_capacity 300000 --num_train_steps 150000 --time_rev True --seed $((i+1)) --eval_freq 2000
wait
done