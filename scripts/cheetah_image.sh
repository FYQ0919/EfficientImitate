CUDA_VISIBLE_DEVICES=0,1,2,3 python ./experiments/obs_ei.py --num_gpus 4 --epoch_repeat 2 --num_workers 4 --env_id cheetah --task_name run --exp_name Cheetah --value_loss_coeff 0.5 --policy_loss_coeff 1.0 --entropy_loss_coeff 0.00 --replay_buffer_size 400 --num_simulations 50 --mcts_num_policy_samples 12 --mcts_num_random_samples 4 --lr_init 0.02 --max_moves 250 --frame_skip 4 --ssl_target 0 --target_update_interval 200 --reward_loss_coeff 0.1 --grad_loss_coeff 1.0 --init_zero 1 --bn_mt 0.1 --td_steps 1 --expert_demo_path ./data/cheetah4_official_demo20.pkl --bc_coeff 0.01 --consistency_loss_coeff 20.0 --seed 5 --bn_mt 0.1
