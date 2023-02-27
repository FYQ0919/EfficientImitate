import datetime
import os
import numpy as np
import gym
import numpy
import torch
from algo.state_ei.main import RZero
from game.dmcontrol import make_dmcontrol_lowdim
from curling_simulator.gym_env import CurlingTwoAgentGymEnv_v1
from arg_utils import *
import pickle_utils

try:
    import cv2
except ModuleNotFoundError:
    raise ModuleNotFoundError('\nPlease run "pip install gym[atari]"')


class CurlingConfig:
    def __init__(self, env_id):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization
        self.env_id = env_id
        self.exp_name = 'Experiment'

        self.seed = 42  # Seed for numpy, torch and the game
        self.max_num_gpus = 4  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available
        self.total_cpus = 128
        self.num_gpus = 4

        ### Expert data
        self.expert_demo_path = './data/walker_walk_demo.pkl'

        self.env = CurlingTwoAgentGymEnv_v1()

        ### Game
        # self.dummy_game = make_dmcontrol_lowdim(env_id, task_name, 0)
        self.observation_shape = (1, 1, 34)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space_size = 1            #  Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 1  # How many observations are stacked together.
        self.env_action_space = self.env.action_space

        ### Self-Play
        self.epoch_repeat = 1
        self.num_workers = 4  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = True
        self.max_moves = 250  # Maximum number of moves if game is not finished before
        self.num_simulations = 50  # Number of future moves self-simulated
        self.discount = 0.99    # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        # Feb 8 Comments by Zhaoheng:
        # This part can work for the walker.
        self.nd = 5
        self.init_zero = 1
        self.mlp_obs_shape = self.env.observation_space.shape[0]
        self.mlp_action_shape = self.env.action_space.shape[0]
        self.act = 'relu'
        self.rep_act = 'identity'
        self.mlp_hidden_shape = 128
        self.mlp_proj_shape = 128
        self.mlp_dyn_shape = [256, 256]
        self.mlp_rep_shape = [256, 256]
        self.mlp_rew_shape = [128]
        self.mlp_val_shape = []
        self.mlp_pi_shape = []
        self.mlp_proj_net_shape = [512, ]
        self.mlp_proj_pred_net_shape = [512, ]

        # THIS IS VERY IMPORTANT!!!!!
        # MCTS Action Samples
        self.mcts_num_policy_samples = 6    # Number of actions that will be sampled from the current policy.
        self.mcts_num_random_samples = 4    # Number of actions that will be sampled randomly.

        ### Training
        self.time_prefix = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results",
                                         os.path.basename(__file__)[:-3])  # Path to store the model weights and TensorBoard logs

        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = int(1000e3)  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 256  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 1000  # TODO(): Set back to 1e3, Number of training steps before using the model for self-playing
        # self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available
        self.save_interval = 10000
        self.frame_skip = 8

        self.target_update_interval = 150
        self.selfplay_update_interval = 50

        self.optimizer = "SGD"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 0.0001  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.001  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 30e3

        ### Replay Buffer
        self.replay_buffer_size = 200  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps_reanalyze = 5  # Number of game moves to keep for every batch element
        self.num_unroll_steps = 5

        self.td_steps = 20  # Number of steps in the future to take into account for calculating the target value
        self.PER = False  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 1  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = True

        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step


        self.ratio_lower_bound = 1.0
        self.ratio_upper_bound = 1.0    # Desired training steps per self played step ratio. Equivalent to a synchronous version,
                                        # training can take much longer. Set it to None to disable it
        # Loss functions
        self.policy_loss_coeff = 1.0 # 1.0
        self.reward_loss_coeff = 1.0
        self.value_loss_coeff  = 0.5    # Switch to 1.0 can be better .?
        self.entropy_loss_coeff = 0.001
        self.grad_loss_coeff = 10.0
        self.consistency_loss_coeff = 2.0
        self.bc_coeff = 0.1
        self.bc_decay = 0.95
        self.bc_decay_interval = 1000
        self.bc_frequency = 2

        self.max_grad_norm = 10

        self.selfplay_model_serve_num = 4
        self.model_worker_serve_num = 2
        self.batch_worker_num = 30
        self.support_size = 200
        self.support_step = 0.5

        self.reward_support_size = 1
        self.reward_support_step = 0.5

        self.ssl_target = 1
        self.explore_type = 'add'
        self.explore_scale = 1.0

        self.reward_delta = 0.2
        self.reward_amp = 1.0

        self.logstd_max = 2.0
        self.logstd_min = -2.0

        self.two_players = False

    def set_result_path(self):
        self.results_path = os.path.join(self.results_path, self.exp_name + '_' + self.time_prefix)
        print("Result path:", self.results_path)

    def get_network_config(self):
        config = {
            'mlp_obs_shape':        self.mlp_obs_shape,
            'mlp_action_shape':     self.mlp_action_shape,
            'mlp_hidden_shape':     self.mlp_hidden_shape,
            'mlp_proj_shape':       self.mlp_proj_shape
        }
        return config

    def new_game(self, seed=0):
        env = CurlingTwoAgentGymEnv_v1()
        return env

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """

        if trained_steps < 50e3:
            return 1.0
        elif trained_steps < 75e3:
            return 0.5
        else:
            return 0.25

    def initialize(self):
        self.mlp_obs_shape = self.env.observation_space.shape[0]
        self.mlp_action_shape = self.env.action_space.shape[0]
        self.env_action_space = self.env.action_space
        self.action_space_size = self.mlp_action_shape

    def sample_random_actions(self, n):
        # acts = [-0.6, -0.4, -0.2, -0.1, 0, 0.1, 0.2, 0.4, 0.6]
        # import random
        # random.shuffle(acts)
        actions = [self.env_action_space.sample() for _ in range(n)]
        return np.array(actions) #.reshape(-1, 1)

    def sample_random_actions_fast(self, n):
        return np.random.randn(n, self.mlp_action_shape)


if __name__ == '__main__':
    config = CurlingConfig(env_id='curling')
    parser = create_parser_from_config(config)
    parser.add_argument('--test', type=int, default=0)
    parser.add_argument('--checkpoint', type=str, default='')
    args = parser.parse_args()
    config = override_config_from_arg(config, args)
    config.set_result_path()
    config.initialize()

    os.makedirs(config.results_path, exist_ok=True)
    dump_config(os.path.join(config.results_path, 'config.json'), config)
    for k, v in config.__dict__.items():
        print(k, v)
    #print(config.__dict__)
    agent = RZero(config)

    if not args.test:
        agent.train_new(log_in_tensorboard=False)
    else:
        rollouts = agent.test_full(args.checkpoint, 8, 2)
        pickle_utils.gsave_data(rollouts, args.checkpoint.replace(".pth", "_rollout.pkl"))

