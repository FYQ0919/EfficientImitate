
import copy
import time
import pickle
import numpy
import ray
import torch

import pickle_utils
import numpy as np
class ExpertGameHistory:
    """
    Store only useful information of a self-play game.
    """
    def __init__(self):
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
        self.reanalysed_predicted_root_values = None

    def subset(self, pos, duration):
        if pos < 0:
            pos = 0

        res = ExpertGameHistory()
        res.observation_history = self.observation_history[pos:pos + duration]
        res.action_history = self.action_history[pos:pos + duration]
        res.reward_history = self.reward_history[pos:pos + duration]

        return res

    def get_stacked_observations(self, index, num_stacked_observations):
        """
        Generate a new observation with the observation at the index position
        and num_stacked_observations past observations and actions stacked.
        """

        # Convert to positive index
        index = index % len(self.observation_history)

        # [t, t-1, t-2, ...]
        # stacked_observations = self.observation_history[index].copy()
        observations = []

        for past_observation_index in reversed(
            range(index + 1 - num_stacked_observations, index + 1)
        ):

            if 0 <= past_observation_index:
                observations.append(self.observation_history[past_observation_index])

            else:
                observations.append(self.observation_history[0])

        stacked_observations = np.concatenate(observations, axis=0)
        return stacked_observations

if __name__ == '__main__':
    path = "./data/curling_traj.pkl"
    # path = "./data/cheetah4_official_demo20.pkl"
    buffer = {}
    expert_trajectories = pickle_utils.load_data(path)
    # curling_traj = []
    # for i in range(4):
    #     path = f"./data/{i}_curling_traj.pkl"
    #     curling_traj.append(pickle_utils.load_data(path))
    #
    # with open(f'experiments/curling_traj.pkl', 'wb') as f:
    #     pickle.dump(curling_traj, f)


    for idx, traj in enumerate(expert_trajectories):
        game_history = ExpertGameHistory()
        print('Expert Trajectory Length =', len(traj['obs']))
        game_history.observation_history = traj['obs']
        game_history.action_history = [traj['act'][0]] + traj['act']  # we need a padding at the front.
        game_history.reward_history = [1.0 for _ in range(len(game_history.action_history))]

        index = - idx - 1  # For convenience, we use the -1, -2, ..., -n to index the expert demos.
        buffer[index] = game_history