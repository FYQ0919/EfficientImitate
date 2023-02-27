# -*- coding: utf-8 -*-
import os, sys, time, pdb
import numpy as np
import taichi as ti
from typing import TypeVar, Iterable, Tuple, Union

from .utils import *  # 辅助函数与类
from .config import *  # 重要的参数
from .curling_sim import *
from enum import Enum
import cv2

from .curling_env import *
import gym
from gym import spaces
from gym.utils import seeding

# 基于规则的智能体
from .rule_based_agent import Rule_based_strag


"""一些有关gym环境的设置项"""
RENDER = True
DEBUG_MODE = False

OBS_GRIDS_X = 32
OBS_GRIDS_Y = 32
ACTION_GRIDS_X = 20
ACTION_GRIDS_Y = 8

OPPONENT_PLAYER = None

OBS_SPACE = spaces.Box(low=np.inf * -1,
                        high=np.inf,
                        shape=(3, OBS_GRIDS_X, OBS_GRIDS_Y),
                        dtype=np.float32)
ACTION_SPACE = spaces.Discrete(ACTION_GRIDS_X * ACTION_GRIDS_Y * 2) # 最后的2代表左旋和右旋

class CurlingTwoAgentGymEnv_v1(gym.Env, CurlingSimuOneEndEnv):
    """gym环境: 两个智能体对打 v1 版本 """
    
    metadata = {'render.modes': ['human']}

    def __init__(self, render=RENDER):
        print(f"[Gym] {self.__doc__}")
        # 初始化仿真环境
        gym.Env.__init__(self, )
        CurlingSimuOneEndEnv.__init__(self, render)
        self._elapsed_steps = 0
        # 设置观测空间大小
        # self.obs_grids = [OBS_GRIDS_X, OBS_GRIDS_Y]
        # self.observation_space = OBS_SPACE
        self.observation_space = spaces.Box(low=np.inf * -1,
                        high=np.inf,
                        shape=(34,),
                        dtype=np.float32)

        ## 设置动作空间大小

        # self.action_space = ACTION_SPACE
        self.action_space = spaces.Box(low=np.array([-1,2.1,-0.2], dtype=np.float32),
                                      high=np.array([1,2.5,0.2], dtype=np.float32),)
        # 使用image进行reset
        self.initialize_img = None
        # obs中是否包含历史信息
        self.history_feature = None
        
        # 将所有可能动作进行反解, 储存起来, 加速程序
        # self.all_actions_dict = list(map(self.decode_action, range(self.action_space.n)))
        pass 

    def observe(self, player: Player):
        """observe接口"""
        """ATTENTION: 为了(可能)减小 muzero dynamic 模块学习难度, 
            obs中的双方球的pos先后顺序不再随着obs的player而变动"""
        # # 获取两队的球的位置信息
        pos_vec = np.concatenate([
            self.observe_stones_pos(Player.A),
            self.observe_stones_pos(Player.B)
        ])
        # 获取当前shot的轮数
        curr_turn_vec = np.array([(self.curr_shot - 2) // 2], dtype=np.float32)
        # 先后手信息, 算法可以确定是哪一个agent
        is_first_vec = np.array([player.value], dtype=np.float32)
        # 合并信息

        obs = np.concatenate([
            pos_vec.flatten(),
            # curr_turn_vec,
            # is_first_vec
        ])

        return obs.astype(np.float32)

        # obs = self.feature_img(self.curr_player)
        # obs = self.simplified_feature_img(self.curr_player)
        # obs = self.layer_his_feature(player=player)
        # return obs

    # def simplified_feature_img(self, player: Player):
    #     def distance(stone):
    #         stone = np.resize(stone, (1, 2))
    #         stone_pos = stone * valid_pos_range + valid_pos_min
    #         return np.linalg.norm(center_pos - stone_pos)

    #     # 先后手信息, 算法可以确定是哪一个agent
    #     is_first_vec = player.value
    #     feat = np.zeros(network_input_size, dtype=np.float32)

    #     if is_first_vec == 0:  # curr Player A
    #         self_stones = self.observe_stones_pos(Player.A)
    #         oppo_stones = self.observe_stones_pos(Player.B)
    #     else:
    #         self_stones = self.observe_stones_pos(Player.B)
    #         oppo_stones = self.observe_stones_pos(Player.A)

    #     for i in range(self_stones.shape[0]):
    #         if (self_stones[i, 0] != invalid_fill_value):
    #             h, w = self.position2point(self_stones[i])  # all
    #             feat[0, h, w] = 1

    #             dis = distance(self_stones[i])
    #             if dis <= r2:  # in house
    #                 h, w = self.position2point(self_stones[i])
    #                 feat[2, h, w] = 1

    #     for i in range(oppo_stones.shape[0]):
    #         if (oppo_stones[i, 0] != invalid_fill_value):
    #             h, w = self.position2point(oppo_stones[i])  # all
    #             feat[1, h, w] = 1

    #             dis = distance(oppo_stones[i])  # in house
    #             if dis <= r2:
    #                 h, w = self.position2point(oppo_stones[i])
    #                 feat[3, h, w] = 1

    #     return feat

    # def layer_6_feature(self, player: Player):
    #     def distance(stone):

    #         stone = np.resize(stone, (1, 2))
    #         stone_pos = stone * valid_pos_range + valid_pos_min
    #         return np.linalg.norm(center_pos - stone_pos)

    #     # 先后手信息, 算法可以确定是哪一个agent
    #     # is_first_vec = player.value
    #     feat = np.zeros(network_input_size, dtype=np.float32)

    #     # if is_first_vec == 0:#curr Player A
    #     #     self_stones = self.observe_stones_pos(Player.A)
    #     #     oppo_stones = self.observe_stones_pos(Player.B)
    #     # else:
    #     #     self_stones = self.observe_stones_pos(Player.B)
    #     #     oppo_stones = self.observe_stones_pos(Player.A)

    #     self_stones = self.observe_stones_pos(0)
    #     oppo_stones = self.observe_stones_pos(1)

    #     for i in range(self_stones.shape[0]):
    #         if (self_stones[i, 0] != invalid_fill_value):
    #             h, w = self.position2point(self_stones[i])  # all
    #             feat[0, h, w] = 1

    #             dis = distance(self_stones[i])
    #             if dis <= r2:  # in house
    #                 h, w = self.position2point(self_stones[i])
    #                 feat[2, h, w] = 1

    #     for i in range(oppo_stones.shape[0]):
    #         if (oppo_stones[i, 0] != invalid_fill_value):
    #             h, w = self.position2point(oppo_stones[i])  # all
    #             feat[1, h, w] = 1

    #             dis = distance(oppo_stones[i])  # in house
    #             if dis <= r2:
    #                 h, w = self.position2point(oppo_stones[i])
    #                 feat[3, h, w] = 1

    #     feat[4, :, :] = self.curr_shot % 2
    #     feat[5, :, :] = self.curr_shot // 2

    #     return feat

    def layer_his_feature(self, player: Player):
        # 创建观测
        feat = np.zeros(self.observation_space.shape, dtype=np.float32)
        # 获取先后手球的位置信息
        stones_pos_A = self.observe_stones_pos(Player.A)
        stones_pos_B = self.observe_stones_pos(Player.B)
        # 去除其中的无效球, 将位置转化为图像格式的信息
        xy_A = (stones_pos_A[stones_pos_A[:,0] != invalid_fill_value] * self.obs_grids).astype(np.int64).T
        xy_B = (stones_pos_B[stones_pos_B[:,0] != invalid_fill_value] * self.obs_grids).astype(np.int64).T
        # feat[0, xy_A[0], xy_A[1]] = 1.0
        # feat[1, xy_B[0], xy_B[1]] = 1.0
        # TODO: 根据当前player调整AB球channel位置, 保证当前player的球一定在第0个channel, 单智能体训练
        feat[player.value, xy_A[0], xy_A[1]] = 1.0
        feat[1-player.value, xy_B[0], xy_B[1]] = 1.0
        # 轮次与先后手信息
        #TODO: 不让agent获得先后手信息, 单智能体训练
        feat[2, :, :] = self.curr_shot // 2 
        # feat[2, :, :] = self.curr_shot % 2
        # feat[3, :, :] = self.curr_shot // 2
        return feat

    # def feature_img(self, player: Player):
    #     def distance(stone):
    #         stone = np.resize(stone, (1, 2))
    #         return np.linalg.norm(center_pos - stone)

    #     # 先后手信息, 算法可以确定是哪一个agent
    #     is_first_vec = player.value

    #     feat = np.zeros(network_input_size, dtype=np.float32)
    #     feat[2, :, :] = 1
    #     # ones
    #     feat[3, :, :] = 1

    #     if player.value == 0:  # curr Player A
    #         A_stones = self.observe_stones_pos(Player.A)
    #         B_stones = self.observe_stones_pos(Player.B)
    #     else:
    #         A_stones = self.observe_stones_pos(Player.B)
    #         B_stones = self.observe_stones_pos(Player.A)

    #     # stone color
    #     for i in range(A_stones.shape[0]):
    #         if (A_stones[i, 0] != invalid_fill_value):
    #             h, w = self.position2point(A_stones[i])
    #             feat[0, h, w] = 1
    #             feat[2, h, w] = 0
    #     for i in range(B_stones.shape[0]):
    #         if (B_stones[i, 0] != invalid_fill_value):
    #             h, w = self.position2point(B_stones[i])
    #             feat[1, h, w] = 1
    #             feat[2, h, w] = 0

    #     # turn num
    #     # 获取当前shot的轮数
    #     curr_turn_vec = np.array([self.curr_shot // 2], dtype=np.float32)  # -2?
    #     if (curr_turn_vec == 0):
    #         feat[4 + is_first_vec * 4, :, :] = 1
    #     elif (curr_turn_vec == 1):
    #         if (is_first_vec == 0):
    #             feat[4, :, :] = 1
    #         else:
    #             feat[9, :, :] = 1
    #     elif (curr_turn_vec == 2 or curr_turn_vec == 3 or curr_turn_vec == 4 or curr_turn_vec == 5):
    #         feat[5 + is_first_vec * 4, :, :] = 1
    #     elif (curr_turn_vec == 6):
    #         feat[6 + is_first_vec * 4:, :] = 1
    #     elif (curr_turn_vec == 7):
    #         feat[7 + is_first_vec * 4:, :] = 1

    #         # in house
    #     for i in range(A_stones.shape[0]):
    #         if (A_stones[i, 0] != -1.0):
    #             dis = distance(A_stones[i])
    #             if dis <= r2:
    #                 h, w = self.position2point(A_stones[i])
    #                 feat[12, h, w] = 1
    #     for i in range(B_stones.shape[0]):
    #         if (B_stones[i, 0] != -1.0):
    #             dis = distance(B_stones[i])
    #             if dis <= r2:
    #                 h, w = self.position2point(B_stones[i])
    #                 feat[12, h, w] = 1

    #     # order to tee
    #     A_valid = []
    #     B_valid = []
    #     for i in range(A_stones.shape[0]):
    #         if (A_stones[i, 0] != -1.0):
    #             A_valid.append(A_stones[i])
    #     for i in range(B_stones.shape[0]):
    #         if (B_stones[i, 0] != -1.0):
    #             B_valid.append(B_stones[i])
    #     total_valid = A_valid + B_valid

    #     A_valid.sort(key=distance)
    #     B_valid.sort(key=distance)
    #     total_valid.sort(key=distance)

    #     # for i in range(len(A_valid)):
    #     #     h, w = self.position2point(A_valid[i])
    #     #     if i<4:
    #     #         feat[13 + i,h,w] = 1
    #     #     else:
    #     #         feat[16,h,w] = 1

    #     # for i in range(len(B_valid)):
    #     #     h, w = self.position2point(B_valid[i])
    #     #     if i<4:
    #     #         feat[17 + i,h,w] = 1
    #     #     else:
    #     #         feat[20,h,w] = 1

    #     for i in range(len(total_valid)):
    #         h, w = self.position2point(total_valid[i])
    #         if i < 8:
    #             feat[13 + i, h, w] = 1
    #         else:
    #             feat[20, h, w] = 1

    #     if np.sum(feat) >= -1e10:
    #         pass
    #     else:
    #         print('feat wrong')
    #         for i in range(21):
    #             print(np.sum(feat[i]))
    #             print(feat[i])
    #     return feat

    def to_play(self):
        """to_play函数示例, 待确认是否有bug..."""
        return self.curr_player.value

    def inverse_time_integrate(self, target_pos:np.ndarray or list, target_w:float):
        """反解动作"""
        target_v = [0, 0]
        target_theta = 0
        target_v_norm = 0
        
        target_pos = np.array(target_pos)
        target_v = np.array(target_v)
        target_pos_ = np.array(target_pos)

        start_pos = np.array([l1+l2+r4, width/2])
        distance = np.linalg.norm(target_pos - start_pos)

        for i in range(100000):
            target_pos = target_pos - target_v * dt * (1 - f_)
            target_v = np.array([target_v_norm * np.cos(target_theta), target_v_norm * np.sin(target_theta)])

            w_p = target_v_norm / r_p
            target_theta = target_theta + w_p * (dt * f_) * target_w

            target_v_norm = np.linalg.norm(target_v) + miu * g * dt * (1 - f_)

            if np.linalg.norm(target_pos_ - target_pos) > distance:
                target_to_current = target_pos_ - target_pos
                target_to_start = target_pos_ - start_pos
                cos_ = np.dot(target_to_current, target_to_start)/(np.linalg.norm(target_to_current)*np.linalg.norm(target_to_start))
                sin_ = np.cross(target_to_current, target_to_start)/(np.linalg.norm(target_to_current)*np.linalg.norm(target_to_start))
                arctan2_ = np.arctan2(sin_, cos_)
                target_theta += arctan2_
                target_v_norm -= miu * g * dt * (1 - f_)
                target_v = np.array([target_v_norm * np.cos(target_theta), target_v_norm * np.sin(target_theta)])
                # print(arctan2_)
                return target_v
        return None
    
    def decode_action(self, action, debug=False):
        """ action (int) 解码, 并执行反解"""
        # 出手位置
        posx, posy = 0, 0.5 #与反解函数中的出手位置保持一致
        # 解码目标点位置, 左右旋信息
        target_x_ratio = (action % ACTION_GRIDS_X) / ACTION_GRIDS_X
        target_y_ratio = ((action // ACTION_GRIDS_X) % ACTION_GRIDS_Y) / ACTION_GRIDS_Y
        target_w = [-1, 1][(action // (ACTION_GRIDS_X * ACTION_GRIDS_Y)) % 2]
        # TODO: 放缩调整目标点位置
        target_x = target_x_ratio * (l3+r4+0.0) + (l1+l2+l3+l4) + 3.0
        target_y = target_y_ratio * (width * 0.7) + width * 0.15
        if debug:
            print(f"解析action结果: target_x_ratio={target_x_ratio:.3f}, target_y_ratio={target_y_ratio:.3f}, target_x={target_x:.3f}, target_y={target_y:.3f}, target_w={target_w:.1f}")
        # 反解出手速度
        vx, vy = self.inverse_time_integrate([target_x, target_y], target_w)
        return posx, posy, target_w, vx, vy, target_x, target_y
        
    
    # def step(self, action:int, inference:bool=False):
    #     """执行动作"""
    #     # if type(action) == torch.Tensor:
    #     #     action = action.item()
    #     # 动作解码并反解
    #     posx, posy, target_w, vx, vy, target_x, target_y = self.all_actions_dict[action]
    #     # posx, posy = 0.0, 0.5
    #     # target_x = float(input("输入target x:"))
    #     # target_y = float(input("输入target y:"))
    #     # target_w = float(input("输入target w:"))
    #     # vx, vy = self.inverse_time_integrate([target_x, target_y], target_w)
    #     # print(f"目标位置 {target_x:.3f}, {target_y:.3f}")
    #     # print(f"出手速度 {vx:.5f}, {vy:.5f}")
    #     # 执行动作
    #     real_action = [posx, posy, target_w, vx, vy]
    #     res, res_str = self.shot_vec(vec=real_action, player=self.curr_player)
    #     # 获取观测和done信号
    #     obs = self.observe(self.curr_player)
    #     done = self.done()
    #     # 计算 reward
    #     if done:
    #         # 游戏规则得分, 得分赋予给后手 Player.B
    #         rew = self.score_zero_sum()[Player.B.value] * 10
    #         if rew > 0:
    #             result = "player 2 wins"
    #         elif rew < 0:
    #             result = "player 1 wins"
    #         else:
    #             result = "Tie"
    #     else:
    #         rew, result = 0.0, None
    #
    #     # 其他奖励得分
    #     # p_idx = 1 - int(self.curr_player == self.first_player)
    #     # dis = np.linalg.norm(self.stones_pos[p_idx::2] - center_pos, axis=1)
    #     # rew += (2.5 - 1.01 * dis[dis <= r4 + radius]).sum()
    #
    #     # 五壶保护扣分项
    #     if res_str == 'FIVE_STONE_PROTECT':
    #         rew -= 2
    #     self._elapsed_steps += 1
    #     if inference:
    #         res, res_str, inference_pos = self.vir_shot_vec(vec=real_action, player=self.curr_player)
    #         return obs, rew, done, {"result": result}, inference_pos
    #     # print(f"真实坐标: {self.stones_pos[self.curr_shot-1]}")
    #     return obs, rew, done, {"result": result}

    def step(self, action:list, inference:bool=False):
        """执行动作"""
        # if type(action) == torch.Tensor:
        #     action = action.item()
        # 动作解码并反解

        obs1 = self.observe(self.curr_player)

        posx, posy = 0.0, 0.5

        if action[0] > 0:
            target_w = 1
        else:
            target_w = -1

        vx = action[1]

        vy = action[2]

        # 执行动作
        real_action = [posx, posy, target_w, vx, vy]
        res, res_str = self.shot_vec(vec=real_action, player=self.curr_player)
        # 获取观测和done信号
        obs2 = self.observe(self.curr_player)
        done = self.done()
        # 计算 reward
        if done:
            # 游戏规则得分, 得分赋予给后手 Player.B
            rew = self.score_zero_sum()[Player.B.value] * 10
            if rew > 0:
                result = "player 2 wins"
            elif rew < 0:
                result = "player 1 wins"
            else:
                result = "Tie"
        else:
            rew, result = 0.0, None

        # 其他奖励得分
        # p_idx = 1 - int(self.curr_player == self.first_player)
        # dis = np.linalg.norm(self.stones_pos[p_idx::2] - center_pos, axis=1)
        # rew += (2.5 - 1.01 * dis[dis <= r4 + radius]).sum()

        # 五壶保护扣分项
        if res_str == 'FIVE_STONE_PROTECT':
            rew -= 2
        self._elapsed_steps += 1

        obs = np.concatenate([obs1.flatten(),obs2.flatten()])

        if inference:
            res, res_str, inference_pos = self.vir_shot_vec(vec=real_action, player=self.curr_player)
            return obs, rew, done, {"result": result}, inference_pos
        # print(f"真实坐标: {self.stones_pos[self.curr_shot-1]}")
        return obs, rew, done, {"result": result}


    def reset(self, render: bool = None, reset_from: list = None):
        # 初始化仿真环境, 确定先后手
        CurlingSimuOneEndEnv.reset_simu(self,
                                        first_player=Player.A,
                                        render=render,
                                        bgr_img=self.initialize_img,
                                        reset_from=reset_from)
        # 获取 obs
        obs = self.observe(self.curr_player)
        self._elapsed_steps = 0
        return obs

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        return None

    def close(self):
        return None

class CurlingSingleAgentGymEnv_v1(CurlingTwoAgentGymEnv_v1):
    """单智能体环境, 对手为过去的自己, 针对先手方训练"""
    def __init__(self, render=RENDER, use_cuda=True):
        super().__init__(render)
        # 设置对手
        # self.oppo = 'NOT_SET_YET'
        self.device = 'cuda' if use_cuda else 'cpu'
    
    def reset(self, render: bool = RENDER, reset_from: list = None):
        # 设置对手
        # self.oppo = oppo
        return super().reset(render, reset_from)
    
    def step(self, action: int, inference: bool = False):
        ## 我方行动
        obs, rew_A, done, info = super().step(action, inference)
        ## 对手行动
        with torch.inference_mode():
            action, log_prob, action_probs = OPPONENT_PLAYER.get_action(torch.as_tensor(obs, device=self.device))
            action = action.detach().cpu().item()
        if DEBUG_MODE:
            print(f"后手行动: {action}")
            
        obs, rew_B, done, info = super().step(action, inference)
        ## 计算我方的奖励
        if done:
            reward = rew_B * -1.0
        else:
            reward = rew_A
        return obs, reward, done, info

class CurlingSingleAgentGymEnv_v2(CurlingTwoAgentGymEnv_v1):
    """单智能体环境, 对手为 rule-based agent, 针对先手方训练"""
    def __init__(self, render=RENDER, use_cuda=True):
        super().__init__(render)
        # 设置对手
        # self.oppo = 'NOT_SET_YET'
        self.device = 'cuda' if use_cuda else 'cpu'
        
        ## 先固定前几投的打法, 2表示先后手各有2各球固定打法
        self.froze_n_stones = 2  
        assert self.froze_n_stones < 5
        
        ## rule based agent
        self.agent_rule = Rule_based_strag()

    def reset(self, render: bool = RENDER, reset_from: list = None):
        # 设置对手
        # self.oppo = oppo
        _ = super().reset(render, reset_from)

        ## 固定前几投打法
        for i in range(self.froze_n_stones * 2):
            start_pos, start_v, target_w, target_pos, noise = self.agent_rule.strategy(self.stones_pos, self.curr_shot, self.curr_player)
            # print(f"出手位置:{start_pos}, 出手速度:{start_v}, 旋转: {target_w}, 目标位置: {target_pos}, 噪声: {noise}")
            real_action = [*start_pos, target_w, *start_v]
            res, res_str = self.shot_vec(vec=real_action, player=self.curr_player)
        
        return self.observe(self.curr_player)
    
    def step(self, action: int, inference: bool = False):
        ## 我方行动
        _obs, rew_A, _done, _info = super().step(action, inference)
        ## 对手行动
        # with torch.inference_mode():
        #     action, log_prob, action_probs = OPPONENT_PLAYER.get_action(torch.as_tensor(obs, device=self.device))
        #     action = action.detach().cpu().item()
        start_pos, start_v, target_w, target_pos, noise = self.agent_rule.strategy(self.stones_pos, self.curr_shot, self.curr_player)
        # print(f"出手位置:{start_pos}, 出手速度:{start_v}, 旋转: {target_w}, 目标位置: {target_pos}, 噪声: {noise}")
        real_action = [*start_pos, target_w, *start_v]
        res, res_str = self.shot_vec(vec=real_action, player=self.curr_player)
        # 判断是否结束
        obs = self.observe(self.curr_player)
        done = self.done()
        # 计算后手奖励
        rew_B = self.score_zero_sum()[Player.B.value] * 10 if done else 0.0
        # 五壶保护扣分项
        if res_str == 'FIVE_STONE_PROTECT':
            rew_B -= 2.0       
        ## 计算我方（先手）的奖励
        if done:
            reward = rew_B * -1.0
        else:
            reward = rew_A
        info = {}
        return obs, reward, done, info

if __name__ == '__main__':
    env = CurlingTwoAgentGymEnv_v1(render=False)
    env.reset(render=False)
    train_x = []
    train_y = []
    done = False
    for i in range(10):
        while not done:
            action = env.action_space.sample()
            train_y.append(np.array(action))
            obs, rew, done, info = env.step(action)
            train_x.append(np.array(obs))

        env.reset(render=False)
        done = False

    np.save("train_x.npy", np.array(train_x))
    np.save("train_y.npy", np.array(train_y))

    # agent_rule = Rule_based_strag()
    # for count in range(10):
    #     print(f"当前第 {count} 球")
    #     start_pos, start_v, target_w, target_pos, noise = agent_rule.strategy(env.stones_pos, env.curr_shot, env.curr_player)
    #     print(f"出手位置:{start_pos}, 出手速度:{start_v}, 旋转: {target_w}, 目标位置: {target_pos}, 噪声: {noise}")
    #     real_action = [*start_pos, target_w, *start_v]
    #     res, res_str = env.shot_vec(vec=real_action, player=env.curr_player)
    #     print(f"目标位置: {target_pos}")
    #     print(f"真实坐标: {env.stones_pos[env.curr_shot-1]}")
    #     pass
    
# if __name__ == '__main__+++++':
#     agent_rule = Rule_based_strag()
#
#     env = CurlingTwoAgentGymEnv_v1(render=True)
#     env.reset()
#
#     for count in range(16):
#         print(f"当前第 {count} 球")
#         start_pos, start_v, target_w, target_pos, noise = agent_rule.strategy(env.stones_pos, env.curr_shot, env.curr_player)
#         print(f"出手位置:{start_pos}, 出手速度:{start_v}, 旋转: {target_w}, 目标位置: {target_pos}, 噪声: {noise}")
#         real_action = [*start_pos, target_w, *start_v]
#         res, res_str = env.shot_vec(vec=real_action, player=env.curr_player)
#         print(f"目标位置: {target_pos}")
#         print(f"真实坐标: {env.stones_pos[env.curr_shot-1]}")
#         pass
    
    # 获取观测和done信号
    # done = env.done()
        

# if __name__ == '__main__++++':
#     pass
#     env1 = CurlingTwoAgentGymEnv_v1(render=True)
#     env2 = CurlingTwoAgentGymEnv_v1(render=True)
#     env1.reset(render=False)
#     env2.reset(render=True)
#
#     actions = [100,200,85,80,90,200,180,170,160,150,210,220]
#
#     for i in range(300):
#         action = actions[i]
#         print(f"执行动作: {action}")
#
#         t = time.time()
#         obs, reward, done, info = env1.step(action)
#         delta = (time.time() - t)
#         print(f"时间间隔 env1 无渲染: {delta:.3f}")
#
#         t = time.time()
#         obs, reward, done, info = env2.step(action)
#         delta = (time.time() - t)
#         print(f"时间间隔 env2 开渲染: {delta:.3f}")
#         print(obs.shape, reward, done, info)
