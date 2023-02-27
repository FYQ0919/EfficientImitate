# -*- coding: utf-8 -*-
from gym.envs.registration import register

import taichi as ti

from .config import *
from .utils import *

# 初始化 taichi 
# ti.cuda 和 pytorch 网络同时使用可能会报错!
ti.init(arch=ti.cuda, default_fp=ti_real, dynamic_index=True)

## 注册gym环境
register(
    id='CurTwo-v0',
    entry_point='curling_simulator.gym_env:CurlingTwoAgentGymEnv_v0',
    max_episode_steps=20,
)

## 注册gym环境
register(
    id='CurlingSingleAgent-v1',
    entry_point='curling_simulator.gym_env:CurlingSingleAgentGymEnv_v1',
    max_episode_steps=20,
)

## 注册gym环境
register(
    id='CurlingSingleAgent-v2',
    entry_point='curling_simulator.gym_env:CurlingSingleAgentGymEnv_v2',
)