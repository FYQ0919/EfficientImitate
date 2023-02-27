import datetime
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import pylab as mpl
from .gym_env import CurlingTwoAgentGymEnv_v1
# import ray
# import datetime
#
# ray.init()

from multiprocessing import Pool
import pickle
def search_wrapper(args):
    times, current_state, final_state, id = args
    fitness_env = CurlingTwoAgentGymEnv_v1()
    us = USearch(times, fitness_env, current_state, final_state, id)
    us.search()

mpl.rcParams['font.sans-serif'] = ['SimHei']


class USearch:
    def __init__(self, times,fitness_env, current_state, final_state, id):
        # 初始化
        self.times = times  # 迭代的代数
        self.result = dict() # 记录搜索结果
        self.g_best = None
        self.fitness_env = fitness_env
        self.id = id
        first_coordinate_current = []
        first_coordinate_final = []
        second_coordinate_current = []
        second_coordinate_final = []
        for i in range(int(current_state[0])):
            if int(current_state[i * 3 + 3]) == 0:
                first_coordinate_current.append([float(current_state[i * 3 + 1]), float(current_state[i * 3 + 2])])
            else:
                second_coordinate_current.append([float(current_state[i * 3 + 1]), float(current_state[i * 3 + 2])])

        current_turn = current_state[-2]
        current_player = final_state[-1]

        for i in range(int(final_state[0])):
            if int(final_state[i * 3 + 3]) == 0:
                first_coordinate_final.append([float(final_state[i * 3 + 1]), float(final_state[i * 3 + 2])])
            else:
                second_coordinate_final.append([float(final_state[i * 3 + 1]), float(final_state[i * 3 + 2])])

        final_turn = final_state[-2]
        final_player = final_state[-2]

        self.current_state = [first_coordinate_current, second_coordinate_current, current_turn, current_player]
        self.final_state = [first_coordinate_final, second_coordinate_final, final_turn, final_player]
        self.fitness_env.reset()
        self.target_obs = self.fitness_env.reset(render=False, reset_from=self.final_state)

        # 初始化第0代初始全局最优解
        # temp = 1000000
        # for i in range(self.size):
        #     for j in range(self.dimension):
        #         self.x[i][j] = random.uniform(self.bound[0][j], self.bound[1][j])
        #         self.v[i][j] = random.uniform(self.v_low, self.v_high)
        #     self.p_best[i] = self.x[i]  # 储存最优的个体
        #     fit = self.fitness(self.p_best[i])
        #     # 做出修改
        #     if fit < temp:
        #         self.g_best = self.p_best[i]
        #         temp = fit

    def fitness(self, x):
        """
        个体适应值计算
        """
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        action = [x1, x2, x3]
        self.fitness_env.reset(render=False, reset_from=self.current_state)
        obs, rew, done, info = self.fitness_env.step(action)

        rmse = np.linalg.norm(obs[:32] - self.target_obs[:32])
        # print(rmse)

        return rmse

    def search(self):
        tmp = 10000
        for i in range(self.times):
            self.fitness_env.reset(reset_from=self.current_state)
            w = [-1, 1]
            vx = [2.1 + 0.03 * i for i in range(15)]
            vy_1 = [0.05 + 0.01 * i for i in range(10)]

            real_w = w[int(i // 150) % 2]
            if real_w < 0:
                real_vy = -(abs(vy_1[int(i) % 10]))
            elif real_w > 0:
                real_vy = (abs(vy_1[int(i) % 10]))

            real_vx = vx[int(i // 10) % 15]

            action = [real_w, real_vx, real_vy]

            current_obs = self.fitness_env.reset(render=False, reset_from=self.current_state)
            obs, rew, done, info = self.fitness_env.step(action)

            rmse = np.linalg.norm(obs[:32] - self.target_obs[:32])
            print(rmse)

            if tmp > rmse:
                self.g_best = action
                self.current_obs = current_obs
                tmp = rmse

        self.result["obs"] = self.current_obs
        self.result["act"] = self.g_best
        self.result["rmse"] = tmp

        with open(f'{self.id}_curling_traj.pkl', 'wb') as f:
            pickle.dump(self.result, f)

        # self.fitness_env.reset(render=True, reset_from=self.current_state)
        # action = self.g_best
        # self.fitness_env.step(action)
        #
        # self.fitness_env.reset()
        # self.fitness_env.reset(render=True, reset_from=self.final_state)
        # time.sleep(10)
        #
        # print(tmp, action)




if __name__ == '__main__':
    # times = 600
    # fitness_env = CurlingTwoAgentGymEnv_v1()
    #
    # current_state = [1, 37.002274, 2.0390556, 0, 0, 0]
    # final_state =[2, 37.00732, 2.0467455, 0, 39.84324, 1.8804194, 1, 0, 1]
    #
    # import datetime
    # start_time = datetime.datetime.now()
    #
    # us = USearch(times, fitness_env, current_state, final_state)
    # us.search()
    # end_time = datetime.datetime.now()
    #
    # print(f"used time: {(end_time - start_time).seconds}s")

    times = 600
    multi_num = 4
    fitness_envs = [CurlingTwoAgentGymEnv_v1() for i in range(multi_num)]
    current_state = [1, 37.002274, 2.0390556, 0, 0, 0]
    final_state = [2, 37.00732, 2.0467455, 0, 39.84324, 1.8804194, 1, 1]

    start_time = datetime.datetime.now()

    # 创建进程池
    pool = Pool(multi_num)

    # 映射多个进程的执行
    pool.map(search_wrapper, [(times, current_state, final_state, i) for i in range(multi_num)] )

    # 关闭进程池
    pool.close()
    pool.join()

    end_time = datetime.datetime.now()
    print("Total time taken: ", end_time - start_time)