import datetime
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import pylab as mpl
from .gym_env import CurlingTwoAgentGymEnv_v1
mpl.rcParams['font.sans-serif'] = ['SimHei']


class PSO:
    def __init__(self, dimension, times, size, low, up, v_low, v_high, fitness_env, current_state, final_state):
        # 初始化
        self.dimension = dimension  # 变量个数
        self.times = times  # 迭代的代数
        self.size = size  # 种群大小
        self.bound = []  # 变量的约束范围
        self.bound.append(low)
        self.bound.append(up)
        self.v_low = v_low
        self.v_high = v_high
        self.x = np.zeros((self.size, self.dimension))  # 所有粒子的位置
        self.v = np.zeros((self.size, self.dimension))  # 所有粒子的速度
        self.p_best = np.zeros((self.size, self.dimension))  # 每个粒子最优的位置
        self.g_best = np.zeros((1, self.dimension))[0]  # 全局最优的位置
        self.fitness_env = fitness_env
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
        temp = 1000000
        for i in range(self.size):
            for j in range(self.dimension):
                self.x[i][j] = random.uniform(self.bound[0][j], self.bound[1][j])
                self.v[i][j] = random.uniform(self.v_low, self.v_high)
            self.p_best[i] = self.x[i]  # 储存最优的个体
            fit = self.fitness(self.p_best[i])
            # 做出修改
            if fit < temp:
                self.g_best = self.p_best[i]
                temp = fit

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
        print(rmse)

        return rmse

    def update(self, size):
        c1 = 2.0  # 学习因子
        c2 = 2.0
        w = 0.8  # 自身权重因子
        for i in range(size):
            # 更新速度(核心公式)
            self.v[i] = w * self.v[i] + c1 * random.uniform(0, 1) * (
                    self.p_best[i] - self.x[i]) + c2 * random.uniform(0, 1) * (self.g_best - self.x[i])
            # 速度限制
            for j in range(self.dimension):
                if self.v[i][j] < self.v_low:
                    self.v[i][j] = self.v_low
                if self.v[i][j] > self.v_high:
                    self.v[i][j] = self.v_high

            # 更新位置
            self.x[i] = self.x[i] + self.v[i]
            # 位置限制
            for j in range(self.dimension):
                if self.x[i][j] < self.bound[0][j]:
                    self.x[i][j] = self.bound[0][j]
                if self.x[i][j] > self.bound[1][j]:
                    self.x[i][j] = self.bound[1][j]
            # 更新p_best和g_best
            if self.fitness(self.x[i]) < self.fitness(self.p_best[i]):
                self.p_best[i] = self.x[i]
            if self.fitness(self.x[i]) < self.fitness(self.g_best):
                self.g_best = self.x[i]

    def pso(self):
        best = []
        self.final_best = np.array([1, 2, 3])
        for gen in range(self.times):
            self.update(self.size)
            if self.fitness(self.g_best) < self.fitness(self.final_best):
                self.final_best = self.g_best.copy()
            print('当前最佳位置：{}'.format(self.final_best))
            temp = self.fitness(self.final_best)
            print('当前的最佳适应度：{}'.format(temp))
            best.append(temp)
            if temp < 0.1:
                break
        t = [i for i in range(self.times)]

        self.fitness_env.reset(render=True, reset_from=self.current_state)
        action = self.g_best
        self.fitness_env.step(action)

        plt.figure()
        plt.grid(axis='both')
        plt.plot(t, best, color='red', marker='.', ms=10)
        plt.rcParams['axes.unicode_minus'] = False
        plt.margins(0)
        plt.xlabel(u"迭代次数")  # X轴标签
        plt.ylabel(u"适应度")  # Y轴标签
        plt.title(u"迭代过程")  # 标题
        plt.show()


if __name__ == '__main__':
    times = 10
    size = 20
    dimension = 3
    v_low = -1
    v_high = 1
    fitness_env = CurlingTwoAgentGymEnv_v1()

    current_state = [1, 37.002274, 2.0390556, 0, 0, 0]
    final_state =[2, 37.00732, 2.0467455, 0, 39.84324, 1.8804194, 1, 1]

    import datetime
    start_time = datetime.datetime.now()
    low = [-1, 2.1, -0.2]
    up = [1, 2.6, 0.2]
    pso = PSO(dimension, times, size, low, up, v_low, v_high, fitness_env, current_state, final_state)
    pso.pso()
    end_time = datetime.datetime.now()

    print(f"used time: {(end_time - start_time).seconds}s")