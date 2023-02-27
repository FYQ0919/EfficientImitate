# -*- coding: utf-8 -*-
import os, sys, time, pdb
import numpy as np
import math
# ti.init(arch=ti.vulkan)
# sys.path.insert(0,'D:\交大文件\AlphaGo+curling\model-based-rl-CNN_sample_muzero\curling_ui\gui_demo')
# from display import *
import random
# import torch

from enum import Enum
from .config import *  # 重要的参数
from .utils import *

# self.target_w = 1
# self.hit_speed = 2
# self.target_theta = -np.pi/36

class Rule_based_strag(object):
    def __init__(self):
        self.target_pos = [0, 0]
        self.shot_para = None
        # self.player = player
        self.corner_guard = None
        self.curr_player = None
        self.curr_shot = None
        self.stones_pos = np.empty((n_curling_stones, 2), dtype=np.float32)

        self.target_w = 1
        self.hit_speed = 2
        self.target_theta = -np.pi/36

    def strategy(self, stones_pos, curr_shot, curr_player):
        self.stones_pos = stones_pos
        self.curr_shot = curr_shot
        _stone_pos_1 = self.stones_pos[1]

        if curr_player == Player.A:
            if curr_shot == 2: #悬壶到中心点前
                self.target_pos = [l1+l2+l3+l4+l3-0.5, width/2+radius]
                self.shot_para = self.get_draw_para(self.target_pos, self.target_w, target_v_norm = 0., target_theta = self.target_theta)
            elif curr_shot == 4: 
                if self.get_distance(self.stones_pos[3], self.stones_pos[2]) <= radius*4: #粘糖葫芦
                    self.target_pos = [self.stones_pos[3][0]-radius*1, self.stones_pos[3][1]]
                    self.shot_para = self.get_draw_para(self.target_pos, self.target_w, target_v_norm = 0., target_theta = self.target_theta)
                elif self.in_defensive_zone(self.stones_pos[3]): #打飞control zone
                    if self.stones_pos[3][1]>=width/2:
                        self.target_pos = [self.stones_pos[3][0], self.stones_pos[3][1]-radius*2/3]
                        self.shot_para = self.get_draw_para(self.target_pos, self.target_w, target_v_norm = 0.6, target_theta = self.target_theta)
                    else:
                        self.target_pos = [self.stones_pos[3][0], self.stones_pos[3][1]+radius*2/3]
                        self.shot_para = self.get_draw_para(self.target_pos, -self.target_w, target_v_norm = 0.6, target_theta = -self.target_theta)
                    
                else:
                    self.target_pos = [l1+l2+l3+l4+l3-r2, width/2+radius/3] #大本营内阻挡
                    self.shot_para = self.get_draw_para(self.target_pos, self.target_w, target_v_norm = 0., target_theta = self.target_theta)

            elif curr_shot == 6 or curr_shot == 8:
                lolipop = 0
                loli_pos = []
                for i in range(self.curr_shot):
                    if self.get_distance(self.stones_pos[3], self.stones_pos[i]) <= radius*3 and self.stones_pos[i][0]<self.stones_pos[3][0]:
                        lolipop = 1
                        loli_pos = self.stones_pos[i]
                        break
                
                if lolipop == 1:
                    self.target_pos = loli_pos
                    self.shot_para = self.get_draw_para(self.target_pos, self.target_w, target_v_norm = 0.0, target_theta = self.target_theta)

                else:
                    hit_flag, hit_pos = self.get_hit_pos()
                    
                    if hit_flag: #打飞中间1/3的壶
                        hit_pos.sort(key=lambda x:x[0])
                        self.target_pos = hit_pos[0]
                        if self.target_pos[1]>=width/2:
                            self.target_pos[1] -= radius*2/3
                            self.shot_para = self.get_draw_para(self.target_pos, self.target_w, target_v_norm = 1.0, target_theta = self.target_theta)
                        else:
                            self.target_pos[1] += radius*2/3
                            self.shot_para = self.get_draw_para(self.target_pos, -self.target_w, target_v_norm = 1.0, target_theta = -self.target_theta)
                    else: #悬壶到中心点前
                        self.target_pos = [l1+l2+l3+l4+l3-r2, width/2]
                        self.shot_para = self.get_draw_para(self.target_pos, self.target_w, target_v_norm = 0., target_theta = self.target_theta)
                
            elif curr_shot == 10:
                draw_flag = 0
                draw_pos = [np.inf, np.inf]

                for i in range(curr_shot):
                    if self.in_inner_circle(self.stones_pos[i]):
                        draw_flag = 1
                        if self.stones_pos[i][0]<draw_pos[0]:
                            draw_pos = self.stones_pos[i]
                if draw_flag: #悬壶到对方壶前
                    self.target_pos = draw_pos
                    if self.target_pos[1]>=width/2:
                        self.shot_para = self.get_draw_para(self.target_pos, self.target_w, target_v_norm = 0., target_theta = self.target_theta)
                    else:
                        self.shot_para = self.get_draw_para(self.target_pos, -self.target_w, target_v_norm = 0., target_theta = -self.target_theta)
                else:
                    self.target_pos = [l1+l2+l3+l4+l3, width/2]
                    self.shot_para = self.get_draw_para(self.target_pos, self.target_w, target_v_norm = 0., target_theta = self.target_theta)

        else:
            if curr_shot == 3: #悬壶在中点前
                self.target_pos = [l1+l2+l3+l4+l3-1.0, width/2+radius/2]
                self.shot_para = self.get_draw_para(self.target_pos, self.target_w, target_v_norm = 0., target_theta = self.target_theta)

            if curr_shot == 5: #corner guard
                corner_guard = [[l1+l2+l3+l4+2*l3/3, width/2+r4*3/4], [l1+l2+l3+l4+2*l3/3, width/2-r4*3/4]]
                hit_pos = []
                draw_pos = []

                for i in range(self.curr_shot):
                    if self.stones_pos[i][0]<l1+l2+l3+l4+l3-r4 and (abs(self.stones_pos[i][1]-corner_guard[0][1]) < 4*radius or \
                        abs(self.stones_pos[i][1]-corner_guard[1][1]) < 4*radius):
                        if (self.curr_shot-i)%2==0:
                            draw_pos.append(self.stones_pos[i])
                        elif (self.curr_shot-i)%2==1:
                            hit_pos.append(self.stones_pos[i])
                
                if len(hit_pos)>0:
                    if hit_pos[0][1] > width/2:
                        self.target_pos = [hit_pos[0][0], hit_pos[0][1]-radius/2]
                        self.shot_para = self.get_draw_para(self.target_pos, self.target_w, target_v_norm = 1.3, target_theta = self.target_theta)
                    elif hit_pos[0][1] < width/2:
                        self.target_pos = [hit_pos[0][0], hit_pos[0][1]+radius/2]
                        self.shot_para = self.get_draw_para(self.target_pos, -self.target_w, target_v_norm = 1.3, target_theta = -self.target_theta)
                
                elif len(draw_pos)>0:
                    self.corner_guard = draw_pos[0]
                    if draw_pos[0][1] > width/2:
                        self.target_pos = [l1+l2+l3+l4+l3-radius*2, draw_pos[0][1]-radius/2]
                        self.shot_para = self.get_draw_para(self.target_pos, -self.target_w, target_v_norm = 0., target_theta = -self.target_theta)
                    elif hit_pos[0][1] < width/2:
                        self.target_pos = [l1+l2+l3+l4+l3-radius*2, draw_pos[0][1]+radius/2]
                        self.shot_para = self.get_draw_para(self.target_pos, self.target_w, target_v_norm = 0., target_theta = self.target_theta)

                else:
                    self.corner_guard = corner_guard[0]
                    self.shot_para = self.get_draw_para(self.corner_guard, self.target_w, target_v_norm = 0., target_theta = self.target_theta)

            if curr_shot == 7 or curr_shot == 9:
                # count = 0
                # target_pos = []
                # for i in range(curr_shot):
                #     if i%2==0 and self.in_defensive_zone(self.stones_pos[i]): #计数control zone里的壶
                #         count+=1
                #         target_pos.append(self.stones_pos[i])
                hit_flag, hit_pos = self.get_hit_pos()
                
                if len(hit_pos) >= 2 or (len(hit_pos)>0 and curr_shot==9): #清除一部分对方壶
                    hit_pos.sort(key=lambda x:x[0])
                    target = hit_pos[0]
                    if self.corner_guard is not None:
                        if self.corner_guard[1]>width/2:
                            target[1] += radius/2
                            self.shot_para = self.get_draw_para(target, self.target_w, target_v_norm = self.hit_speed, target_theta = self.target_theta)
                        else:
                            target[1] -= radius/2
                            self.shot_para = self.get_draw_para(target, -self.target_w, target_v_norm = self.hit_speed, target_theta = -self.target_theta)
                    else:
                        target[1] += radius/2
                        self.shot_para = self.get_draw_para(target, self.target_w, target_v_norm = self.hit_speed, target_theta = self.target_theta)

                elif self.corner_guard is not None: #悬壶到corner guard后
                    already_draw = 0
                    for i in range(curr_shot):
                        if (curr_shot-i)%2==0 and abs(self.corner_guard[1]-self.stones_pos[i][1])<radius*3 and self.corner_guard[0]<self.stones_pos[i][0]\
                             and self.stones_pos[i][0]<l1+l2+l3+l4+l3:
                            already_draw = 1
                            break
                    # print("already_draw: ", already_draw)
                    if already_draw==0:
                        if self.corner_guard[1]>width/2:
                            self.target_pos = [l1+l2+l3+l4+l3-radius, self.corner_guard[1]-radius/2]
                            self.shot_para = self.get_draw_para(self.target_pos, -self.target_w, target_v_norm = 0., target_theta = -self.target_theta)
                        elif self.corner_guard[1]<width/2:
                            self.target_pos = [l1+l2+l3+l4+l3-radius, self.corner_guard[1]+radius/2]
                            self.shot_para = self.get_draw_para(self.target_pos, self.target_w, target_v_norm = 0., target_theta = self.target_theta)
                    else:
                        self.target_pos = [l1+l2+l3+l4+l3-radius*2, width-self.corner_guard[1]]
                        if self.target_pos[1]>width/2:
                            self.target_pos[1] -=radius/2
                            self.shot_para = self.get_draw_para(self.target_pos, -self.target_w, target_v_norm = 0., target_theta = -self.target_theta)
                        else:
                            self.target_pos[1] +=radius/2
                            self.shot_para = self.get_draw_para(self.target_pos, self.target_w, target_v_norm = 0., target_theta = self.target_theta)

                elif len(hit_pos)==1:
                    target = hit_pos[0]
                    if target[1]>=width/2:
                        target[1] -= radius*2/3
                        self.shot_para = self.get_draw_para(target, self.target_w, target_v_norm = self.hit_speed/2, target_theta = self.target_theta)
                    else:
                        target[1] += radius*2/3
                        self.shot_para = self.get_draw_para(target, -self.target_w, target_v_norm = self.hit_speed/2, target_theta = -self.target_theta)
                else:
                    target = [l1+l2+l3+l4+l3-r2, width-radius]


            if curr_shot == 11:
                hit_flag = 0
                hit_pos = []
                available_traj = self.detect_center_guard()

                for i in range(curr_shot):
                    if i%2==0 and self.in_inner_circle(self.stones_pos[i]):
                        hit_flag = 1
                        hit_pos.append(self.stones_pos[i])
                if hit_flag:
                    
                    if "ABOVE" in available_traj:
                        hit_pos.sort(key=lambda x:x[1], reverse=True)
                        self.target_pos = hit_pos[0]
                        self.shot_para = self.get_draw_para(self.target_pos, self.target_w, target_v_norm = self.hit_speed, target_theta = self.target_theta)
                    elif "BELOW" in available_traj:
                        hit_pos.sort(key=lambda x:x[1])
                        self.target_pos = hit_pos[0]
                        self.shot_para = self.get_draw_para(self.target_pos, -self.target_w, target_v_norm = self.hit_speed, target_theta = -self.target_theta)
                    else:
                        hit_pos.sort(key=lambda x:x[1], reverse=True)
                        self.target_pos = hit_pos[0]
                        self.shot_para = self.get_draw_para(self.target_pos, self.target_w, target_v_norm = self.hit_speed, target_theta = self.target_theta)
                else:
                    self.target_pos = [l1+l2+l3+l4+l3, width/2]
                    if "ABOVE" in available_traj:
                        self.shot_para = self.get_draw_para(self.target_pos, self.target_w, target_v_norm = 0, target_theta = self.target_theta)
                    elif "BELOW" in available_traj:
                        self.shot_para = self.get_draw_para(self.target_pos, -self.target_w, target_v_norm = 0, target_theta = -self.target_theta)
                    else:
                        self.shot_para = self.get_draw_para(self.target_pos, self.target_w, target_v_norm = 0, target_theta = self.target_theta)

        return self.shot_para

    def in_defensive_zone(self, stone_pos):
        width_range = [width/3, 2*width/3]
        length_range = [l1+l2+l3+l4+l3-r4, l1+l2+l3+l4+l3]

        if stone_pos[0] >= length_range[0] and stone_pos[0] <= length_range[1] \
            and stone_pos[1] >= width_range[0] and stone_pos[1] <= width_range[1]:
            return True
        else:
            return False

    def in_inner_circle(self, stone_pos):
        distance = self.get_distance(stone_pos, center_pos)
        if distance <= r2:
            return True
        else:
            return False
    
    def get_distance(self, point1, point2): #计算两点间距离
        point1 = np.array(point1)
        point2 = np.array(point2)
        return np.linalg.norm(point1-point2)
    
    def get_draw_para(self, target_pos, target_w, target_v_norm = 0., target_theta = 0):
        answer = inverse_draw(target_pos, target_w, target_v_norm, target_theta)

        if(answer!=None):
            start_pos, target_v, target_w, pos_traj, noise = answer 
            start_pos = (start_pos[0] - (l1+l2+r4), start_pos[1] / width)
            for i in range(len(pos_traj)):
                pos_traj[i] = [(pos_traj[i][0]*scale+1)/plane_length, (1 - ((width - pos_traj[i][1])*scale+1) / gui_width)]

            # answer = (start_pos, target_v, target_w, pos_traj, noise) # 出手位置, 出手速度, 旋转, 轨迹 GUI, 噪声 (击打点已经加过噪声)
            return start_pos, target_v, target_w, target_pos, noise
        else:
            return

    def detect_center_guard(self):
        # res = ["CENTER", "ABOVE", "BELOW"]
        res = ["ABOVE", "BELOW"]

        width_range = [width/3, 2*width/3]
        center_width_range = [width/2-radius/2, width/2+radius/2]
        length_range = [l1+l2+l3+l4+l3/3, l1+l2+l3+l4+l3-r4]
        for i in range(self.curr_shot):
            if self.stones_pos[i][0]>=length_range[0] and self.stones_pos[i][0]<=length_range[1] \
                and self.stones_pos[i][1]>=width_range[0] and self.stones_pos[i][1]<=width_range[1]: #in center guard range

                # if self.stones_pos[i][1]>=center_width_range[0] and self.stones_pos[i][1]<=center_width_range[1]:
                #     res.pop("CENTER")
                if self.stones_pos[i][1]>=center_width_range[1] and self.stones_pos[i][1]<=width_range[1]:
                    if "ABOVE" in res:
                        res.remove("ABOVE")
                if self.stones_pos[i][1]>=width_range[0] and self.stones_pos[i][1]<=center_width_range[0]:
                    if "BELOW" in res:
                        res.remove("BELOW")

        return res

    def get_hit_pos(self):
        hit_flag = 0
        hit_pos_dic = {"ABOVE":[], "CENTER":[], "BELOW":[]}
        available_traj = self.detect_center_guard()
        hit_pos = []

        for i in range(self.curr_shot):
            if (self.curr_shot-i)%2==1 and self.in_defensive_zone(self.stones_pos[i]):
                if self.stones_pos[i][1] > width/2:
                    hit_pos_dic["ABOVE"].append(self.stones_pos[i])
                elif self.stones_pos[i][1] < width/2:
                    hit_pos_dic["BELOW"].append(self.stones_pos[i])    
                else:
                    hit_pos_dic["CENTER"].append(self.stones_pos[i])  

        if len(available_traj) > 0:
            for choice in available_traj:
                hit_pos += hit_pos_dic[choice]
        
        if len(hit_pos) > 0:
            hit_flag = 1
        
        # print(hit_flag, hit_pos)
        return (hit_flag, hit_pos)


def inverse_draw(target_pos, target_w, target_v_norm = 0., target_theta = 0):
    r_p = 0.117
    f_avg = 0.00037 # 0.0001
    # f_avg = 0.001
    # miu_avg = 2.315162/26.0297/g/(1-f_avg)
    miu_avg = 1/125

    def pos_noise():
        mu = 0
        sigma = radius*2/4 #误差1/4个壶
        noise = [0, 0]

        noise[0] = random.gauss(mu, sigma)
        noise[1] = random.gauss(mu, sigma)

        return np.array(noise)

    # rendering_init()
    noise = np.array([0, 0])
    noise = pos_noise()

    target_pos = np.array(target_pos) + noise #添加噪声
    target_pos = np.array(target_pos)
    target_v = np.array([target_v_norm*np.cos(target_theta), target_v_norm*np.sin(target_theta)])

    target_pos_ = np.array(target_pos)
    target_v_ = np.array(target_v)
    target_v_norm_ = target_v_norm   
    target_theta_ = target_theta 

    start_pos = np.array([l1+l2+l3, width/2])
    distance = np.linalg.norm(target_pos - start_pos)
    record_pos = []

    for i in range(100000):
        target_pos = target_pos - target_v * dt * (1 - f_avg)
        target_v = np.array([target_v_norm * np.cos(target_theta), target_v_norm * np.sin(target_theta)])

        w_p = target_v_norm / r_p
        target_theta = target_theta + w_p * (dt * f_avg) * target_w #altered -

        target_v_norm = np.linalg.norm(target_v) + miu_avg * g * dt * (1 - f_avg)
        # pos_record.append([target_pos[0], target_pos[1]])
        if i % 48 == 0:
            record_pos.append([target_pos[0], target_pos[1]])
            # ax.scatter(target_pos[0], target_pos[1], s=1, color='#808A87', zorder=3)
            # plt.pause(0.00000000001)

        # if start_pos[0] - target_pos[0] > 0:
        #     if 0.8*width > target_pos[1] > 0.2*width:
        #         record_pos = np.array(record_pos)
        #         # ax.scatter(record_pos[:,0], record_pos[:,1], s=1, color='#808A87', zorder=3)
        #         return target_pos, target_v, target_w
        #     else:
        #         print("invalid target pos")
        #         # raise ValueError
        #         return

        if start_pos[0] - target_pos[0] > 0:
            # if target_pos[1] >= 0.7*width or target_w >= 0:
            if target_w >= 0:
                theta = clockwise_angle(target_pos_-target_pos, target_pos_-[start_pos[0], 0.8*width])
                target_theta = target_theta_ + theta
                # if target_pos_[1] < 0.6*width:#0.75
                #     angle1 = np.arctan((target_pos[1]-target_pos_[1])/(target_pos_[0]-start_pos[0]))
                #     angle2 = np.arctan((width*0.6-target_pos_[1])/(target_pos_[0]-start_pos[0]))
                #     target_theta = target_theta + angle1 - angle2
                # else:
                #     angle1 = np.arctan((target_pos[1]-target_pos_[1])/(target_pos_[0]-start_pos[0]))
                #     angle2 = np.arctan((target_pos_[1]-width*0.6)/(target_pos_[0]-start_pos[0]))
                #     target_theta = target_theta + angle2 + angle1

            # elif target_pos[1] <= 0.3*width or target_w < 0:
            else:
                theta = clockwise_angle(target_pos_-target_pos, target_pos_-[start_pos[0], 0.2*width])
                target_theta = target_theta_ + theta
                # if target_pos_[1] > 0.4*width:#0.25
                #     angle1 = np.arctan((-target_pos[1]+target_pos_[1])/(target_pos_[0]-start_pos[0]))
                #     angle2 = np.arctan((-width*0.4+target_pos_[1])/(target_pos_[0]-start_pos[0]))
                #     target_theta = target_theta + angle2 - angle1
                # else:
                #     angle1 = np.arctan((-target_pos[1]+target_pos_[1])/(target_pos_[0]-start_pos[0]))
                #     angle2 = np.arctan((width*0.4-target_pos_[1])/(target_pos_[0]-start_pos[0]))
                #     target_theta = target_theta - angle1 - angle2
            # else:
            #     print("SUCESS")
            #     if target_w >= 0 :
            #         target_pos[1]
            #     record_pos.append([target_pos[0], target_pos[1]])
            #     return target_pos, target_v, target_w, record_pos 
            break

    target_pos = target_pos_
    target_v_norm = target_v_norm_
    target_v = np.array([target_v_norm_*np.cos(target_theta), target_v_norm_*np.sin(target_theta)])
    record_pos = []
    for i in range(100000):
        target_pos = target_pos - target_v * dt * (1 - f_avg)
        target_v = np.array([target_v_norm * np.cos(target_theta), target_v_norm * np.sin(target_theta)])

        w_p = target_v_norm / r_p
        target_theta = target_theta + w_p * (dt * f_avg) * target_w #altered -

        target_v_norm = np.linalg.norm(target_v) + miu_avg * g * dt * (1 - f_avg)
        # pos_record.append([target_pos[0], target_pos[1]])
        if i % 48 == 0:
            record_pos.append([target_pos[0], target_pos[1]])
            # ax.scatter(target_pos[0], target_pos[1], s=1, color='#808A87', zorder=3)
            # plt.pause(0.00000000001)

        if start_pos[0] - target_pos[0] > 0:
            if 0.85*width > target_pos[1] > 0.15*width:
                # print("SUCESS")
                record_pos.append([target_pos[0], target_pos[1]])
                return target_pos, target_v, target_w, record_pos, noise
            else:
                return

def clockwise_angle(v1, v2):
    x1,y1 = v1
    x2,y2 = v2
    dot = x1*x2+y1*y2
    det = x1*y2-y1*x2
    theta = np.arctan2(det, dot)
    theta = theta if theta>0 else 2*np.pi+theta
    return theta

if __name__ == '__main__':
    stra = Rule_based_strag()
