# Copyright (c) 2025, Williams.Wang. All rights reserved. Use restricted under LICENSE terms.

"""
GridWorld 环境实现
一个 5x5 的网格世界，包含起点、终点和障碍物
"""

from typing import Tuple, List, Optional, Dict, Any
import numpy as np


class GridWorld:
    """
    5x5 GridWorld 环境
    
    状态表示：(row, col) 坐标
    动作空间：0=上, 1=下, 2=左, 3=右
    """
    
    # 动作定义
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化 GridWorld 环境
        
        Args:
            config: 配置字典，包含网格大小、起点、终点、障碍物等信息
        """
        self.grid_size = config['grid_size']
        self.start_pos = tuple(config['start'])
        self.goal_pos = tuple(config['goal'])
        self.holes = [tuple(hole) for hole in config['holes']]
        
        # 奖励设置
        self.reward_goal = config['reward_goal']
        self.reward_hole = config['reward_hole']
        self.reward_step = config['reward_step']
        
        # 动作映射
        self.action_map = {
            self.UP: (-1, 0),
            self.DOWN: (1, 0),
            self.LEFT: (0, -1),
            self.RIGHT: (0, 1)
        }
        
        # 当前状态
        self.current_pos = None
        
        # 状态和动作空间
        self.num_states = self.grid_size * self.grid_size
        self.num_actions = 4
        
    def reset(self) -> Tuple[int, int]:
        """
        重置环境到初始状态
        
        Returns:
            初始状态 (row, col)
        """
        self.current_pos = self.start_pos
        return self.current_pos
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict]:
        """
        执行动作，返回下一状态、奖励、是否结束、额外信息
        
        Args:
            action: 动作编号 (0=上, 1=下, 2=左, 3=右)
            
        Returns:
            next_state: 下一个状态 (row, col)
            reward: 奖励值
            done: 是否结束
            info: 额外信息字典
        """
        if self.current_pos is None:
            raise ValueError("Environment must be reset before stepping")
        
        # 计算下一个位置
        dr, dc = self.action_map[action]
        next_row = self.current_pos[0] + dr
        next_col = self.current_pos[1] + dc
        next_pos = (next_row, next_col)
        
        # 检查边界
        if (next_row < 0 or next_row >= self.grid_size or 
            next_col < 0 or next_col >= self.grid_size):
            # 越界，保持原位置，给予负奖励
            reward = self.reward_hole
            done = False
            info = {'reason': 'out_of_bounds'}
            return self.current_pos, reward, done, info
        
        # 检查是否碰到障碍物
        if next_pos in self.holes:
            # 碰到障碍物，保持原位置，给予负奖励
            reward = self.reward_hole
            done = False
            info = {'reason': 'hit_obstacle'}
            return self.current_pos, reward, done, info
        
        # 更新位置
        self.current_pos = next_pos
        
        # 检查是否到达终点
        if self.current_pos == self.goal_pos:
            reward = self.reward_goal
            done = True
            info = {'reason': 'goal_reached'}
        else:
            reward = self.reward_step
            done = False
            info = {'reason': 'normal_step'}
        
        return self.current_pos, reward, done, info
    
    def get_state_index(self, pos: Tuple[int, int]) -> int:
        """
        将坐标转换为状态索引
        
        Args:
            pos: (row, col) 坐标
            
        Returns:
            状态索引 (0 到 num_states-1)
        """
        row, col = pos
        return row * self.grid_size + col
    
    def get_pos_from_index(self, state_index: int) -> Tuple[int, int]:
        """
        将状态索引转换为坐标
        
        Args:
            state_index: 状态索引
            
        Returns:
            (row, col) 坐标
        """
        row = state_index // self.grid_size
        col = state_index % self.grid_size
        return (row, col)
    
    def is_terminal(self, pos: Tuple[int, int]) -> bool:
        """
        检查给定位置是否为终端状态
        
        Args:
            pos: (row, col) 坐标
            
        Returns:
            是否为终端状态
        """
        return pos == self.goal_pos
    
    def get_valid_actions(self, pos: Tuple[int, int]) -> List[int]:
        """
        获取给定位置的有效动作列表
        
        Args:
            pos: (row, col) 坐标
            
        Returns:
            有效动作列表
        """
        valid_actions = []
        row, col = pos
        
        for action in range(self.num_actions):
            dr, dc = self.action_map[action]
            next_row, next_col = row + dr, col + dc
            next_pos = (next_row, next_col)
            
            # 检查是否在边界内且不是障碍物
            if (0 <= next_row < self.grid_size and 
                0 <= next_col < self.grid_size and 
                next_pos not in self.holes):
                valid_actions.append(action)
        
        return valid_actions if valid_actions else list(range(self.num_actions))
    
    def render(self) -> str:
        """
        渲染当前环境状态为字符串
        
        Returns:
            环境的字符串表示
        """
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # 标记起点
        grid[self.start_pos[0]][self.start_pos[1]] = 'S'
        
        # 标记终点
        grid[self.goal_pos[0]][self.goal_pos[1]] = 'G'
        
        # 标记障碍物
        for hole in self.holes:
            grid[hole[0]][hole[1]] = '#'
        
        # 标记当前位置
        if self.current_pos and self.current_pos != self.start_pos and self.current_pos != self.goal_pos:
            grid[self.current_pos[0]][self.current_pos[1]] = 'A'
        
        # 转换为字符串
        result = []
        for row in grid:
            result.append(' '.join(row))
        
        return '\n'.join(result)
