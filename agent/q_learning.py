# Copyright (c) 2025, Williams.Wang. All rights reserved. Use restricted under LICENSE terms.

"""
Q-Learning 智能体实现
使用表格式 Q-learning 算法进行强化学习
"""

from typing import Tuple, Dict, Any
import numpy as np
import random


class QLearningAgent:
    """
    Q-Learning 智能体
    
    使用 ε-greedy 策略进行探索与利用的平衡
    """
    
    def __init__(self, config: Dict[str, Any], num_states: int, num_actions: int):
        """
        初始化 Q-Learning 智能体
        
        Args:
            config: 配置字典，包含学习率、折扣因子、探索参数等
            num_states: 状态空间大小
            num_actions: 动作空间大小
        """
        self.num_states = num_states
        self.num_actions = num_actions
        
        # 学习参数
        self.alpha = config['alpha']  # 学习率
        self.gamma = config['gamma']  # 折扣因子
        
        # 探索参数
        self.epsilon = config['epsilon_start']  # 当前探索率
        self.epsilon_start = config['epsilon_start']  # 初始探索率
        self.epsilon_end = config['epsilon_end']  # 最小探索率
        self.epsilon_decay = config['epsilon_decay']  # 探索率衰减因子
        
        # 初始化 Q-table
        self.q_table = np.zeros((num_states, num_actions))
        
        # 用于收敛判断的变量
        self.last_q_table = None
        
    def choose_action(self, state_index: int) -> int:
        """
        根据 ε-greedy 策略选择动作
        
        Args:
            state_index: 当前状态索引
            
        Returns:
            选择的动作
        """
        if random.random() < self.epsilon:
            # 探索：随机选择动作
            return random.randint(0, self.num_actions - 1)
        else:
            # 利用：选择 Q 值最大的动作
            return np.argmax(self.q_table[state_index])
    
    def update(self, state_index: int, action: int, reward: float, 
               next_state_index: int, done: bool) -> float:
        """
        根据 Q-learning 规则更新 Q-table
        
        Args:
            state_index: 当前状态索引
            action: 执行的动作
            reward: 获得的奖励
            next_state_index: 下一状态索引
            done: 是否为终端状态
            
        Returns:
            Q值的变化量（用于收敛判断）
        """
        # 保存旧的Q值
        old_q_value = self.q_table[state_index, action]
        
        if done:
            # 终端状态，没有未来奖励
            target = reward
        else:
            # 计算目标Q值：r + γ * max(Q(s', a'))
            target = reward + self.gamma * np.max(self.q_table[next_state_index])
        
        # Q-learning 更新规则：Q(s,a) ← Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        self.q_table[state_index, action] += self.alpha * (target - old_q_value)
        
        # 返回Q值变化量
        return abs(self.q_table[state_index, action] - old_q_value)
        
    def decay_epsilon(self) -> None:
        """
        衰减探索率 epsilon
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def get_max_q_change(self) -> float:
        """
        计算相对于上一次的最大Q值变化
        
        Returns:
            最大Q值变化量，如果没有历史记录则返回float('inf')
        """
        if self.last_q_table is None:
            return float('inf')
        
        diff = np.abs(self.q_table - self.last_q_table)
        return np.max(diff)
    
    def save_q_table_snapshot(self) -> None:
        """
        保存当前Q-table的快照，用于下次比较
        """
        self.last_q_table = self.q_table.copy()
    
    def get_v_table(self) -> np.ndarray:
        """
        从Q-table计算V-table（状态价值函数）
        
        Returns:
            V-table，形状为 (num_states,)
        """
        return np.max(self.q_table, axis=1)
    
    def get_policy(self) -> np.ndarray:
        """
        从Q-table提取贪心策略
        
        Returns:
            策略数组，形状为 (num_states,)，每个元素是该状态下的最优动作
        """
        return np.argmax(self.q_table, axis=1)
    
    def get_action_probabilities(self, state_index: int) -> np.ndarray:
        """
        获取在给定状态下的动作概率分布（基于当前ε-greedy策略）
        
        Args:
            state_index: 状态索引
            
        Returns:
            动作概率分布，形状为 (num_actions,)
        """
        probs = np.ones(self.num_actions) * (self.epsilon / self.num_actions)
        best_action = np.argmax(self.q_table[state_index])
        probs[best_action] += (1 - self.epsilon)
        return probs
    
    def reset_exploration(self) -> None:
        """
        重置探索率到初始值
        """
        self.epsilon = self.epsilon_start
    
    def set_epsilon(self, epsilon: float) -> None:
        """
        手动设置探索率
        
        Args:
            epsilon: 新的探索率值
        """
        self.epsilon = max(0.0, min(1.0, epsilon))  # 确保在[0,1]范围内
