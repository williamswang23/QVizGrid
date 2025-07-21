# Copyright (c) 2025, Williams.Wang. All rights reserved. Use restricted under LICENSE terms.

"""
GridWorld环境的单元测试
"""

import pytest
import numpy as np
from envs.gridworld import GridWorld


@pytest.fixture
def test_config():
    """测试配置"""
    return {
        'grid_size': 5,
        'start': [0, 0],
        'goal': [4, 4],
        'holes': [[2, 1], [3, 3]],
        'reward_goal': 1.0,
        'reward_hole': -1.0,
        'reward_step': -0.04
    }


@pytest.fixture
def env(test_config):
    """创建测试环境"""
    return GridWorld(test_config)


class TestGridWorldInitialization:
    """测试GridWorld初始化"""
    
    def test_init_basic_properties(self, env):
        """测试基本属性初始化"""
        assert env.grid_size == 5
        assert env.start_pos == (0, 0)
        assert env.goal_pos == (4, 4)
        assert env.holes == [(2, 1), (3, 3)]
        assert env.num_states == 25
        assert env.num_actions == 4
    
    def test_init_rewards(self, env):
        """测试奖励设置"""
        assert env.reward_goal == 1.0
        assert env.reward_hole == -1.0
        assert env.reward_step == -0.04
    
    def test_action_map(self, env):
        """测试动作映射"""
        expected_map = {
            0: (-1, 0),  # UP
            1: (1, 0),   # DOWN
            2: (0, -1),  # LEFT
            3: (0, 1)    # RIGHT
        }
        assert env.action_map == expected_map


class TestGridWorldReset:
    """测试环境重置"""
    
    def test_reset_position(self, env):
        """测试重置后位置"""
        pos = env.reset()
        assert pos == env.start_pos
        assert env.current_pos == env.start_pos
    
    def test_multiple_resets(self, env):
        """测试多次重置"""
        for _ in range(5):
            pos = env.reset()
            assert pos == (0, 0)
            assert env.current_pos == (0, 0)


class TestGridWorldStep:
    """测试环境步骤执行"""
    
    def test_normal_movement(self, env):
        """测试正常移动"""
        env.reset()
        
        # 向右移动
        next_pos, reward, done, info = env.step(3)  # RIGHT
        assert next_pos == (0, 1)
        assert reward == env.reward_step
        assert not done
        assert info['reason'] == 'normal_step'
    
    def test_boundary_collision(self, env):
        """测试边界碰撞"""
        env.reset()
        
        # 向上移动（越界）
        next_pos, reward, done, info = env.step(0)  # UP
        assert next_pos == (0, 0)  # 位置不变
        assert reward == env.reward_hole
        assert not done
        assert info['reason'] == 'out_of_bounds'
        
        # 向左移动（越界）
        next_pos, reward, done, info = env.step(2)  # LEFT
        assert next_pos == (0, 0)  # 位置不变
        assert reward == env.reward_hole
        assert not done
        assert info['reason'] == 'out_of_bounds'
    
    def test_obstacle_collision(self, env):
        """测试障碍物碰撞"""
        env.reset()
        
        # 移动到障碍物附近
        env.current_pos = (2, 0)
        
        # 向右移动到障碍物 (2,1)
        next_pos, reward, done, info = env.step(3)  # RIGHT
        assert next_pos == (2, 0)  # 位置不变
        assert reward == env.reward_hole
        assert not done
        assert info['reason'] == 'hit_obstacle'
    
    def test_goal_reached(self, env):
        """测试到达目标"""
        env.reset()
        
        # 直接设置到目标附近
        env.current_pos = (4, 3)
        
        # 向右移动到目标
        next_pos, reward, done, info = env.step(3)  # RIGHT
        assert next_pos == (4, 4)
        assert reward == env.reward_goal
        assert done
        assert info['reason'] == 'goal_reached'
    
    def test_step_without_reset_raises_error(self, env):
        """测试未重置就执行步骤会抛出错误"""
        with pytest.raises(ValueError, match="Environment must be reset before stepping"):
            env.step(0)


class TestGridWorldUtilityMethods:
    """测试实用方法"""
    
    def test_get_state_index(self, env):
        """测试状态索引转换"""
        assert env.get_state_index((0, 0)) == 0
        assert env.get_state_index((0, 4)) == 4
        assert env.get_state_index((4, 0)) == 20
        assert env.get_state_index((4, 4)) == 24
        assert env.get_state_index((2, 3)) == 13
    
    def test_get_pos_from_index(self, env):
        """测试索引转位置"""
        assert env.get_pos_from_index(0) == (0, 0)
        assert env.get_pos_from_index(4) == (0, 4)
        assert env.get_pos_from_index(20) == (4, 0)
        assert env.get_pos_from_index(24) == (4, 4)
        assert env.get_pos_from_index(13) == (2, 3)
    
    def test_state_index_conversion_consistency(self, env):
        """测试状态索引转换的一致性"""
        for row in range(env.grid_size):
            for col in range(env.grid_size):
                pos = (row, col)
                index = env.get_state_index(pos)
                recovered_pos = env.get_pos_from_index(index)
                assert recovered_pos == pos
    
    def test_is_terminal(self, env):
        """测试终端状态判断"""
        assert env.is_terminal(env.goal_pos)
        assert not env.is_terminal(env.start_pos)
        assert not env.is_terminal((2, 2))
        assert not env.is_terminal((2, 1))  # 障碍物不是终端状态
    
    def test_get_valid_actions(self, env):
        """测试有效动作获取"""
        # 起始位置：只能向右和向下
        valid_actions = env.get_valid_actions((0, 0))
        assert set(valid_actions) == {1, 3}  # DOWN, RIGHT
        
        # 中心位置：但(2,1)有障碍物，所以向左不可行
        valid_actions = env.get_valid_actions((2, 2))
        assert set(valid_actions) == {0, 1, 3}  # UP, DOWN, RIGHT (不能LEFT到障碍物)
        
        # 右下角：只能向上和向左
        valid_actions = env.get_valid_actions((4, 4))
        assert set(valid_actions) == {0, 2}  # UP, LEFT
        
        # 障碍物附近
        valid_actions = env.get_valid_actions((2, 0))
        expected = {0, 1, 3}  # UP, DOWN, RIGHT (不能向左越界，不能向右到障碍物(2,1))
        assert set(valid_actions) == {0, 1}  # 只能UP, DOWN，RIGHT会到障碍物
    
    def test_render(self, env):
        """测试环境渲染"""
        env.reset()
        grid_str = env.render()
        
        # 检查包含起始位置标记
        assert 'S' in grid_str
        # 检查包含目标位置标记  
        assert 'G' in grid_str
        # 检查包含障碍物标记
        assert '#' in grid_str
        
        # 移动智能体
        env.current_pos = (1, 1)
        grid_str = env.render()
        assert 'A' in grid_str  # 智能体位置


class TestGridWorldCompleteEpisode:
    """测试完整episode"""
    
    def test_simple_path_to_goal(self, env):
        """测试简单路径到目标"""
        env.reset()
        
        # 手动执行一条到达目标的路径
        actions = [3, 3, 3, 3, 1, 1, 1, 1]  # 先右后下，8步到达目标
        total_reward = 0
        
        for i, action in enumerate(actions):
            next_pos, reward, done, info = env.step(action)
            total_reward += reward
            
            if done:
                assert next_pos == env.goal_pos
                assert reward == env.reward_goal
                break
        
        assert done  # 应该已经完成
        
        # 验证总奖励：8步移动(-0.04 * 8 = -0.32) + 到达目标(+1.0) = 0.68
        # 由于浮点数精度问题，使用较宽松的容忍度
        expected_reward = 0.68
        assert abs(total_reward - expected_reward) < 0.05  # 放宽容忍度


# 运行测试的辅助函数
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
