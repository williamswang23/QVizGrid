# Copyright (c) 2025, Williams.Wang. All rights reserved. Use restricted under LICENSE terms.

"""
Q-Learning GridWorld 主训练脚本
实时可视化 Q-table 和 V-table 的演化过程
"""

import yaml
import numpy as np
from typing import Dict, List, Tuple, Any
import os
from collections import deque

from rich.live import Live
from rich.console import Console

from envs.gridworld import GridWorld
from agent.q_learning import QLearningAgent
from viz.live_view import render_layout, save_q_table_snapshot, create_gif_from_frames, cleanup_frames


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def check_convergence(agent: QLearningAgent, episode_rewards: deque, 
                     config: Dict[str, Any]) -> Tuple[bool, str]:
    """
    检查是否满足收敛条件
    
    Args:
        agent: Q-Learning智能体
        episode_rewards: 最近episodes的奖励历史
        config: 训练配置
        
    Returns:
        (是否收敛, 收敛原因)
    """
    # 检查Q值变化收敛
    max_q_change = agent.get_max_q_change()
    if max_q_change != float('inf') and max_q_change < config['convergence_tol']:
        return True, f"Q-value convergence: max change {max_q_change:.6f} < {config['convergence_tol']}"
    
    # 检查平均回报稳定（需要足够的历史数据）
    if len(episode_rewards) >= 100:
        recent_rewards = list(episode_rewards)
        recent_mean = np.mean(recent_rewards[-50:])
        earlier_mean = np.mean(recent_rewards[-100:-50])
        
        if abs(recent_mean - earlier_mean) < 1e-3:
            return True, f"Reward stability: recent={recent_mean:.4f}, earlier={earlier_mean:.4f}"
    
    return False, ""


def run_episode(env: GridWorld, agent: QLearningAgent) -> Tuple[float, int]:
    """
    运行一个完整的episode
    
    Args:
        env: GridWorld环境
        agent: Q-Learning智能体
        
    Returns:
        (总奖励, 步数)
    """
    # 重置环境
    state_pos = env.reset()
    state_index = env.get_state_index(state_pos)
    
    total_reward = 0.0
    steps = 0
    max_steps = 200  # 防止无限循环
    
    while steps < max_steps:
        # 选择动作
        action = agent.choose_action(state_index)
        
        # 执行动作
        next_state_pos, reward, done, info = env.step(action)
        next_state_index = env.get_state_index(next_state_pos)
        
        # 更新Q-table
        q_change = agent.update(state_index, action, reward, next_state_index, done)
        
        # 累计奖励和步数
        total_reward += reward
        steps += 1
        
        # 更新状态
        state_index = next_state_index
        
        # 检查是否结束
        if done:
            break
    
    return total_reward, steps


def print_episode_info(episode: int, total_reward: float, steps: int, 
                      agent: QLearningAgent, episode_rewards: deque) -> None:
    """
    打印episode信息
    
    Args:
        episode: episode编号
        total_reward: 总奖励
        steps: 步数
        agent: 智能体
        episode_rewards: 奖励历史
    """
    if len(episode_rewards) >= 100:
        avg_reward = np.mean(list(episode_rewards)[-100:])
        print(f"Episode {episode:4d} | Reward: {total_reward:6.2f} | Steps: {steps:3d} | "
              f"Avg100: {avg_reward:6.2f} | ε: {agent.epsilon:.4f}")
    else:
        print(f"Episode {episode:4d} | Reward: {total_reward:6.2f} | Steps: {steps:3d} | "
              f"ε: {agent.epsilon:.4f}")


def main():
    """
    主训练函数
    """
    console = Console()
    console.print("🚀 Starting Q-Learning GridWorld Training...", style="bold green")
    
    # 加载配置
    config = load_config()
    console.print("📋 Configuration loaded from config.yaml")
    
    # 创建环境
    env = GridWorld(config['env'])
    console.print(f"🌍 GridWorld environment created: {env.grid_size}x{env.grid_size} grid")
    console.print(f"   Start: {env.start_pos}, Goal: {env.goal_pos}")
    console.print(f"   Obstacles: {env.holes}")
    
    # 创建智能体
    agent = QLearningAgent(config['agent'], env.num_states, env.num_actions)
    console.print("🤖 Q-Learning agent created")
    console.print(f"   Learning rate: {agent.alpha}")
    console.print(f"   Discount factor: {agent.gamma}")
    console.print(f"   Epsilon: {agent.epsilon_start} → {agent.epsilon_end}")
    
    # 训练配置
    train_config = config['train']
    max_episodes = train_config['max_episodes']
    render_interval = train_config.get('render_interval', 10)
    use_visualization = train_config.get('use_visualization', True)
    
    # GIF录制配置
    snapshot_interval = train_config.get('snapshot_interval', 100)
    create_gif = train_config.get('create_gif', True)
    frames_dir = "outputs/frames"
    gif_output_path = "outputs/q_evolution.gif"
    
    # 训练统计
    episode_rewards = deque(maxlen=1000)
    convergence_patience = train_config.get('convergence_patience', 20)
    convergence_count = 0
    
    console.print(f"\n🎯 Training Configuration:")
    console.print(f"   Max episodes: {max_episodes}")
    console.print(f"   Render interval: {render_interval}")
    console.print(f"   Use visualization: {use_visualization}")
    console.print(f"   Snapshot interval: {snapshot_interval}")
    console.print(f"   Create GIF: {create_gif}")
    console.print(f"   Convergence tolerance: {train_config['convergence_tol']}")
    console.print(f"   Convergence patience: {convergence_patience}")
    
    # 清理旧的帧文件
    if create_gif:
        cleanup_frames(frames_dir)
    
    if use_visualization:
        console.print("\n🎬 Starting live visualization...", style="bold cyan")
        
        # 初始化可视化
        v_table = agent.get_v_table()
        policy = agent.get_policy()
        initial_stats = {
            'episode': 0,
            'max_episodes': max_episodes,
            'epsilon': agent.epsilon,
            'total_reward': 0.0,
            'avg_reward': 0.0,
            'max_q_change': 0.0,
            'steps': 0
        }
        
        # 使用Rich Live进行实时可视化
        with Live(render_layout(agent.q_table, v_table, policy, env, initial_stats), 
                  refresh_per_second=2, console=console) as live:
            
            # 主训练循环
            for episode in range(1, max_episodes + 1):
                # 保存Q-table快照用于收敛检查
                if episode > 1:
                    agent.save_q_table_snapshot()
                
                # 运行一个episode
                total_reward, steps = run_episode(env, agent)
                episode_rewards.append(total_reward)
                
                # 衰减epsilon
                agent.decay_epsilon()
                
                # 更新可视化
                if episode % render_interval == 0 or episode <= 10:
                    v_table = agent.get_v_table()
                    policy = agent.get_policy()
                    max_q_change = agent.get_max_q_change()
                    avg_reward = np.mean(list(episode_rewards)[-100:]) if len(episode_rewards) >= 100 else np.mean(list(episode_rewards))
                    
                    stats = {
                        'episode': episode,
                        'max_episodes': max_episodes,
                        'epsilon': agent.epsilon,
                        'total_reward': total_reward,
                        'avg_reward': avg_reward,
                        'max_q_change': max_q_change,
                        'steps': steps
                    }
                    
                    live.update(render_layout(agent.q_table, v_table, policy, env, stats))
                
                # 保存GIF帧
                if create_gif and episode % snapshot_interval == 0:
                    v_table = agent.get_v_table()
                    policy = agent.get_policy()
                    max_q_change = agent.get_max_q_change()
                    avg_reward = np.mean(list(episode_rewards)[-100:]) if len(episode_rewards) >= 100 else np.mean(list(episode_rewards))
                    
                    stats = {
                        'episode': episode,
                        'max_episodes': max_episodes,
                        'epsilon': agent.epsilon,
                        'total_reward': total_reward,
                        'avg_reward': avg_reward,
                        'max_q_change': max_q_change,
                        'steps': steps
                    }
                    
                    frame_path = os.path.join(frames_dir, f"frame_{episode:04d}.png")
                    save_q_table_snapshot(agent.q_table, v_table, policy, env, stats, frame_path)
                
                # 检查收敛条件
                if episode > 50:
                    converged, reason = check_convergence(agent, episode_rewards, train_config)
                    if converged:
                        convergence_count += 1
                        if convergence_count >= convergence_patience:
                            console.print(f"\n🎉 Training converged after {episode} episodes!", style="bold green")
                            console.print(f"   Reason: {reason}")
                            break
                    else:
                        convergence_count = 0
    else:
        console.print("\n📊 Training without visualization...")
        console.print("-" * 80)
        
        # 主训练循环（无可视化）
        for episode in range(1, max_episodes + 1):
            # 保存Q-table快照用于收敛检查
            if episode > 1:
                agent.save_q_table_snapshot()
            
            # 运行一个episode
            total_reward, steps = run_episode(env, agent)
            episode_rewards.append(total_reward)
            
            # 衰减epsilon
            agent.decay_epsilon()
            
            # 打印训练信息
            if episode % render_interval == 0 or episode <= 10:
                print_episode_info(episode, total_reward, steps, agent, episode_rewards)
            
            # 保存GIF帧（无可视化模式）
            if create_gif and episode % snapshot_interval == 0:
                v_table = agent.get_v_table()
                policy = agent.get_policy()
                max_q_change = agent.get_max_q_change()
                avg_reward = np.mean(list(episode_rewards)[-100:]) if len(episode_rewards) >= 100 else np.mean(list(episode_rewards))
                
                stats = {
                    'episode': episode,
                    'max_episodes': max_episodes,
                    'epsilon': agent.epsilon,
                    'total_reward': total_reward,
                    'avg_reward': avg_reward,
                    'max_q_change': max_q_change,
                    'steps': steps
                }
                
                frame_path = os.path.join(frames_dir, f"frame_{episode:04d}.png")
                save_q_table_snapshot(agent.q_table, v_table, policy, env, stats, frame_path)
            
            # 检查收敛条件
            if episode > 50:
                converged, reason = check_convergence(agent, episode_rewards, train_config)
                if converged:
                    convergence_count += 1
                    if convergence_count >= convergence_patience:
                        console.print(f"\n🎉 Training converged after {episode} episodes!", style="bold green")
                        console.print(f"   Reason: {reason}")
                        break
                else:
                    convergence_count = 0
    
    # 训练完成统计
    console.print(f"\n✅ Training completed!")
    console.print(f"   Total episodes: {episode}")
    console.print(f"   Final epsilon: {agent.epsilon:.6f}")
    
    if len(episode_rewards) >= 100:
        final_avg = np.mean(list(episode_rewards)[-100:])
        console.print(f"   Final 100-episode average reward: {final_avg:.4f}")
    
    # 显示最终策略
    console.print(f"\n🗺️  Final Policy (Best Actions):")
    policy = agent.get_policy()
    action_symbols = ['↑', '↓', '←', '→']
    
    for row in range(env.grid_size):
        row_str = ""
        for col in range(env.grid_size):
            pos = (row, col)
            if pos == env.start_pos:
                row_str += " S "
            elif pos == env.goal_pos:
                row_str += " G "
            elif pos in env.holes:
                row_str += " # "
            else:
                state_idx = env.get_state_index(pos)
                action = policy[state_idx]
                row_str += f" {action_symbols[action]} "
        console.print(row_str)
    
    # 创建GIF动画
    if create_gif:
        console.print(f"\n🎬 Creating GIF animation...")
        create_gif_from_frames(frames_dir, gif_output_path, fps=config.get('viz', {}).get('fps', 2))
        console.print(f"   GIF saved to: {gif_output_path}")
    
    console.print(f"\n🏁 Training session finished.", style="bold green")


if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs("outputs/frames", exist_ok=True)
    
    main()
