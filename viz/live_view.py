# Copyright (c) 2025, Williams.Wang. All rights reserved. Use restricted under LICENSE terms.

"""
使用 Rich 库实时可视化 Q-table 和 V-table
包含GIF录制功能
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import os
from typing import Dict, Any

from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text

from envs.gridworld import GridWorld


def get_color_for_value(value: float, max_abs_val: float) -> str:
    """
    根据Q值或V值获取颜色
    
    Args:
        value: 当前值
        max_abs_val: 用于归一化的最大绝对值
        
    Returns:
        Rich库兼容的颜色字符串
    """
    if max_abs_val == 0 or abs(value) < 1e-6:
        return "grey50"
    
    # 将值标准化到 0-1 之间
    norm_val = min(abs(value) / max_abs_val, 1.0)
    
    if value > 0:
        if norm_val > 0.7:
            return "bright_green"
        elif norm_val > 0.4:
            return "green"
        else:
            return "dark_green"
    elif value < 0:
        if norm_val > 0.7:
            return "bright_red"
        elif norm_val > 0.4:
            return "red"
        else:
            return "dark_red"
    return "grey50"


def create_q_table_grid(q_table: np.ndarray, env: GridWorld) -> Panel:
    """
    创建Q-table的可视化网格
    
    Args:
        q_table: Q-table (num_states, num_actions)
        env: GridWorld环境实例
        
    Returns:
        包含Q-table的Rich Panel
    """
    layout = Layout()
    layout.split_row(
        Layout(name="left"),
        Layout(name="right")
    )
    
    action_names = ["↑ UP", "↓ DOWN", "← LEFT", "→ RIGHT"]
    max_abs_q = np.max(np.abs(q_table)) if q_table.size > 0 else 1.0
    
    tables = []
    for action in range(env.num_actions):
        table = Table(title=action_names[action], show_header=False, box=None, padding=(0, 1))
        
        # 添加列
        for _ in range(env.grid_size):
            table.add_column(justify="center")
        
        # 添加行
        for row in range(env.grid_size):
            row_data = []
            for col in range(env.grid_size):
                pos = (row, col)
                state_idx = env.get_state_index(pos)
                q_value = q_table[state_idx, action]
                
                if pos == env.goal_pos:
                    cell_text = Text("GOAL", style="bold yellow")
                elif pos in env.holes:
                    cell_text = Text("####", style="bright_black")
                else:
                    cell_text = Text(f"{q_value:5.2f}")
                    color = get_color_for_value(q_value, max_abs_q)
                    cell_text.stylize(color)
                
                row_data.append(cell_text)
            table.add_row(*row_data)
        
        tables.append(Panel(table, border_style="cyan"))
    
    # 布局：上左、下左、上右、下右
    layout["left"].split_column(tables[0], tables[2])  # UP, LEFT
    layout["right"].split_column(tables[1], tables[3])  # DOWN, RIGHT
    
    return Panel(layout, title="[bold cyan]Q-Table (State-Action Values)[/bold cyan]", border_style="cyan")


def create_v_table_and_policy_grid(v_table: np.ndarray, policy: np.ndarray, env: GridWorld) -> Panel:
    """
    创建V-table热力图和策略图
    
    Args:
        v_table: V-table (num_states,)
        policy: 策略数组 (num_states,)
        env: GridWorld环境实例
        
    Returns:
        包含V-table和策略的Rich Panel
    """
    layout = Layout()
    layout.split_row(Layout(name="v_table"), Layout(name="policy"))
    
    # V-Table
    v_table_viz = Table(title="V-Table", show_header=False, box=None, padding=(0, 1))
    max_abs_v = np.max(np.abs(v_table)) if v_table.size > 0 else 1.0
    
    for _ in range(env.grid_size):
        v_table_viz.add_column(justify="center")
    
    for row in range(env.grid_size):
        row_data = []
        for col in range(env.grid_size):
            pos = (row, col)
            state_idx = env.get_state_index(pos)
            v_value = v_table[state_idx]
            
            if pos == env.goal_pos:
                cell_text = Text("GOAL", style="bold yellow")
            elif pos in env.holes:
                cell_text = Text("####", style="bright_black")
            else:
                cell_text = Text(f"{v_value:5.2f}")
                color = get_color_for_value(v_value, max_abs_v)
                cell_text.stylize(color)
            
            row_data.append(cell_text)
        v_table_viz.add_row(*row_data)
    
    # Policy Grid
    policy_viz = Table(title="Policy", show_header=False, box=None, padding=(0, 2))
    action_symbols = ['↑', '↓', '←', '→']
    
    for _ in range(env.grid_size):
        policy_viz.add_column(justify="center")
    
    for row in range(env.grid_size):
        row_data = []
        for col in range(env.grid_size):
            pos = (row, col)
            if pos == env.goal_pos:
                cell_text = Text("G", style="bold yellow")
            elif pos == env.start_pos:
                cell_text = Text("S", style="bold cyan")
            elif pos in env.holes:
                cell_text = Text("#", style="bright_black")
            else:
                state_idx = env.get_state_index(pos)
                action = policy[state_idx]
                cell_text = Text(action_symbols[action], style="bold magenta")
            
            row_data.append(cell_text)
        policy_viz.add_row(*row_data)
    
    layout["v_table"].update(Panel(v_table_viz, border_style="yellow"))
    layout["policy"].update(Panel(policy_viz, border_style="magenta"))
    
    return Panel(layout, title="[bold green]State Values & Policy[/bold green]", border_style="green")


def create_info_panel(stats: Dict[str, Any]) -> Panel:
    """
    创建信息展示面板
    
    Args:
        stats: 包含统计数据的字典
        
    Returns:
        包含统计信息的Rich Panel
    """
    info_text = Text()
    info_text.append(f"Episode: ", style="bold white")
    info_text.append(f"{stats.get('episode', 0):>5d}", style="bold yellow")
    info_text.append(f" / {stats.get('max_episodes', 0)}\n")
    
    info_text.append(f"Epsilon: ", style="bold white")
    info_text.append(f"{stats.get('epsilon', 0):.4f}", style="bold cyan")
    info_text.append("\n")
    
    info_text.append(f"Total Reward: ", style="bold white")
    info_text.append(f"{stats.get('total_reward', 0):>8.2f}", style="bold green")
    info_text.append("\n")
    
    info_text.append(f"Avg Reward: ", style="bold white")
    info_text.append(f"{stats.get('avg_reward', 0):>8.2f}", style="bold green")
    info_text.append("\n")
    
    info_text.append(f"Max Q Change: ", style="bold white")
    info_text.append(f"{stats.get('max_q_change', 0):.6f}", style="bold red")
    info_text.append("\n")
    
    info_text.append(f"Steps: ", style="bold white")
    info_text.append(f"{stats.get('steps', 0):>5d}", style="bold blue")
    
    return Panel(info_text, title="[bold]Training Statistics[/bold]", border_style="blue")


def render_layout(q_table: np.ndarray, v_table: np.ndarray, policy: np.ndarray, 
                 env: GridWorld, stats: Dict[str, Any]) -> Layout:
    """
    构建完整的终端可视化布局
    
    Args:
        q_table: Q-table
        v_table: V-table  
        policy: 当前策略
        env: GridWorld环境实例
        stats: 包含统计数据的字典
        
    Returns:
        完整的Rich Layout对象
    """
    layout = Layout()
    
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1)
    )
    
    layout["main"].split_row(
        Layout(name="left_side", ratio=3),
        Layout(name="right_side", ratio=1)
    )
    
    layout["left_side"].split(
        Layout(name="q_table", ratio=2),
        Layout(name="v_policy", ratio=1)
    )
    
    # 设置各个部分的内容
    header_text = Text("🤖 GridWorld Q-Learning Real-time Visualization 🎯", 
                      justify="center", style="bold white on blue")
    layout["header"].update(Panel(header_text))
    
    layout["q_table"].update(create_q_table_grid(q_table, env))
    layout["v_policy"].update(create_v_table_and_policy_grid(v_table, policy, env))
    layout["right_side"].update(create_info_panel(stats))
    
    return layout


# ==================== GIF录制功能 ====================

def save_q_table_snapshot(q_table: np.ndarray, v_table: np.ndarray, policy: np.ndarray,
                         env: GridWorld, stats: Dict[str, Any], 
                         save_path: str) -> None:
    """
    保存Q-table和V-table的静态图像快照
    
    Args:
        q_table: Q-table
        v_table: V-table
        policy: 策略数组
        env: GridWorld环境
        stats: 统计信息
        save_path: 保存路径
    """
    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Episode {stats.get('episode', 0)} - Q-Learning Evolution", fontsize=16)
    
    # 颜色映射
    cmap_positive = LinearSegmentedColormap.from_list("positive", ["white", "green"])
    cmap_negative = LinearSegmentedColormap.from_list("negative", ["red", "white"])
    
    action_names = ["UP ↑", "DOWN ↓", "LEFT ←", "RIGHT →"]
    
    # 绘制四个Q-table子图
    max_abs_q = np.max(np.abs(q_table)) if q_table.size > 0 else 1.0
    
    for action in range(4):
        row = action // 2
        col = action % 2
        ax = axes[row, col]
        
        # 重塑Q值为网格形状
        q_grid = q_table[:, action].reshape(env.grid_size, env.grid_size)
        
        # 绘制热力图
        if max_abs_q > 0:
            im = ax.imshow(q_grid, cmap='RdYlGn', vmin=-max_abs_q, vmax=max_abs_q)
        else:
            im = ax.imshow(q_grid, cmap='RdYlGn')
        
        # 添加数值标注
        for i in range(env.grid_size):
            for j in range(env.grid_size):
                pos = (i, j)
                if pos == env.goal_pos:
                    text = "GOAL"
                elif pos in env.holes:
                    text = "####"
                else:
                    text = f"{q_grid[i, j]:.2f}"
                
                ax.text(j, i, text, ha="center", va="center", 
                       color="white" if abs(q_grid[i, j]) > max_abs_q * 0.5 else "black",
                       fontsize=8)
        
        ax.set_title(f"Q-Table {action_names[action]}")
        ax.set_xticks(range(env.grid_size))
        ax.set_yticks(range(env.grid_size))
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    # V-table热力图
    ax_v = axes[0, 2]
    v_grid = v_table.reshape(env.grid_size, env.grid_size)
    max_abs_v = np.max(np.abs(v_table)) if v_table.size > 0 else 1.0
    
    if max_abs_v > 0:
        im_v = ax_v.imshow(v_grid, cmap='RdYlGn', vmin=-max_abs_v, vmax=max_abs_v)
    else:
        im_v = ax_v.imshow(v_grid, cmap='RdYlGn')
    
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            pos = (i, j)
            if pos == env.goal_pos:
                text = "G"
                color = "gold"
            elif pos in env.holes:
                text = "#"
                color = "black"
            else:
                text = f"{v_grid[i, j]:.2f}"
                color = "white" if abs(v_grid[i, j]) > max_abs_v * 0.5 else "black"
            
            ax_v.text(j, i, text, ha="center", va="center", color=color, fontsize=8)
    
    ax_v.set_title("V-Table (State Values)")
    ax_v.set_xticks(range(env.grid_size))
    ax_v.set_yticks(range(env.grid_size))
    plt.colorbar(im_v, ax=ax_v, shrink=0.8)
    
    # 策略图
    ax_policy = axes[1, 2]
    policy_grid = np.zeros((env.grid_size, env.grid_size))
    action_symbols = ['↑', '↓', '←', '→']
    
    ax_policy.imshow(policy_grid, cmap='gray', alpha=0.3)
    
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            pos = (i, j)
            if pos == env.goal_pos:
                text = "G"
                color = "gold"
            elif pos == env.start_pos:
                text = "S"
                color = "cyan"
            elif pos in env.holes:
                text = "#"
                color = "red"
            else:
                state_idx = env.get_state_index(pos)
                action = policy[state_idx]
                text = action_symbols[action]
                color = "blue"
            
            ax_policy.text(j, i, text, ha="center", va="center", 
                          color=color, fontsize=12, fontweight='bold')
    
    ax_policy.set_title("Greedy Policy")
    ax_policy.set_xticks(range(env.grid_size))
    ax_policy.set_yticks(range(env.grid_size))
    
    # 添加统计信息
    stats_text = (
        f"Episode: {stats.get('episode', 0)}\n"
        f"Epsilon: {stats.get('epsilon', 0):.4f}\n"
        f"Total Reward: {stats.get('total_reward', 0):.2f}\n"
        f"Avg Reward: {stats.get('avg_reward', 0):.2f}\n"
        f"Max Q Change: {stats.get('max_q_change', 0):.6f}"
    )
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def create_gif_from_frames(frames_dir: str, output_path: str, fps: float = 2.0) -> None:
    """
    从保存的帧创建GIF动画
    
    Args:
        frames_dir: 帧图像目录
        output_path: 输出GIF路径
        fps: 帧率
    """
    import imageio
    import glob
    
    # 获取所有帧文件
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    
    if not frame_files:
        print("No frame files found!")
        return
    
    # 创建GIF
    with imageio.get_writer(output_path, mode='I', duration=1.0/fps) as writer:
        for frame_file in frame_files:
            image = imageio.imread(frame_file)
            writer.append_data(image)
    
    print(f"GIF created: {output_path}")
    print(f"Total frames: {len(frame_files)}")


# 清理帧文件的辅助函数
def cleanup_frames(frames_dir: str) -> None:
    """
    清理帧文件
    
    Args:
        frames_dir: 帧目录
    """
    import glob
    
    frame_files = glob.glob(os.path.join(frames_dir, "frame_*.png"))
    for file in frame_files:
        os.remove(file)
    
    print(f"Cleaned up {len(frame_files)} frame files.")
