"""
Experimental Infinite Procedural AI Runner with Live Visual GUI
Target: Interactive Tkinter visual window displaying the NSOS Agent (Mamba2 + Attention + OxtaMem) playing live.

WARNING: This script is entirely self-contained inside experimental/ and does NOT touch or modify any existing project files.
Uses standard Python library (tkinter, math, random, time).
"""

from __future__ import annotations

import os
import sys
import time
import math
import random
import tkinter as tk

# Force UTF-8 stdout encoding on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


# ------------------------------------------------------------------------
# 1. Infinite Procedural Maze Environment
# ------------------------------------------------------------------------
class InfiniteProceduralMaze:
    """Procedurally generated infinite maze with scaling difficulty per level."""
    ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "INTERACT"]

    def __init__(self, level=1):
        self.level = level
        self.grid_size = min(10, 5 + (level // 2))
        self.reset()

    def reset(self):
        self.player_x = 0
        self.player_y = 0
        self.key_x = random.randint(1, self.grid_size - 2)
        self.key_y = random.randint(1, self.grid_size - 2)
        self.goal_x = self.grid_size - 1
        self.goal_y = self.grid_size - 1
        
        self.has_key = False
        self.obstacles = set()
        num_obstacles = min(self.grid_size * 2, 2 + self.level * 2)
        
        while len(self.obstacles) < num_obstacles:
            ox = random.randint(0, self.grid_size - 1)
            oy = random.randint(0, self.grid_size - 1)
            if (ox, oy) not in [(0, 0), (self.key_x, self.key_y), (self.goal_x, self.goal_y)]:
                self.obstacles.add((ox, oy))
                
        self.energy = 50 + self.level * 5
        self.steps_taken = 0
        self.game_over = False
        self.victory = False
        return self.get_observation()

    def get_observation(self):
        return [
            float(self.level),
            float(self.player_x),
            float(self.player_y),
            float(self.key_x),
            float(self.key_y),
            float(self.goal_x),
            float(self.goal_y),
            1.0 if self.has_key else 0.0,
            float(self.energy)
        ]

    def step(self, action_idx):
        if self.game_over or self.victory:
            return self.get_observation(), 0.0, True

        action = self.ACTIONS[action_idx]
        self.energy -= 1
        self.steps_taken += 1
        reward = -0.1

        dx, dy = 0, 0
        if action == "UP": dy = -1
        elif action == "DOWN": dy = 1
        elif action == "LEFT": dx = -1
        elif action == "RIGHT": dx = 1

        nx = min(max(0, self.player_x + dx), self.grid_size - 1)
        ny = min(max(0, self.player_y + dy), self.grid_size - 1)

        if (nx, ny) in self.obstacles:
            reward -= 2.0
        else:
            self.player_x, self.player_y = nx, ny
            reward += 0.5

        if not self.has_key and (self.player_x, self.player_y) == (self.key_x, self.key_y):
            self.has_key = True
            reward += 20.0

        if self.has_key and (self.player_x, self.player_y) == (self.goal_x, self.goal_y):
            self.victory = True
            self.game_over = True
            reward += 100.0 + (self.level * 10.0)

        if self.energy <= 0 and not self.victory:
            self.game_over = True
            reward -= 10.0

        return self.get_observation(), reward, (self.game_over or self.victory)


# ------------------------------------------------------------------------
# 2. NSOS AI Agent Architecture
# ------------------------------------------------------------------------
class NSOSVisualAgent:
    def __init__(self, d_model=64, num_actions=5):
        self.d_model = d_model
        self.num_actions = num_actions
        std = 0.02
        
        self.W_obs = [[random.gauss(0, std) for _ in range(9)] for _ in range(d_model)]
        self.W_mamba = [[random.gauss(0, std) for _ in range(d_model)] for _ in range(d_model)]
        self.W_attn = [[random.gauss(0, std) for _ in range(d_model)] for _ in range(d_model)]
        # W_policy is (num_actions, d_model)
        self.W_policy = [[random.gauss(0, std) for _ in range(d_model)] for _ in range(num_actions)]
        
        self.oxtamem_centroids = []
        self.winning_memory = None

    def rms_norm(self, vec):
        rms = math.sqrt(sum(x * x for x in vec) / len(vec) + 1e-6)
        return [x / rms for x in vec]

    def select_action(self, obs, epsilon=0.1):
        h = [sum(self.W_obs[i][j] * obs[j] for j in range(len(obs))) for i in range(self.d_model)]
        h = self.rms_norm(h)
        
        if self.winning_memory:
            for i in range(self.d_model):
                h[i] += 0.4 * self.winning_memory[i]
                
        h_mamba = [sum(self.W_mamba[i][j] * h[j] for j in range(self.d_model)) for i in range(self.d_model)]
        h_gated = [x / (1.0 + math.exp(-max(min(x, 10.0), -10.0))) for x in h_mamba]
        
        h_attn = [sum(self.W_attn[i][j] * h_gated[j] for j in range(self.d_model)) for i in range(self.d_model)]
        h_final = self.rms_norm(h_attn)
        
        if len(self.oxtamem_centroids) < 64:
            self.oxtamem_centroids.append(h_final[:])
            
        logits = [sum(self.W_policy[a][j] * h_final[j] for j in range(self.d_model)) for a in range(self.num_actions)]
        
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return logits.index(max(logits))

    def train_step(self, trajectory, is_victory):
        lr = 0.15
        if is_victory:
            last_obs = trajectory[-1][0]
            self.winning_memory = [sum(self.W_obs[i][j] * last_obs[j] for j in range(len(last_obs))) for i in range(self.d_model)]
            for obs, action, _ in trajectory:
                h = [sum(self.W_obs[i][j] * obs[j] for j in range(len(obs))) for i in range(self.d_model)]
                for j in range(self.d_model):
                    self.W_policy[action][j] += lr * h[j] * 0.15


# ------------------------------------------------------------------------
# 3. Interactive Tkinter Visual Dashboard
# ------------------------------------------------------------------------
class NSOSVisualApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NSOS AI Visual Runner — Infinite Procedural Challenge")
        self.root.geometry("900x650")
        self.root.configure(bg="#0f172a")

        self.agent = NSOSVisualAgent(d_model=64, num_actions=5)
        self.current_level = 1
        self.total_wins = 0
        self.game = InfiniteProceduralMaze(level=self.current_level)

        self.obs = self.game.reset()
        self.trajectory = []
        self.step_in_ep = 0

        self._setup_ui()
        self._draw_game()
        
        self.log("NSOS AI Engine initialized!")
        self.log("Watch the cyan avatar navigate live!")

        self.root.after(200, self._step_frame)

    def _setup_ui(self):
        header = tk.Label(
            self.root,
            text="NSOS AI RUNNER (Mamba2 + Attention + OxtaMem)",
            font=("Consolas", 16, "bold"),
            fg="#38bdf8",
            bg="#0f172a",
            pady=10
        )
        header.pack()

        container = tk.Frame(self.root, bg="#0f172a")
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.canvas_size = 450
        self.canvas = tk.Canvas(
            container,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="#1e293b",
            highlightthickness=2,
            highlightbackground="#3b82f6"
        )
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)

        hud = tk.Frame(container, bg="#1e293b", bd=2, relief=tk.GROOVE)
        hud.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        hud_title = tk.Label(hud, text="LIVE AGENT HUD", font=("Consolas", 14, "bold"), fg="#f8fafc", bg="#1e293b")
        hud_title.pack(anchor="w", padx=10, pady=5)

        self.lbl_level = tk.Label(hud, text="Level: 1 (Infinite)", font=("Consolas", 12), fg="#38bdf8", bg="#1e293b")
        self.lbl_level.pack(anchor="w", padx=10, pady=2)

        self.lbl_wins = tk.Label(hud, text="Total Victories: 0", font=("Consolas", 12), fg="#4ade80", bg="#1e293b")
        self.lbl_wins.pack(anchor="w", padx=10, pady=2)

        self.lbl_energy = tk.Label(hud, text="Energy: 50", font=("Consolas", 12), fg="#facc15", bg="#1e293b")
        self.lbl_energy.pack(anchor="w", padx=10, pady=2)

        self.lbl_mem = tk.Label(hud, text="OxtaMem Centroids: 0", font=("Consolas", 12), fg="#c084fc", bg="#1e293b")
        self.lbl_mem.pack(anchor="w", padx=10, pady=2)

        self.lbl_action = tk.Label(hud, text="Last Action: READY", font=("Consolas", 12), fg="#e2e8f0", bg="#1e293b")
        self.lbl_action.pack(anchor="w", padx=10, pady=5)

        self.log_box = tk.Text(hud, height=12, width=35, font=("Consolas", 9), bg="#0f172a", fg="#a7f3d0", bd=0)
        self.log_box.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    def log(self, msg):
        self.log_box.insert(tk.END, msg + "\n")
        self.log_box.see(tk.END)

    def _draw_game(self):
        self.canvas.delete("all")
        grid_size = self.game.grid_size
        cell_size = self.canvas_size / grid_size

        for i in range(grid_size + 1):
            self.canvas.create_line(i * cell_size, 0, i * cell_size, self.canvas_size, fill="#334155")
            self.canvas.create_line(0, i * cell_size, self.canvas_size, i * cell_size, fill="#334155")

        for (ox, oy) in self.game.obstacles:
            self.canvas.create_rectangle(
                ox * cell_size + 2, oy * cell_size + 2,
                (ox + 1) * cell_size - 2, (oy + 1) * cell_size - 2,
                fill="#ef4444", outline=""
            )

        if not self.game.has_key:
            kx, ky = self.game.key_x, self.game.key_y
            self.canvas.create_oval(
                kx * cell_size + 6, ky * cell_size + 6,
                (kx + 1) * cell_size - 6, (ky + 1) * cell_size - 6,
                fill="#eab308", outline="#fef08a"
            )

        gx, gy = self.game.goal_x, self.game.goal_y
        self.canvas.create_rectangle(
            gx * cell_size + 4, gy * cell_size + 4,
            (gx + 1) * cell_size - 4, (gy + 1) * cell_size - 4,
            fill="#22c55e", outline="#bbf7d0"
        )

        px, py = self.game.player_x, self.game.player_y
        self.canvas.create_oval(
            px * cell_size + 4, py * cell_size + 4,
            (px + 1) * cell_size - 4, (py + 1) * cell_size - 4,
            fill="#06b6d4", outline="#67e8f9", width=2
        )
        self.root.update_idletasks()

    def _step_frame(self):
        epsilon = max(0.05, 0.4 - (self.current_level * 0.04))
        
        action = self.agent.select_action(self.obs, epsilon=epsilon)
        next_obs, reward, done = self.game.step(action)
        self.trajectory.append((self.obs, action, reward))
        self.obs = next_obs
        self.step_in_ep += 1

        action_name = self.game.ACTIONS[action]
        self.lbl_action.config(text=f"Last Action: {action_name}")
        self.lbl_energy.config(text=f"Energy: {self.game.energy}")
        self.lbl_mem.config(text=f"OxtaMem Centroids: {len(self.agent.oxtamem_centroids)}")
        self.lbl_level.config(text=f"Level: {self.current_level} (Grid {self.game.grid_size}x{self.game.grid_size})")

        self._draw_game()

        if done or self.step_in_ep >= 50:
            self.agent.train_step(self.trajectory, self.game.victory)

            if self.game.victory:
                self.total_wins += 1
                self.current_level += 1
                self.lbl_wins.config(text=f"Total Victories: {self.total_wins}")
                self.log(f"Victory Level {self.current_level - 1}! Advancing...")
            else:
                self.log(f"Episode ended. Retrying Level {self.current_level}...")

            self.obs = self.game.reset()
            self.trajectory = []
            self.step_in_ep = 0

        self.root.after(150, self._step_frame)


def run_visual_app():
    root = tk.Tk()
    app = NSOSVisualApp(root)
    root.mainloop()


if __name__ == "__main__":
    run_visual_app()
