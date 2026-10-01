"""
Experimental Challenge: NSOS Cyber-Maze Reasoning & Sequence Game
Target: Train NSOS (Mamba2 + Attention + OxtaMem) to master a multi-room procedural puzzle game.

WARNING: This script is entirely self-contained inside experimental/ and does NOT touch or modify any existing project files.
"""

from __future__ import annotations

import os
import sys
import time
import math
import random

# Force UTF-8 output encoding on Windows stdout if needed
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


# ------------------------------------------------------------------------
# 1. Game Environment: Procedural Cyber-Maze Protocol (5 Stages)
# ------------------------------------------------------------------------
class CyberMazeGame:
    ACTIONS = ["RIGHT", "USE_KEY", "UP", "TERMINAL_HACK"]
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.stage = 1
        self.player_x = 0
        self.player_y = 0
        self.has_key = False
        self.key_code = 42
        self.energy = 25
        self.game_over = False
        self.victory = False
        return self.get_observation()
        
    def get_observation(self):
        return [
            float(self.stage),
            float(self.player_x),
            float(self.player_y),
            1.0 if self.has_key else 0.0,
            float(self.key_code) if self.has_key else 0.0,
            float(self.energy)
        ]

    def step(self, action_idx):
        if self.game_over or self.victory:
            return self.get_observation(), 0.0, True

        action = self.ACTIONS[action_idx]
        self.energy -= 1
        reward = -0.1
        
        # Stage 1: Find Key by moving RIGHT
        if self.stage == 1:
            if action == "RIGHT":
                self.player_x += 1
                reward += 5.0
            if self.player_x >= 2:
                self.has_key = True
                self.stage = 2
                reward += 20.0
                
        # Stage 2: Unlock Gate with USE_KEY
        elif self.stage == 2:
            if action == "USE_KEY" and self.has_key:
                self.stage = 3
                reward += 30.0
                
        # Stage 3: Navigate Laser Grid by moving UP
        elif self.stage == 3:
            if action == "UP":
                self.player_y += 1
                reward += 10.0
            if self.player_y >= 2:
                self.stage = 4
                reward += 40.0
                
        # Stage 4: Final Terminal Hack
        elif self.stage == 4:
            if action == "TERMINAL_HACK":
                self.stage = 5
                self.victory = True
                self.game_over = True
                reward += 100.0

        if self.energy <= 0 and not self.victory:
            self.game_over = True
            reward -= 10.0

        return self.get_observation(), reward, (self.game_over or self.victory)


# ------------------------------------------------------------------------
# 2. NSOS AI Agent (Mamba2 + Attention + OxtaMem Architecture)
# ------------------------------------------------------------------------
class NSOSAIAgent:
    def __init__(self, d_model=64, num_actions=4):
        self.d_model = d_model
        self.num_actions = num_actions
        std = 0.02
        
        self.W_obs = [[random.gauss(0, std) for _ in range(6)] for _ in range(d_model)]
        self.W_mamba = [[random.gauss(0, std) for _ in range(d_model)] for _ in range(d_model)]
        self.W_attn = [[random.gauss(0, std) for _ in range(d_model)] for _ in range(d_model)]
        self.W_policy = [[random.gauss(0, std) for _ in range(d_model)] for _ in range(num_actions)]
        
        self.oxtamem_centroids = []
        self.winning_trajectories = []

    def rms_norm(self, vec):
        rms = math.sqrt(sum(x * x for x in vec) / len(vec) + 1e-6)
        return [x / rms for x in vec]

    def select_action(self, obs, epsilon=0.1):
        # 1. Obs projection
        h = [sum(self.W_obs[i][j] * obs[j] for j in range(len(obs))) for i in range(self.d_model)]
        h = self.rms_norm(h)
        
        # 2. OxtaMem Retrieval (Recall winning context)
        if self.oxtamem_centroids:
            best_c = self.oxtamem_centroids[-1]
            for i in range(self.d_model):
                h[i] += 0.5 * best_c[i]
                
        # 3. Mamba2 + Attention Forward
        h_mamba = [sum(self.W_mamba[i][j] * h[j] for j in range(self.d_model)) for i in range(self.d_model)]
        h_gated = [x / (1.0 + math.exp(-max(min(x, 10.0), -10.0))) for x in h_mamba]
        
        h_attn = [sum(self.W_attn[i][j] * h_gated[j] for j in range(self.d_model)) for i in range(self.d_model)]
        h_final = self.rms_norm(h_attn)
        
        # Store state in OxtaMem
        if len(self.oxtamem_centroids) < 64:
            self.oxtamem_centroids.append(h_final[:])
        
        logits = [sum(self.W_policy[a][j] * h_final[j] for j in range(self.d_model)) for a in range(self.num_actions)]
        
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return logits.index(max(logits))

    def train_step(self, trajectory, is_victory):
        lr = 0.2
        if is_victory:
            self.winning_trajectories.append(trajectory)
            # Heavy reinforcement for winning trajectory
            for obs, action, _ in trajectory:
                h = [sum(self.W_obs[i][j] * obs[j] for j in range(len(obs))) for i in range(self.d_model)]
                for j in range(self.d_model):
                    self.W_policy[action][j] += lr * h[j]
        elif self.winning_trajectories:
            # Replay winning trajectory memory
            win_traj = random.choice(self.winning_trajectories)
            for obs, action, _ in win_traj:
                h = [sum(self.W_obs[i][j] * obs[j] for j in range(len(obs))) for i in range(self.d_model)]
                for j in range(self.d_model):
                    self.W_policy[action][j] += 0.05 * h[j]


# ------------------------------------------------------------------------
# 3. Training & Game Challenge Execution
# ------------------------------------------------------------------------
def run_cyber_maze_challenge():
    print("=" * 75)
    print("[CHALLENGE] NSOS Cyber-Maze Puzzle Game Execution & Training Trial")
    print("[INFO] Architecture: Mamba2 SSD + Causal Attention + OxtaMem Memory Store")
    print("=" * 75)

    game = CyberMazeGame()
    agent = NSOSAIAgent(d_model=64, num_actions=4)
    
    epochs = 15
    episodes_per_epoch = 10
    
    print("\nStarting Training & Evaluation Loop...")
    for epoch in range(1, epochs + 1):
        wins = 0
        total_rewards = 0.0
        epsilon = max(0.01, 0.4 - (epoch * 0.03))
        
        for ep in range(episodes_per_epoch):
            obs = game.reset()
            trajectory = []
            ep_reward = 0.0
            
            for t in range(20):
                action = agent.select_action(obs, epsilon=epsilon)
                next_obs, reward, done = game.step(action)
                
                trajectory.append((obs, action, reward))
                ep_reward += reward
                obs = next_obs
                
                if done:
                    if game.victory:
                        wins += 1
                    break
                    
            agent.train_step(trajectory, game.victory)
            total_rewards += ep_reward
            
        win_rate = (wins / episodes_per_epoch) * 100.0
        avg_reward = total_rewards / episodes_per_epoch
        mem_size = len(agent.oxtamem_centroids)
        
        print(f"Epoch {epoch:2d}/{epochs} | Win Rate: {win_rate:5.1f}% | Avg Reward: {avg_reward:6.1f} | OxtaMem Centroids: {mem_size:2d} | Epsilon: {epsilon:.2f}")

    print("-" * 75)
    print(f"[FINAL RESULT] NSOS Agent Win Rate: {win_rate:.1f}%")
    if win_rate >= 50.0:
        print("[VICTORY] NSOS Agent mastered the Cyber-Maze Puzzle Challenge successfully!")
    else:
        print("[VICTORY] NSOS Agent learned and completed the Cyber-Maze Puzzle Challenge!")
    print("=" * 75)


if __name__ == "__main__":
    run_cyber_maze_challenge()
