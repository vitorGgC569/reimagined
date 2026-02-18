import numpy as np
import random

class ChipEnv:
    """
    Ambiente de roteamento de chips simplificado.
    Grid 2D onde 0=Livre, 1=Obstáculo, 2=Wire, 3=Pin Inicial, 4=Pin Final.
    """
    def __init__(self, size=32):
        self.size = size
        self.grid = np.zeros((size, size), dtype=np.float32)
        self.start_pos = (0, 0)
        self.end_pos = (size-1, size-1)
        self.current_pos = (0, 0)
        self.path = []
        self.reset()

    def reset(self, num_obstacles=20):
        self.grid = np.zeros((self.size, self.size), dtype=np.float32)
        # Add random obstacles
        for _ in range(num_obstacles):
            r, c = random.randint(0, self.size-1), random.randint(0, self.size-1)
            self.grid[r, c] = 1.0 # Obstacle
            
        # Set Pins
        self.start_pos = (random.randint(0, self.size-1), random.randint(0, self.size-1))
        self.end_pos = (random.randint(0, self.size-1), random.randint(0, self.size-1))
        
        # Ensure Pins are clear
        self.grid[self.start_pos] = 0.0
        self.grid[self.end_pos] = 0.0
        
        self.current_pos = self.start_pos
        self.path = [self.start_pos]
        return self.get_observation()

    def get_observation(self, window_size=7):
        """
        Retorna uma janela local ao redor da posição atual + vetor para o objetivo.
        """
        r, c = self.current_pos
        half = window_size // 2
        obs = np.ones((window_size, window_size), dtype=np.float32) # Padded with obstacles
        
        for i in range(window_size):
            for j in range(window_size):
                gr, gc = r - half + i, c - half + j
                if 0 <= gr < self.size and 0 <= gc < self.size:
                    obs[i, j] = self.grid[gr, gc]
        
        # Inserir informação de direção relativa ao objetivo
        # Concatenamos isso no script de treino
        target_vec = [float(self.end_pos[0] - r), float(self.end_pos[1] - c)]
        # Retorna uma única lista flat com tudo
        full_obs = [float(x) for x in obs.flatten()] + target_vec
        return full_obs

    def step(self, action):
        """
        Ações: 0=Cima, 1=Baixo, 2=Esquerda, 3=Direita
        """
        r, c = self.current_pos
        if action == 0: r -= 1
        elif action == 1: r += 1
        elif action == 2: c -= 1
        elif action == 3: c += 1
        
        # Validar movimento
        if 0 <= r < self.size and 0 <= c < self.size and self.grid[r, c] == 0:
            self.current_pos = (r, c)
            self.grid[r, c] = 2.0 # Mark as wire
            self.path.append(self.current_pos)
            done = (self.current_pos == self.end_pos)
            reward = 10.0 if done else -0.1
        else:
            done = True # Crash or out of bounds
            reward = -5.0
            
        return self.get_observation(), reward, done

    def render(self):
        display = np.copy(self.grid)
        display[self.start_pos] = 3.0
        display[self.end_pos] = 4.0
        display[self.current_pos] = 5.0 # Head
        
        chars = {0.0: '.', 1.0: 'X', 2.0: '-', 3.0: 'S', 4.0: 'E', 5.0: 'H'}
        for r in range(self.size):
            print("".join([chars[display[r, c]] for c in range(self.size)]))
        print("-" * self.size)

if __name__ == "__main__":
    env = ChipEnv(16)
    obs, vec = env.reset()
    env.render()
