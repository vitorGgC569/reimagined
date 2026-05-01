from oxta_mem import GeodesicClient
import time
import random
import os

class CausalMazeSolver:
    def __init__(self, size=8):
        self.size = size
        self.maze = [[0 for _ in range(size)] for _ in range(size)]
        
        # Add moderate traps
        trap_count = int(size * 1.2)
        random.seed(42) # Deterministic for this test
        for _ in range(trap_count):
            r, c = random.randint(0, size-1), random.randint(0, size-1)
            if (r, c) != (0, 0) and (r, c) != (size-1, size-1):
                self.maze[r][c] = -1
        
        self.maze[size-1][size-1] = 1 # Goal
        self.db_path = "causal_maze_final.db"
        
        # Cleanup
        for f in [self.db_path, f"{self.db_path}.usearch_index", f"{self.db_path}.usearch_index.meta"]:
            if os.path.exists(f): os.remove(f)

        self.client = GeodesicClient(driver="native", db_path=self.db_path, size_mb=100)
        self.pos = (0, 0)
        self.path = [(0, 0)]
        self.failed_moves = {} # (pos) -> list of moves that led to trauma

    def solve(self):
        print("--- Starting Advanced Causal Maze Solver ---")
        print(f"Goal: ({self.size-1}, {self.size-1})")
        
        step = 0
        while self.pos != (self.size-1, self.size-1):
            step += 1
            state = {"pos": self.pos, "path": list(self.path)}
            
            # Identify possible moves
            adj = []
            if self.pos[0] < self.size-1: adj.append((self.pos[0]+1, self.pos[1]))
            if self.pos[1] < self.size-1: adj.append((self.pos[0], self.pos[1]+1))
            if self.pos[0] > 0: adj.append((self.pos[0]-1, self.pos[1]))
            if self.pos[1] > 0: adj.append((self.pos[0], self.pos[1]-1))
            
            # Filter: Don't go back in THIS path and don't pick known failures
            possible = [m for m in adj if m not in self.path and m not in self.failed_moves.get(self.pos, [])]

            if len(possible) > 1:
                # CAUSAL FORK: Store this decision point in Geodesic Memory
                fork_id = f"fork_{self.pos}"
                self.client.save(fork_id, state)
                self.client.save("latest_fork_id", fork_id)

            if possible:
                # Heuristic: Move towards goal
                possible.sort(key=lambda p: (self.size-1 - p[0]) + (self.size-1 - p[1]))
                next_pos = possible[0]
                
                if self.maze[next_pos[0]][next_pos[1]] != -1:
                    print(f"Step {step}: {self.pos} -> {next_pos}")
                    self.pos = next_pos
                    self.path.append(next_pos)
                else:
                    print(f"Step {step}: {self.pos} -> {next_pos} (💥 TRAP!)")
                    # Register failure
                    if self.pos not in self.failed_moves: self.failed_moves[self.pos] = []
                    self.failed_moves[self.pos].append(next_pos)
                    self.rewind()
            else:
                print(f"Step {step}: No moves from {self.pos}")
                self.rewind()
            
            if step > 300: break

        if self.pos == (self.size-1, self.size-1):
            print(f"\n🏆 SUCCESS! Reached goal in {step} steps.")
            print(f"Final Path: {self.path}")

    def rewind(self):
        print("🕒 Recalling last valid timeline from Geodesic Engine...")
        fork_id = self.client.load_latest("latest_fork_id")
        if fork_id:
            checkpoint = self.client.load_latest(fork_id)
            if checkpoint:
                self.pos = checkpoint["pos"]
                self.path = list(checkpoint["path"])
                print(f"   Back at {self.pos}. Retrying from new timeline.")
                return
        
        # Absolute fallback
        self.pos, self.path = (0, 0), [(0, 0)]

if __name__ == "__main__":
    solver = CausalMazeSolver(size=5)
    solver.solve()
