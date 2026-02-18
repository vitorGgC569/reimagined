
import time
import sys
import os
import psutil
import random

class Dashboard:
    def __init__(self, log_file="dashboard_telemetry.log"):
        self.log_file = log_file
        self.process = psutil.Process(os.getpid())
        self.start_time = time.time()

    def log(self, step, loss, lr, grad_norm, activation_stats=None):
        timestamp = time.time() - self.start_time

        # System Vitality
        cpu_usage = self.process.cpu_percent()
        ram_usage = self.process.memory_info().rss / 1024 / 1024 # MB

        # Activation Analysis (Simulated Histogram summary)
        act_mean = 0.0
        act_std = 0.0
        if activation_stats:
            act_mean = activation_stats.get('mean', 0.0)
            act_std = activation_stats.get('std', 0.0)

        # Log Line
        # Time, Step, Loss, LR, GradNorm, CPU%, RAM(MB), ActMean, ActStd
        entry = f"{timestamp:.2f},{step},{loss:.4f},{lr:.6f},{grad_norm:.4f},{cpu_usage:.1f},{ram_usage:.1f},{act_mean:.4f},{act_std:.4f}\n"

        with open(self.log_file, "a") as f:
            f.write(entry)

    def print_status(self):
        # Print a cool ASCII dashboard snapshot
        print("\033[H\033[J") # Clear screen (if supported)
        print("=== 📡 OXTA/NSOS Real-Time Telemetry ===")
        cpu = self.process.cpu_percent()
        ram = self.process.memory_info().rss / 1024 / 1024
        print(f"SYSTEM: CPU {cpu}% | RAM {ram:.1f} MB")
        print("----------------------------------------")

# Integration Test
if __name__ == "__main__":
    dash = Dashboard()
    print("Logging dummy data...")
    for i in range(10):
        dash.log(i, 5.0 - i*0.1, 0.001, 1.5, {'mean': 0.1, 'std': 1.0})
        time.sleep(0.1)
    print(f"Log saved to {dash.log_file}")
