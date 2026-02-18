import time
import math
import statistics

class StatsTracker:
    def __init__(self, name="Metric"):
        self.name = name
        self.values = []
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        if self.start_time:
            self.values.append(time.time() - self.start_time)
            self.start_time = None

    def add(self, val):
        self.values.append(val)

    def report(self):
        if not self.values:
            print(f"[{self.name}] No data collected.")
            return

        n = len(self.values)
        mean = statistics.mean(self.values)
        stdev = statistics.stdev(self.values) if n > 1 else 0.0
        min_v = min(self.values)
        max_v = max(self.values)
        p95 = sorted(self.values)[int(n * 0.95)]

        print(f"📊 {self.name} Stats (N={n}):")
        print(f"   Mean:   {mean:.6f}")
        print(f"   StdDev: {stdev:.6f}")
        print(f"   Min:    {min_v:.6f}")
        print(f"   Max:    {max_v:.6f}")
        print(f"   P95:    {p95:.6f}")
        return mean

class AcceptanceCriteria:
    def __init__(self):
        # Defaults
        self.max_latency = 0.1 # seconds
        self.min_throughput = 10.0 # tokens/s (CPU assumption)
        self.max_error = 1e-4

    def check(self, metric_name, value, threshold, op="<"):
        passed = False
        if op == "<": passed = value < threshold
        elif op == ">": passed = value > threshold

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   Criteria [{metric_name}]: {value:.6f} {op} {threshold} -> {status}")
        return passed
