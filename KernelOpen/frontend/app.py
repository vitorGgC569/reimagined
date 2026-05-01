import os
import sys
import random
import time
from flask import Flask, render_template, jsonify, request

# Add project root to python path to access SDK
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Try importing the SDK (if built), else use mock
try:
    import uhk.uhk_core as uhk_core
    HAS_SDK = True
    print("[INFO] UHK SDK Loaded successfully.")
except ImportError:
    HAS_SDK = False
    print("[WARN] UHK SDK not found. Running in simulation mode for UI demo.")

app = Flask(__name__)

# Runtime Instance
runtime = None
if HAS_SDK:
    runtime = uhk_core.UniversalKernelRuntime()

# Global State for Visualization
kernel_state = {
    "status": "IDLE",
    "throughput": 0.0,
    "latency": 0.0,
    "active_sms": 0,
    "ring_buffer_usage": 0,
    "tasks_processed": 0
}

@app.route('/')
def index():
    return render_template('index.html', sdk_status=HAS_SDK)

@app.route('/api/status')
def get_status():
    global runtime

    if HAS_SDK and runtime:
        # Fetch Real Metrics from C++ Runtime
        kernel_state["throughput"] = runtime.get_throughput()
        kernel_state["latency"] = runtime.get_latency()
        kernel_state["active_sms"] = runtime.get_active_sms()
        kernel_state["ring_buffer_usage"] = runtime.get_ring_buffer_usage()
        kernel_state["tasks_processed"] = runtime.get_tasks_processed()
    else:
        # Simulate dynamic updates if logic is running (Fallback)
        if kernel_state["status"] == "RUNNING":
            # Fluctuate metrics for visual effect
            kernel_state["throughput"] = 4000.0 + random.uniform(-100, 100) # TOPS
            kernel_state["latency"] = 1.1 + random.uniform(-0.1, 0.1)     # us
            kernel_state["active_sms"] = random.randint(120, 132)
            kernel_state["ring_buffer_usage"] = random.randint(10, 80)
            kernel_state["tasks_processed"] += random.randint(100, 500)
        else:
            kernel_state["active_sms"] = 0
            kernel_state["ring_buffer_usage"] = 0

    return jsonify(kernel_state)

@app.route('/api/start_job', methods=['POST'])
def start_job():
    mode = request.json.get('mode', 'BITNET')
    kernel_state["status"] = "RUNNING"

    # In a real scenario, this would trigger a specific workload generator loop
    # For now, the persistent kernel (or its simulation) is already running upon instantiation
    # We could send a "Start Benchmark" command here if the runtime supported it.

    return jsonify({"message": f"Job {mode} Started", "status": "OK"})

@app.route('/api/stop_job', methods=['POST'])
def stop_job():
    kernel_state["status"] = "IDLE"
    # We don't shut down the runtime because re-instantiating it might be heavy,
    # just logically 'stop' the job visualization or send a stop command.
    return jsonify({"message": "Job Stopped", "status": "OK"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
