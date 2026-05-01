import sys
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add the build directory to python path
sys.path.append(".")
sys.path.append("build")

try:
    import pantheon
except ImportError:
    print("Could not import pantheon module.")
    sys.exit(1)

# 1. Define Real Models (Tiny for prototype)
class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.fc = nn.Linear(4 * 8 * 8, 10) # Assume input 8x8

    def forward(self, x):
        feat = F.relu(self.conv1(x))
        out = self.fc(feat.flatten(1))
        return feat, out, self.fc.weight

class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1) # Same channels for simplicity in proto
        self.fc = nn.Linear(4 * 8 * 8, 10)

    def forward(self, x):
        feat = F.relu(self.conv1(x))
        out = self.fc(feat.flatten(1))
        return feat, out, self.fc.weight

def test_omni_integrated():
    print("=== Running Omni-Distiller Integrated Prototype ===")
    print("1. Initializing Pantheon Engine (C++)...")

    engine = pantheon.Engine()
    engine.initialize()

    # Verify Engine Health
    status = engine.get_status()
    print(f"   Status: {status}")

    print("2. Instantiating PyTorch Models (Teacher & Student)...")
    teacher = TeacherModel()
    student = StudentModel()

    # Dummy Input: Batch 2, 1 Channel, 8x8 Image
    inputs = torch.randn(2, 1, 8, 8)

    print("3. Executing Training Step (Forward Pass)...")
    # Teacher Inference
    with torch.no_grad():
        t_feat, t_logits, t_fc_w = teacher(inputs)

    # Student Inference
    s_feat, s_logits, s_fc_w = student(inputs)

    # Convert to flat lists for C++ engine
    # Features: [Batch, Channel, Height, Width] -> Flat
    s_feat_flat = s_feat.flatten().tolist()
    t_feat_flat = t_feat.flatten().tolist()

    # Logits: [Batch, Classes] -> Flat
    s_logits_flat = s_logits.flatten().tolist()
    t_logits_flat = t_logits.flatten().tolist()

    # Weights: [Classes, InFeatures] -> Flat
    s_w_flat = s_fc_w.flatten().tolist()
    t_w_flat = t_fc_w.flatten().tolist()

    print("4. Computing Omni-Distiller Losses (C++ Engines)...")

    # --- Part 1: Response & Structure ---
    print("   [Part 1] Computing Logit & Structure Losses...")
    # NTCE: s, t, target=0, beta=2.0
    loss_ntce = engine.compute_ntce_loss(s_logits_flat, t_logits_flat, 0, 2.0)
    print(f"      NTCE Loss: {loss_ntce:.4f}")

    # MSDCRD: s, t, batch=2, c=4, h=8, w=8, ph=2, pw=2
    loss_msdcrd = engine.compute_msdcrd_loss(s_feat_flat, t_feat_flat, 2, 4, 8, 8, 2, 2)
    print(f"      MSDCRD Loss: {loss_msdcrd:.4f}")

    # CAM: s_f, s_w, t_f, t_w, b=2, c=4, h=8, w=8, cls=10, tgt=0
    loss_cam = engine.compute_cam_loss(s_feat_flat, s_w_flat, t_feat_flat, t_w_flat,
                                     2, 4, 8, 8, 10, 0)
    print(f"      CAT-KD Loss: {loss_cam:.4f}")

    # --- Part 2: Cognition ---
    print("   [Part 2] Simulating CoT & Symbolic Verification...")
    # Simulate a reasoning trace (tokens)
    t_trace = [1.0, 2.0, 3.0, 4.0, 5.0]
    s_trace = [1.0, 2.1, 2.9, 4.0, 5.2]
    # CoT Chunk: s, t, chunk=2
    loss_cot = engine.compute_chunk_wise_loss(s_trace, t_trace, 2)
    print(f"      CoT Chunk-Wise Loss: {loss_cot:.4f}")

    # Symbolic: s_out, expected_sum=0.0
    loss_sym = engine.compute_symbolic_loss(s_logits_flat, 0.0)
    print(f"      Symbolic Penalty: {loss_sym:.4f}")

    # --- Part 3: Physics ---
    print("   [Part 3] Computing Physical Dynamics Losses...")
    # Gradient Matching (Simulated grads)
    loss_task = nn.CrossEntropyLoss()(s_logits, torch.tensor([0, 0])) # Target class 0
    loss_task.backward()

    # Extract grads from first conv layer
    s_grads = student.conv1.weight.grad.flatten().tolist()
    # Teacher grads (simulated as student grads + noise)
    t_grads = [g + 0.01 for g in s_grads]

    loss_gkd = engine.compute_gradient_loss(t_grads, s_grads)
    print(f"      Gradient Matching Loss: {loss_gkd:.4f}")

    # Topology: s, t, n=8
    loss_topo = engine.compute_topology_loss(s_feat_flat, t_feat_flat, 8)
    print(f"      Topological Loss: {loss_topo:.4f}")

    # --- Part 4 & 5: Frontier & Hardware ---
    print("   [Part 4/5] Applying Frontier Constraints...")
    # Meta: curr=1.0, loss
    new_temp = engine.update_meta_policy(1.0, loss_ntce)
    print(f"      Meta-Policy Temp Update: 1.0 -> {new_temp:.4f}")

    # Photonic: w, width=16, height=16
    loss_pho = engine.compute_photonic_loss(s_w_flat, 16, 16)
    print(f"      Photonic Weight Constraint: {loss_pho:.4f}")

    print("5. Integrating KernelOpen Task...")
    engine.submit_dummy_task()
    time.sleep(0.5)
    tps = engine.get_throughput()
    print(f"   Kernel Throughput: {tps} TOPS")

    print("6. Shutdown...")
    engine.shutdown()

    print("=== Integrated Prototype Test Completed Successfully ===")

if __name__ == "__main__":
    test_omni_integrated()
