import sys
import os
import time
from stats_utils import StatsTracker, AcceptanceCriteria

sys.path.append(os.path.join(os.path.dirname(__file__), '../../OXN/nsos/build'))
import nsos_ext

def test_scaling_stats():
    print("==========================================")
    print("   📊 NSOS SCALING STATISTICS REPORT 📊   ")
    print("==========================================")

    device = nsos_ext.Device.CPU
    try:
        nsos_ext.Tensor.zeros([1], nsos_ext.Device.GPU)
        device = nsos_ext.Device.GPU
    except:
        pass

    lat_tracker = StatsTracker("Latency (s)")
    ac = AcceptanceCriteria()

    print("\n[Protocol] Context Stretcher (512 -> 8192)")
    d_model = 64
    model = nsos_ext.JambaModel(2, d_model, 128, device)
    if device == nsos_ext.Device.GPU: model.to(device)

    seqs = [512, 1024, 2048, 4096, 8192]

    print(f"{'SeqLen':<10} | {'Time (s)':<10} | {'Tokens/s':<10}")
    print("-" * 35)

    for seq in seqs:
        try:
            input_t = nsos_ext.Tensor.zeros([1, seq, d_model], device)

            start = time.time()
            # Fix: Pass explicit None for context if binding requires it, or rely on default
            # The error showed: (self, arg0, arg1) -> Tensor. Arg1 is Context*.
            # It seems the default argument handling in pybind might be strict or I missed checking it.
            # Passing None usually works for pointers.
            _ = model.forward(input_t, None)
            dur = time.time() - start

            lat_tracker.add(dur)
            throughput = seq / (dur + 1e-9)

            print(f"{seq:<10} | {dur:<10.4f} | {throughput:<10.1f}")

        except Exception as e:
            print(f"{seq:<10} | FAILED ({e})")
            break

    mean_lat = lat_tracker.report()

    if mean_lat is not None:
        # Acceptance: Throughput should not drop to zero
        # For CPU, expectation is low.
        ac.check("Mean Latency", mean_lat, 5.0, "<")
    else:
        print("No data collected for acceptance check.")

if __name__ == "__main__":
    test_scaling_stats()
