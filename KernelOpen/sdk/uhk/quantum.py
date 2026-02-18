import numpy as np

# Try importing Qiskit and UHK Core
try:
    import uhk.uhk_core as uhk_core
except ImportError:
    raise ImportError("UHK Core SDK not built. Please run 'pip install .'")

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    # Optional: from qiskit_ibm_runtime import QiskitRuntimeService
except ImportError:
    raise ImportError("Qiskit not found. Please install via 'pip install qiskit qiskit-aer'")


class QuantumBridge:
    """
    Connects the UHK Runtime to IBM Q (or Simulator).
    """
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.backend = None

        if self.api_key:
            print("[Quantum] Connecting to IBM Cloud...")
            # service = QiskitRuntimeService(channel="ibm_quantum", token=api_key)
            # self.backend = service.least_busy(operational=True, simulator=False)
            print("[Quantum] (Mock) Authenticated with IBM Cloud.")
            # Fallback to Sim for this demo even with key
            self.backend = AerSimulator()
        else:
            print("[Quantum] No API Key. Using Aer Simulator (Local).")
            self.backend = AerSimulator()

    def run_bell_state(self):
        """
        Runs a Bell State (Entanglement) circuit on the backend.
        """
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        print("[Quantum] Submitting Circuit to Backend...")

        # Transpile for the specific backend
        tqc = transpile(qc, self.backend)

        # Run job
        job = self.backend.run(tqc, shots=1024)
        result = job.result()
        counts = result.get_counts(tqc)

        print(f"[Quantum] Result Counts: {counts}")
        return counts
