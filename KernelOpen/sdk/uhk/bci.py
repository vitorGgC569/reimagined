import time
import threading
import numpy as np

# Try importing BrainFlow and UHK Core
try:
    import uhk.uhk_core as uhk_core
except ImportError:
    raise ImportError("UHK Core SDK not built. Please run 'pip install .'")

try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
except ImportError:
    raise ImportError("BrainFlow not found. Please install via 'pip install brainflow'")


class BCIController:
    """
    Connects a real or synthetic BCI device to the UHK Spike Encoder.
    """
    def __init__(self, serial_port=None, use_synthetic=True):
        self.params = BrainFlowInputParams()
        # BrainFlow expects empty string instead of None for unused string params
        self.params.serial_port = serial_port if serial_port else ""

        if use_synthetic:
            self.board_id = BoardIds.SYNTHETIC_BOARD.value
        else:
            # Default to Cyton Board (Daisy)
            self.board_id = BoardIds.CYTON_DAISY_BOARD.value

        self.board = BoardShim(self.board_id, self.params)

        # Initialize UHK Spike Encoder (Threshold=1.0, Decay=0.9)
        self.encoder = uhk_core.SpikeEncoder(1.0, 0.9)

        self.is_streaming = False
        self.thread = None

    def start(self):
        print(f"[BCI] Connecting to Board ID {self.board_id}...")
        self.board.prepare_session()
        self.board.start_stream()
        self.is_streaming = True

        self.thread = threading.Thread(target=self._process_stream)
        self.thread.start()
        print("[BCI] Stream Started.")

    def _process_stream(self):
        while self.is_streaming:
            # Fetch latest data (e.g., last 20 samples)
            data = self.board.get_current_board_data(20)
            if data.shape[1] > 0:
                # Use Channel 1 (EEG)
                eeg_channel = BoardShim.get_eeg_channels(self.board_id)[0]
                signal = data[eeg_channel]

                # Convert to Float Vector for C++ Binding
                signal_vec = [float(x) for x in signal]

                # Encode Spikes via C++
                spikes = self.encoder.encode_signal(signal_vec)

                # In a real app, these spikes would go to the SNN
                # print(f"[BCI] Encoded Spikes: {spikes[:10]}...")

            time.sleep(0.01)

    def stop(self):
        self.is_streaming = False
        if self.thread:
            self.thread.join()

        if self.board.is_prepared():
            self.board.stop_stream()
            self.board.release_session()
        print("[BCI] Stream Stopped.")
