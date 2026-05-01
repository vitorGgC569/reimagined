import collections
import time
import numpy as np

# --- 1. IO_URING Simulator ---
class CompletionQueueEntry:
    def __init__(self, user_data, res):
        self.user_data = user_data
        self.res = res

class IOUring:
    """
    Simulates the Ring Buffer architecture of io_uring.
    Submission Queue (SQ) -> Kernel Thread -> Completion Queue (CQ).
    """
    def __init__(self, entries=4096):
        self.sq = collections.deque(maxlen=entries)
        self.cq = collections.deque(maxlen=entries)
        self.is_setup = True

    def submit_read(self, fd, buffer, offset, user_data):
        # Enqueue generic "Read" Op
        self.sq.append({
            'op': 'READ', 'fd': fd, 'buffer': buffer,
            'offset': offset, 'user_data': user_data
        })

    def submit_write(self, fd, buffer, offset, user_data):
        self.sq.append({
            'op': 'WRITE', 'fd': fd, 'buffer': buffer,
            'offset': offset, 'user_data': user_data
        })

    def process_sq(self):
        """Simulates the Kernel Polling Thread (SQPOLL)."""
        processed = 0
        while self.sq:
            op = self.sq.popleft()
            # Execute Op (Simulated latency?)
            # Just immediate for functional test
            if op['op'] == 'READ':
                # Fake Read
                pass
            elif op['op'] == 'WRITE':
                pass

            # Post Completion
            cqe = CompletionQueueEntry(op['user_data'], 0) # 0 = Success
            self.cq.append(cqe)
            processed += 1
        return processed

    def peek_cq(self):
        if self.cq:
            return self.cq[0]
        return None

    def pop_cq(self):
        if self.cq:
            return self.cq.popleft()
        return None

# --- 2. GPUDirect Storage (GDS) Simulator ---
class GDSDriver:
    """
    Mocks cuFile APIs.
    """
    def cuFileDriverOpen(self):
        return "GDS_HANDLE_OK"

    def cuFileBufRegister(self, gpu_ptr, size):
        # Maps GPU memory to DMA engine
        return f"REGISTERED_PTR_{id(gpu_ptr)}"

    def cuFileRead(self, handle, devPtr_base, size, file_offset, devPtr_offset):
        # Logic: Disk DMA -> GPU Memory (Zero Copy)
        # In python simulation: copy file bytes -> "GPU buffer"
        return size # bytes read

# --- 3. CXL Memory Pool Simulator ---
class CXLMemoryPool:
    """
    Simulates a disaggregated memory pool connected via CXL.
    Key Characteristic: Higher latency than local DRAM, but huge capacity.
    """
    def __init__(self, capacity_gb, latency_ns=200):
        self.capacity = capacity_gb * 1024**3
        self.used = 0
        self.latency_ns = latency_ns # ~3x Local DRAM
        self.data_store = {} # ObjID -> Data

    def alloc(self, size):
        if self.used + size > self.capacity:
            raise MemoryError("CXL Pool Full")
        addr = self.used
        self.used += size
        return addr

    def read(self, addr, size):
        # Simulate latency
        # time.sleep(self.latency_ns / 1e9)
        # Return dummy data
        return b'\x00' * size

# --- 4. Processing-In-Memory (PIM) Simulator ---
class PIMUnit:
    """
    Simulates computation happening INSIDE memory chips.
    Ideal for Scan/Filter/Aggregate.
    """
    def __init__(self, memory_slice):
        self.memory = np.array(memory_slice)

    def scan_filter(self, threshold):
        """
        Executes 'x > threshold' entirely in memory.
        Returns bitmap of matches.
        No data moves to CPU caches.
        """
        return self.memory > threshold

    def aggregate_sum(self):
        """
        Returns sum. CPU only receives 1 scalar result.
        Bandwidth savings: Huge.
        """
        return np.sum(self.memory)
