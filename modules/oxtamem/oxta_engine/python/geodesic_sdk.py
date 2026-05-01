import json
import numpy as np
import io

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class GeodesicClient:
    """
    High-level Python SDK for Geodesic Memory Engine.
    Handles serialization of complex objects (NumPy, PyTorch) automatically.
    """
    def __init__(self, driver="redis", **kwargs):
        """
        driver: 'redis' (connect via network) or 'native' (direct rust binding)
        kwargs: arguments for the driver (host, port, db_path, etc)
        """
        self.driver_type = driver

        if driver == "redis":
            import redis
            host = kwargs.get("host", "localhost")
            port = kwargs.get("port", 6379)
            self.conn = redis.Redis(host=host, port=port, decode_responses=False) # Binary mode
        elif driver == "native":
            # Assumes geodesic_engine is built and installed in python path
            from geodesic_engine import PyGeodesicEngine
            path = kwargs.get("db_path", "geodesic.db")
            size = kwargs.get("size_mb", 100)
            self.conn = PyGeodesicEngine(path, size)
        else:
            raise ValueError("Unknown driver. Use 'redis' or 'native'")

    def _serialize(self, data):
        """Converts Tensors/Arrays/Objects to bytes."""
        if HAS_TORCH and isinstance(data, torch.Tensor):
            buff = io.BytesIO()
            torch.save(data, buff)
            return b"TORCH:" + buff.getvalue()

        if isinstance(data, np.ndarray):
            buff = io.BytesIO()
            np.save(buff, data)
            return b"NUMPY:" + buff.getvalue()

        if isinstance(data, (bytes, bytearray, memoryview)):
            return b"BYTES:" + bytes(data)

        if isinstance(data, str):
            return b"TEXT:" + data.encode("utf-8")

        try:
            return b"JSON:" + json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        except TypeError as error:
            raise TypeError(
                "Unsupported payload type for GeodesicClient. Use bytes, str, JSON-compatible "
                "objects, NumPy arrays, or PyTorch tensors."
            ) from error

    def _deserialize(self, data: bytes):
        """Restores bytes to objects."""
        if data.startswith(b"TORCH:"):
            buff = io.BytesIO(data[6:])
            return torch.load(buff, map_location="cpu")

        if data.startswith(b"NUMPY:"):
            buff = io.BytesIO(data[6:])
            return np.load(buff, allow_pickle=False)

        if data.startswith(b"JSON:"):
            return json.loads(data[5:].decode("utf-8"))

        if data.startswith(b"TEXT:"):
            return data[5:].decode("utf-8")

        if data.startswith(b"BYTES:"):
            return data[6:]

        return data # Return as raw bytes if unknown prefix

    def save(self, key: str, value):
        """Saves any object (Tensor, Array, Dict) to the timeline."""
        payload = self._serialize(value)

        if self.driver_type == "redis":
            return self.conn.set(key, payload)
        else:
            return self.conn.write(key, payload)

    def load_latest(self, key: str):
        """Retrieves the latest state of the object."""
        if self.driver_type == "redis":
            data = self.conn.get(key)
        else:
            data = self.conn.read_latest(key)

        if data is None: return None
        return self._deserialize(data)

    def recall_history(self, key: str, depth: int):
        """Returns a list of states (time travel)."""
        if self.driver_type == "redis":
            raw_list = self.conn.execute_command("RECALL", key, depth)
        else:
            raw_list = self.conn.recall(key, depth)
        if raw_list is None:
            return []
        return [self._deserialize(x) for x in raw_list]
