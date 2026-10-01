import numpy as np
import io
import json

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

MAX_PAYLOAD_BYTES = 16 * 1024 * 1024
MAX_RECALL_DEPTH = 1024
MAX_KEY_BYTES = 1024

class GeodesicClient:
    """
    High-level Python SDK for Geodesic Memory Engine.
    Handles safe serialization for bytes, text, JSON values, NumPy arrays, and
    PyTorch tensors. Generic pickle deserialization is intentionally not used.
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
            if not isinstance(host, str) or not host or not (1 <= int(port) <= 65535):
                raise ValueError("redis host/port are invalid")
            password = kwargs.get("password", kwargs.get("auth_token"))
            self.conn = redis.Redis(
                host=host,
                port=port,
                password=password,
                decode_responses=False,
                socket_connect_timeout=float(kwargs.get("connect_timeout", 5.0)),
                socket_timeout=float(kwargs.get("socket_timeout", 10.0)),
                ssl=bool(kwargs.get("ssl", False)),
            )
        elif driver == "native":
            # Assumes geodesic_engine is built and installed in python path
            from .native import PyGeodesicEngine
            path = kwargs.get("db_path", "geodesic.db")
            size = kwargs.get("size_mb", 100)
            if not isinstance(path, str) or not path or "\x00" in path:
                raise ValueError("db_path must be a non-empty string without NUL")
            if not isinstance(size, int) or isinstance(size, bool) or not 1 <= size <= 32768:
                raise ValueError("size_mb must be an integer in 1..32768")
            self.conn = PyGeodesicEngine(path, size)
        else:
            raise ValueError("Unknown driver. Use 'redis' or 'native'")

    def _serialize(self, data):
        """Converts Tensors/Arrays/Objects to bytes."""
        if HAS_TORCH and isinstance(data, torch.Tensor):
            buff = io.BytesIO()
            np.save(buff, data.detach().cpu().numpy(), allow_pickle=False)
            payload = b"TORCHTENSOR:" + buff.getvalue()
            if len(payload) > MAX_PAYLOAD_BYTES:
                raise ValueError("serialized payload exceeds the 16 MiB limit")
            return payload

        if isinstance(data, np.ndarray):
            buff = io.BytesIO()
            if data.dtype == object:
                raise TypeError("object dtype arrays are not supported by the safe serializer")
            np.save(buff, data, allow_pickle=False)
            payload = b"NUMPY:" + buff.getvalue()
            if len(payload) > MAX_PAYLOAD_BYTES:
                raise ValueError("serialized payload exceeds the 16 MiB limit")
            return payload

        if isinstance(data, bytes):
            payload = b"BYTES:" + data
            if len(payload) > MAX_PAYLOAD_BYTES:
                raise ValueError("serialized payload exceeds the 16 MiB limit")
            return payload

        if isinstance(data, str):
            payload = b"TEXT:" + data.encode("utf-8")
            if len(payload) > MAX_PAYLOAD_BYTES:
                raise ValueError("serialized payload exceeds the 16 MiB limit")
            return payload

        if data is None or isinstance(data, (bool, int, float, list, dict)):
            payload = b"JSON:" + json.dumps(
                data, separators=(",", ":"), ensure_ascii=False, allow_nan=False
            ).encode("utf-8")
            if len(payload) > MAX_PAYLOAD_BYTES:
                raise ValueError("serialized payload exceeds the 16 MiB limit")
            return payload

        raise TypeError(
            "unsupported value type for safe OxtaMem serialization; use bytes, str, JSON values, "
            "numpy arrays, or torch tensors"
        )

    def _deserialize(self, data: bytes):
        """Restores bytes to objects."""
        if len(data) > MAX_PAYLOAD_BYTES:
            raise ValueError("stored payload exceeds the 16 MiB limit")
        if data.startswith(b"TORCHTENSOR:"):
            buff = io.BytesIO(data[12:])
            array = np.load(buff, allow_pickle=False)
            if not HAS_TORCH:
                return array
            return torch.from_numpy(array)

        if data.startswith(b"NUMPY:"):
            buff = io.BytesIO(data[6:])
            return np.load(buff, allow_pickle=False)

        if data.startswith(b"BYTES:"):
            return data[6:]

        if data.startswith(b"TEXT:"):
            return data[5:].decode("utf-8")

        if data.startswith(b"JSON:"):
            return json.loads(data[5:].decode("utf-8"))

        return data # Return as raw bytes if unknown prefix

    def save(self, key: str, value):
        """Saves any object (Tensor, Array, Dict) to the timeline."""
        self._validate_key(key)
        payload = self._serialize(value)

        if self.driver_type == "redis":
            return self.conn.set(key, payload)
        else:
            return self.conn.write(key, payload)

    def load_latest(self, key: str):
        """Retrieves the latest state of the object."""
        self._validate_key(key)
        if self.driver_type == "redis":
            data = self.conn.get(key)
        else:
            data = self.conn.read_latest(key)

        if data is None: return None
        return self._deserialize(data)

    def recall_history(self, key: str, depth: int):
        """Returns a list of states (time travel)."""
        self._validate_key(key)
        depth = int(depth)
        if depth < 0 or depth > MAX_RECALL_DEPTH:
            raise ValueError("depth must be between 0 and 1024")
        if self.driver_type == "redis":
            raw_list = self.conn.execute_command("RECALL", key, depth)
        else:
            raw_list = self.conn.recall(key, depth)
        return [self._deserialize(x) for x in raw_list]

    @staticmethod
    def _validate_key(key: str):
        if not isinstance(key, str) or not key or "\x00" in key:
            raise ValueError("key must be a non-empty string without NUL")
        if len(key.encode("utf-8")) > MAX_KEY_BYTES:
            raise ValueError("UTF-8 key exceeds 1024 bytes")
