import functools
import sys
import os

# Try to import C++ extension
try:
    import nsos_ext
except ImportError:
    # Fallback for dev environment
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../build'))
    import nsos_ext

class ContractViolation(Exception):
    pass

def validate_tensor(t, name="tensor"):
    if not hasattr(t, "shape") or not hasattr(t, "device"):
        raise ContractViolation(f"{name} is not a valid Tensor object.")
    if t.shape is None or len(t.shape) == 0:
        raise ContractViolation(f"{name} has invalid shape (None or scalar). Expects >=1 dims.")

def ensure_device(t, target_device, name="tensor"):
    if t.device != target_device:
        raise ContractViolation(f"{name} device mismatch. Expected {target_device}, got {t.device}.")

def contiguous_check(t, name="tensor"):
    if hasattr(t, "is_contiguous") and not t.is_contiguous():
        raise ContractViolation(f"{name} must be contiguous.")
    if hasattr(t, "strides") and hasattr(t, "shape"):
        expected = 1
        for dim, stride in zip(reversed(t.shape), reversed(t.strides)):
            if stride != expected:
                raise ContractViolation(f"{name} must be contiguous.")
            expected *= dim

# Decorator for Forward Methods
def shield(forward_func):
    @functools.wraps(forward_func)
    def wrapper(self, *args, **kwargs):
        # 1. Inspect Inputs
        if len(args) > 0:
            input_tensor = args[0]
            # Check if input is Tensor-like (nsos_ext.Tensor)
            # We assume first arg is input x
            if hasattr(input_tensor, "shape"):
                validate_tensor(input_tensor, "Input")
                contiguous_check(input_tensor, "Input")
                # Optional: Strict Check - Input device must match model device
                if hasattr(self, "device"):
                    ensure_device(input_tensor, self.device, "Input")

        # 2. Execute C++ Core
        try:
            result = forward_func(self, *args, **kwargs)
        except RuntimeError as e:
            # Catch C++ exceptions and re-raise cleanly
            raise RuntimeError(f"[NSOS Core Error] {e}") from e
        except Exception as e:
            raise ContractViolation(f"[Shield] Unexpected Error: {e}") from e

        # 3. Inspect Output
        if hasattr(result, "shape"):
            validate_tensor(result, "Output")
            contiguous_check(result, "Output")

        return result
    return wrapper

# Base Layer (Python Side Wrapper)
class Module:
    def __init__(self):
        self.device = nsos_ext.Device.CPU
        self._cpp_module = None

    def to(self, device):
        self.device = device
        if self._cpp_module and hasattr(self._cpp_module, "to"):
            self._cpp_module.to(device)
        return self

    @shield
    def forward(self, x):
        if not self._cpp_module:
            raise NotImplementedError("Module not initialized")
        return self._cpp_module.forward(x)

    def state_dict(self):
        if not self._cpp_module or not hasattr(self._cpp_module, "parameters"):
            return {}
        state = {}
        for index, parameter in enumerate(self._cpp_module.parameters()):
            name = getattr(parameter, "name", f"param_{index}")
            state[name] = getattr(parameter, "data", parameter)
        return state
