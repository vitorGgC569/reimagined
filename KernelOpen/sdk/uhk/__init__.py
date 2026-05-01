# This file exposes the bindings
# Note: uhk_core is a compiled extension module (.so) located in this directory after build
try:
    from .uhk_core import UniversalKernelRuntime, OpType
except ImportError:
    # If not built yet, don't crash immediately, allow setup.py to run
    pass
