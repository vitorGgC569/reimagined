import numpy as np

# Abstract Base Class for RMI Models
class Model:
    def train(self, X, Y):
        raise NotImplementedError
    def predict(self, key):
        raise NotImplementedError

class LinearModel(Model):
    def __init__(self):
        self.slope = 0.0
        self.intercept = 0.0

    def train(self, X, Y):
        n = len(X)
        if n > 1:
            A = np.vstack([X, np.ones(n)]).T
            self.slope, self.intercept = np.linalg.lstsq(A, Y, rcond=None)[0]
        elif n == 1:
            self.slope = 0
            self.intercept = Y[0]

    def predict(self, key):
        return self.slope * key + self.intercept

class CubicSplineModel(Model):
    """
    Simulates a Cubic Spline segment.
    Uses numpy.polyfit(deg=3) to approximate the segment.
    """
    def __init__(self):
        self.coeffs = None # ax^3 + bx^2 + cx + d

    def train(self, X, Y):
        if len(X) < 4:
            # Fallback to linear if not enough points
            self.coeffs = np.polyfit(X, Y, 1)
            # Pad with zeros for consistency? No, handled in predict
        else:
            self.coeffs = np.polyfit(X, Y, 3)

    def predict(self, key):
        if self.coeffs is None: return 0
        return np.polyval(self.coeffs, key)

class RadixLayer(Model):
    """
    Partitions keys based on significant bits (Prefix).
    This is effectively a directory / lookup table layer.
    """
    def __init__(self, num_bits=10):
        self.num_bits = num_bits
        self.shift = 0

    def train(self, X, Y):
        if len(X) == 0: return
        max_val = np.max(X)
        # Calculate how much to shift to get the top `num_bits`
        # Assuming X are integers.
        # Determine bit width of max_val
        width = int(max_val).bit_length()
        self.shift = max(0, width - self.num_bits)

    def predict(self, key):
        # Return the index of the next model
        return int(key) >> self.shift
