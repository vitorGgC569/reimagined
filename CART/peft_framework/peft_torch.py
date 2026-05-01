import torch
import peft_torch_backend

class TurboFusion(torch.nn.Module):
    """
    TurboFusion Layer: The 'Trump Card' of PEFT.
    Combines DoRA (Weight Decomposition) and IA3 (Learned Vector Scaling).

    Backend: Optimized C++ Extension via LibTorch.
    Supports: CPU, CUDA (GPU), ROCm natively.
    """
    def __init__(self, in_features, out_features, rank=8):
        super().__init__()
        # Initialize the C++ module
        self.core = peft_torch_backend.TurboFusionLinear(in_features, out_features, rank)
        self.current_rank = rank

    def forward(self, x):
        return self.core.forward(x)

    def resize_rank(self, new_rank):
        """
        Dynamically resizes the rank of the internal adapters.
        Note: You MUST re-create your optimizer after calling this,
        as parameter shapes change.
        """
        self.core.resize_rank(new_rank)
        self.current_rank = new_rank

    @staticmethod
    def estimate_optimal_rank(gradient, threshold=1e-4):
        """
        Estimates the optimal rank based on the singular value decomposition (SVD)
        of the gradient matrix. (Logic inspired by MiSS).

        Args:
            gradient (torch.Tensor): The gradient of the weight matrix (out, in).
            threshold (float): Singular value threshold to consider 'important'.

        Returns:
            int: The estimated rank.
        """
        # SVD is expensive, so we use it carefully.
        # U, S, Vh = torch.linalg.svd(gradient, full_matrices=False)
        # Using svdvals is faster if we only need singular values
        if gradient.dim() > 2:
            # Handle batched gradients if necessary, or just mean
            gradient = gradient.mean(dim=0)

        S = torch.linalg.svdvals(gradient.float()) # Ensure float for svd

        # Count values > threshold
        # Normalize S to be relative to max singular value for better thresholding
        if S[0] > 0:
            S = S / S[0]

        new_rank = (S > threshold).sum().item()
        return max(1, int(new_rank))
