import torch
import peft_torch
import time
import math

# Set fixed seed
torch.manual_seed(42)

def generate_dataset(num_samples, input_dim, output_dim):
    true_weights = torch.rand(output_dim, input_dim) * 2 - 1
    X = torch.rand(num_samples, input_dim) * 2 - 1
    Y = torch.mm(X, true_weights.t())
    noise = (torch.rand(num_samples, output_dim) * 2 - 1) * 0.01
    Y += noise
    return X, Y

def train_dynamic_model():
    input_dim = 64
    output_dim = 64
    initial_rank = 8
    num_samples = 2000
    epochs = 20 # More epochs to allow rank adaptation
    batch_size = 32
    lr = 0.01
    svd_interval = 5 # Check rank every 5 epochs

    print("--- Dynamic TurboFusion Benchmark (MiSS Logic) ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    X, Y = generate_dataset(num_samples, input_dim, output_dim)
    X = X.to(device)
    Y = Y.to(device)

    # Start with a conservative rank
    model = peft_torch.TurboFusion(input_dim, output_dim, initial_rank).to(device)

    # We need a function to re-create optimizer because parameters change
    def create_optimizer(m):
        return torch.optim.AdamW(m.parameters(), lr=lr)

    optimizer = create_optimizer(model)
    loss_fn = torch.nn.MSELoss()

    print(f"Initial Rank: {initial_rank}")
    start_time = time.time()

    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(X.size(0))
        epoch_loss = 0.0

        # Accumulate gradients for SVD check?
        # Actually, MiSS checks gradient of the *weight*.
        # In LoRA, we approximate weight grad via A and B grads or just check AB grad.
        # Simplest proxy: Check singular values of the learned update (BA) or just one of the matrices.
        # MiSS paper usually checks the gradient of the *frozen* weights if we were fine-tuning them,
        # but here we are training adapters.
        # "Adapting rank based on gradient importance" -> usually implies checking grad of the adapter or the target layer.
        # Let's check the effective weight gradient.
        # dL/dW_eff approx dL/dW_pre * ...
        # Let's use the accumulated gradient on 'lora_A' (r x in) or 'lora_B' (out x r) as proxy?
        # Or better: construct the full gradient matrix (out x in) conceptually? Too expensive.
        # Heuristic: Check singular values of `lora_B @ lora_A` (the delta W).
        # If it has low rank structure, we can prune?
        # Actually, `estimate_optimal_rank` takes a gradient.
        # Let's pass the gradient of the output projection (lora_B) as a proxy for "how many dimensions of rank we are using".

        accumulated_grad_B = None

        for i in range(0, X.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X[indices], Y[indices]

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Capture grad of B for SVD check (just from last batch is noisy, but maybe ok for demo)
            # Accessing parameter via name or traversal
            if (epoch + 1) % svd_interval == 0 and i == 0:
                 for name, param in model.named_parameters():
                     if "lora_B" in name and param.grad is not None:
                         accumulated_grad_B = param.grad.clone()

        avg_loss = epoch_loss / (num_samples/batch_size)

        # Dynamic Rank Adjustment Logic
        if (epoch + 1) % svd_interval == 0 and accumulated_grad_B is not None:
            # Check rank of lora_B's gradient (out x r).
            # If gradients lie in a smaller subspace, we can reduce rank.
            # If gradients fill the rank, maybe we need more?
            # Standard MiSS prunes. Let's try to find optimal rank.

            # We use a loose threshold to encourage finding structure
            suggested_rank = peft_torch.TurboFusion.estimate_optimal_rank(accumulated_grad_B, threshold=0.01)

            # Constraint: Don't grow too wild, don't vanish.
            suggested_rank = min(max(1, suggested_rank), 32)

            if suggested_rank != model.current_rank:
                print(f"[Epoch {epoch+1}] Resizing Rank: {model.current_rank} -> {suggested_rank}")
                model.resize_rank(suggested_rank)
                # Important: Optimizer must be reset because param shapes changed
                optimizer = create_optimizer(model)
            else:
                print(f"[Epoch {epoch+1}] Rank Stable: {model.current_rank}")

    end_time = time.time()
    duration = end_time - start_time

    print(f"Final Result: Time = {duration:.5f}s, Loss = {avg_loss:.5f}, Final Rank = {model.current_rank}")

if __name__ == "__main__":
    train_dynamic_model()
