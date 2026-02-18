import torch
import peft_torch
import time

# Set fixed seed
torch.manual_seed(42)

def generate_dataset(num_samples, input_dim, output_dim):
    true_weights = torch.rand(output_dim, input_dim) * 2 - 1

    # Generate all data at once (Vectorized generation)
    X = torch.rand(num_samples, input_dim) * 2 - 1
    Y = torch.mm(X, true_weights.t())
    noise = (torch.rand(num_samples, output_dim) * 2 - 1) * 0.01
    Y += noise

    return X, Y

def train_model():
    input_dim = 64
    output_dim = 64
    rank = 8
    num_samples = 2000 # Increased samples to make the speedup evident
    epochs = 10
    batch_size = 32
    lr = 0.01

    print("--- PyTorch Optimized Benchmark (Batched) ---")
    print(f"Task: Multivariate Regression")
    print(f"Samples: {num_samples}, Epochs: {epochs}, Batch Size: {batch_size}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    X, Y = generate_dataset(num_samples, input_dim, output_dim)
    X = X.to(device)
    Y = Y.to(device)

    model = peft_torch.TurboFusion(input_dim, output_dim, rank).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    print("Training TurboFusion...")
    start_time = time.time()

    model.train()
    for epoch in range(epochs):
        # Mini-batch loop
        permutation = torch.randperm(X.size(0))
        epoch_loss = 0.0

        for i in range(0, X.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X[indices], Y[indices]

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # print(f"Epoch {epoch+1} Loss: {epoch_loss / (num_samples/batch_size)}")

    end_time = time.time()
    duration = end_time - start_time

    print(f"Result: Time = {duration:.5f}s")

    # Calculate throughput
    total_samples = num_samples * epochs
    throughput = total_samples / duration
    print(f"Throughput: {throughput:.2f} samples/sec")

if __name__ == "__main__":
    train_model()
