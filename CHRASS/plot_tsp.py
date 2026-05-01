import matplotlib.pyplot as plt

def load_tour(filename):
    x, y = [], []
    try:
        with open(filename, 'r') as f:
            for line in f:
                parts = line.split()
                x.append(int(parts[0]))
                y.append(int(parts[1]))
    except: pass
    return x, y

def plot_tours():
    x_spat, y_spat = load_tour("tour_spatial.txt")
    x_spec, y_spec = load_tour("tour_spectral.txt")

    if not x_spat: return

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    # Spatial
    axs[0].plot(x_spat, y_spat, linewidth=0.5, alpha=0.8, color='blue')
    axs[0].set_title("Spatial Sort (Naive X+Y)")
    axs[0].set_aspect('equal')
    axs[0].grid(False)

    # Spectral
    axs[1].plot(x_spec, y_spec, linewidth=0.5, alpha=0.8, color='red')
    axs[1].set_title("Spectral Sort (Chebyshev Heat)")
    axs[1].set_aspect('equal')
    axs[1].grid(False)

    plt.suptitle("CHRASS TSP: The Topology of Intelligence")
    plt.savefig("tsp_comparison.png")
    print("Saved tsp_comparison.png")

if __name__ == "__main__":
    plot_tours()
