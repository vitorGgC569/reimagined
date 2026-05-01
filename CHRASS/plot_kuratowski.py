import matplotlib.pyplot as plt
import numpy as np

def plot_kuratowski():
    try:
        with open("kuratowski_map.txt", "r") as f:
            side = int(f.readline())
            data = list(map(int, f.read().split()))
    except:
        print("No map file found.")
        return

    grid = np.array(data).reshape(side, side)

    # Mask empty (-1)
    # Nets 0-8.

    plt.figure(figsize=(10, 10))

    # Custom colormap
    cmap = plt.cm.get_cmap('tab10', 10)

    # Plot masked array
    masked_grid = np.ma.masked_where(grid == -1, grid)
    plt.imshow(masked_grid, cmap=cmap, vmin=-0.5, vmax=9.5)

    # Terminals
    H = [(20, 20), (20, 50), (20, 80)] # (Row, Col) -> in plot (x=Col, y=Row)
    S = [(80, 20), (80, 50), (80, 80)]

    for i, (r, c) in enumerate(H):
        plt.plot(c, r, 's', markersize=15, markeredgecolor='black', label=f'H{i+1}')
        plt.text(c, r-3, f'H{i+1}', ha='center')

    for i, (r, c) in enumerate(S):
        plt.plot(c, r, '^', markersize=15, markeredgecolor='black', label=f'S{i+1}')
        plt.text(c, r+5, f'S{i+1}', ha='center')

    plt.title(f"Kuratowski Challenge (K3,3 on 2D Grid)")
    plt.colorbar(ticks=range(9), label='Net ID')
    plt.grid(True, which='both', color='lightgray', linestyle='--', linewidth=0.5)
    plt.savefig('kuratowski_result.png')
    print("Saved kuratowski_result.png")

if __name__ == "__main__":
    plot_kuratowski()
