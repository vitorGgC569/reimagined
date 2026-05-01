import matplotlib.pyplot as plt
import numpy as np

def plot_matrix_bug():
    try:
        with open("matrix_bug_map.txt", "r") as f:
            side = int(f.readline())
            data = list(map(int, f.read().split()))
    except:
        print("No map file found.")
        return

    grid = np.array(data).reshape(side, side)

    plt.figure(figsize=(10, 10))

    # Custom colormap
    # -2 = Bug (Red)
    # -1 = Empty (White)
    # 0-8 = Nets (Colors)
    cmap = plt.cm.get_cmap('tab10', 10)
    cmap.set_under('white') # -1
    cmap.set_over('red')    # -2

    # We need to map values to ranges:
    # -2 -> 10 (Over)
    # -1 -> -1 (Under)
    # 0-8 -> 0-8
    plot_grid = np.copy(grid)
    plot_grid[grid == -2] = 10

    plt.imshow(plot_grid, cmap=cmap, vmin=-0.5, vmax=9.5)

    # Terminals
    H = [(20, 20), (20, 50), (20, 80)]
    S = [(80, 20), (80, 50), (80, 80)]

    for i, (r, c) in enumerate(H):
        plt.plot(c, r, 's', markersize=15, markeredgecolor='black', label=f'H{i+1}')
    for i, (r, c) in enumerate(S):
        plt.plot(c, r, '^', markersize=15, markeredgecolor='black', label=f'S{i+1}')

    plt.title(f"Matrix Bug Challenge (9/9 Routed via Reality Distortion)")

    # Highlight Bugs explicitly
    bugs = np.argwhere(grid == -2)
    if len(bugs) > 0:
        plt.scatter(bugs[:, 1], bugs[:, 0], c='red', s=50, marker='x', label='Reality Distortion (Crossing)')

    plt.legend(loc='upper right')
    plt.grid(False)
    plt.savefig('matrix_bug_result.png')
    print("Saved matrix_bug_result.png")

if __name__ == "__main__":
    plot_matrix_bug()
