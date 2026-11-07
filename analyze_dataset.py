import numpy as np
import matplotlib.pyplot as plt
from bungee_jump import quantities, reference_quantities, energy

# Load dataset
data = np.load("bungee_jump.npz", allow_pickle=True)
t = data["t"]
Y = data["Y"]
meta_str = str(data["meta"]).replace("true", "True").replace("false", "False")
metas = eval(meta_str)

print("Number of trajectories:", Y.shape[0])
print("Time steps per trajectory:", Y.shape[1])

x = Y[:, :, 0]
v = Y[:, :, 1]
lowest_points = np.max(x, axis=1)

print("\n=== Lowest point statistics (200 jumps) ===")
print(f"Mean:   {lowest_points.mean():.3f} m")
print(f"Std:    {lowest_points.std():.3f} m")
print(f"Min:    {lowest_points.min():.3f} m")
print(f"Max:    {lowest_points.max():.3f} m")

# No-drag cases
cd0_idx = [i for i, m in enumerate(metas) if m["params"].get("c_d", 0) == 0]
print(f"\nNo-drag cases: {len(cd0_idx)}")

diffs = []
for i in cd0_idx:
    p = metas[i]["params"]
    init = metas[i]["initial"]
    ref = reference_quantities(p, init)
    if ref:
        diffs.append(abs(lowest_points[i] - ref["lowest_point"]))

if diffs:
    print(f"Mean |sim - ref|: {np.mean(diffs):.4f} m")
    print(f"Max  |sim - ref|: {np.max(diffs):.4f} m")

# Density plot
from scipy.stats import gaussian_kde
density = gaussian_kde(lowest_points)
xs = np.linspace(lowest_points.min(), lowest_points.max(), 300)

plt.figure(figsize=(8, 4))
plt.plot(xs, density(xs), linewidth=2)
plt.fill_between(xs, density(xs), alpha=0.25)
plt.xlabel("Lowest point (m)")
plt.ylabel("Density")
plt.title("Distribution of lowest points over 200 random jumps")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Sample trajectories
plt.figure(figsize=(9, 4))
for i in range(5):
    plt.plot(t, x[i], alpha=0.8, label=f"jump {i+1}")
plt.xlabel("t (s)")
plt.ylabel("x (m)")
plt.title("Sample trajectories (first 5 jumps)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()