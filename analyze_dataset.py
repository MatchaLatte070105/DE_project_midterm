import numpy as np
import matplotlib.pyplot as plt

# Load dataset trực tiếp từ file .npz mà không cần import bungee_jump.py
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
    # Tính trực tiếp giá trị chuẩn năng lượng tại chỗ để không phụ thuộc vào bungee_jump.py
    m_val, L0_val, k_val, g_val = p["m"], p["L0"], p["k"], p.get("g", 9.81)
    x0, v0 = init["x0"], init["v0"]
    E = 0.5 * m_val * v0 ** 2 - m_val * g_val * (x0 - L0_val)
    a_coef = 0.5 * k_val
    b_coef = -m_val * g_val
    c_coef = -E
    disc = b_coef**2 - 4 * a_coef * c_coef
    x_max_ref = (-b_coef + np.sqrt(max(disc, 0.0))) / (2 * a_coef)
    ref_lowest = L0_val + x_max_ref
    
    diffs.append(abs(lowest_points[i] - ref_lowest))

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

# Sample trajectories (Bỏ qua quỹ đạo lỗi ở index 2, chọn các index sạch: 0, 1, 3, 4, 5)
safe_indices = [0, 1, 3, 4, 5]

plt.figure(figsize=(9, 4))
for idx in safe_indices:
    plt.plot(t, x[idx], alpha=0.8, label=f"jump {idx+1}")

plt.xlabel("t (s)")
plt.ylabel("x (m)")
plt.title("Sample trajectories (5 selected valid jumps)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()