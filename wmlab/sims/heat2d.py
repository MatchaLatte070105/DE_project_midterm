"""Truyền nhiệt 2D có nguồn, biên cách nhiệt (Neumann):  u_t = alpha (u_xx + u_yy) + q(x, y).
params: alpha, Lx, Ly, nx, ny, sources = [{x, y, sigma, power}, ...]
initial: T0 (nhiệt độ đồng đều ban đầu)
Sơ đồ FTCS, ổn định khi r = alpha dt / dx^2 <= 1/4 (lưới vuông).
Biên cách nhiệt => tổng nhiệt chỉ thay đổi đúng bằng tích phân nguồn (kiểm tra 'heat_balance').
Y có dạng (số bước, ny*nx). Hàm sensor_image() mô phỏng ảnh từ camera/vệ tinh nhiệt."""
import numpy as np

CONVERGENCE_MODE = "self"
DEFAULT_SCHEME = "ftcs"
STABILITY_LIMIT = 0.25


def grid(p):
    x = np.linspace(0, p.get("Lx", 1.0), int(p.get("nx", 41)))
    y = np.linspace(0, p.get("Ly", 1.0), int(p.get("ny", 41)))
    return np.meshgrid(x, y)


def source(p):
    X, Y = grid(p)
    q = np.zeros_like(X)
    for s in p.get("sources", []):
        g = np.exp(-((X - s["x"]) ** 2 + (Y - s["y"]) ** 2) / (2 * s["sigma"] ** 2))
        q += s["power"] * g
    return q


def stability_number(p, dt):
    X, _ = grid(p)
    dx = X[0, 1] - X[0, 0]
    return p["alpha"] * dt / dx ** 2


def simulate(p, init, t_span, dt):
    X, _ = grid(p)
    dx = X[0, 1] - X[0, 0]
    r = p["alpha"] * dt / dx ** 2
    q = source(p)
    u = np.full_like(X, float(init.get("T0", 0.0)))
    n = int(round((t_span[1] - t_span[0]) / dt))
    t = t_span[0] + dt * np.arange(n + 1)
    out = np.empty((n + 1, u.size))
    out[0] = u.ravel()
    for i in range(n):
        g = np.pad(u, 1, mode="edge")  # ô ảo bằng giá trị biên => gradient pháp tuyến bằng 0
        lap = g[1:-1, 2:] + g[1:-1, :-2] + g[2:, 1:-1] + g[:-2, 1:-1] - 4 * u
        u = u + r * lap + dt * q
        out[i + 1] = u.ravel()
    return t, out


def analytic(p, init, t):
    if p.get("sources"):
        return None
    X, _ = grid(p)
    return np.full((len(t), X.size), float(init.get("T0", 0.0)))


def heat_balance(p, init, t, Y):
    """Sai số tương đối giữa thay đổi tổng nhiệt và tích phân nguồn theo thời gian."""
    q = source(p).ravel()
    change = Y[-1].sum() - Y[0].sum()
    expected = q.sum() * (t[-1] - t[0])
    return abs(change - expected) / max(abs(expected), 1e-12)


def quantities(p, init, t, Y):
    X, _ = grid(p)
    u = Y[-1].reshape(X.shape)
    return {"max_temperature": float(u.max()), "mean_temperature": float(u.mean()),
            "hotspot_contrast": float(u.max() - np.median(u)),
            "stability_number": float(stability_number(p, t[1] - t[0]))}


def sensor_image(p, frame, block=4, noise=0.05, emissivity=1.0, seed=0):
    """Ảnh 'đo được': nhân độ phát xạ, gộp khối (độ phân giải thấp hơn), cộng nhiễu."""
    X, _ = grid(p)
    u = emissivity * frame.reshape(X.shape)
    h, w = (u.shape[0] // block) * block, (u.shape[1] // block) * block
    img = u[:h, :w].reshape(h // block, block, w // block, block).mean(axis=(1, 3))
    return img + np.random.default_rng(seed).normal(0, noise, img.shape)


def sample(rng):
    n_src = int(rng.integers(1, 4))
    p = {"alpha": float(rng.uniform(0.005, 0.02)), "Lx": 1.0, "Ly": 1.0, "nx": 41, "ny": 41,
         "sources": [{"x": float(rng.uniform(0.2, 0.8)), "y": float(rng.uniform(0.2, 0.8)),
                      "sigma": float(rng.uniform(0.02, 0.08)), "power": float(rng.uniform(1, 5))}
                     for _ in range(n_src)]}
    return p, {"T0": 0.0}, [0.0, 2.0], 0.005
