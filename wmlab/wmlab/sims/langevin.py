"""Dao động điều hoà Langevin (có quán tính):
    dx = v dt,   m dv = (-k x - gamma v) dt + sqrt(2 gamma kT) dW  [+ nhiễu ngoài cân bằng tuỳ chọn]
params: m, k, gamma, kT, seed, active (cường độ nhiễu "hoạt động" không kèm ma sát; 0 = cân bằng)
initial: x0, v0
Ở cân bằng (active=0): phân bố đều năng lượng <k x^2> = <m v^2> = kT, và hệ thức
dao động–đáp ứng (FDT) thoả mãn: tỉ số X = 1. Khi active > 0, X lệch khỏi 1."""
import numpy as np

STATE = ["x", "v"]


def _run(p, x, v, n, dt, rng, force=None, start=0):
    """Euler–Maruyama vector hoá cho nhiều quỹ đạo. force: lực ngoài h bật từ bước 'start'."""
    m, k, g, kT = p["m"], p["k"], p["gamma"], p["kT"]
    act = p.get("active", 0.0)
    amp = np.sqrt(2 * g * kT * dt) / m
    xs = np.empty((n + 1,) + np.shape(x))
    xs[0] = x
    for i in range(n):
        noise = rng.standard_normal(np.shape(x))
        a = (-k * x - g * v) / m
        if force is not None and i >= start:
            a = a + force / m
        v = v + a * dt + amp * noise
        if act:
            v = v + np.sqrt(2 * act * dt) / m * rng.standard_normal(np.shape(x))
        x = x + v * dt
        xs[i + 1] = x
    return xs, x, v


def simulate(p, init, t_span, dt):
    rng = np.random.default_rng(p.get("seed", 0))
    n = int(round((t_span[1] - t_span[0]) / dt))
    m, k, g, kT = p["m"], p["k"], p["gamma"], p["kT"]
    act = p.get("active", 0.0)
    amp = np.sqrt(2 * g * kT * dt) / m
    t = t_span[0] + dt * np.arange(n + 1)
    Y = np.empty((n + 1, 2))
    x, v = float(init["x0"]), float(init["v0"])
    Y[0] = x, v
    for i in range(n):
        v = v + (-k * x - g * v) / m * dt + amp * rng.standard_normal()
        if act:
            v = v + np.sqrt(2 * act * dt) / m * rng.standard_normal()
        x = x + v * dt
        Y[i + 1] = x, v
    return t, Y


def analytic(p, init, t):
    return None


def equipartition(p, Y, burn=0.2):
    s = Y[int(burn * len(Y)):]
    return {"kx2_over_kT": float(p["k"] * np.mean(s[:, 0] ** 2) / p["kT"]),
            "mv2_over_kT": float(p["m"] * np.mean(s[:, 1] ** 2) / p["kT"])}


def fdt_ratio(p, n_traj=4000, dt=0.002, t_eq=20.0, tau_max=3.0, h=0.05, seed=1):
    """Thí nghiệm hai quỹ đạo (cùng nhiễu): một nhánh bật lực bậc thang h tại t_w.
    So đáp ứng chi(tau) = d<x>/dh với [C(0) - C(tau)] / kT. Trả về hệ số X (=1 khi FDT đúng)."""
    rng = np.random.default_rng(seed)
    m, k = p["m"], p["k"]
    x = rng.normal(0, np.sqrt(p["kT"] / k), n_traj)
    v = rng.normal(0, np.sqrt(p["kT"] / m), n_traj)
    _, x, v = _run(p, x, v, int(t_eq / dt), dt, rng)  # đưa về trạng thái dừng
    n = int(tau_max / dt)
    state = rng.bit_generator.state
    free, _, _ = _run(p, x.copy(), v.copy(), n, dt, rng)
    rng.bit_generator.state = state  # dùng lại đúng chuỗi nhiễu cho nhánh có lực
    pert, _, _ = _run(p, x.copy(), v.copy(), n, dt, rng, force=h, start=0)
    chi = (pert - free).mean(axis=1) / h
    C = (free * free[0]).mean(axis=1)
    rhs = (C[0] - C) / p["kT"]
    X = float(np.dot(chi, rhs) / np.dot(rhs, rhs))
    return X, chi, rhs


def quantities(p, init, t, Y):
    q = equipartition(p, Y)
    q["fdt_X"] = fdt_ratio(p)[0]
    return q


def sample(rng):
    p = {"m": 1.0, "k": float(rng.uniform(0.5, 4)), "gamma": float(rng.uniform(0.2, 2)),
         "kT": float(rng.uniform(0.5, 2)), "seed": int(rng.integers(1e9)),
         "active": float(rng.choice([0.0, rng.uniform(0.2, 1.0)]))}
    return p, {"x0": 0.0, "v0": 0.0}, [0.0, 50.0], 0.005
