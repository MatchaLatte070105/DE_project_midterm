"""Đồ án AMCS — điểm xuất phát: học một "world model" (bộ dự báo một bước) từ dữ liệu
mô phỏng, rồi KIỂM CHỨNG nó bằng định luật vật lý thay vì chỉ nhìn loss.

Hệ: dao động điều hoà không cản (c = 0) -> năng lượng phải bảo toàn.
So sánh hai mô hình:
  (A) hồi quy tuyến tính  — đúng cấu trúc của hệ tuyến tính
  (B) MLP 2-64-64-2 (numpy thuần, Adam) — mô hình "hộp đen"
Cả hai đều có loss một bước nhỏ. Câu hỏi của đồ án: khi chạy dài (rollout),
mô hình nào còn tôn trọng bảo toàn năng lượng và pha dao động?
Chạy:  python examples/amcs_surrogate_starter.py
"""
import numpy as np

from wmlab.sims import oscillator

rng = np.random.default_rng(0)
P = {"m": 1.0, "c": 0.0, "k": 4.0, "F0": 0.0, "w": 0.0}
DT = 0.05

# 1) Sinh dữ liệu: nhiều quỹ đạo ngắn, cặp (s_t, s_{t+1})
X, Yn = [], []
for _ in range(200):
    init = {"x0": rng.uniform(-1, 1), "v0": rng.uniform(-2, 2)}
    _, S = oscillator.simulate(P, init, [0, 2.0], DT)
    X.append(S[:-1]); Yn.append(S[1:])
X, Yn = np.concatenate(X), np.concatenate(Yn)

# 2A) Hồi quy tuyến tính s_{t+1} = A s_t
A = np.linalg.lstsq(X, Yn, rcond=None)[0].T

# 2B) MLP dự báo phần dư s_{t+1} - s_t
def init_layer(i, o):
    return rng.normal(0, np.sqrt(1 / i), (i, o)), np.zeros(o)

W = [init_layer(2, 64), init_layer(64, 64), init_layer(64, 2)]
m_ = [[np.zeros_like(w), np.zeros_like(b)] for w, b in W]
v_ = [[np.zeros_like(w), np.zeros_like(b)] for w, b in W]

def forward(x):
    hs = [x]
    for i, (w, b) in enumerate(W):
        x = x @ w + b
        if i < len(W) - 1:
            x = np.tanh(x)
        hs.append(x)
    return x, hs

target = Yn - X
lr, b1, b2 = 1e-3, 0.9, 0.999
for step in range(1, 4001):
    idx = rng.integers(0, len(X), 256)
    out, hs = forward(X[idx])
    g = 2 * (out - target[idx]) / len(idx)
    for i in reversed(range(len(W))):
        w, b = W[i]
        gw, gb = hs[i].T @ g, g.sum(0)
        if i > 0:
            g = (g @ w.T) * (1 - hs[i] ** 2)
        for j, gr in enumerate((gw, gb)):
            m_[i][j] = b1 * m_[i][j] + (1 - b1) * gr
            v_[i][j] = b2 * v_[i][j] + (1 - b2) * gr ** 2
            mh, vh = m_[i][j] / (1 - b1 ** step), v_[i][j] / (1 - b2 ** step)
            upd = lr * mh / (np.sqrt(vh) + 1e-8)
            W[i] = (W[i][0] - upd, W[i][1]) if j == 0 else (W[i][0], W[i][1] - upd)
mse_mlp = np.mean((forward(X)[0] - target) ** 2)
mse_lin = np.mean((X @ A.T - Yn) ** 2)

# 3) Kiểm chứng bằng vật lý: rollout dài từ một điều kiện đầu mới
init = {"x0": 0.8, "v0": 0.0}
n = 2000  # 100 giây ~ 32 chu kỳ
_, truth = oscillator.simulate(P, init, [0, n * DT], DT)
s_lin = np.empty((n + 1, 2)); s_mlp = np.empty((n + 1, 2))
s_lin[0] = s_mlp[0] = truth[0]
for i in range(n):
    s_lin[i + 1] = A @ s_lin[i]
    s_mlp[i + 1] = s_mlp[i] + forward(s_mlp[i][None])[0][0]

E = lambda S: oscillator.energy(P, S)
for name, S, mse in [("Tuyến tính", s_lin, mse_lin), ("MLP", s_mlp, mse_mlp)]:
    drift = np.abs(E(S) - E(truth)[0]).max() / E(truth)[0]
    err = np.abs(S - truth).max()
    print(f"{name:<11} loss 1 bước={mse:.2e} | trôi năng lượng sau 100 s={drift:.2%} | sai số quỹ đạo lớn nhất={err:.3f}")
print("\nGợi ý cho đồ án: thêm ràng buộc (mạng Hamiltonian, tích phân symplectic, phạt vi phạm năng lượng),"
      " rồi lặp lại cùng phép kiểm chứng. Báo cáo bằng chỉ số vật lý, không chỉ loss.")
