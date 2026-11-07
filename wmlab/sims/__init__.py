"""Simulator tham chiếu. Mỗi module cung cấp:
    simulate(p, init, t_span, dt) -> (t, Y)
    analytic(p, init, t)          -> Y hoặc None (khi không có nghiệm giải tích)
    quantities(p, init, t, Y)     -> dict các đại lượng có thể hỏi trong 'question'
    sample(rng)                   -> (p, init, t_span, dt) ngẫu nhiên để sinh bộ dữ liệu
và tuỳ chọn energy(p, Y).
Sinh viên thêm hệ mới bằng cách viết một module cùng giao diện và đăng ký vào REGISTRY."""
from ._integ import rk4  # noqa: F401
from . import projectile_drag, oscillator, heat1d, lorenz, langevin, lorentz, heat2d

REGISTRY = {
    "projectile_drag": projectile_drag,
    "oscillator": oscillator,
    "heat1d": heat1d,
    "lorenz": lorenz,
    "langevin": langevin,
    "lorentz": lorentz,
    "heat2d": heat2d,
}
