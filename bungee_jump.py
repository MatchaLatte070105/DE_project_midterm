"""Bungee jump — free fall then piecewise-linear elastic rope (optional quadratic drag).

    m x'' = m g                           when x <= L0   (free fall / slack rope)
    m x'' = m g - k (x - L0) - c_d v |v|  when x > L0    (rope stretched)

params : m [kg] jumper mass, L0 [m] unstretched rope length, k [N/m] rope stiffness,
         c_d [kg/m] quadratic drag coefficient, g [m/s^2] gravity,
         method [str] integrator ("euler"|"heun"|"rk4"), event_handling [bool] land on x = L0
initial: x0 [m] initial downward position, v0 [m/s] initial downward velocity
Analytic solution: only when c_d = 0 (exact lowest point and peak deceleration via energy
                   conservation; piecewise closed-form trajectory in each phase)

Group: 7 (B4 - Bungee Jump) 
"""
import numpy as np

NAME = "bungee_jump"          # used as system: in scenario files
STATE = ["x", "v"]            # state variables in the order of vector y = [x, v]
CONVERGENCE_MODE = "self"     # self-convergence across step sizes (covers both c_d = 0 and c_d > 0)
INVARIANT_NAME = "energy"     # conserved mechanical energy when c_d = 0
x is measured downward from the platform [m]:
    m * x'' = m * g                         when x <= L0   (free fall / slack rope)
    m * x'' = m * g - k*(x - L0) - c_d*v*|v| when x > L0   (rope stretched)

Group: <??????>
"""
import numpy as np

NAME = "bungee_jump"
STATE = ["x", "v"]
CONVERGENCE_MODE = "self"
INVARIANT_NAME = "energy"


# 1. THE MODEL
def rhs(t, y, p):
    """Right-hand side f(t, y) of the first-order system y' = f(t, y).
    A second-order equation x'' = g(t, x, x') must be rewritten as y = [x, v], y' = [v, a]."""
    """Right-hand side of the first-order system y' = f(t, y), y = [x, v]."""
    x, v = float(y[0]), float(y[1])
    m = p["m"]
    g = p.get("g", 9.81)
    L0 = p["L0"]
    k = p["k"]
    c_d = p.get("c_d", 0.0)

    # Piecewise equation of motion using if/else for slack (x <= L0) vs. taut (x > L0) rope
    if x <= L0:
        a = g - (c_d / m) * v * abs(v)
    else:
        a = g - (k / m) * (x - L0) - (c_d / m) * v * abs(v)

    return np.array([v, a])


# 2. NUMERICAL METHODS (provided)
    a = g
    if x > L0:
        a -= (k / m) * (x - L0)
    if c_d > 0.0:
        a -= (c_d / m) * v * abs(v)
    return np.array([v, a])


# 2. NUMERICAL METHODS
def _step(f, t, y, h, p, method):
    if method == "euler":
        return y + h * f(t, y, p)
    if method == "heun":
        k1 = f(t, y, p)
        k2 = f(t + h, y + h * k1, p)
        return y + h / 2 * (k1 + k2)
    k1 = f(t, y, p)
    k2 = f(t + h / 2, y + h / 2 * k1, p)
    k3 = f(t + h / 2, y + h / 2 * k2, p)
    k4 = f(t + h, y + h * k3, p)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def _find_taut_time(t, y, dt, p, method):
    """Bisection search for the exact instant within [t, t + dt] when x crosses L0."""
    L0 = p["L0"]
    h_lo, h_hi = 0.0, dt
    going_down = y[0] <= L0

    for _ in range(52):
        h_mid = 0.5 * (h_lo + h_hi)
        y_mid = _step(rhs, t, y, h_mid, p, method)
        if (going_down and y_mid[0] < L0) or (not going_down and y_mid[0] > L0):
            h_lo = h_mid
        else:
            h_hi = h_mid

    y_cross = _step(rhs, t, y, h_hi, p, method)
    y_cross[0] = L0 + (1e-15 if going_down else -1e-15)
    return t + h_hi, y_cross


def simulate(p, init, t_span, dt):
    """Returns t (n+1,) and Y (n+1, len(STATE))."""
        return y + 0.5 * h * (k1 + k2)
    # classical RK4
    k1 = f(t, y, p)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1, p)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2, p)
    k4 = f(t + h, y + h * k3, p)
    return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _find_taut_time(t, y, dt, p, method):
    """Binary-search the exact time when the rope becomes taut (x crosses L0)."""
    L0 = p["L0"]
    t_lo, y_lo = t, y.copy()
    t_hi = t + dt
    y_hi = _step(rhs, t, y, dt, p, method)

    for _ in range(40):
        h_mid = 0.5 * (t_hi - t_lo)
        y_mid = _step(rhs, t_lo, y_lo, h_mid, p, method)
        t_mid = t_lo + h_mid
        if y_mid[0] < L0:
            t_lo, y_lo = t_mid, y_mid
        else:
            t_hi, y_hi = t_mid, y_mid
        if abs(y_mid[0] - L0) < 1e-12:
            break
    y_hi[0] = L0
    return t_hi, y_hi


def simulate(p, init, t_span, dt):
    """Integrate. When event_handling=True the step lands exactly on the taut instant."""
    n = int(round((t_span[1] - t_span[0]) / dt))
    t = t_span[0] + dt * np.arange(n + 1)
    Y = np.empty((n + 1, len(STATE)))
    Y[0] = [init["x0"], init["v0"]]
    method = p.get("method", "rk4")
    use_event = bool(p.get("event_handling", True))
    use_event = bool(p.get("event_handling", False))
    L0 = p["L0"]

    for i in range(n):
        y_next = _step(rhs, t[i], Y[i], dt, p, method)
        # Event handling at the kink x = L0 (restores RK4 4th-order convergence)
        if use_event:
            cross_down = (Y[i, 0] <= L0 < y_next[0])
            cross_up = (Y[i, 0] > L0 >= y_next[0])
            if cross_down or cross_up:
                t_cross, y_cross = _find_taut_time(t[i], Y[i], dt, p, method)
                h_rem = t[i] + dt - t_cross
                if h_rem > 1e-15:
                    y_next = _step(rhs, t_cross, y_cross, h_rem, p, method)
                else:
                    y_next = y_cross
        if use_event and Y[i, 0] <= L0 < y_next[0]:
            t_cross, y_cross = _find_taut_time(t[i], Y[i], dt, p, method)
            h_rem = t[i] + dt - t_cross
            if h_rem > 1e-15:
                y_next = _step(rhs, t_cross, y_cross, h_rem, p, method)
            else:
                y_next = y_cross
        Y[i + 1] = y_next
    return t, Y


#  3. ANALYTIC SOLUTION
def analytic(p, init, t):
    return None


#  4. QUANTITIES OF INTEREST
# 3. ANALYTIC / REFERENCE
def analytic(p, init, t):
    """No closed-form time history → return None (convergence uses mode='self')."""
    return None


def _energy_lowest_point(p, init):
    """Exact lowest point from mechanical energy (valid only when c_d = 0)."""
    m = p["m"]
    g = p.get("g", 9.81)
    L0 = p["L0"]
    k = p["k"]
    x0 = init["x0"]
    v0 = init["v0"]

    # Energy at release (elastic = 0)
    E = 0.5 * m * v0 ** 2 - m * g * (x0 - L0)

    # At bottom: v = 0, x = L0 + delta
    a = 0.5 * k
    b = -m * g
    c = -E
    disc = b * b - 4 * a * c
    delta = (-b + np.sqrt(max(disc, 0.0))) / (2 * a)
    return float(L0 + delta)


def reference_quantities(p, init):
    """Independent energy-based formulas (only when c_d = 0)."""
    if p.get("c_d", 0.0) != 0.0:
        return {}
    m = p["m"]
    g = p.get("g", 9.81)
    k = p["k"]
    lowest = _energy_lowest_point(p, init)
    x_max = lowest - p["L0"]
    max_decel_g = (k * x_max / m - g) / g
    return {
        "lowest_point": lowest,
        "max_stretch": float(x_max),
        "max_deceleration_g": float(max_decel_g),
    }


# 4. QUANTITIES OF INTEREST
def quantities(p, init, t, Y):
    x = Y[:, 0]
    v = Y[:, 1]
    L0 = p["L0"]
    m = p["m"]
    k = p["k"]
    g = p.get("g", 9.81)

    lowest_point = float(np.max(x))
    x_max = max(lowest_point - L0, 0.0)
    max_decel_g = float((k * x_max / m - g) / g)

    # Count local maxima of x that exceed L0 + 1 m when drag is present
    # count local maxima of x that exceed L0 + 1 m
    n_bounces = 0
    if p.get("c_d", 0.0) > 0.0:
        for i in range(1, len(x) - 1):
            if x[i] > x[i - 1] and x[i] > x[i + 1] and x[i] > L0 + 1.0:
                n_bounces += 1
        if n_bounces > 0:
            n_bounces -= 1  # last peak is settling

    return {
        "lowest_point": lowest_point,
        "max_stretch": float(x_max),
        "max_deceleration_g": max_decel_g,
        "n_bounces": int(n_bounces),
        "x_final": float(x[-1]),
        "v_final": float(v[-1]),
    }


def _energy_lowest_point(p, init):
    """Exact lowest point from mechanical energy conservation (valid when c_d = 0)."""
    m = p["m"]
    g = p.get("g", 9.81)
    L0 = p["L0"]
    k = p["k"]
    x0 = init["x0"]
    v0 = init["v0"]

    # Mechanical energy at release (with gravitational potential reference at x = L0)
    E = 0.5 * m * v0 ** 2 - m * g * (x0 - L0)

    # At the lowest point: v = 0, x = L0 + x_max -> 0.5 * k * x_max^2 - m * g * x_max - E = 0
    a = 0.5 * k
    b = -m * g
    c = -E
    disc = b * b - 4 * a * c
    x_max = (-b + np.sqrt(max(disc, 0.0))) / (2 * a)
    return float(L0 + x_max)


def reference_quantities(p, init):
    """OPTIONAL: the same quantities from an INDEPENDENT formula (exact energy method),
    used by the `reference_match` check. Return {} when no formula applies."""
    if p.get("c_d", 0.0) != 0.0:
        return {}
    m = p["m"]
    g = p.get("g", 9.81)
    k = p["k"]
    lowest = _energy_lowest_point(p, init)
    x_max = lowest - p["L0"]
    max_decel_g = (k * x_max / m - g) / g
    return {
        "lowest_point": lowest,
        "max_stretch": float(x_max),
        "max_deceleration_g": float(max_decel_g),
    }


#  5. PHYSICAL CONSTRAINTS
def energy(p, Y):
    """OPTIONAL: total energy along the trajectory (enables energy_conserved / energy_nonincreasing).
    Zero of gravitational potential is taken at x = L0 so E(0) = m * g * L0 > 0."""
    x, v = Y[:, 0], Y[:, 1]
# 5. PHYSICAL CONSTRAINTS
def energy(p, Y):
    """Mechanical energy. Zero of gravitational potential is at x = L0."""
    x = Y[:, 0]
    v = Y[:, 1]
    m = p["m"]
    g = p.get("g", 9.81)
    L0 = p["L0"]
    k = p["k"]
    stretch = np.maximum(x - L0, 0.0)
    return 0.5 * m * v ** 2 - m * g * (x - L0) + 0.5 * k * stretch ** 2


def invariant(p, Y):
    """OPTIONAL: any other conserved quantity."""
    return energy(p, Y)


def bounds(p):
    """OPTIONAL: {column index: (min, max)} for physically allowed values; None = no limit."""
    return {}


EXTRA_CHECKS = {}


#  6. RANDOM SCENARIOS (dataset)
def sample(rng):
    """Return (params, initial, t_span, dt) drawn from PHYSICALLY REASONABLE ranges"""
    m = float(rng.uniform(50.0, 100.0))
    L0 = float(rng.uniform(15.0, 30.0))
    # Static stretch m*g/k between 3 m and 12 m
    E = 0.5 * m * v ** 2 - m * g * (x - L0)
    stretch = np.maximum(x - L0, 0.0)
    E += 0.5 * k * stretch ** 2
    return E


def bounds(p):
    return {}


# 6. RANDOM SCENARIOS
def sample(rng):
    """Physically reasonable draws."""
    m = float(rng.uniform(50.0, 100.0))
    L0 = float(rng.uniform(15.0, 30.0))
    # static stretch mg/k between 3 m and 12 m
    k = float(m * 9.81 / rng.uniform(3.0, 12.0))
    c_d = float(rng.choice([0.0, rng.uniform(0.5, 3.0)]))
    p = {
        "m": m,
        "L0": L0,
        "k": k,
        "g": 9.81,
        "c_d": c_d,
        "method": "rk4",
        "event_handling": True,
    }
    init = {"x0": 0.0, "v0": 0.0}
    return p, init, [0.0, 60.0], 0.01


def validate_run(p, init, t, Y, tol_energy=1e-4, tol_ref=1e-3):
    """Quick helper check of energy conservation and reference match."""
# 7. HELPERS
def validate_run(p, init, t, Y, tol_energy=1e-4, tol_ref=1e-3):
    """Quick check of energy conservation and reference match."""
    qty = quantities(p, init, t, Y)
    ref = reference_quantities(p, init)
    E = energy(p, Y)
    e0 = abs(E[0]) if abs(E[0]) > 1e-12 else 1.0
    e_rel = (E.max() - E.min()) / e0

    out = {
        "lowest_point": qty["lowest_point"],
        "energy_rel_variation": float(e_rel),
        "energy_ok": e_rel < tol_energy,
    }
    if ref:
        diff = abs(qty["lowest_point"] - ref["lowest_point"])
        out.update({
            "ref_lowest": ref["lowest_point"],
            "ref_diff": float(diff),
            "ref_ok": diff < tol_ref,
        })
    return out
    return out