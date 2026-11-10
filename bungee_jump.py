import numpy as np


def _energy_lowest_point(p, init):
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


def quantities(p, init, t, Y):
    """Compute key quantities from the trajectory Y."""
    x = Y[:, 0]
    return {
        "lowest_point": float(np.min(x))
    }


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