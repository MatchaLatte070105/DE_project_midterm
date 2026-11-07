import numpy as np
import matplotlib.pyplot as plt
from bungee_jump import simulate, energy, quantities, reference_quantities

# 1. Baseline parameters
p = {
    "m": 70.0,
    "L0": 20.0,
    "k": 100.0,
    "g": 9.81,
    "c_d": 0.0,
    "method": "rk4",
    "event_handling": True,
}
init = {"x0": 0.0, "v0": 0.0}
t_span = [0.0, 40.0]
dt = 0.005

# 2. Run simulation
t, Y = simulate(p, init, t_span, dt)
x = Y[:, 0]
v = Y[:, 1]

# 3. Compute quantities
qty = quantities(p, init, t, Y)
ref = reference_quantities(p, init)
E = energy(p, Y)

print("=== Simulation results ===")
print(qty)
print("=== Reference values (energy method) ===")
print(ref)

# 4. Plot trajectory
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(t, x)
plt.axhline(p["L0"], color="gray", linestyle="--", label="L0 (rope becomes taut)")
plt.xlabel("t (s)")
plt.ylabel("x (m)")
plt.title("Position vs time")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t, v)
plt.xlabel("t (s)")
plt.ylabel("v (m/s)")
plt.title("Velocity vs time")
plt.grid(True)

plt.tight_layout()
plt.show()

