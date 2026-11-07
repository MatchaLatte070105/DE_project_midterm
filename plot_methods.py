import matplotlib.pyplot as plt
import numpy as np

# Plot style (falls back to default if seaborn style is unavailable)
style = 'seaborn-v0_8-whitegrid'
plt.style.use(style if style in plt.style.available else 'default')

# Use a moderate dpi for on-screen display; high dpi is only used when saving
fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
fig.subplots_adjust(top=0.88, bottom=0.14, left=0.11, right=0.96)


# Example ODE: y' = f(t, y) = y - t + 1, with exact solution y(t) = e^t + t
def f(t, y):
    return y - t + 1


def exact(t):
    return np.exp(t) + t


# Exact solution curve
t_fine = np.linspace(0, 1.6, 300)
ax.plot(t_fine, exact(t_fine), color='#1f77b4', linewidth=2,
        label='Exact Solution $y(t)$')

# Current point (t_n, y_n) and step size
tn = 0.5
yn = exact(tn)
h = 0.9
t_next = tn + h

# Euler step: follow the tangent line at (t_n, y_n)
slope = f(tn, yn)
y_euler = yn + h * slope
y_true = exact(t_next)

ax.plot([tn, t_next], [yn, y_euler], color='#ff7f0e', linestyle='--',
        linewidth=2, label='Euler Approximation')

# Points (zorder keeps them on top of the lines)
ax.plot(tn, yn, 'o', color='red', markersize=7, zorder=5,
        label='Current Point $(t_n, y_n)$')
ax.plot(t_next, y_euler, 's', color='#ff7f0e', markersize=7, zorder=5,
        label='Euler $y_{n+1}$')
ax.plot(t_next, y_true, 'o', color='#1f77b4', markersize=7, zorder=5,
        label='Exact $y(t_{n+1})$')

# Local error annotation
ax.annotate('', xy=(t_next, y_true), xytext=(t_next, y_euler),
            arrowprops=dict(arrowstyle='<->', color='black', linewidth=1.2))
ax.text(t_next + 0.03, (y_true + y_euler) / 2, 'local error',
        fontsize=10, va='center')

# Labels and title
ax.set_title("Bungee Jump Model: Euler's Method vs Exact Solution",
             fontsize=13, fontweight='bold', pad=10)
ax.set_xlabel('Time $t$', fontsize=11)
ax.set_ylabel('State $y$', fontsize=11)
ax.legend(frameon=True, facecolor='white', edgecolor='lightgray',
          fontsize=9, loc='upper left')

# Fixed axis limits
ax.set_xlim(0.0, 1.6)
ax.set_ylim(0.0, 6.0)

plt.show()