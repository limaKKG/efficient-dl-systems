import numpy as np
import matplotlib.pyplot as plt

# точки
t2 = np.arange(1, 11)

# оценка корреляционной функции
Kx_star = 16.56 * (t2 + 2)**2

# теоретическая корреляционная функция
Kx = 1.84 * (1 + 2)**2 * (t2 + 2)**2  # (1+2)^2 = 9

plt.figure(figsize=(10, 6))

plt.plot(t2, Kx_star, label="Kx*(1, t2) — оценка", linewidth=2)
plt.plot(t2, Kx, label="Kx(1, t2) — теория", linestyle="--", linewidth=2)

plt.title("Корреляционная функция: Kx*(1, t2) и Kx(1, t2)")
plt.xlabel("t2")
plt.ylabel("Значение функции")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
