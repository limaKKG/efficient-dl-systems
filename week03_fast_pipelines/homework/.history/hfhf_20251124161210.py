import numpy as np
import matplotlib.pyplot as plt

# Функция процесса
def X(U, t):
    return U*(t+2)**2 - 2*t - 2

t = np.arange(1, 11)

# Реализации
x1 = X(-1, t)
x2 = X(1, t)
x3 = X(2, t)

# Оценка математического ожидания
mx_est = (x1 + x2 + x3) / 3

# Теоретическое математическое ожидание
M_U = 1.2  # Мы его ранее вычислили: M[U] = 1.2
mx = M_U*(t+2)**2 - 2*t - 2

plt.figure(figsize=(10,6))

# Графики реализаций
plt.plot(t, x1, label='x1(t) при U = -1')
plt.plot(t, x2, label='x2(t) при U = 1')
plt.plot(t, x3, label='x3(t) при U = 2')

# Теоретическое мат.ожидание — штрихованное
plt.plot(t, mx, linestyle='--', label='mX(t) теоретическое')

# Оценка мат.ожидания — точки
plt.scatter(t, mx_est, color='black', label='mX*(t) оценка')

plt.xlabel('t')
plt.ylabel('Значение функции')
plt.title('Реализации x1, x2, x3 и оценки mX(t), mX*(t)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
