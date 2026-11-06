import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Ваш начальный массив данных
u_initial = np.array([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
])

steps = 255
N = len(u_initial)
x = np.arange(N)

# Создаем фигуру и оси
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 50)
ax.set_ylim(0, 10.5)
ax.grid(True)
ax.set_xlabel('Индекс элемента (x)')
ax.set_ylabel('Значение u(x)')

# Рисуем начальное состояние (для сравнения)
ax.plot(x, u_initial, color='gray', linestyle='--', alpha=0.5, label='Начальное состояние')

# Создаем пустую линию, которую будем обновлять
line, = ax.plot([], [], color='tab:blue', linewidth=2, label='Текущее состояние')
title = ax.set_title('Шаг: 0')
ax.legend(loc='upper right')

# Текущее состояние массива (копируем начальное)
u_current = u_initial.copy()

# Функция инициализации (пустой кадр)
def init():
    line.set_data([], [])
    return line, title

# Функция обновления для каждого кадра
def update(frame):
    global u_current
    
    # Делаем один шаг уравнения
    # u = u - 0.5 * (u - np.roll(u, 1))
    u_current = u_current - 0.5 * (u_current - np.roll(u_current, 1))
    
    # Обновляем данные на графике
    line.set_data(x, u_current)
    title.set_text(f'Шаг: {frame + 1} / {steps}')
    
    return line, title

# Создаем анимацию
ani = animation.FuncAnimation(
    fig, update, frames=steps, init_func=init, 
    interval=30, blit=True
)

# Сохраняем анимацию в GIF
print("Сохранение GIF... Это может занять несколько секунд.")
ani.save('advection.gif', writer='pillow', fps=30)
print("Готово! Файл advection.gif сохранен.")

