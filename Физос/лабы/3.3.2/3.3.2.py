import matplotlib.pyplot as plt
import numpy as np

# ---------- Функции МНК ----------
def mnk_zero(x, y):
    k = np.sum(x*y) / np.sum(x**2)
    return k

def dmnk_zero(x, y, k):
    n = len(x)
    residuals = y - k*x
    sigma = np.sqrt(np.sum(residuals**2) / (n - 1))
    delta_k = sigma / np.sqrt(np.sum(x**2))
    return delta_k

# ---------- Данные ----------
U = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50])

I_13 = np.array([2.68e-3, 8.59e-3, 20.03e-3, 32.75e-3, 47.28e-3, 66.01e-3, 83.59e-3, 101.65e-3, 122.48e-3, 144.06e-3, 167.7e-3, 193.2e-3, 247.7e-3, 305.25e-3, 368.8e-3, 429.15e-3, 0.8384, 1.3265, 1.8897, 2.5054, 3.173, 3.8888, 4.6336, 5.485])
I_14 = np.array([4.29e-3, 13.4e-3, 25.67e-3, 39.46e-3, 58.35e-3, 74.48e-3, 94.05e-3, 114.71e-3, 138.46e-3, 158.65e-3, 181.58e-3, 208.81e-3, 267.05e-3, 328.78e-3, 391.37e-3, 458.75e-3, 0.8872, 1.3852, 1.9657, 2.5965, 3.2952, 4.0269, 4.8268, 5.746])
I_15 = np.array([8.81e-3, 20.65e-3, 34.59e-3, 50.49e-3, 69.95e-3, 90.55e-3, 110.756e-3, 130.2e-3, 156.49e-3, 178.02e-3, 205.35e-3, 230.63e-3, 288.4e-3, 353.81e-3, 417.78e-3, 0.5146, 0.9243, 1.4352, 2.028, 2.673, 3.3688, 4.1307, 4.9309, 5.867])
I_16 = np.array([14.01e-3, 28.05e-3, 43.43e-3, 62.53e-3, 82.56e-3, 104.15e-3, 125.71e-3, 147.9e-3, 174.91e-3, 198.7e-3, 224.41e-3, 255.11e-3, 312.33e-3, 377.17e-3, 448.48e-3, 0.5532, 0.9696, 1.4894, 2.0893, 2.751, 3.4547, 4.2222, 5.0253, 5.976])

# Погрешности (оценочные)
dU = 0.01 * U + 0.05
dI_13 = 0.005 * np.abs(I_13) + 0.002
dI_14 = 0.005 * np.abs(I_14) + 0.002
dI_15 = 0.005 * np.abs(I_15) + 0.002
dI_16 = 0.005 * np.abs(I_16) + 0.002

# Фильтр для аппроксимации (U >= 5 В)
mask = U >= 5
x_fit = U[mask] ** 1.5

currents = [I_13, I_14, I_15, I_16]
errors = [dI_13, dI_14, dI_15, dI_16]
labels = ['1.3 A', '1.4 A', '1.5 A', '1.6 A']
colors = ['blue', 'green', 'red', 'orange']

k_values = []
k_errors = []

# ---------- 1. Графики I от U^(3/2) с продлением до нуля ----------
fig1, axs1 = plt.subplots(2, 2, figsize=(12, 10))
axs1 = axs1.flatten()

for idx, (I_data, dI_data) in enumerate(zip(currents, errors)):
    y_fit = I_data[mask]
    k = mnk_zero(x_fit, y_fit)
    delta_k = dmnk_zero(x_fit, y_fit, k)
    k_values.append(k)
    k_errors.append(delta_k)

    ax = axs1[idx]
    x_all = U ** 1.5
    y_all = I_data
    dx_all = 1.5 * np.sqrt(U) * dU
    ax.errorbar(x_all, y_all, xerr=dx_all, yerr=dI_data, fmt='o', color='k', capsize=2, label='Эксперимент')
    ax.scatter(x_fit, y_fit, color=colors[idx], s=30, label='Точки для аппроксимации (U>=5 В)')
    x_plot = np.linspace(0, max(x_all)*1.05, 100)
    ax.plot(x_plot, k * x_plot, '--', color=colors[idx], label=f'k = {k:.3e} мА/В^(3/2)')
    ax.set_xlabel('$U^{3/2}$, В^(3/2)')
    ax.set_ylabel('I, мА')
    ax.set_title(f'I_накала = {labels[idx]}')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig('plots_individual.png', dpi=300)
plt.show()

# ---------- 2. Совмещённый график I от U^(3/2) с продлением ----------
plt.figure(figsize=(10, 6))
for idx, (I_data, dI_data) in enumerate(zip(currents, errors)):
    x_all = U ** 1.5
    y_all = I_data
    dx_all = 1.5 * np.sqrt(U) * dU
    plt.errorbar(x_all, y_all, xerr=dx_all, yerr=dI_data, fmt='o', color=colors[idx], capsize=2, label=labels[idx])
    k = k_values[idx]
    x_plot = np.linspace(0, max(x_all)*1.05, 100)
    plt.plot(x_plot, k * x_plot, '--', color=colors[idx])

plt.xlabel('$U^{3/2}$, В^(3/2)')
plt.ylabel('I, мА')
plt.title('Зависимость анодного тока от $U^{3/2}$ (все точки с погрешностями)')
plt.legend()
plt.grid(True)
plt.savefig('plots_combined.png', dpi=300)
plt.show()

# ---------- 3. Сырые данные: I vs U с погрешностями ----------
plt.figure(figsize=(10, 6))
for idx, (I_data, dI_data) in enumerate(zip(currents, errors)):
    plt.errorbar(U, I_data, xerr=dU, yerr=dI_data, fmt='o', color=colors[idx], capsize=2, label=labels[idx])
plt.xlabel('U, В')
plt.ylabel('I, мА')
plt.title('Вольт-амперные характеристики диода (сырые данные)')
plt.legend()
plt.grid(True)
plt.savefig('plots_raw.png', dpi=300)
plt.show()

# ---------- 4. Область малых напряжений (U < 10 В) в координатах I vs U^(3/2) ----------
plt.figure(figsize=(10, 6))
mask_small = U <= 10
U_small = U[mask_small]
x_small = U_small ** 1.5

for idx, (I_data, dI_data) in enumerate(zip(currents, errors)):
    I_small = I_data[mask_small]
    dI_small = dI_data[mask_small]
    dx_small = 1.5 * np.sqrt(U_small) * dU[mask_small]
    plt.errorbar(x_small, I_small, xerr=dx_small, yerr=dI_small, fmt='o', color=colors[idx], capsize=2, label=f'Эксп. {labels[idx]}')
    k = k_values[idx]
    x_plot = np.linspace(0, max(x_small)*1.05, 100)
    plt.plot(x_plot, k * x_plot, '--', color=colors[idx], linewidth=1.5, label=f'Теор. {labels[idx]} (экстраполяция)')

plt.xlabel('$U^{3/2}$, В^(3/2)')
plt.ylabel('I, мА')
plt.title('Область малых напряжений (U < 10 В) – расхождение с законом 3/2 в координатах I vs U^(3/2)')
plt.legend()
plt.grid(True)
plt.savefig('plots_small.png', dpi=300)
plt.show()

# ---------- Вывод результатов ----------
print("Результаты аппроксимации I = k * U^(3/2):")
for idx, label in enumerate(labels):
    print(f"{label}: k = {k_values[idx]:.3e} ± {k_errors[idx]:.3e} мА/В^(3/2)")

# Параметры установки
rk = 0.9e-3
ra = 9.5e-3
l = 9e-3
alpha = 0.98
epsilon0 = 8.854187817e-12

def calc_em(k_mA_per_V32):
    k_A = k_mA_per_V32 * 1e-3
    K = (4/9) * epsilon0 * (2 * np.pi * l / ra) * alpha
    sqrt_2em = k_A / K
    em = 0.5 * sqrt_2em**2
    return em

em_values = []
em_errors = []
for k, err in zip(k_values, k_errors):
    em = calc_em(k)
    em_values.append(em)
    delta_em = em * 2 * (err / k)
    em_errors.append(delta_em)

print("\nВычисленные значения e/m:")
for idx, label in enumerate(labels):
    print(f"{label}: e/m = {em_values[idx]:.3e} ± {em_errors[idx]:.3e} Кл/кг")

em_mean = np.mean(em_values)
em_error_mean = np.sqrt(np.sum(np.array(em_errors)**2)) / len(em_errors)
print(f"\nСреднее e/m = {em_mean:.3e} ± {em_error_mean:.3e} Кл/кг")
print(f"Табличное значение e/m = 1.758820e11 Кл/кг")
