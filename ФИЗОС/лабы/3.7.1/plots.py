"""
Графики и расчёты для лабораторной работы 3.7.1 (скин-эффект).
Выбросы отфильтрованы:
- для фазовых измерений: отбрасываются точки с x > x0;
- для индуктивности: отбрасываются L < 0 и точки, где L начинает расти.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.4,
})

# ============================================================
# Параметры установки
# ============================================================
a = 22.5e-3        # радиус экрана, м
h = 1.5e-3         # толщина стенки, м
mu0 = 4 * np.pi * 1e-7
sigma_table = 5e7  # табличная проводимость

# ============================================================
# 1. Низкие частоты: xi(nu)
# ============================================================
freq_low = np.array([22.5, 23, 33, 43, 53, 63, 73, 83, 93, 103, 113])
I_low = np.array([452.5, 451.63, 447.85, 442.9, 436.91, 430.34,
                  423.53, 416.63, 409.86, 403.31, 397.05]) * 1e-3  # А
U_low = np.array([0.1471, 0.1499, 0.211, 0.2684, 0.3212, 0.369,
                  0.4121, 0.4506, 0.4846, 0.5146, 0.541])  # В
xi_low = U_low / (freq_low * I_low)

# Погрешности
d_freq = 1.0
d_I = 0.1e-3
d_U = 1e-3
d_xi = xi_low * np.sqrt((d_U/U_low)**2 + (d_freq/freq_low)**2 + (d_I/I_low)**2)

# 1/xi^2 и nu^2
y_low = 1 / xi_low**2
x_low = freq_low**2
dy_low = 2 * d_xi / xi_low**3
dx_low = 2 * freq_low * d_freq

# Линейная регрессия y = K x + b
slope, intercept, r_val, p_val, std_err = linregress(x_low, y_low)
K = slope
b = intercept
xi0 = 1 / np.sqrt(b)
sigma1 = np.sqrt(K / b) / (np.pi * a * h * mu0)

# Погрешности K и b (приближённо)
# из linregress: std_err для slope, для intercept можно оценить по остаткам
resid = y_low - (K * x_low + b)
s2 = np.sum(resid**2) / (len(x_low) - 2)
s_K = std_err
s_b = np.sqrt(s2 * np.sum(x_low**2) / (len(x_low) * np.sum((x_low - np.mean(x_low))**2)))
# sigma1 погрешность
s_sigma1 = sigma1 * 0.5 * np.sqrt((s_K/K)**2 + (s_b/b)**2)

print("=== 1. Низкие частоты ===")
print(f"xi0 = {xi0:.4e} В/(Гц·А)")
print(f"K = {K:.3e} ± {s_K:.1e}")
print(f"b = {b:.3e} ± {s_b:.1e}")
print(f"sigma1 = ({sigma1:.2e} ± {s_sigma1:.1e}) См/м")

# График
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.errorbar(x_low, y_low, xerr=dx_low, yerr=dy_low, fmt='o', capsize=4,
            label='Эксперимент')
x_fit = np.linspace(0, x_low.max(), 100)
ax.plot(x_fit, K * x_fit + b, 'r-', label='Аппроксимация')
ax.set_xlabel(r'$\nu^2$, Гц$^2$')
ax.set_ylabel(r'$1/\xi^2$, (Гц$\cdot$А/В)$^2$')
ax.set_title('Низкие частоты: $1/\\xi^2$ от $\\nu^2$')
ax.legend()
fig.tight_layout()
fig.savefig('plot_3_7_1_xi_low.png', dpi=150)
plt.close(fig)

# ============================================================
# 2. Средние частоты: tg(psi) от nu
# ============================================================
freq_mid = np.array([113, 129, 145, 161, 177, 193, 209, 225,
                     315, 405, 495, 585, 675, 765, 855, 945, 1035, 1125])
I_mid = np.array([393.91, 384.56, 376.32, 368.95, 362.53, 356.82,
                  351.77, 347.31, 329.73, 318.67, 310.45, 303.21,
                  296.49, 289.94, 284.03, 277.29, 270.54, 263.86]) * 1e-3
U_mid = np.array([0.5384, 0.574, 0.6031, 0.6264, 0.6452, 0.6607,
                  0.6734, 0.6837, 0.7149, 0.7236, 0.7221, 0.7151,
                  0.7053, 0.6945, 0.6814, 0.6672, 0.6523, 0.6371])
x0_mid = np.array([4.4, 3.9, 3.4, 3.1, 2.8, 2.6, 2.4, 2.2,
                   3.2, 2.5, 2.0, 1.7, 1.5, 3.3, 2.9, 5.3, 4.8, 4.4])
x_mid = np.array([3.0, 2.7, 2.5, 2.3, 2.2, 2.0, 1.9, 1.8,
                  2.7, 2.2, 1.9, 1.6, 1.4, 3.2, 2.8, 5.1, 4.7, 4.4])

# Фильтрация: x <= x0 (все точки проходят)
mask = x_mid <= x0_mid
freq_mid = freq_mid[mask]
I_mid = I_mid[mask]
U_mid = U_mid[mask]
x0_mid = x0_mid[mask]
x_mid = x_mid[mask]

delta_phi = (x_mid / x0_mid) * np.pi
psi = delta_phi - np.pi/2
tg_psi = np.tan(psi)

# Погрешность tg(psi)
d_x = 0.2
d_delta = (np.pi / x0_mid) * np.sqrt((d_x)**2 + (x_mid/x0_mid * d_x)**2)
d_psi = d_delta  # так как psi = delta - pi/2
d_tg = d_psi / np.cos(psi)**2

# Линейная регрессия tg_psi = k * nu
slope_tg, intercept_tg, r_val_tg, p_val_tg, std_err_tg = linregress(freq_mid, tg_psi)
k_tg = slope_tg
sigma2 = k_tg / (np.pi * a * h * mu0)
s_sigma2 = sigma2 * (std_err_tg / k_tg)

print("\n=== 2. Средние частоты (tg psi) ===")
print(f"k = {k_tg:.3e} ± {std_err_tg:.1e} Гц^-1")
print(f"sigma2 = ({sigma2:.2e} ± {s_sigma2:.1e}) См/м")

# График
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.errorbar(freq_mid, tg_psi, xerr=d_freq, yerr=d_tg, fmt='o', capsize=4,
            label='Эксперимент')
x_fit = np.linspace(0, freq_mid.max(), 100)
ax.plot(x_fit, k_tg * x_fit + intercept_tg, 'r-', label='Аппроксимация')
ax.set_xlabel(r'$\nu$, Гц')
ax.set_ylabel(r'$\operatorname{tg}\psi$')
ax.set_title('Средние частоты: $\\operatorname{tg}\\psi$ от $\\nu$')
ax.legend()
fig.tight_layout()
fig.savefig('plot_3_7_1_tgpsi.png', dpi=150)
plt.close(fig)

# ============================================================
# 3. Высокие частоты: xi(nu) и сравнение с теорией
# ============================================================
freq_high = np.array([23000, 24000, 25000, 26000, 27000, 27250,
                      27375, 27453, 27625, 27750, 28000, 29000,
                      30000, 31000, 32000])
I_high = np.array([15.068, 13.566, 12.131, 10.755, 9.432, 9.11,
                   8.949, 8.847, 8.63, 8.471, 8.155, 6.924,
                   5.784, 4.677, 3.5609]) * 1e-3
U_high = np.array([0.0226, 0.0205, 0.0183, 0.016, 0.0139, 0.0134,
                   0.0132, 0.0131, 0.0129, 0.0126, 0.0123, 0.0115,
                   0.0108, 0.0105, 0.01])
xi_high = U_high / (freq_high * I_high)
d_xi_high = xi_high * np.sqrt((d_U/U_high)**2 + (d_freq/freq_high)**2 + (d_I/I_high)**2)

# Коэффициент ослабления
ratio_high = xi_high / xi0
d_ratio = ratio_high * np.sqrt((d_xi_high/xi_high)**2 + (0.003/xi0)**2)  # погрешность xi0 ~0.3%

# Теоретическая кривая
nu_theory = np.logspace(np.log10(20), np.log10(35000), 300)
omega = 2 * np.pi * nu_theory
alpha = (1 + 1j) * np.sqrt(omega * sigma1 * mu0 / 2)
H1_H0 = 1 / (np.cosh(alpha * h) + 0.5 * alpha * a * np.sinh(alpha * h))
ratio_theory = np.abs(H1_H0)

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.errorbar(freq_high, ratio_high, yerr=d_ratio, fmt='o', capsize=4,
            label='Эксперимент (высокие частоты)')
# Добавим низкие и средние частоты для полноты
ratio_low = xi_low / xi0
ratio_mid = (U_mid / (freq_mid * I_mid)) / xi0
ax.plot(freq_low, ratio_low, 's', color='C2', label='Эксперимент (низкие)')
ax.plot(freq_mid, ratio_mid, '^', color='C3', label='Эксперимент (средние)')
ax.plot(nu_theory, ratio_theory, 'k-', label='Теория (σ = %.1e)' % sigma1)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\nu$, Гц')
ax.set_ylabel(r'$|H_1|/|H_0|$')
ax.set_title('Коэффициент ослабления поля от частоты')
ax.legend()
fig.tight_layout()
fig.savefig('plot_3_7_1_xi_all.png', dpi=150)
plt.close(fig)

# ============================================================
# 4. Индуктивность от частоты
# ============================================================
freq_L = np.array([40, 50, 60, 75, 100, 120, 150, 200, 250, 300,
                   400, 500, 600, 800, 1500, 2000, 2500, 3000, 4000])
L_L = np.array([10.337, 10.163, 9.9219, 9.5223, 8.7916, 8.2242,
                7.4176, 6.2853, 5.4682, 4.8817, 4.1563, 3.7575,
                3.5194, 3.2655, 3.0107, 2.9656, 2.9472, 2.9376, 2.9326])
# Исключаем точку 750 Гц (L=0.2156) и 1000 Гц (L=-3.74) - они не входят в freq_L

Lmax = L_L[0]
Lmin = L_L[-1]

y_L = (Lmax - L_L) / (L_L - Lmin)
x_L = freq_L**2

# Линейная регрессия y_L = K_L * x_L
slope_L, intercept_L, r_val_L, p_val_L, std_err_L = linregress(x_L, y_L)
K_L = slope_L
sigma3 = np.sqrt(K_L) / (np.pi * a * h * mu0)
s_sigma3 = sigma3 * (std_err_L / (2 * K_L))

print("\n=== 4. Индуктивность ===")
print(f"Lmax = {Lmax} мГн, Lmin = {Lmin} мГн")
print(f"K_L = {K_L:.3e} ± {std_err_L:.1e} Гц^-2")
print(f"sigma3 = ({sigma3:.2e} ± {s_sigma3:.1e}) См/м")

# График
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.errorbar(x_L, y_L, xerr=2*freq_L*d_freq, yerr=None, fmt='o', capsize=4,
            label='Эксперимент')
x_fit = np.linspace(0, x_L.max(), 100)
ax.plot(x_fit, K_L * x_fit + intercept_L, 'r-', label='Аппроксимация')
ax.set_xlabel(r'$\nu^2$, Гц$^2$')
ax.set_ylabel(r'$(L_{\max}-L)/(L-L_{\min})$')
ax.set_title('Индуктивность: $(L_{\\max}-L)/(L-L_{\\min})$ от $\\nu^2$')
ax.legend()
fig.tight_layout()
fig.savefig('plot_3_7_1_L.png', dpi=150)
plt.close(fig)

print("\nСводка:")
print(f"  sigma1 = {sigma1:.2e} См/м")
print(f"  sigma2 = {sigma2:.2e} См/м")
print(f"  sigma3 = {sigma3:.2e} См/м")
print(f"  табличное = {sigma_table:.2e} См/м")
