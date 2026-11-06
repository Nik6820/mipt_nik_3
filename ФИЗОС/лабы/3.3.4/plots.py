"""
Обработка данных и графики для лабораторной работы 3.3.4.
Исправлены mathtext-ошибки: заголовки/подписи без \\text{}, без $^2$/$^3$
без основания; степени записаны Unicode-символами (², ³) вне math-режима.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.interpolate import interp1d

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.4,
})

# ============================================================
# Параметры образца и калибровки
# ============================================================
a = 1.5e-3            # толщина, м
l = 1.7e-3            # ширина, м
L35 = 3.0e-3          # расстояние 3–5, м
e = 1.602e-19         # Кл
SN_cm2 = 75.0         # SN = 75 см²
SN = SN_cm2 * 1e-4    # 7.5e-3 м²
r_outer_max = 5.0     # Ом

# ============================================================
# Калибровка ATE-8702
# ============================================================
I_ATE = np.array([0, 0.24, 0.50, 0.74, 0.98, 1.25, 1.49, 1.77, 1.91])
B_ATE_raw = np.array([21.9, 191.9, 399.0, 577.4, 738.5,
                      887.8, 968.9, 1037.2, 1064.0])
B_ATE = B_ATE_raw - B_ATE_raw[0]

# ============================================================
# Калибровка милливеберметром (SN = 75 см²)
# ============================================================
I_web = np.array([0, 0.24, 0.48, 0.75, 1.00, 1.25, 1.50, 1.75, 1.84])
dPhi_web_mWb = np.array([0.0, 1.1, 2.3, 3.4, 4.4, 5.3, 6.0, 6.4, 6.3])
B_web_mTl = dPhi_web_mWb / SN_cm2 * 1e4

ratio = B_web_mTl[1:] / B_ATE[1:]
print(f"B_web/B_ATE = {ratio.mean():.3f} ± {ratio.std():.3f}")

B_from_I = interp1d(I_ATE, B_ATE, kind='linear', fill_value='extrapolate')

# ============================================================
# Данные Холла
# ============================================================
data_forward = {
    0.14: ([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 1.86],
           [0, -0.011, -0.022, -0.033, -0.042, -0.051, -0.057, -0.060, -0.061]),
    0.20: ([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 1.85],
           [0, -0.016, -0.032, -0.048, -0.063, -0.076, -0.084, -0.091, -0.093]),
    0.30: ([0, 0.24, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 1.86],
           [0, -0.023, -0.049, -0.072, -0.095, -0.114, -0.127, -0.137, -0.140]),
    0.40: ([0, 0.24, 0.49, 0.75, 1, 1.25, 1.5, 1.75, 1.87],
           [0, -0.032, -0.064, -0.096, -0.125, -0.150, -0.168, -0.181, -0.186]),
    0.50: ([0, 0.25, 0.49, 0.75, 1, 1.25, 1.5, 1.75, 1.88],
           [0, -0.038, -0.080, -0.118, -0.155, -0.186, -0.209, -0.223, -0.230]),
    0.60: ([0, 0.25, 0.49, 0.75, 1, 1.25, 1.49, 1.75, 1.88],
           [0, -0.047, -0.098, -0.143, -0.187, -0.225, -0.252, -0.271, -0.278]),
    0.70: ([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 1.88],
           [0, -0.056, -0.113, -0.168, -0.220, -0.265, -0.295, -0.317, -0.327]),
    0.80: ([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 1.88],
           [0, -0.064, -0.131, -0.195, -0.249, -0.300, -0.338, -0.361, -0.373]),
    0.90: ([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 1.88],
           [0, -0.070, -0.147, -0.215, -0.286, -0.340, -0.380, -0.409, -0.421]),
    1.00: ([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 1.88],
           [0, -0.076, -0.164, -0.241, -0.311, -0.377, -0.420, -0.453, -0.465]),
}
I_M_rev = np.array([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 1.87])
U_rev = np.array([0, 0.073, 0.154, 0.229, 0.298, 0.361, 0.403, 0.432, 0.444])

EXCLUDE = 0.14

# ============================================================
# Обработка
# ============================================================
currents = sorted(data_forward.keys())
results = []

print("=" * 70)
print("Линейный фит U_H = α·B для каждого тока")
print(f"{'I, мА':>6} | {'α, мВ/Тл':>12} | {'R_H, 10^-3 м³/Кл':>22}")
print("-" * 70)

for I_mA in currents:
    I_M, U = data_forward[I_mA]
    I_M = np.array(I_M, dtype=float)
    U = np.array(U, dtype=float)
    B_T = B_from_I(I_M) * 1e-3
    slope, intercept, r, p, se = linregress(B_T, U)
    I_A = I_mA * 1e-3
    R_H = slope * 1e-3 * a / I_A
    R_H_err = se * 1e-3 * a / I_A
    mark = "  <-- выброс" if I_mA == EXCLUDE else ""
    results.append((I_mA, slope, R_H, R_H_err))
    print(f"{I_mA:6.2f} | {slope:12.4f} | {R_H*1e3:9.4f} ± {R_H_err*1e3:.4f}{mark}")

R_H_arr = np.array([r[2] for r in results])
R_H_err_arr = np.array([r[3] for r in results])
mask = np.array([r[0] != EXCLUDE for r in results])
w = 1.0 / R_H_err_arr[mask]**2
R_H_avg = np.sum(w * R_H_arr[mask]) / np.sum(w)
R_H_avg_err = np.sqrt(1.0 / np.sum(w))

print("-" * 70)
print(f"Среднее R_H (без I = {EXCLUDE} мА) = "
      f"({R_H_avg*1e3:.4f} ± {R_H_avg_err*1e3:.4f})·10^-3 м³/Кл")

n = 1.0 / (abs(R_H_avg) * e)
n_err = n * (R_H_avg_err / abs(R_H_avg))
print(f"n = ({n:.3e} ± {n_err:.2e}) м^-3 = "
      f"({n*1e-6:.2e} ± {n_err*1e-6:.2e}) см^-3")

U35 = 1.839e-3
I_cond = 1.0e-3
sigma = I_cond * L35 / (U35 * a * l)
sigma_err = sigma * np.sqrt(0.01**2 + 0.033**2 + 0.0005**2 + 0.067**2 + 0.059**2)
print(f"σ = ({sigma:.3e} ± {sigma_err:.2e}) См/м")

mu = abs(R_H_avg) * sigma
mu_err = mu * np.sqrt((R_H_avg_err/R_H_avg)**2 + (sigma_err/sigma)**2)
print(f"μ = ({mu:.4f} ± {mu_err:.4f}) м²/(В·с) = "
      f"({mu*1e4:.1f} ± {mu_err*1e4:.1f}) см²/(В·с)")

# ============================================================
# График 1: калибровка (без mathtext-конструкций с кириллицей)
# ============================================================
fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.plot(I_ATE, B_ATE, 'o-', color='C0', ms=8, label='ATE-8702')
ax.plot(I_web, B_web_mTl, 's--', color='C1', ms=7,
        label=f'Милливеберметр (SN = {SN_cm2:.0f} см²)')
ax.set_xlabel('I_M, А')
ax.set_ylabel('B, мТл')
ax.set_title(f'Калибровка электромагнита\n'
             f'B_web / B_ATE = {ratio.mean():.2f} '
             f'(систематика ≈ {(1-ratio.mean())*100:.0f}%)')
ax.legend(loc='upper left')
fig.tight_layout()
fig.savefig('calib.png', dpi=150)
plt.close(fig)

# ============================================================
# График 2: U_H(B) — все токи
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6.5))
cmap = plt.cm.viridis
for i, I_mA in enumerate(currents):
    I_M, U = data_forward[I_mA]
    B = B_from_I(np.array(I_M, dtype=float))
    color = cmap(i / (len(currents) - 1))
    ax.plot(B, U, 'o-', color=color, ms=4, lw=1.2,
            label=f'I = {I_mA:.2f} мА')

B_rev = -B_from_I(I_M_rev)
ax.plot(B_rev, U_rev, 's--', color='C5', ms=5, lw=1.2,
        label='I = 1.00 мА (обр.)')

ax.axhline(0, color='k', lw=0.5)
ax.axvline(0, color='k', lw=0.5)
ax.set_xlabel('B, мТл')
ax.set_ylabel('U_H, мВ')
ax.set_title('ЭДС Холла от индукции магнитного поля')
ax.legend(loc='upper left', fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig('hall.png', dpi=150)
plt.close(fig)

# ============================================================
# График 3: R_H(I)
# ============================================================
fig, ax = plt.subplots(figsize=(8.5, 5.5))
I_arr = np.array([r[0] for r in results])
mask_use = I_arr != EXCLUDE
mask_out = ~mask_use

ax.errorbar(I_arr[mask_use], R_H_arr[mask_use]*1e3,
            yerr=R_H_err_arr[mask_use]*1e3,
            fmt='o', ms=7, capsize=4, color='C0',
            label='Использовано в среднем')
ax.errorbar(I_arr[mask_out], R_H_arr[mask_out]*1e3,
            yerr=R_H_err_arr[mask_out]*1e3,
            fmt='s', ms=8, capsize=4, color='C3',
            label=f'Выброс (I = {EXCLUDE} мА)')
ax.axhline(R_H_avg*1e3, color='k', ls='--', lw=1.2,
           label=f'Среднее = {R_H_avg*1e3:.3f}')

ax.set_xlabel('I, мА')
ax.set_ylabel('R_H, 10⁻³ м³/Кл')
ax.set_title('Постоянная Холла от тока через образец')
ax.legend(loc='lower right')
fig.tight_layout()
fig.savefig('Rh.png', dpi=150)
plt.close(fig)

print("\nСохранены файлы: calib.png, hall.png, Rh.png")
