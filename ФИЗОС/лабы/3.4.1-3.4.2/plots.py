"""
Графики и расчёты для лабораторных работ 3.4.1 и 3.4.2.
ИСПРАВЛЕНО:
  - 3.4.1: перевод массы из мг в кг (1e-6 вместо 1e-3).
  - 3.4.2: фит по T >= 22 °C, исправлена mathtext-ошибка в легенде.
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.4,
})

# ============================================================
# 3.4.1  Магнитная восприимчивость
# ============================================================
I_cal   = np.array([0.20, 0.50, 0.80, 1.10, 1.40, 1.70, 2.00, 2.30, 2.60, 2.90, 3.07])
B_kGs   = np.array([0.5, 1.2, 1.8, 2.4, 3.1, 3.6, 4.2, 4.9, 5.5, 5.9, 6.2])
B_T     = B_kGs * 0.1                    # Тл
B2      = B_T ** 2

delta_B_rel = 0.02                       # 2% систематика калибровки
delta_B2    = 2 * B_T * delta_B_rel * B_T
delta_m_mg  = 1.0                        # мг
delta_F     = delta_m_mg * 1e-6 * 9.81   # Н

dm_Al_mg = np.array([0, 1, 4, 7, 12, 18, 24, 30, 37, 44, 47])
dm_Cu_mg = np.array([0, 1, 2, 4, 6, 9, 12, 14, 20, 21, 22])

F_Al = dm_Al_mg * 1e-6 * 9.81
F_Cu = dm_Cu_mg * 1e-6 * 9.81

def ols_origin(x, y, sy):
    w = 1.0 / sy**2
    k = np.sum(w * x * y) / np.sum(w * x**2)
    var_k = 1.0 / np.sum(w * x**2)
    return k, np.sqrt(var_k)

k_Al, sig_k_Al_stat = ols_origin(B2, F_Al, np.full_like(B2, delta_F))
k_Cu, sig_k_Cu_stat = ols_origin(B2, F_Cu, np.full_like(B2, delta_F))

sig_k_Al_syst = k_Al * 2 * delta_B_rel
sig_k_Cu_syst = k_Cu * 2 * delta_B_rel
sig_k_Al = np.hypot(sig_k_Al_stat, sig_k_Al_syst)
sig_k_Cu = np.hypot(sig_k_Cu_stat, sig_k_Cu_syst)

mu0 = 4 * np.pi * 1e-7
r_s = 5.0e-3
delta_r = 0.1e-3
S = np.pi * r_s**2
delta_S = 2 * np.pi * r_s * delta_r

chi_Al = 2 * mu0 * k_Al / S
chi_Cu = -2 * mu0 * k_Cu / S
sig_chi_Al = abs(chi_Al) * np.hypot(sig_k_Al/k_Al, delta_S/S)
sig_chi_Cu = abs(chi_Cu) * np.hypot(sig_k_Cu/k_Cu, delta_S/S)

print("=" * 60)
print("3.4.1")
print(f"  k_Al = ({k_Al:.3e} ± {sig_k_Al:.1e}) Н/Тл²")
print(f"  k_Cu = ({k_Cu:.3e} ± {sig_k_Cu:.1e}) Н/Тл²")
print(f"  χ_Al = ({chi_Al:.2e} ± {sig_chi_Al:.1e})")
print(f"  χ_Cu = ({chi_Cu:.2e} ± {sig_chi_Cu:.1e})")

def plot_341(material, F, k, chi, sig_chi, color, marker, fname):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(B2 * 1e3, F * 1e6,
                xerr=delta_B2 * 1e3, yerr=delta_F * 1e6,
                fmt=marker, color=color, capsize=4,
                label='Эксперимент', zorder=3)
    x_fit = np.linspace(0, B2.max(), 200)
    ax.plot(x_fit * 1e3, k * x_fit * 1e6,
            '-', color=color, lw=1.8, label='Аппроксимация')
    ax.set_xlabel('$B^2$, мТл$^2$')
    ax.set_ylabel('$|F|$, мкН')
    ax.set_title(material)
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)

plot_341('Алюминий (парамагнетик)', F_Al, k_Al, chi_Al, sig_chi_Al,
         'C0', 'o', 'plot_3_4_1_Al.png')
plot_341('Медь (диамагнетик)', F_Cu, k_Cu, chi_Cu, sig_chi_Cu,
         'C2', 's', 'plot_3_4_1_Cu.png')

# ============================================================
# 3.4.2  Закон Кюри–Вейсса
# ============================================================
T   = np.array([14.2, 16.0, 18.01, 20.0, 22.0, 24.0, 26.0, 28.0,
                30.0, 32.0, 34.0, 36.0, 38.0, 40.0])
tau = np.array([10.4082, 9.958738, 9.776049, 9.53297, 9.189143,
                8.778818, 8.62145, 8.535055, 8.485958, 8.452457,
                8.426839, 8.406604, 8.392871, 8.38079])
tau0 = 8.252

delta_T    = 0.5
delta_tau  = 0.001
delta_tau0 = 0.001

u       = tau**2 - tau0**2
delta_u = np.sqrt((2 * tau * delta_tau)**2 + (2 * tau0 * delta_tau0)**2)
y       = 1.0 / u
delta_y = delta_u / u**2

mask = T >= 22
xw, yw, dw = T[mask], y[mask], delta_y[mask]
w  = 1.0 / dw**2
S_w  = np.sum(w)
Sx   = np.sum(w * xw)
Sy   = np.sum(w * yw)
Sxx  = np.sum(w * xw**2)
Sxy  = np.sum(w * xw * yw)
D    = S_w * Sxx - Sx**2

b = (S_w * Sxy - Sx * Sy) / D
a = (Sxx * Sy - Sx * Sxy) / D
sig_b = np.sqrt(S_w   / D)
sig_a = np.sqrt(Sxx   / D)

Theta_p = -a / b
sig_Theta = Theta_p * np.hypot(sig_a/a, sig_b/b)

y_pred = a + b * xw
chi2 = np.sum(((yw - y_pred) / dw)**2)
ndf  = len(xw) - 2

print("\n" + "=" * 60)
print("3.4.2")
print(f"  b = ({b:.3e} ± {sig_b:.1e}) мкс⁻²·°C⁻¹")
print(f"  a = ({a:.3e} ± {sig_a:.1e}) мкс⁻²")
print(f"  Θ_p = ({Theta_p:.2f} ± {sig_Theta:.2f}) °C")
print(f"  χ²/ndf = {chi2/ndf:.2f}")

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.errorbar(T, y, xerr=delta_T, yerr=delta_y,
            fmt='o', color='C3', capsize=4, label='Эксперимент', zorder=3)
x_fit = np.linspace(21, 41, 200)
ax.plot(x_fit, a + b * x_fit, '-', color='C3', lw=1.8,
        label=r'Аппроксимация ($T \geq 22$ °C)')
ax.axvline(Theta_p, color='k', ls='--', lw=1.2, alpha=0.7,
           label=r'$\Theta_p$')
ax.set_xlabel('$T$, °C')
ax.set_ylabel(r'$1/(\tau^2 - \tau_0^2)$, мкс$^{-2}$')
ax.set_title('Гадолиний: закон Кюри–Вейсса')
ax.legend(loc='upper left')
ax.set_xlim(13, 42)
fig.tight_layout()
fig.savefig('plot_3_4_2.png', dpi=150)
plt.close(fig)

print("\nСохранены файлы: plot_3_4_1_Al.png, plot_3_4_1_Cu.png, plot_3_4_2.png")
