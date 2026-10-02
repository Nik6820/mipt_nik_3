import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Параметры установки
# ============================================================
d_probe = 0.2e-3
delta_d = 0.1e-3
l_probe = 5.2e-3
delta_l = 0.1e-3
S_probe = np.pi * d_probe * l_probe
delta_S = S_probe * np.sqrt((delta_d/d_probe)**2 + (delta_l/l_probe)**2)

e     = 1.6e-19
m_e   = 9.11e-31
m_i   = 22 * 1.66e-27
k_B   = 1.38e-23
T_i   = 300
P     = 2 * 133.3
U_ign = 2140.000

delta_U3 = 0.001
delta_I3 = 0.01
delta_Ip = 0.0001
delta_Up = 0.1

# ============================================================
# 2. ВАХ разряда (×10)
# ============================================================
U_p_inc = 10 * np.array([34.992, 34.118, 32.853, 30.160, 26.863,
                         25.428, 24.083, 22.950, 22.192, 21.106,
                         20.071, 19.738, 19.135, 18.660, 18.492])
I_p_inc = np.array([0.5409, 0.8118, 1.0885, 1.2976, 1.6015, 1.8553,
                    2.1580, 2.5116, 2.7408, 3.0531, 3.4761, 3.9168,
                    4.3513, 4.7599, 4.9737])

U_p_dec = 10 * np.array([18.600, 18.975, 19.576, 19.771, 20.056, 21.128,
                         22.486, 24.564, 26.425, 33.159, 33.911, 34.560, 35.070])
I_p_dec = np.array([4.8384, 4.4253, 4.0370, 3.7497, 3.4466, 3.0404, 2.6517,
                    2.0325, 1.6676, 1.0515, 0.8764, 0.6741, 0.5193])

# ============================================================
# 3. Зондовые характеристики
# ============================================================
U_3_1 = np.array([25.013, 22.033, 19.076, 16.113, 13.055, 10.020, 8.034, 6.007,
                  4.0115, 2.0486, 1.0674, 0.5316,
                  -0.4975, -1.0407, -2.0612, -4.0126, -6.033, -7.994,
                  -10.048, -13.085, -16.206, -19.178, -22.047, -25.023])
I_3_1 = np.array([109.04, 106.55, 103.64, 99.02, 91.52, 80.22, 70.55, 58.98,
                  46.38, 32.82, 24.80, 21.14,
                  -32.37, -36.31, -44.30, -57.97, -70.69, -81.97,
                  -92.17, -104.02, -112.48, -117.86, -121.36, -124.31])

U_3_2 = np.array([25.022, 22.059, 19.007, 16.048, 13.050, 10.008, 8.011, 6.021,
                  4.0427, 2.0250, 1.0000, 0.5198,
                  -0.5126, -1.030, -2.0178, -4.0178, -5.998, -8.034,
                  -10.030, -13.047, -16.088, -19.009, -22.040, -25.021])
I_3_2 = np.array([63.77, 61.92, 60.04, 57.80, 54.22, 47.85, 41.66, 33.70,
                  24.03, 12.68, 6.25, 3.23,
                  -11.33, -14.66, -20.62, -31.82, -41.46, -49.52,
                  -55.70, -62.04, -65.84, -68.19, -70.23, -72.19])

U_3_3 = np.array([25.024, 22.023, 19.016, 16.059, 13.006, 10.030, 8.009, 6.047,
                  4.031, 2.0011, 1.0071, 0.5408,
                  -0.4644, -1.0500, -2.0763, -3.9505, -4.0612, -6.102,
                  -8.174, -10.012, -13.013, -16.014, -19.057, -22.048, -25.025])
I_3_3 = np.array([31.99, 30.87, 29.78, 28.64, 27.02, 24.20, 21.27, 17.38,
                  12.16, 5.68, 2.07, 0.37,
                  -4.40, -6.54, -10.17, -16.18, -16.51, -21.72,
                  -25.80, -28.48, -31.44, -33.15, -34.46, -35.70, -37.01])

probes = [
    (U_3_1, I_3_1, 4.9877, 'r'),
    (U_3_2, I_3_2, 3.0041, 'g'),
    (U_3_3, I_3_3, 1.5041, 'b'),
]

# ============================================================
# 4. Обработка зондовой характеристики (метод асимптот)
# ============================================================
def analyze_probe(U, I, n_asymp=4, name=''):
    p_pos = np.polyfit(U[:n_asymp],  I[:n_asymp],  1)
    p_neg = np.polyfit(U[-n_asymp:], I[-n_asymp:], 1)

    I_pos_at_0 = p_pos[1]
    I_neg_at_0 = p_neg[1]

    I_center = (I_pos_at_0 + I_neg_at_0) / 2.0
    I_in     = (I_pos_at_0 - I_neg_at_0) / 2.0

    I_c = I - I_center
    mask_zero = np.abs(I_c) < 0.5 * I_in
    p_zero = np.polyfit(U[mask_zero], I_c[mask_zero], 1)
    slope  = p_zero[0]

    dI_in = 0.05 * I_in
    dslope = 0.10 * slope
    kTe_eV = I_in / (2.0 * slope)
    dkTe = kTe_eV * np.sqrt((dI_in/I_in)**2 + (dslope/slope)**2)

    print(f"  {name}:")
    print(f"    I_center = {I_center:+.2f} мкА,  I_in = {I_in:.2f} ± {dI_in:.2f} мкА")
    print(f"    Наклон: {slope:.2f} ± {dslope:.2f} мкА/В")
    print(f"    kTe = {kTe_eV:.2f} ± {dkTe:.2f} эВ\n")

    return {'I_in': I_in, 'dI_in': dI_in,
            'I_center': I_center, 'slope': slope, 'dslope': dslope,
            'kTe': kTe_eV, 'dkTe': dkTe,
            'p_pos': p_pos, 'p_neg': p_neg, 'p_zero': p_zero,
            'mask_zero': mask_zero}

print("=" * 65)
print("ОБРАБОТКА ЗОНДОВЫХ ХАРАКТЕРИСТИК")
print("=" * 65)

results = []
for U, I, Ip, c in probes:
    r = analyze_probe(U, I, n_asymp=4, name=f"Ip = {Ip} мА")
    r.update({'Ip': Ip, 'U': U, 'I': I, 'color': c})
    results.append(r)

# ============================================================
# 5. Параметры плазмы
# ============================================================
print("=" * 65)
print("ПАРАМЕТРЫ ПЛАЗМЫ")
print("=" * 65)
e_CGS = 4.8e-10

for r in results:
    I_in_A = r['I_in'] * 1e-6
    kTe_J  = r['kTe'] * e

    n_e = I_in_A / (0.4 * e * S_probe * np.sqrt(2 * kTe_J / m_i))
    n_e_cm3 = n_e * 1e-6

    dn_e = n_e_cm3 * np.sqrt((r['dI_in']/r['I_in'])**2
                            + 0.5*(r['dkTe']/r['kTe'])**2
                            + (delta_S/S_probe)**2)

    omega_p = 5.6e4 * np.sqrt(n_e_cm3)
    d_omega_p = 0.5 * omega_p * (dn_e/n_e_cm3)

    kTe_erg = r['kTe'] * e * 1e7
    r_De = np.sqrt(kTe_erg / (4 * np.pi * n_e_cm3 * e_CGS**2))
    dr_De = 0.5 * r_De * np.sqrt((r['dkTe']/r['kTe'])**2 + (dn_e/n_e_cm3)**2)

    kTi_erg = T_i * 1.38e-16
    r_D = np.sqrt(kTi_erg / (4 * np.pi * n_e_cm3 * e_CGS**2))
    dr_D = 0.5 * r_D * (dn_e/n_e_cm3)

    N_D = (4/3) * np.pi * r_D**3 * n_e_cm3
    dN_D = N_D * np.sqrt(3*(dr_D/r_D)**2 + (dn_e/n_e_cm3)**2)

    n_neutral = P / (k_B * T_i) * 1e-6
    alpha = n_e_cm3 / n_neutral
    dalpha = alpha * (dn_e/n_e_cm3)

    print(f"\nIp = {r['Ip']} мА:")
    print(f"  kTe = {r['kTe']:.2f} ± {r['dkTe']:.2f} эВ")
    print(f"  n_e = ({n_e_cm3:.2e} ± {dn_e:.1e}) см⁻³")
    print(f"  omega_p = {omega_p:.2e} рад/с")
    print(f"  r_De = {r_De:.2e} см, r_D = {r_D:.2e} см")
    print(f"  N_D = {N_D:.1f}, alpha = {alpha:.2e}")

    r.update({'n_e': n_e_cm3, 'dn_e': dn_e,
              'omega_p': omega_p, 'r_De': r_De, 'r_D': r_D,
              'N_D': N_D, 'alpha': alpha})

# ============================================================
# 6. R_диф
# ============================================================
dU = np.diff(U_p_inc)
dI = np.diff(I_p_inc)
R_diff = -dU / dI
R_max = np.max(np.abs(R_diff))
idx_max = np.argmax(np.abs(R_diff))
print(f"\nR_диф (макс.) ≈ {R_max:.2e} Ом ≈ {R_max/1e3:.1f} кОм "
      f"при I ≈ {I_p_inc[idx_max+1]:.2f} мА")

# ============================================================
# 7. Графики
# ============================================================
fig1, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(U_p_inc, I_p_inc, xerr=delta_Up, yerr=delta_Ip,
            fmt='bo-', capsize=3, label='Нарастание тока')
ax.errorbar(U_p_dec, I_p_dec, xerr=delta_Up, yerr=delta_Ip,
            fmt='rs--', capsize=3, label='Убывание тока')
ax.set_xlabel('$U_p$, В')
ax.set_ylabel('$I_p$, мА')
ax.set_title('ВАХ тлеющего разряда в неоне')
ax.grid(True, alpha=0.3); ax.legend()
plt.tight_layout(); plt.savefig('vac.png', dpi=300); plt.close()
print("✓ vac.png")

fig2, ax = plt.subplots(figsize=(9, 6))
for r in results:
    ax.errorbar(r['U'], r['I'], xerr=delta_U3, yerr=delta_I3,
                fmt='o-', color=r['color'], markersize=4,
                capsize=2, label=f"$I_p$ = {r['Ip']} мА")
ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
ax.set_xlabel('$U_3$, В'); ax.set_ylabel('$I_3$, мкА')
ax.set_title('Зондовые характеристики')
ax.grid(True, alpha=0.3); ax.legend()
plt.tight_layout(); plt.savefig('probes.png', dpi=300); plt.close()
print("✓ probes.png")

fig3, ax = plt.subplots(figsize=(9, 6))
r = results[0]
ax.errorbar(r['U'], r['I'], xerr=delta_U3, yerr=delta_I3,
            fmt='o', color=r['color'], markersize=5,
            capsize=2, label='Эксперимент')
U_line = np.linspace(-30, 30, 200)
ax.plot(U_line, r['p_pos'][0]*U_line + r['p_pos'][1],
        'k--', lw=1.2, alpha=0.7, label='Асимптота (+)')
ax.plot(U_line, r['p_neg'][0]*U_line + r['p_neg'][1],
        'k--', lw=1.2, alpha=0.7, label='Асимптота (−)')
ax.axhline(r['I_in'], color='green', ls=':', lw=1.2,
           label=f"$I_{{in}}$ = {r['I_in']:.1f} мкА")
ax.axhline(r['I_center'], color='purple', ls=':', lw=1.2,
           label=f"$I_{{center}}$ = {r['I_center']:.1f} мкА")
U_t = np.linspace(-3, 3, 50)
I_t = r['p_zero'][0]*U_t + r['p_zero'][1]
ax.plot(U_t, I_t + r['I_center'], 'r-', lw=1.5,
        label=f"Касательная ({r['p_zero'][0]:.1f} мкА/В)")
U_star = r['I_in'] / r['slope']
ax.plot([0], [r['I_center'] + r['I_in']], 'g*', markersize=14)
ax.plot([U_star], [r['I_center'] + r['I_in']], 'r*', markersize=14)
ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
ax.set_xlabel('$U_3$, В'); ax.set_ylabel('$I_3$, мкА')
ax.set_title(f'Обработка зондовой характеристики '
             f'($k_BT_e$ = {r["kTe"]:.2f} эВ)')
ax.grid(True, alpha=0.3); ax.legend(loc='lower right', fontsize=8)
plt.tight_layout(); plt.savefig('probe_analysis.png', dpi=300); plt.close()
print("✓ probe_analysis.png")

fig4, ax = plt.subplots(figsize=(9, 6))
Ip_arr  = np.array([r['Ip']  for r in results])
kTe_arr = np.array([r['kTe'] for r in results])
dkTe_arr = np.array([r['dkTe'] for r in results])
n_e_arr = np.array([r['n_e'] for r in results])
dn_e_arr = np.array([r['dn_e'] for r in results])

ax2 = ax.twinx()
ax.errorbar(Ip_arr, kTe_arr, yerr=dkTe_arr, fmt='ro-',
            capsize=4, label='$k_BT_e$, эВ')
ax2.errorbar(Ip_arr, n_e_arr, yerr=dn_e_arr, fmt='bs--',
             capsize=4, label='$n_e$, см$^{-3}$')
ax.set_xlabel('$I_p$, мА')
ax.set_ylabel('$k_BT_e$, эВ', color='r')
ax2.set_ylabel('$n_e$, см$^{-3}$', color='b')
ax.set_title('Зависимости $k_BT_e$ и $n_e$ от тока разряда')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left'); ax2.legend(loc='upper right')
plt.tight_layout(); plt.savefig('plasma_params.png', dpi=300); plt.close()
print("✓ plasma_params.png")

# ============================================================
# 8. Итоговая таблица
# ============================================================
print("\n" + "=" * 100)
print("ИТОГОВАЯ ТАБЛИЦА")
print("=" * 100)
print(f"{'Ip, мА':>8} {'kTe, эВ':>14} {'n_e, см⁻³':>22} "
      f"{'ωp, рад/с':>14} {'r_De, см':>12} {'r_D, см':>12} "
      f"{'N_D':>8} {'α':>12}")
for r in results:
    print(f"{r['Ip']:>8.3f} {r['kTe']:>7.2f}±{r['dkTe']:<5.2f} "
          f"{r['n_e']:>9.2e}±{r['dn_e']:<8.1e} "
          f"{r['omega_p']:>14.2e} {r['r_De']:>12.2e} {r['r_D']:>12.2e} "
          f"{r['N_D']:>8.1f} {r['alpha']:>12.2e}")
