import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Параметры установки
# ============================================================
mu0 = 4 * np.pi * 1e-7
Nt0, Nt1 = 1750, 300
D, dt = 0.10, 0.01
Nc0, Nc1 = 825, 435
dc, lc = 0.07, 0.8

R_T, R_C, R0 = 0.0, 60.0, 0.0
R_M_first  = 50.0   # ПЕРВЫЙ (правильный)
R_M_second = 60.0   # ВТОРОЙ (ошибочный)
R_M_calib  = 10.0

R_first  = R_T + R_M_first  + R0    # 50 Ом
R_second = R_T + R_M_second + R0    # 60 Ом
R_calib  = R_C + R_M_calib  + R0    # 70 Ом

dI_c, dx_c = 1.4566, 9.3
delta_I, delta_x = 0.001, 0.2

# ============================================================
# 2. Коэффициенты пересчёта
# ============================================================
def compute_K(R_loop):
    return mu0 * (dc/dt)**2 * (Nc0*Nc1/Nt1) * (R_loop/R_calib) * (1/lc) * (dI_c/dx_c)

K_first  = compute_K(R_first)
K_second = compute_K(R_second)

dK_rel = np.sqrt((delta_I/dI_c)**2 + (delta_x/dx_c)**2)
delta_K_first  = K_first  * dK_rel
delta_K_second = K_second * dK_rel
delta_dB_step  = K_first * delta_x
delta_H_step   = Nt0 / (np.pi * D) * delta_I

print("=" * 60)
print("ПАРАМЕТРЫ РАСЧЁТА")
print("=" * 60)
print(f"R_first  = {R_first} Ом,  R/R_c = {R_first/R_calib:.3f}")
print(f"R_second = {R_second} Ом, R/R_c = {R_second/R_calib:.3f}")
print(f"K_first  = ({K_first*1000:.4f} ± {delta_K_first*1000:.4f}) Тл/см × 10⁻³")
print(f"K_second = ({K_second*1000:.4f} ± {delta_K_second*1000:.4f}) Тл/см × 10⁻³")
print(f"δ(ΔB) на шаг = {delta_dB_step*1000:.4f} мТл")
print(f"δH = {delta_H_step:.3f} А/м")
print(f"δK/K = {dK_rel*100:.2f} %")
print("=" * 60)

# ============================================================
# 3. Токи на ступенях
# ============================================================
I_steps = np.array([0, 0.0152, 0.0279, 0.0389, 0.0442, 0.0556,
                    0.0663, 0.0945, 0.1567, 0.2498, 0.5128, 1.4560])

# ============================================================
# 4. Данные
# ============================================================
# Первый проход — ПРАВИЛЬНЫЙ (флип только П2)
first_attempt = [
    (11,10,20.6), (10,9,19.4), (9,8,11.7), (8,7,8.1), (7,6,4.1),
    (6,5,2.5), (5,4,2.3), (4,3,1.3), (3,2,1.6), (2,1,1.8), (1,0,4.5),
    (0,1,8.1), (1,2,11.1), (2,3,24.6), (3,4,17.5), (4,5,29.2),
    (5,6,16.0), (6,7,25.6), (7,8,26.2), (8,9,18.2), (9,10,23.0), (10,11,20.9),
    (11,10,20.1), (10,9,19.2), (9,8,11.6), (8,7,11.2), (7,6,3.9),
    (6,5,2.9), (5,4,1.9), (4,3,0.7), (3,2,1.0), (2,1,1.3), (1,0,3.8),
    (0,1,8.3), (1,2,11.2), (2,3,24.4), (3,4,17.9), (4,5,28.9),
    (5,6,16.3), (6,7,26.0), (7,8,26.3), (8,9,18.2), (9,10,23.1), (10,11,20.9)
]

# Второй проход — НЕПРАВИЛЬНЫЙ (флип П1+П2), исправленная версия
second_attempt = [
    (11,10,20.3), (10,9,17.9), (9,8,11.6), (8,7,9.2), (7,6,4.3),
    (6,5,2.5), (5,4,1.4), (4,3,0.9), (3,2,1.1), (2,1,1.4), (1,0,2.6),
    (0,1,7.7), (1,2,10.4), (2,3,22.8), (3,4,16.2), (4,5,26.2),
    (5,6,14.8), (6,7,23.5), (7,8,23.9), (8,9,16.9), (9,10,21.0), (10,11,19.0),
    (11,10,19.1), (10,9,20.4), (9,8,12.1), (8,7,10.0), (7,6,3.7),
    (6,5,1.5), (5,4,1.3), (4,3,0.8), (3,2,1.2), (2,1,1.0), (1,0,2.4),
    (0,1,7.8), (1,2,10.7), (2,3,24.3), (3,4,16.6), (4,5,27.8),
    (5,6,15.1), (6,7,25.0), (7,8,25.2), (8,9,17.1), (9,10,21.9), (10,11,19.8)
]

initial_curve = [
    (0,1,3.4), (1,2,6.6), (2,3,11.5), (3,4,5.9), (4,5,11.3),
    (5,6,8.6), (6,7,16.8), (7,8,21.0), (8,9,16.1), (9,10,21.0), (10,11,19.4)
]

# ============================================================
# 5. Обработка
# ============================================================
def process_loop(data, K_local, delta_x_step=delta_x):
    dH_step = Nt0 / (np.pi * D) * delta_I
    H_list, B_list, dH_list, dB_list = [], [], [], []
    H_start = Nt0 * I_steps[data[0][0]] / (np.pi * D)
    H_list.append(H_start); B_list.append(0.0)
    dH_list.append(dH_step); dB_list.append(0.0)

    B = 0.0; var_dB = 0.0
    for i, (from_step, to_step, dx) in enumerate(data):
        I = I_steps[to_step]
        if i < 11:
            H =  Nt0 * I / (np.pi * D);  dB = -K_local * dx
        elif i < 22:
            H = -Nt0 * I / (np.pi * D);  dB = -K_local * dx
        elif i < 33:
            H = -Nt0 * I / (np.pi * D);  dB =  K_local * dx
        else:
            H =  Nt0 * I / (np.pi * D);  dB =  K_local * dx
        B += dB
        var_dB += (K_local * delta_x_step)**2
        H_list.append(H); B_list.append(B)
        dH_list.append(dH_step); dB_list.append(np.sqrt(var_dB))

    H_arr = np.array(H_list); B_arr = np.array(B_list)
    B_center = (np.max(B_arr) + np.min(B_arr)) / 2.0
    B_arr = B_arr - B_center
    return H_arr, B_arr, np.array(dH_list), np.array(dB_list)

def process_initial(data, K_local, delta_x_step=delta_x):
    dH_step = Nt0 / (np.pi * D) * delta_I
    H_list, B_list, dH_list, dB_list = [], [], [], []
    H_list.append(0.0); B_list.append(0.0)
    dH_list.append(dH_step); dB_list.append(0.0)
    B = 0.0; var_dB = 0.0
    for (from_step, to_step, dx) in data:
        I = I_steps[to_step]
        H = Nt0 * I / (np.pi * D)
        dB = K_local * dx
        B += dB
        var_dB += (K_local * delta_x_step)**2
        H_list.append(H); B_list.append(B)
        dH_list.append(dH_step); dB_list.append(np.sqrt(var_dB))
    return (np.array(H_list), np.array(B_list),
            np.array(dH_list), np.array(dB_list))

print("\nПРАВИЛЬНЫЙ проход (флип только П2):")
H_loop, B_loop, dH_loop, dB_loop = process_loop(first_attempt, K_first)
print(f"  B ∈ [{np.min(B_loop):.3f}, {np.max(B_loop):.3f}] Тл")

H_wrong, B_wrong, dH_wrong, dB_wrong = process_loop(second_attempt, K_second)
print(f"  B (ошибочный) ∈ [{np.min(B_wrong):.3f}, {np.max(B_wrong):.3f}] Тл")

H_init, B_init, dH_init, dB_init = process_initial(initial_curve, K_first)

# ============================================================
# 6. Характеристики
# ============================================================
B_s = np.max(np.abs(B_loop))
B_s_err = np.sqrt((np.sqrt(len(B_loop))*delta_dB_step)**2 + (dK_rel*B_s)**2)

B_r = B_loop[np.argmin(np.abs(H_loop))]
H_c = H_loop[np.argmin(np.abs(B_loop))]
H_c_err = np.sqrt(delta_H_step**2 + (B_s_err/(mu0*5000))**2)

print("\n" + "=" * 60)
print("ХАРАКТЕРИСТИКИ (по правильному проходу)")
print("=" * 60)
print(f"B_s = ({B_s:.3f} ± {B_s_err:.3f}) Тл")
print(f"B_r = ({B_r:.3f} ± 0.020) Тл")
print(f"H_c = ({H_c:.1f} ± {H_c_err:.1f}) А/м  (|H_c| ≈ {abs(H_c):.1f} А/м)")

I_big = 0.2
idx_hot_loop = [i+1 for i, (a,b,_) in enumerate(first_attempt)
                if I_steps[a] > I_big or I_steps[b] > I_big]
idx_hot_init = [i+1 for i, (a,b,_) in enumerate(initial_curve)
                if I_steps[a] > I_big or I_steps[b] > I_big]

# ============================================================
# 7. Графики
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
ax.errorbar(H_loop, B_loop, xerr=dH_loop, yerr=dB_loop,
            fmt='b.-', capsize=2, alpha=0.7, label='Правильный проход')
ax.plot(H_loop[idx_hot_loop], B_loop[idx_hot_loop],
        'r.', markersize=12, label='Большие токи (тепловой дрейф)')
ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
ax.set_xlabel('$H$, А/м'); ax.set_ylabel('$B$, Тл')
ax.set_title('Петля гистерезиса (правильный проход)')
ax.grid(True, alpha=0.3); ax.legend()

ax = axes[0, 1]
ax.errorbar(H_init, B_init, xerr=dH_init, yerr=dB_init,
            fmt='go-', capsize=2, alpha=0.7, label='Начальная кривая')
ax.plot(H_init[idx_hot_init], B_init[idx_hot_init],
        'r.', markersize=12, label='Большие токи (тепловой дрейф)')
ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
ax.set_xlabel('$H$, А/м'); ax.set_ylabel('$B$, Тл')
ax.set_title('Начальная кривая намагничивания')
ax.grid(True, alpha=0.3); ax.legend()

ax = axes[1, 0]
ax.plot(H_loop, B_loop, 'b.-', alpha=0.7, label='Петля (правильная)')
ax.plot(H_init, B_init, 'go-', alpha=0.7, label='Начальная кривая')
ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
ax.set_xlabel('$H$, А/м'); ax.set_ylabel('$B$, Тл')
ax.set_title('Петля гистерезиса и начальная кривая')
ax.grid(True, alpha=0.3); ax.legend()

ax = axes[1, 1]
ax.plot(H_wrong, B_wrong, 'r.-', alpha=0.7, label='Ошибочный (П1+П2)')
ax.plot(H_loop,  B_loop,  'b.-', alpha=0.9, label='Правильный (только П2)')
ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
ax.set_xlabel('$H$, А/м'); ax.set_ylabel('$B$, Тл')
ax.set_title('Сравнение проходов')
ax.grid(True, alpha=0.3); ax.legend()

plt.tight_layout()
plt.savefig('hysteresis_plots.png', dpi=300)
plt.show()

# ============================================================
# 8. Проверка замкнутости
# ============================================================
segs = [slice(0,11), slice(11,22), slice(22,33), slice(33,44)]
print("\nСуммы Δx по сегментам:")
print("  Правильный проход:")
for i, s in enumerate(segs):
    print(f"    Сегмент {i+1}: {sum(d for _,_,d in first_attempt[s]):.1f} см")
print("  Ошибочный проход:")
for i, s in enumerate(segs):
    print(f"    Сегмент {i+1}: {sum(d for _,_,d in second_attempt[s]):.1f} см")
