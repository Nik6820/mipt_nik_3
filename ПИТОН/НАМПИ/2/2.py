import numpy as np
import matplotlib.pyplot as plt

files = ["signal01.dat", "signal02.dat", "signal03.dat"]

for file_name in files:
    try:
        # Считываем данные
        data = np.loadtxt(file_name)
        
        # Кумулятивная сумма для быстрого вычисления среднего
        c = np.cumsum(data)
        filtered = np.zeros_like(data)
        
        # 1. Стартовые точки, где окно меньше 10 (индексы от 0 до 9)
        # Делим накопленную сумму на количество элементов (1, 2, 3... 10)
        n_start = min(10, len(data))
        counts = np.arange(1, n_start + 1)
        filtered[:n_start] = c[:n_start] / counts
        
        # 2. Остальные точки, где окно ровно 10 (индексы от 10 до конца)
        # Это разность текущей кумулятивной суммы и суммы 10 шагов назад, деленная на 10
        if len(data) > 10:
            filtered[10:] = (c[10:] - c[:-10]) / 10.0
            
        # --- Отрисовка графиков ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # График сырого сигнала
        axes[0].plot(data, color='steelblue')
        axes[0].set_title(f'Сырой сигнал ({file_name})')
        axes[0].grid(True)
        
        # График отфильтрованного сигнала
        axes[1].plot(filtered, color='steelblue')
        axes[1].set_title('После фильтра')
        axes[1].grid(True)
        
        # Красиво размещаем элементы
        plt.tight_layout()
        
        # Сохраняем результат в картинку
        out_img = file_name.replace(".dat", "_plot.png")
        plt.savefig(out_img)
        plt.close(fig) # Закрываем фигуру, чтобы не забивать память
        
        print(f"Файл {file_name} обработан. График сохранен как {out_img}")

    except FileNotFoundError:
        print(f"Файл {file_name} не найден в папке.")
    except Exception as e:
        print(f"Ошибка при обработке {file_name}: {e}")
