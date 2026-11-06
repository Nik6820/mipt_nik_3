import numpy as np
from PIL import Image

# Список файлов
files = ["lunar01_raw.jpg", "lunar02_raw.jpg", "lunar03_raw.jpg"]

for file_name in files:
    # Считываем картинку
    img = Image.open(file_name)
    data = np.array(img)

    # Находим минимум и максимум
    min_val = data.min()
    max_val = data.max()

    print(f"Файл: {file_name} | Исходный диапазон: от {min_val} до {max_val}")

    # Защита от деления на ноль
    if max_val == min_val:
        updated_data = np.zeros_like(data)
    else:
        # Линейно "растягиваем" диапазон
        updated_data = (data.astype(np.float64) - min_val) / (max_val - min_val) * 255.0
        updated_data = np.round(updated_data).astype(np.uint8)

    # Формируем имя для выходного файла
    out_name = file_name.replace(".jpg", "_enhanced.png")

    # Сохраняем
    res_img = Image.fromarray(updated_data)
    res_img.save(out_name)
    print(f"Сохранено: {out_name}\n")

print("Обработка всех файлов завершена.")
