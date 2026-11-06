import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("flights.csv")
df.columns = [c.strip() for c in df.columns]

stats = df.groupby("CARGO").agg(
    flights=("PRICE", "size"),
    total_price=("PRICE", "sum"),
    total_weight=("WEIGHT", "sum"),
)
print(stats)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col, title in zip(
    axes,
    ["flights", "total_price", "total_weight"],
    ["Количество рейсов", "Суммарная стоимость груза", "Суммарная масса груза"],
):
    stats[col].plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel("Авиакомпания")
    ax.set_ylabel("Значение")
    ax.tick_params(axis="x", rotation=0)
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.0f}",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center", va="bottom", fontsize=9,
        )

plt.tight_layout()
plt.show()
