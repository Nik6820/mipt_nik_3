import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

lines = [l for l in open('data.dat') if l.strip()]
frames = []
for i in range(0, len(lines), 2):
    x = list(map(float, lines[i].split()))
    y = list(map(float, lines[i + 1].split()))
    frames.append((x, y))

# общие пределы, одинаковые для всех кадров
xs = [v for x, _ in frames for v in x]
ys = [v for _, y in frames for v in y]
xlim = (min(xs), max(xs))
ylim = (min(ys), max(ys))

fig, ax = plt.subplots(figsize=(8, 8))
writer = PillowWriter(fps=1)

with writer.saving(fig, 'process.gif', dpi=100):
    for i, (x, y) in enumerate(frames):
        ax.plot(x, y, 'o-', color='crimson')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.set_title(f'Кадр {i + 1} из {len(frames)}')

        ax.minorticks_on()
        ax.grid(True, which='major', color='gray', linewidth=0.6)
        ax.grid(True, which='minor', color='gray', linewidth=0.3, alpha=0.5)

        writer.grab_frame()
