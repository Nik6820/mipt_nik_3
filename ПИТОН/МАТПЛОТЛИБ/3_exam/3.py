import matplotlib.pyplot as plt

data = [line.strip().split(';') for line in open('students.csv') if line.strip()]
grades = sorted({row[2] for row in data}, key=lambda s: float(s.replace(',', '.')))

def stats(index):
    keys = sorted({row[index] for row in data})
    out = {}
    for k in keys:
        marks = [float(row[2].replace(',', '.')) for row in data if row[index] == k]
        n = len(marks)
        out[k] = {'n': n, 'avg': sum(marks) / n}
    return out

preps  = stats(0)
groups = stats(1)

def show(title, d):
    print('\n===', title, '===')
    for k in sorted(d, key=lambda k: -d[k]['avg']):
        s = d[k]
        print(f"{k:>10}  n={s['n']:>3}  avg={s['avg']:.2f}")

show('Преподаватели', preps)
show('Группы', groups)

def top(d, key, reverse=True):
    return sorted(d, key=lambda k: d[k][key], reverse=reverse)

print('\n--- Выводы ---')
h   = top(preps,  'avg', reverse=True)[:3]
nh  = top(preps,  'avg', reverse=False)[:3]
bad = top(groups, 'avg', reverse=False)[:3]

print('Самые халявные препы: ', ', '.join(f"{k} ({preps[k]['avg']:.2f})" for k in h))
print('Самые нехалявные препы:', ', '.join(f"{k} ({preps[k]['avg']:.2f})" for k in nh))
print('Самые раздолбайские группы:', ', '.join(f"{k} ({groups[k]['avg']:.2f})" for k in bad))

def draw(index, title, out_png):
    counts = {}
    for row in data:
        counts.setdefault(row[index], {})
        counts[row[index]][row[2]] = counts[row[index]].get(row[2], 0) + 1

    keys = sorted(counts)
    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = [0] * len(keys)

    for g in grades:
        vals = [counts[k].get(g, 0) for k in keys]
        ax.bar(keys, vals, bottom=bottom, label=g)
        bottom = [b + v for b, v in zip(bottom, vals)]

    ax.set_title(title)
    ax.set_ylabel('Количество студентов')
    ax.legend(title='Оценка')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()

draw(0, 'Распределение оценок по преподавателям', 'by_prep.png')
draw(1, 'Распределение оценок по группам', 'by_group.png')
