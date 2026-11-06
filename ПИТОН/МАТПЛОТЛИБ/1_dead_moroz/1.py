import matplotlib.pyplot as plt

for i in range(1, 6):
    name = '00' + str(i) + '.dat'
    f = open(name)
    n = int(f.readline())
    x = []
    y = []
    for _ in range(n):
        a, b = map(float, f.readline().split()[:2])
        x.append(a)
        y.append(b)

    plt.scatter(x, y, s=0.5)
    plt.title('график по набору ' + name)
    plt.axis('equal')
    plt.savefig(name + '.png', dpi=500, bbox_inches='tight')
    plt.close()
