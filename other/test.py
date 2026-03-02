import matplotlib.pyplot as plt
import numpy as np

y = np.random.uniform(0, 1, 10000)
x = np.linspace(0, 1, 10000)

plt.scatter(x, y)
plt.savefig("img/plot.eps", format='eps', bbox_inches='tight')
plt.close()
