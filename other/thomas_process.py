import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.neighbors import KernelDensity
import scipy.stats as stats

### Simulation
L = 100  # Size of the region (L*L)

#parameters
lambda_p = 0.05
v = 10
sigma = 2

# Centers of the clusters
N_c = np.random.poisson(lambda_p * L * L) #Number of clusters
centers_x = np.random.uniform(0, L, N_c)
centers_y = np.random.uniform(0, L, N_c)

# Secondary points
points_x, points_y = [], []
for i in range(N_c):
    L_x=[]
    L_y=[]
    N_points = np.random.poisson(v)  # Number of points for the cluster
    x_offsets = np.random.normal(0, sigma, N_points)
    y_offsets = np.random.normal(0, sigma, N_points)

    L_x= [x+centers_x[i] for x in x_offsets]
    L_y= [y+centers_y[i] for y in y_offsets]

    points_x.append(L_x)
    points_y.append(L_y)

# Plot
plt.figure(figsize=(8,8))

plt.scatter(centers_x, centers_y, s=50, color='green', marker='x', label="Centers of clusters")
for i in range(N_c):
    plt.scatter(points_x[i], points_y[i], s=5)


plt.xlim(0, L)
plt.ylim(0, L)
plt.legend()
plt.title("Thomas Process")
plt.show()

### pair correlation function

def estimate_pcf(points, L=100):

    distances= distance.pdist(points) #Distances between each pair of points

    r_values = np.linspace(0, 200, 200)

    #Estimation of product density with the kernel

    #kde = KernelDensity(bandwidth=0.01, kernel='gaussian')
    #kde.fit(distances[:, None])
    #density = np.exp(kde.score_samples(r_values[:, None]))

    kde = stats.gaussian_kde(distances, bw_method=0.2)
    density = kde(r_values)

    #Theorical density

    lambda_hat= len(points)/(L**2)
    print(lambda_hat)

    g_r = density / (lambda_hat**2)

    return r_values, g_r


points=[]
for i in range(len(points_x)):
    for j in range(len(points_x[i])):
        L=np.array([points_x[i][j],points_y[i][j]])
        points.append(L)

points=np.array(points)

r_values, g_r = estimate_pcf(points)

plt.plot(r_values, g_r, label=" Estimation of g(r)", color='blue')
plt.xlabel("r")
plt.ylabel("Pair Correlation Function g(r)")
plt.legend()
plt.show()