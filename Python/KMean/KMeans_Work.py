import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

n_samples = 43346
random_state = 200
X, y = make_blobs(n_samples=n_samples, random_state=random_state)

y_predK = KMeans(n_clusters=8, random_state=random_state ).fit_predict(X)

#plt.scatter(X[:,0], X[:,1])
plt.scatter(X[:,0], X[:,1], c = y_predK)

plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
plt.show()