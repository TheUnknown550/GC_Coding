import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

n_samples = 2000
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state)

y_predK3 = KMeans(n_clusters=3, random_state=random_state).fit_predict(X)

y_predK4 = KMeans(n_clusters=4, random_state=random_state ).fit_predict(X)

#plt.scatter(X[:,0], X[:,1])
#plt.scatter(X[:,0], X[:,1], c = y_predK3)
plt.scatter(X[:,0], X[:,1], c = y_predK4)
plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
plt.show()