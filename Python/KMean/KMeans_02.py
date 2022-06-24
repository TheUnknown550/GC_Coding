import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

n_samples = 2000
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state)

# k means determine k
sse = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(X)
    sse.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Plot "Elbow curve"
plt.plot(K, sse,'bx-',linewidth=2,markersize=8)
plt.xlabel('K')
plt.ylabel('Sum of Squared Disctance')
plt.title('The Elbow Method presents the optimal K')
plt.show()



