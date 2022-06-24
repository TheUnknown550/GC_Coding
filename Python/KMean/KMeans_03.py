import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

iris = datasets.load_iris()

X = iris.data
y = iris.target

print (iris.feature_names)
print (iris.target_names)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

km = KMeans(n_clusters = 3, random_state=21)
km.fit(X_scaled)

centers = km.cluster_centers_
print(centers)
centers_original = scaler.inverse_transform(centers)
print(centers_original)


#this will tell us to which cluster does the data observations belong.
new_labels = km.labels_
# Plot the identified clusters and compare with the answers
fig, axes = plt.subplots(1, 2, figsize=(16,8))
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='gist_rainbow',
edgecolor='k', s=150)
axes[1].scatter(X[:, 0], X[:, 1], c=new_labels, cmap='Wistia',
edgecolor='k', s=150)
axes[0].set_xlabel('Sepal length', fontsize=18)
axes[0].set_ylabel('Sepal width', fontsize=18)
axes[1].set_xlabel('Sepal length', fontsize=18)
axes[1].set_ylabel('Sepal width', fontsize=18)
axes[0].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
axes[1].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
axes[0].set_title('Actual', fontsize=18)
axes[1].set_title('Predicted', fontsize=18)

plt.scatter(X[:,0], X[:,1], c=y, cmap='gist_rainbow')

plt.scatter(X[:,0], X[:,1], c=km.labels_, cmap='Wistia')
plt.plot(centers_original[:,0], centers_original[:,1], 'k+')

plt.xlabel('Sepal length', fontsize=18)
plt.ylabel('Sepal width', fontsize=18)
plt.title('K-Means', fontsize=18)
plt.title('Actual', fontsize=18)
plt.show()