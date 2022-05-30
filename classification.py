from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) read data
iris_dataset = load_iris()

XX = pd.DataFrame(iris_dataset.data, columns=iris_dataset.feature_names)
yy = pd.DataFrame(iris_dataset.target)

# 2) Split data -> train / test
from sklearn.model_selection import train_test_split
X_train, X_test ,y_train,y_test = train_test_split(XX, yy, test_size=0.3,random_state=0)

# 3) Normalise data
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn import tree
# 4) create tree object
decision_tree = tree.DecisionTreeClassifier(criterion="gini")

# 5) train
decision_tree.fit(X_train_std,y_train)

# 6) predict
y_predict = decision_tree.predict(X_test_std)

# 7) performace 
print("accuracy DT train :{:.2f}".format(decision_tree.score(X_train_std, y_train)))
print("accuracy DT test  :{:.2f}".format(decision_tree.score(X_test_std, y_test)))


# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=6)#K = 1

knn.fit(X_train_std,y_train)

print("accuracy KNN train :{:.2f}".format(knn.score(X_train_std, y_train)))
print("accuracy KNN test  :{:.2f}".format(knn.score(X_test_std, y_test)))


# NN 
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(5,2),random_state=0)
mlp.fit(X_train_std,y_train)
print("accuracy MLP train :{:.2f}".format(mlp.score(X_train_std, y_train)))
print("accuracy MLP test  :{:.2f}".format(mlp.score(X_test_std, y_test)))

import graphviz 
dot_data = tree.export_graphviz(decision_tree, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("iris") 

dot_data = tree.export_graphviz(decision_tree, out_file=None, 
                     feature_names=iris_dataset.feature_names,  
                     class_names=iris_dataset.target_names,  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 

# NN 
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(4,),random_state=0)
y_train = y_train
mlp.fit(X_train_std,y_train.values.ravel())
print("accuracy MLP (train) :{:.2f}".format(mlp.score(X_train_std, y_train)))
print("accuracy MLP (test)  :{:.2f}".format(mlp.score(X_test_std, y_test)))

print(y_train.shape)

