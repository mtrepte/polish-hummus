from sklearn.decomposition import PCA as sklearn_PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def cross_validate(model, X, y):
	folds = 10

	# print scores for each fold
	for fold in range(folds):
		score = model(X, y, fold, folds)
		print(fold, score)


def PCA(X, y, fold, folds):
	# hyperparameters
	n_dim = 2
	
	# color mapping
	labels = ['chill', 'hate']
	colors = ['blue', 'red']

	# apply PCA to featurized tweets
	pca_model = sklearn_PCA(n_components=n_dim)
	X = pca_model.fit_transform(X)

	# plot data
	for i in range(len(X)):
		x1, x2 = X[i, 0], X[i, 1]
		label = y[i]
		if i % 100 == 0:
			print(i, x1, x2, label)
		plt.scatter(x1, x2, c=colors[label], cmap=plt.cm.Paired)

	# legend
	patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
	plt.legend(handles=patches, fontsize=10)

	plt.show()

def knn(X, y, fold, folds):
	# hyperparameters
	n_neighbors = 5
	n_components = 100 # vec dim unrreduced is 28113

	print("fold: ", fold)

	print("PCA-ing")

	# reduce dimensions of vectors to expedite prediction and feature select
	pca_model = sklearn_PCA(n_components=n_components)
	X = pca_model.fit_transform(X)

	print("partitioning")

	# partition dataset
	X_train, y_train, X_test, y_test = partition(X, y, fold, folds)

	print("training")

	# train
	knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
	knn_model.fit(X_train, y_train)

	print("score")

	# compute accuracy
	score = knn_model.score(X_test, y_test)

	return score

def random_forest(X, y, fold, folds):
	# hyperparameters
	forests_num = 20

	# partition dataset
	X_train, y_train, X_test, y_test = partition(X, y, fold, folds)
	
	# train
	random_forest = RandomForestClassifier(n_estimators=forests_num)
	random_forest.fit(X_train, y_train)

	# computer accuracy
	score = random_forest.score(X_test, y_test)

	return score

def partition(X, y, fold, num_folds):
	partition_start = int(fold / num_folds * len(X))
	partition_end = int((fold + 1) / num_folds * len(X))

	X_train = X[partition_start:partition_end]
	y_train = y[partition_start:partition_end]

	print(type(X))
	X = list(X)
	X_test = np.array(X[:partition_start] + X[partition_end:])
	y = list(y)
	y_test = np.array(y[:partition_start] + y[partition_end:])

	print(len(X_train), len(y_train), len(X_test), len(y_test))

	return [X_train, y_train, X_test, y_test]