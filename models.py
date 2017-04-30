from sklearn.decomposition import PCA as sklearn_PCA
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import math
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def PCA(X, y):
    # hyperparameters
    n_dim = 2
    
    # label color mapping
    labels = ['hate', 'chill']
    colors = ['red', 'blue']

    
    # apply PCA to featurized tweets
    model = sklearn_PCA(n_components=n_dim)
    X = .fit_transform(X)

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

def knn(X, y):
	#hyperparameters
	n_neighbors = 5




# def random_forest(X, y):
#     # partition dataset
#     training_percent = .70
#     partition_index = int(training_percent * len(X))
#     X_train = X[:partition_index]
#     X_test = X[partition_index:]
#     y_train = y[:partition_index]
#     y_test = y[partition_index:]
    
#     # hyperparameters
#     forests_num = 20
    
#     # train
#     random_forest = RandomForestClassifier(n_estimators=forests_num)
#     random_forest.fit(X_train, y_train)

#     # predict
#     length = len(X_test)
#     X_predicts = [random_forest.predict(X_test[i], y_test[i]) for i in length]
    
#     return X_predicts

# def rnn(X, y):
#     #hyperparameters
#     look_back = 1
    
#     # normalizing featurized tweets
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     X = scalar.fit_transform(X)
    
#     # partition dataset
#     training_percent = .70
#     partition_index = int(training_percent * len(X))
#     X_train = X[:partition_index]
#     X_test = X[partition_index:]
#     y_train = y[:partition_index]
#     y_test = y[partition_index:]
    
# # convert an array of values into a dataset matrix
# def create_dataset(dataset, look_back=1):
#     dataX, dataY = [], []
#     for i in range(len(dataset)-look_back-1):
#         a = dataset[i:(i+look_back), 0]
#         dataX.append(a)
#         dataY.append(dataset[i + look_back, 0])
#     return numpy.array(dataX), numpy.array(dataY)