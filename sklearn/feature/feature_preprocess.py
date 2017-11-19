from sklearn import preprocessing
import numpy as np

x_array = np.array([[1., 0., 2.], [-2.,-1.,0.],[1.1, 0.5,-1.2]])
x_scale = preprocessing.scale(x_array)
print(x_scale)
print(x_scale.mean(axis=0))
print(x_scale.std(axis=0))

scaler = preprocessing.StandardScaler().fit(x_array)
print(scaler)
print(scaler.mean_)
print(scaler.scale_)
print(scaler.std_)

x_train = np.array([[1,0,-1],[2,0,0],[-1,0,1]])
minMaxScaler = preprocessing.MinMaxScaler()
transform_train =  minMaxScaler.fit_transform(x_train)
print(transform_train)

maxAbsScaler = preprocessing.MaxAbsScaler()
x_train_max_abs = maxAbsScaler.fit_transform(x_train)
print(x_train_max_abs)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
# X_train_trans = quantile_transformer.fit_transform(X_train)
# print(X_train_trans)

oneHotEnc = preprocessing.OneHotEncoder()
print(oneHotEnc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])  )
print( oneHotEnc.transform([[0,1,2]]).toarray())