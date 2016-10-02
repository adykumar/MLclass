import mltools as ml
import numpy as np

iris = np.genfromtxt("data/iris.txt",delimiter=None) # loading data
Y = iris[:,-1]
X = iris[:,0:-1]
X,Y = ml.shuffleData(X,Y)
Xtr,Xte,Ytr,Yte= ml.splitData(X,Y,0.75) #75/25 train/test

knn= ml.knn.knnClassify()
knn.train(Xtr,Ytr,2)
YteHat= knn.predict(Xte)

ml.plotClassify2D(knn,Xtr,Ytr)
