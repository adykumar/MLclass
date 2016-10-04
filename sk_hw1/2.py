import mltools as ml
import numpy as np
import matplotlib.pyplot as plt

iris = np.genfromtxt("data/iris.txt",delimiter=None) # loading data
Y = iris[:,-1]
X = iris[:,0:-1]
X,Y = ml.shuffleData(X,Y)
Xtr,Xte,Ytr,Yte= ml.splitData(X,Y,0.75) #75/25 train/test

Xf2 = iris[:,0:2]
Xtr_2,Xte_2,Ytr_2,Yte_2= ml.splitData(Xf2,Y,0.75) #75/25 train/test

knn= ml.knn.knnClassify()
knn.train(Xtr,Ytr,2)
YteHat= knn.predict(Xte)

k_list= [1, 5, 10, 50]
for i in range(4):
    knn.train(Xtr_2,Ytr_2, k_list[i])
    res= knn.predict(Xte_2)
    ml.plotClassify2D(knn, Xte_2, res)
    plt.show()
