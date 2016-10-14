import mltools as ml
import numpy as np
import matplotlib.pyplot as plt

iris = np.genfromtxt("data/iris.txt",delimiter=None) # loading data
Y = iris[:,-1]
X = iris[:,0:-1]
X2 = iris[:,0:2]
X2,Y = ml.shuffleData(X2,Y)
Xtr , Xte , Ytr , Yte = ml.splitData(X2,Y,0.75) #75/25 train/test


#knn.train(Xtr_,Ytr,2)
#YteHat= knn.predict(Xte)

k_list= [1, 5, 10, 50]
for k in k_list:
    knn= ml.knn.knnClassify()
    knn.train(Xtr,Ytr, k)
    #result= knn.predict(Xte_2f)
    ml.plotClassify2D(knn, Xtr, Ytr)
    plt.show()
'''
'''
K=[1,2,5,10,50,100,200];

errTrain=[]
for i,k in enumerate(K):
    learner = ml.knn.knnClassify()
    learner.train(Xtr,Ytr,k)
    Yhat = learner.predict(Xtr)
    errnum= np.mean(Yhat!=Ytr)
    print Xtr.shape, Xte.shape, Ytr.shape, Yte.shape, Yhat.shape, errnum
    errTrain.append(errnum)
print errTrain
plt.semilogx(errTrain)
plt.show()

errTrain = []
errTest = []
K=[1,2,5,10,50,100,200]
for i,k in enumerate(K):

        learner = ml.knn.knnClassify(Xtr, Ytr, k )
        Yhat_tr = learner.predict(Xtr)
        errTrain.append(np.mean(Ytr != Yhat_tr))

        learner_tst = ml.knn.knnClassify(Xte, Yte, k )
        Yhat_tst = learner.predict(Xte)
        errTest.append(np.mean(Yte != Yhat_tst))
#print errTrain, errTest
plt.semilogx(errTrain,'r')
plt.semilogx(errTest,'g')
plt.show()
