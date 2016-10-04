import numpy as np
import matplotlib.pyplot as plt
iris = np.genfromtxt("data/iris.txt",delimiter=None) # load the text file
Y = iris[:,-1] # target value is the last column
X = iris[:,0:-1] # features are the other columns

Xpart = X[:5]

''' 1a '''
print "Data Points:",X.shape[0]
print "Features:",X.shape[1]
#print X

''' 1b '''
plt.hist(X)
plt.show()

#print Xpart

'''1c '''
a= np.array(X)
xmean= np.mean(a, axis=0)
print "xmean",xmean

'''1d'''
xvar= np.var(a, axis=0)
print "xvar",xvar
xstd= np.std(a, axis=0)
print "xstd",xstd

'''1e'''
# subtract mean value from each feature
print a-xmean
xnorm= (a-xmean)/xstd
print xnorm

'''1f'''

plt.scatter(xnorm[:,0],xnorm[:,1], c=Y)
plt.gray()
plt.show()

plt.scatter(xnorm[:,0],xnorm[:,2], c=Y)
plt.gray()
plt.show()

plt.scatter(xnorm[:,0],xnorm[:,3], c=Y)
plt.gray()
plt.show()
