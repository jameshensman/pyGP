# -*- coding: utf-8 -*-
import sys
import numpy as np
import pylab as pb

sys.path.append('../')
import pyGP

#sigmoid helper
sigmoid = lambda x : 1./(1+np.exp(-x))

#generate some 2D test data.  this is done by sampling from a GP, pushing the result through a sigmoid, and sampling (binomial) from the value to get a label.
X = np.random.rand(50,2)*2 -1
k = pyGP.kernels.full_RBF(10,np.ones(2))
myGP = pyGP.regression.basic(kernel=k)
proportion = 0
#ensure roughly 50-50 split
while (proportion < .40) or (proportion >.60):
	print 'resampling'
	F = myGP.sample(X)
	Y = np.random.binomial(1,sigmoid(F))
	Y *= 2# convert to +-1
	Y -= 1
	Y = Y.reshape(Y.size,1)
	proportion = np.sum(Y<0)/float(Y.size)

#build classifier
myGP_bl = pyGP.classification.binary_laplace(X = X, Y = Y, kernel = k)
myGP_bl.laplace_approximation()

#build meshgrid
xx,yy = np.mgrid[-1:1:100j,-1:1:100j]
Xtest = np.vstack((xx.flatten(),yy.flatten())).T

#test mesh points
Ygrid = myGP_bl.predict_MAP(Xtest)
Ygrid = Ygrid.reshape(xx.shape)

pb.figure()
pb.scatter(X[:,0],X[:,1],60,Y,cmap=pb.cm.gray)
pb.contour(xx,yy,Ygrid,[0.5],linewidths=2,colors='k')
pb.contour(xx,yy,Ygrid,[0.75,0.25],linewidths=1,colors='k')
pb.title("'True' kernel parameters")

myGP_bl.set_params(np.random.randn(myGP_bl.kernel.nparams))
myGP_bl.update()
Ygrid = myGP_bl.predict_MAP(Xtest)
Ygrid = Ygrid.reshape(xx.shape)
pb.figure()
pb.scatter(X[:,0],X[:,1],60,Y,cmap=pb.cm.gray)
pb.contour(xx,yy,Ygrid,[0.5],linewidths=2,colors='k')
pb.contour(xx,yy,Ygrid,[0.75,0.25],linewidths=1,colors='k')
pb.title("Random kernel parameters")

myGP_bl.find_kernel_parameters()
myGP_bl.update()
Ygrid = myGP_bl.predict_MAP(Xtest)
Ygrid = Ygrid.reshape(xx.shape)

pb.figure()
pb.scatter(X[:,0],X[:,1],60,Y,cmap=pb.cm.gray)
pb.contour(xx,yy,Ygrid,[0.5],linewidths=2,colors='k')
pb.contour(xx,yy,Ygrid,[0.75,0.25],linewidths=1,colors='k')
pb.title("Learnt (MAP) kernel parameters")

pb.show()

