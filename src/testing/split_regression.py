# -*- coding: utf-8 -*-
import sys
import numpy as np
import pylab as pb

sys.path.append('../')
import pyGP

class split_kernel_1D:
	def __init__(self,ln_alpha=0.,ln_gamma=0., split=0):
		"""A kernel with two parts - two regions which are both smooth but independent
		
		Notes
		----------
		This only works for 1D at the moment and the regions are defined by a split-point. higher dimensional versions are possible in future..."""
		
		self.k = pyGP.kernels.RBF(ln_alpha,ln_gamma)
		self.split = split
		
		self.nparams = self.k.nparams
	
	def __call__(self,x1,x2):
		N1,D1 = x1.shape
		N2,D2 = x2.shape
		assert (D1==1) and (D2==1), "This kernel can only handle 1D data"
		
		K = self.k(x1,x2)
		
		zeros =  np.zeros((N1,N2))
		in_lhs = (x1<self.split)*(x2.T<self.split)
		in_rhs = (x1>self.split)*(x2.T>self.split)
		return np.where(in_lhs,K,zeros) + np.where(in_rhs,K,zeros)
	
	def gradients(self,x1):
		N1,D1 = x1.shape
		assert (D1==1) , "This kernel can only handle 1D data"
		
		K = self.k.gradients(x1)
		
		zeros =  np.zeros((self.nparams,N1,N1))
		in_lhs = (x1<self.split)*(x1.T<self.split)
		in_rhs = (x1>self.split)*(x1.T>self.split)
		return np.where(in_lhs,K,zeros) + np.where(in_rhs,K,zeros)
		
	def get_params(self):
		return self.k.get_params()
	def set_params(self,params):
		self.k.set_params(params)
		
		

#generate some data
X = np.linspace(-5,5,100).reshape(100,1)
Y = np.where(X<0,np.cos(X),np.sin(X-1)) + np.random.randn(100,1)*0.05

#stuff for plotting
def plot(GP):
	pb.figure()
	pb.plot(X,Y,'r.')
	mu,var = GP.predict(X)
	pb.plot(X,mu,'b',linewidth=1.5)
	upper,lower = mu + 2*np.sqrt(var), mu - 2*np.sqrt(var)
	pb.plot(X,upper,'k--',X,lower,'k--')
	pb.title('GP Regression with a discontinuity')
	pb.xlabel('X')
	pb.ylabel('Y')
	
#initialise pyGP objects
k = split_kernel_1D()
myGP = pyGP.regression.basic_regression.GP(X=X,Y=Y,kernel=k)

#line-search for split position (uniform prior)
Nsearch = 200
splits = np.linspace(myGP.X.min(),myGP.X.max(),Nsearch)
marginals = np.zeros(Nsearch)
for i,s in enumerate(splits):
	k.split = s
	myGP.find_kernel_params()
	
	marginals[i] = -myGP.ll()
best_split = splits[np.argmax(marginals)]

#final solution  and plotting
k.split = best_split
myGP.find_kernel_params()
plot(myGP)

#plot likelihood of split position
pb.figure()
pb.plot(X,Y,'r.')#
pb.ylabel('data (red)')
pb.twinx()
pb.plot((splits+myGP.xmean)*myGP.xstd,marginals,'g')
pb.ylabel('log likelihood of split position (green)')

pb.show()
		
		
		