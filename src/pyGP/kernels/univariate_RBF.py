# -*- coding: utf-8 -*-
import numpy as np

class RBF:
	"""Radial Basis Funcion (or 'Squared Exponential') kernel, with the same scale in all directions...
	k(x_i,x_j) = \alpha \exp \{ -\gamma ||x_1-x_2||^2 \}
	"""
	def __init__(self,alpha,gamma):
		self.alpha = np.exp(alpha)
		self.gamma = np.exp(gamma)
		self.nparams = 2
		
	def set_params(self,new_params):
		assert new_params.size == self.nparams
		# try to unpack np array safely
		self.alpha,self.gamma = np.exp(new_params).copy().flatten()  
		
	def get_params(self):
		# return np.array([self.alpha, self.gamma])
		return np.log(np.array([self.alpha, self.gamma]))
		
	def __call__(self,x1,x2):
		"""
		Evaluates the kernel at a set of location pairs
		
		Arguments
		----------
		x1, x2 : array
			these are locations at which to evaluate the kernel
			
		Notes
		----------
		The inputs are n x d arrays where n is the number of locations and d
		is the dimension of the space. Therefore, the second dimension of 
		these arrays has to match!
		"""
		if len(x1.shape)>1:
			N1,D1 = x1.shape
			N2,D2 = x2.shape
		else:
			N1,D1 = x1.shape[0],1
			N2,D2 = x2.shape[0],1
			
		assert D1==D2, "Vectors must be of matching dimension"
		# use broadcasting to avoid for loops. should be uber fast
		diff = x1.reshape(N1,1,D1) - x2.reshape(1,N2,D2)
		# evaluate the kernel at each point
		K = self.alpha*np.exp(-np.sum(np.square(diff),-1)*self.gamma)
		return K
		
	def gradients(self,x1):
		"""Calculate the gradient of the matrix K wrt the (log of the) free 
		parameters"""
		N1,D1 = x1.shape
		diff = x1.reshape(N1,1,D1)-x1.reshape(1,N1,D1)
		diff = np.sum(np.square(diff),-1)
		#dalpha = np.exp(-diff*self.gamma)
		dalpha = self.alpha*np.exp(-diff*self.gamma)
		#dgamma = -self.alpha*diff*np.exp(-diff*self.gamma)
		dgamma = -self.alpha*self.gamma*diff*np.exp(-diff*self.gamma)
		return (dalpha, dgamma)
		
	def gradients_wrt_data(self,x1,indexn=None,indexd=None):
		"""compute the derivative matrix of the kernel wrt the _data_. This 
		returns a list of matrices: each matrix is NxN, and there are N*D 
		of them."""
		N1,D1 = x1.shape
		diff = x1.reshape(N1,1,D1)-x1.reshape(1,N1,D1)
		diff = np.sum(np.square(diff),-1)
		expdiff = np.exp(-self.gamma*diff)
		
		if (indexn==None) and(indexd==None):#calculate all gradients
			rets = []
			for n in range(N1):
				for d in range(D1):
					K = np.zeros((N1,N1))
					K[n,:] = -2*self.alpha*self.gamma*(x1[n,d]-x1[:,d])*expdiff[n,:]
					K[:,n] = K[n,:]
					rets.append(K.copy())
			return rets
		else:
			K = np.zeros((N1,N1))
			K[indexn,:] = -2*self.alpha*self.gamma*(x1[indexn,indexd]-x1[:,indexd])*expdiff[indexn,:]
			K[:,indexn] = K[indexn,:]
			return 
