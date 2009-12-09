# -*- coding: utf-8 -*-
class full_RBF:
	def __init__(self,alpha,gammas):
		self.gammas = np.exp(gammas.flatten())
		self.dim = gammas.size
		self.alpha = np.exp(alpha)
		self.nparams = self.dim+1
		
	def set_params(self,params):
		assert params.size==self.nparams
		self.alpha = np.exp(params.flatten()[0])
		self.gammas = np.exp(params.flatten()[1:])
	
	def get_params(self):
		return np.log(np.hstack((self.alpha,self.gammas)))
		
	def __call__(self,x1,x2):
		N1,D1 = x1.shape
		N2,D2 = x2.shape
		assert D1==D2, \
			"Vectors must be of matching dimension"
		assert D1==self.dim, \
			"That data does not match the dimensionality of this kernel"
		diff = x1.reshape(N1,1,D1)-x2.reshape(1,N2,D2)
		diff = self.alpha*np.exp(-np.sum(np.square(diff)*self.gammas,-1))
		return diff
		
	def gradients(self,x1):
		"""Calculate the gradient of the matrix K wrt the (log of the) free parameters"""
		N1,D1 = x1.shape
		diff = x1.reshape(N1,1,D1)-x1.reshape(1,N1,D1)
		sqdiff = np.sum(np.square(diff)*self.gammas,-1)
		expdiff = np.exp(-sqdiff)
		grads = [-g*np.square(diff[:,:,i])*self.alpha*expdiff for i,g in enumerate(self.gammas)]
		grads.insert(0, self.alpha*expdiff)
		return grads
	
	def gradients_wrt_data(self,x1,indexn=None,indexd=None):
		"""compute the derivative matrix of the kernel wrt the _data_. Crazy
		This returns a list of matices: each matrix is NxN, and there are N*D of them!"""
		N1,D1 = x1.shape
		diff = x1.reshape(N1,1,D1)-x1.reshape(1,N1,D1)
		sqdiff = np.sum(np.square(diff)*self.gammas,-1)
		expdiff = np.exp(-sqdiff)
		
		if (indexn==None) and(indexd==None):#calculate all gradients
			rets = []
			for n in range(N1):
				for d in range(D1):
					K = np.zeros((N1,N1))
					K[n,:] = -2*self.alpha*self.gammas[d]*(x1[n,d]-x1[:,d])*expdiff[n,:]
					K[:,n] = K[n,:]
					rets.append(K.copy())
			return rets
		else:
			K = np.zeros((N1,N1))
			K[indexn,:] = -2*self.alpha*self.gammas[indexd]*(x1[indexn,indexd]-x1[:,indexd])*expdiff[indexn,:]
			K[:,indexn] = K[indexn,:]
			return K.copy()