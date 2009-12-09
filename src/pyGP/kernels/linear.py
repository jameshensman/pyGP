# -*- coding: utf-8 -*-
class linear:
	"""effectively the inner product, I think"""
	def __init__(self,alpha,bias):
		self.alpha = np.exp(alpha)
		self.bias = np.exp(bias)
		self.nparams = 2
	def set_params(self,new_params):
		assert new_params.size == self.nparams
		self.alpha,self.bias = np.exp(new_params).flatten()#try to unpack np array safely.  
	def get_params(self):
		return np.log(np.array([self.alpha,self.bias]))
	def __call__(self,x1,x2):
		N1,D1 = x1.shape
		N2,D2 = x2.shape
		assert D1==D2, "Vectors must be of matching dimension"
		prod = x1.reshape(N1,1,D1)*x2.reshape(1,N2,D2)
		prod = self.alpha*np.sum(prod,-1) + self.bias
		#diff = self.alpha*np.sqrt(np.square(np.sum(diff,-1)))
		return prod
	def gradients(self,x1):
		"""Calculate the gradient of the kernel matrix wrt the (log of the) parameters"""
		dalpha = self(x1,x1)-self.bias
		dbias = np.ones((x1.shape[0],x1.shape[0]))*self.bias
		return dalpha, dbias