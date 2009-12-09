# -*- coding: utf-8 -*-
class polynomial:
	def __init__(self,alpha,order):
		"""Order of the polynomila is considered fixed...TODO: make the order optimisable..."""
		self.alpha = alpha
		self.order = order
		self.nparams = 1
	def set_params(self,new_params):
		assert new_params.size == self.nparams
		self.alpha, = new_params.flatten()
	def __call__(self,x1,x2):
		N1,D1 = x1.shape
		N2,D2 = x2.shape
		assert D1==D2, "Vectors must be of matching dimension"
		prod = x1.reshape(N1,1,D1)*x2.reshape(1,N2,D2)
		prod = self.alpha*np.power(np.sum(prod,-1) + 1, self.order)
		return prod