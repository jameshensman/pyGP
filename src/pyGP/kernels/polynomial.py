# -*- coding: utf-8 -*-
import numpy as np
class polynomial:
	def __init__(self,ln_alpha,order):
		"""Order of the polynomila is considered fixed...TODO: make the order optimisable..."""
		self.alpha = np.exp(ln_alpha)
		self.order = order
		self.nparams = 1
	def set_params(self,new_params):
		assert new_params.size == self.nparams
		self.alpha, = np.exp(new_params.flatten())
	def get_params(self):
		return np.log([self.alpha])
	def __call__(self,x1,x2):
		N1,D1 = x1.shape
		N2,D2 = x2.shape
		assert D1==D2, "Vectors must be of matching dimension"
		prod = x1.reshape(N1,1,D1)*x2.reshape(1,N2,D2)
		prod = self.alpha*np.power(np.sum(prod,-1) + 1, self.order)
		return prod
	def gradients(self,x1):
		N,D = x1.shape
		return np.reshape(self(x1,x1),(1,N,N))
	