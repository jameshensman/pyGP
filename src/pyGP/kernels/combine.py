# -*- coding: utf-8 -*-
class combine_addition:
	"""make a new kernel by adding two existing ones"""
	def __init__(self,A,B):
		self.A = A
		self.B = B
		self.nparams = A.nparams + B.nparams
	def set_params(self,new_params):
		assert new_params.size == self.nparams
		self.A.set_params(new_params[:self.A.nparams])
		self.B.set_params(new_params[self.A.nparams:])
	def __call__(self,x1,x2):
		return self.A(x1,x2) + self.B(x1,x2)
	def gradients(self,x1):
		return np.hstack([self.A.gradients(x1),self.B.gradients(x1)])