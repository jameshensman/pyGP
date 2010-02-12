# -*- coding: utf-8 -*-
import numpy as np
import pylab
from scipy.optimize import fmin, fmin_ncg, fmin_cg
from scipy import linalg	
from .. import kernels

# a few helper functions *see R&W page 43*
class logistic:
	def __init__(self):
		"""A class for a logistic sigmoid. Expects the values Y (class labels) as +-1 ."""
		pass
	def __call__(self,y,f):
		""" standard logistic sigmoid - effectively giving p(y|f)"""
		return 1./(1+np.exp(-y*f))
	def log(self,y,f):
		"""log p(y|f)"""
		return -np.log(1+np.exp(-y*f))
	def log_gradient(self,y,f):
		"""d/df log p(y|f)"""
		t = (y+1)/2.
		pi = self(np.ones(f.shape),f)
		return t - pi
	def log_gradient2(self,y,f):
		"""d^2/df^2 log p(y|f)"""
		pi = self(np.ones(f.shape),f)
		return -pi*(1-pi)

class cum_gaus:
	def __init__(self):
		""""""
		pass
	

class GP:
	"""A simple GP for binary classification. See Rassmusen & Williams 2005 Chapter 3.  
	The posterior distribution is expanded about the maximum such that the second moments 
	(of the true distribution and the Gaussian approximation) match. This is known as the Laplace approximation.  
	
	Optimisation of the hyperparameters occurs via the marginal 
	likelihood approach. There is a Univariate Gaussian Prior on the 
	hyperparameters (the kernel parameters and the noise parameter). SCG is 
	used to optimise the parameters (MAP estimate)"""
	
	def __init__(self, X=None, Y=None, kernel=None, parameter_priors=None, prior_mean=None, sigmoid=None):
		"""
		Parameters
		----------
		X : array (None)
			input
		Y : array (None)
			observations
		kernel : Kernel object (None)
			The covariance function
		parameter_prior : (None)
			
		
		Notes
		----------
		If you don't supply a kernel then a multivariate squared exponential 
		kernel will be generated, with the appropriate dimension taken from X.
		So if you want to use the default kernel, then you must supply some
		input/observation data.
		
		See Also
		----------
		pyGP.kernels : for a selection of covariance functions
		"""
		if (X is not None) and (Y is not None):
			self.N = Y.shape[0]
			self.setY(Y)
			self.setX(X)
			
		if kernel==None:
			self.kernel = kernels.full_RBF(-1,-np.ones(self.Xdim))
		else:
			self.kernel = kernel
		if parameter_priors==None:
			self.parameter_prior_widths = np.ones(self.kernel.nparams)
		else:
			assert parameter_priors.size==(self.kernel.nparams)
			self.parameter_prior_widths = np.array(parameter_priors).flatten()
			
		if prior_mean is None:
			self.prior_mean = lambda X: np.zeros(len(X))
		else:
			self.prior_mean = prior_mean
			
		if (X is not None) and (Y is not None):
			self.update()
			# constant in the marginal. precompute for convenience. 
			self.n2ln2pi = 0.5*self.N*np.log(2*np.pi) 
			
		#constant in the kernel parameter prior. precompute for convenience
		self.prior_const = -0.5*(self.kernel.nparams+1)*np.log(2*np.pi) - 0.5*np.log(np.prod(self.parameter_prior_widths))
		
		#default to a logistic sigmoid
		self.sigmoid = sigmoid or logistic()
		
		#initialise fhat to something sensible
		self.f_hat = self.Y.copy()
		
	def setX(self,newX):
		"""
		zero means and normalises X
		
		Parameters
		----------
		newX :
		"""
		self.X = newX.copy()
		N,self.Xdim = newX.shape
		assert N == self.N, "bad shape"
		# zero mean and normalise...
		self.xmean = self.X.mean(0)
		self.xstd = self.X.std(0)
		self.X -= self.xmean
		self.X /= self.xstd

	def setY(self,newY):
		"""
		Make sure Y is valid - should be +/- 1.
		
		Parameters
		----------
		newY :
		"""
		N,Ydim = newY.shape
		assert Ydim ==1, "Binary classification only with this class (Y.shape is wrong)."
		assert N == self.N, "bad shape"
		self.Y = newY.copy()
		
	def hyper_prior(self):
		"""return the log of the current hyper paramters under their prior"""
		return  self.prior_const - 0.5*np.dot(self.parameter_prior_widths,np.square(self.get_params()))
			
	def hyper_prior_grad(self):
		"""return the gradient of the (log of the) hyper prior for the current
		parameters"""
		return -self.parameter_prior_widths*self.get_params()
		
	def get_params(self):
		"""return the parameters of this GP: that is the kernel parameters and
		the beta value"""
		return self.kernel.get_params()
		
	def set_params(self,params):
		""" set the kernel parameters and the noise parameter beta"""
		self.kernel.set_params(params)
	
	def fcost(self,f):
		f = f.reshape(f.size,1)
		return -np.sum(self.sigmoid.log(self.Y,f)) + 0.5*np.dot(f.T,np.dot(self.Kinv,f))
	def fcost_grad(self,f):
		f = f.reshape(f.size,1)
		ret =  -self.sigmoid.log_gradient(self.Y,f) + np.dot(self.Kinv,f)
		return ret.flatten()
	def fcost_hessian(self,f):
		f = f.reshape(f.size,1)
		return -np.diag(self.sigmoid.log_gradient2(self.Y,f)[:,0]) + self.Kinv
		
	def laplace_approximation(self):
		"""find the mode and hessian of the (probabiliy of) f, the latent function variables"""
		self.update()
		#self.f_hat = fmin(self.fcost,self.Y.copy().flatten()+np.random.randn(self.N))
		#self.f_hat = fmin_cg(self.fcost,self.f_hat.copy().flatten(),fprime=self.fcost_grad)
		try:
			self.f_hat = fmin_ncg(self.fcost,self.f_hat.copy().flatten(),fprime=self.fcost_grad,fhess=self.fcost_hessian)
		except(ValueError):
			print 'ncg method barfed'
			self.f_hat = fmin_cg(self.fcost,self.f_hat.copy().flatten(),fprime=self.fcost_grad)
			
		self.f_hat = self.f_hat.reshape(self.N,1)
			
	def marginal(self):
		"""return the log of the marginal likelihood (see R&W page 48)"""
		Wsqrt = np.diag(np.sqrt(-self.sigmoid.log_gradient2(self.Y,self.f_hat)[:,0]))
		return -0.5*np.dot(self.f_hat.T,np.dot(self.Kinv,self.f_hat))\
			+np.sum(self.sigmoid.log(self.Y,self.f_hat)) \
			-0.5*np.log(np.linalg.det(np.eye(self.N) + np.dot(Wsqrt,np.dot(self.K,Wsqrt)) ))
	
	def ll(self,params):
		self.set_params(params)
		self.laplace_approximation()
		ret =  -(self.marginal() + self.hyper_prior())
		print ret
		return ret
		
	def ll_grad(self):
		pass
	
	def find_kernel_parameters(self):
		#new_params = fmin(self.ll,np.random.randn(self.kernel.nparams))
		new_params = fmin(self.ll,self.get_params())
		self.set_params(new_params)
	
	def update(self):
		""""""
		self.K = self.kernel(self.X,self.X) + np.eye(self.X.shape[0])*1e-6
		self.L = linalg.cho_factor(self.K)
		self.Kinv = linalg.cho_solve(self.L,np.eye(self.L[0].shape[0]))
		
	def predict_MAP(self,Xtest):
		"""Make a prediction for a test case. Just uses the sigmoid of the MAP estimate of F. Fine for doing 'most probable' classification."""
		Xtest = (Xtest - self.xmean)/self.xstd
		Ktest_train = self.kernel(Xtest,self.X)
		#Ktest_test = self.kernel(Xtest,Xtest)
		Ftest = np.dot(Ktest_train,np.dot(self.Kinv,self.f_hat))
		return self.sigmoid(np.ones(Ftest.shape),Ftest)
	
	
		
		
		
		
		
		