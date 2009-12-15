# -*- coding: utf-8 -*-
import numpy as np
import pylab
from scipy.optimize import fmin, fmin_ncg, fmin_cg
from scipy import linalg	
from .. import kernels

class GP:
	"""A simple GP with optimisation of the hyperparameters via the marginal 
	likelihood approach. There is a Univariate Gaussian Prior on the 
	hyperparameters (the kernel parameters and the noise parameter). SCG is 
	used to optimise the parameters (MAP estimate)"""
	
	def __init__(self, X=None, Y=None, kernel=None, parameter_priors=None, 
		beta=0.1, prior_mean=None):
		"""
		Arguments
		----------
		X : array (None)
			input
		Y : array (None)
			observations
		kernel : Kernel object (None)
			The covariance function
		parameter_prior : (None)
			
		beta : float (0.1)
		
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
			self.parameter_prior_widths = np.ones(self.kernel.nparams+1)
		else:
			assert parameter_priors.size==(self.kernel.nparams+1)
			self.parameter_prior_widths = np.array(parameter_priors).flatten()
		if prior_mean is None:
			self.prior_mean = lambda X: np.zeros(len(X))
		else:
			self.prior_mean = prior_mean
			
		self.beta=beta
		if (X is not None) and (Y is not None):
			self.update()
			# constant in the marginal. precompute for convenience. 
			self.n2ln2pi = 0.5*self.Ydim*self.N*np.log(2*np.pi) 

	def setX(self,newX):
		"""
		zero means and normalises X
		
		Arguments
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
		zero means and normalises Y
		
		Arguments
		----------
		newY :
		"""
		self.Y = newY.copy()
		N,self.Ydim = newY.shape
		assert N == self.N, "bad shape"
		#normalise...
		self.ymean = self.Y.mean(0)
		self.ystd = self.Y.std(0)
		self.Y -= self.ymean
		self.Y /= self.ystd
		
	def hyper_prior(self):
		"""return the log of the current hyper paramters under their prior"""
		return -0.5*np.dot(self.parameter_prior_widths,np.square(self.get_params()))
	
	def hyper_prior_grad(self):
		"""return the gradient of the (log of the) hyper prior for the current
		parameters"""
		return -self.parameter_prior_widths*self.get_params()
		
	def get_params(self):
		"""return the parameters of this GP: that is the kernel parameters and
		the beta value"""
		return np.hstack((self.kernel.get_params(),np.log(self.beta)))
		
	def set_params(self,params):
		""" set the kernel parameters and the noise parameter beta"""
		assert params.size==self.kernel.nparams+1
		self.beta = np.exp(params[-1])
		self.kernel.set_params(params[:-1])
		
	def ll(self,params=None):
		"""A cost function to optimise for setting the kernel parameters. 
		Uses current parameter values if none are passed """
		if not params == None:
			self.set_params(params)
		try:
			self.update()
		except:
			return np.inf
		return -self.marginal() - self.hyper_prior()
		
	def ll_grad(self,params=None):
		""" the gradient of the ll function, for use with conjugate gradient 
		optimisation. uses current values of parameters if none are passed """
		if not params == None:
			self.set_params(params)
		try:
			self.update()
		except:
			return np.ones(params.shape)*np.NaN
		self.update_grad()
		matrix_grads = [e for e in self.kernel.gradients(self.X)]
		#noise gradient matrix
		matrix_grads.append(-np.eye(self.K.shape[0])/self.beta) 
		
		grads = [0.5*np.trace(np.dot(self.alphalphK,e)) for e in matrix_grads]
			
		return -np.array(grads) - self.hyper_prior_grad()
		
	def find_kernel_params(self,iters=1000):
		"""
		Optimise the marginal likelihood.
		
		Arguments
		----------
		iters : int (1000)
			number of iterations to use in optimisation
		
		See Also
		----------
		scipy.optimize.fmin_cg : the conjugate graident algorithm
		""" 
		
		#work with the log of beta - fmin works better that way.
		new_params = fmin_cg(
			f = self.ll,
			x0 = np.hstack((self.kernel.get_params(), np.log(self.beta))),
			fprime = self.ll_grad,
			maxiter = iters
		)
		final_ll = self.ll(new_params) # sets variables - required!
		
	def update(self):
		"""do the Cholesky decomposition as required to make predictions and 
		calculate the marginal likelihood"""
		self.K = self.kernel(self.X,self.X) 
		self.K += np.eye(self.K.shape[0])/self.beta
		self.L = np.linalg.cholesky(self.K)
		self.A = linalg.cho_solve((self.L,1),self.Y)
	
	def update_grad(self):
		"""do the matrix manipulation required in order to calculate
		gradients"""
		self.Kinv = np.linalg.solve(
			self.L.T,
			np.linalg.solve(self.L,np.eye(self.L.shape[0]))
		)
		self.alphalphK = np.dot(self.A,self.A.T)-self.Ydim*self.Kinv
		
	def marginal(self):
		"""The Marginal Likelihood. Useful for optimising Kernel parameters"""
		return -self.Ydim*np.sum(np.log(np.diag(self.L)))\
		 	- 0.5*np.trace(np.dot(self.Y.T,self.A)) - self.n2ln2pi

	def predict(self,x_star):
		"""Make a prediction upon new data points"""
		x_star = (np.asarray(x_star)-self.xmean)/self.xstd

		#Kernel matrix k(X_*,X)
		k_x_star_x = self.kernel(x_star,self.X) 
		k_x_star_x_star = self.kernel(x_star,x_star) 
		
		#find the means and covs of the projection...
		means = np.dot(k_x_star_x, self.A)
		means *= self.ystd
		means += self.ymean
		
		v = np.linalg.solve(self.L,k_x_star_x.T)
		variances = (
			np.diag( 
				k_x_star_x_star - np.dot(v.T,v)
			).reshape(x_star.shape[0],1) + 1./self.beta
		) * self.ystd.reshape(1,self.Ydim)
		return means,variances
	
	def sample(self,X):
		if hasattr(self,'X'):
			mean, variance = self.predict(X)
			X2 = (X-self.xmean)/self.xstd # normalised version of points to sample
			v = np.linalg.solve(self.L,self.kernel(X2,self.X).T)
			covariance = (self.kernel(X2,X2) - np.dot(v.T,v))
		else:
			mean = self.prior_mean(X)
			covariance = self.kernel(X,X)
		return np.random.multivariate_normal(mean.flatten(),covariance)