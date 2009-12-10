import sys
import numpy as np
import pylab as pb

sys.path.append('../')
import pyGP

#generate data:
Ndata = 50
X = np.linspace(-3,3,Ndata).reshape(Ndata,1)
Y = np.sin(X) + np.random.standard_normal(X.shape)/20

#create GP object
try:
	myGP = pyGP.GP(X,Y)#,kernels.linear(-1,-1))
except AttributeError:
	print dir(pyGP)
	raise
	
#stuff for plotting
xx = np.linspace(-4,4,200).reshape(200,1)
def plot():
	pb.figure()
	pb.plot(X,Y,'r.')
	yy,cc = myGP.predict(xx)
	pb.plot(xx,yy,scaley=False)
	pb.plot(xx,yy + 2*np.sqrt(cc),'k--',scaley=False)
	pb.plot(xx,yy - 2*np.sqrt(cc),'k--',scaley=False)

plot()
# draw some samples
for i in range(4):
	pb.plot(xx,myGP.sample(xx),scaley=False,alpha=0.3)
myGP.find_kernel_params()
plot()
# draw some samples
for i in range(4):
	pb.plot(xx,myGP.sample(xx),scaley=False,alpha=0.3)
pb.show()
	