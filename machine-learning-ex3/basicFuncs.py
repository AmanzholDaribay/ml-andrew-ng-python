import numpy as np

def sigmoid(x, theta):
	"""
	x.shape = (m, n)
	theta.shape = (n,)
	output.shape = (m,)
	"""
	z = x @ theta
	h = 1/(1+np.exp(-z))
	return h, z


def cost(h, y):
	"""
	h.shape = (m,1)
	y.shpe = (m,1)
	output.shape = (,)
	"""
	m = y.shape[0]
	cost = (-1/m) * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h))
	return cost



def GDS(x, theta, y, lamb, alpha):
	"""
	x.shape = (m, n)
	theta.shape = (n,)
	y.shape = (m,1)
	outputs.shape = (n,)

	Usage Example:
	for i in range(iterations):
		Theta = Theta - GDS(X,Theta, Y,lamb,alpha)
	"""
	m = y.shape[0]
	dif = (1/m) * x.T @ (sigmoid(x,theta) - y) #classical differentiation term, output is (n,1)
	dif_reg = (lamb/m) * theta #regularization term
	return (dif + dif_reg) * alpha

def augment(input):
	"""
	augments input with one
	"""
	return np.insert(input, 0, 1, axis = 1)



	#np.set_printoptions(suppress=True) 