from collections.abc import Callable
from typing import TypeAlias
import numpy as np
Value:TypeAlias = np.ndarray
class Layer:
	def backward(self,cache:list[Value],ygrad:list[Value])->list[Value]:
		"""applies the backward pass operation"""
		raise NotImplementedError
	def forward(self,cache:list[Value],x:list[Value])->list[Value]:
		"""applies the forward pass operation"""
		raise NotImplementedError
	def infer(self,x:list[Value])->list[Value]:
		return self.forward(None,x)
	def momentum_sgd(self,lr:float,persistence:float):
		"""
		Simple SGD with momentum
		"""
		self.opt_step(lambda param,grad:tuple([grad*lr+param,grad*persistence]))
	def opt_step(self,f:callable([[Value,Value],tuple[Value,Value]])):
		"""applies the optimization step to each parameter according to f(parameter, gradient) -> (newparameter, newgradient) """
		raise NotImplementedError
class Linear(Layer):
	def __init__(self,inputdimension:int,outputdimension:int):
#			//TODO randomly initialize the weights and bias too if you decide to put it. Also make relevant gradients initially set to None
	def backward(self,cache:list[Value],ygrad:list[Value])->list[Value]:
		x=cache[len(cache)-len(ygrad):]
		xgrad=[]
		for x,dy in zip(x,ygrad):
#			//TODO accumulate the parameter gradients given input x and output gradient dy, then comput the input gradient dx to append to the input gradient list
			xgrad.append(dx)
		del cache[len(cache)-len(ygrad):]
		return xgrad
	def forward(self,cache:list[Value],x:list[Value])->list[Value]:
		if cache is not None:
			cache.extend(x)
#			//TODO for every input matrix multiply the weight by the input, and then add bias if you have it
		return x
	def opt_step(self,f:callable([[Value,Value],tuple[Value,Value]])):
#			/TODO apply the optimization function f to each parameter and it's gradient, then set the parameter and gradient to the returned value afterwards
class Tanh(Layer):
	def backward(self,cache:list[Value],ygrad:list[Value])->list[Value]:
		y=cache[len(cache)-len(ygrad):]
#		//TODO this layer has no parameter so you only need to find the input gradient. given y=tanh(x), dy/dx = 1.0-y^2. conveniently, since the outputs were cached rather than the inputs. so to compute xgrad you just need to subtract the square of every output from one, then multiply the result by the corresponding output gradient
		del cache[len(cache)-len(ygrad):]
		return xgrad
	def forward(self,cache:list[Value],x:list[Value])->list[Value]:
#		//TODO apply tanh to each component.
		if cache is not None:
			cache=cache.extend(y)
		return y
	def opt_step(self,f:callable([[Value,Value],tuple[Value,Value]])):
#		// nothing to do here since the tanh layer has no parameter
		pass
