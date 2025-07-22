from layers import Layer, Linear, Tanh
from typing import TypeAlias
import numpy as np
Value:TypeAlias = np.ndarray
class NN(Layer):
	"""basic neural network example struct"""
	def __init__(self,inputdimension:int,intermediatedimension:int,outputdimension:int):
#		//TODO initialize the layers
	def backward(self,cache:list[Value],ygrad:list[Value])->list[Value]:
#		//TODO call backward on each layer in reverse order
	def forward(self,cache:list[Value],x:list[Value])->list[Value]:
#		//TODO call forward on each layer
	def opt_step(self,f:callable([[Value,Value],tuple[Value,Value]])):
#		//TODO call opt_step on each layer
#// If you implement everything correctly, this should learn the xor function and output something close to 0.0 1.0 1.0 0.0 for these inputs
def main():
	i=np.array([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]]);
	lr=0.01
	persistence=0.5
	target=np.array([[0.0],[1.0],[1.0],[0.0]])
	nn=NN(2,10,1)

	for n in range(0,1000):
		cache=[]
		output=nn.forward(cache,[i])
		gradient=[(target-output[0])*2.0]
		nn.backward(cache,gradient)
		nn.momentum_sgd(lr,persistence)

	print(nn.infer([i])[0])
main()
