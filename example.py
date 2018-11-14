import numpy as np
from nn_dax import NNDAX

if __name__=="__main__":

	np.random.seed(0)
	
	
	#Name of your input measures in power bi
	input_features=["[x1]","[x2]","[x3]"]

	
	#Define weights,biases and activation functions of pre-trained neural network
	#List of dicts
	n_hidden=4
	layers=[]
	layers.append({"W":np.random.randn(len(input_features),n_hidden), "b":np.random.randn(1,n_hidden),"activation":"relu"})
	layers.append({"W":np.random.randn(n_hidden,1), "b":np.random.randn(1,1),"activation":""})
	
	#Create NNDAX object
	nnd=NNDAX(input_features,layers)
	
	#Generate dax code
	print(nnd.generate_dax())
	
	
	#Run a input sample through the network.
	#Can double check this value with calculated values from PowerBI to ensure everything works correctly.
	print( "-----------")
	print( "TEST")
	print( "input=[1,2,3]:")
	print( "output=" + str(nnd.calculate(np.array([[1,2,3]]))))
	

