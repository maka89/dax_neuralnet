# dax_neuralnet

Generates code for neural network measures in DAX for a pre-trained neural network.
Currently support only for standard MLPs.

The neural network is represented as a list of layers.
Each layer is a dict with members:

	- "W": Weights. Numpy 2d array
	- "b": biases. Numpy 2d array (First dim always 1)
	- "activation": string representing activation function. Default is linear. "relu","tanh" or "sigmoid"

Each layers outputs are calculated as a_(i) = act_fn ( dot( a_(i-1) , W) + b)

## Example

Let's define the names of the input measures in DAX, and define a neural network with 4 hidden relu units:
'''python

#Name of your input measures in power bi
input_features=["[x1]","[x2]","[x3]"]


layers=[]
layers.append({"W":np.random.randn(len(input_features),n_hidden), "b":np.random.randn(1,4),"activation":"relu"})
layers.append({"W":np.random.randn(4,1), "b":np.random.randn(1,1),"activation":""})

'''
Let's create a NNDAX object and generate code:

	nnd=NNDAX(input_features,layers)
	
	#Generate dax code
	print(nnd.generate_dax())

Additionally, you can run a sample (or several) through the network to confirm that you get similar outputs in python and Power BI:

 	print( "TEST")
	print( "input=[1,2,3]:")
	print( "output=" + str(nnd.calculate(np.array([[1,2,3]]))))
