# dax_neuralnet

Generates code for neural network measures in DAX for a pre-trained neural network.
Currently support only for standard MLPs.

The neural network is represented as a list of layers.
Each layer is a dict with members:
	- "W": Weights. Numpy 2d array
	- "b": biases. Numpy 2d array (First dim always 1)
	- "activation": string representing activation function. Default is linear. "relu","tanh" or "sigmoid"

Each layer calculated as l_(i) = act_fn ( dot( l_(i-1) , W) + b)