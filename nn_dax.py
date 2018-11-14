import numpy as np

class NNDAX:

	def __init__(self,input_features,layers):
		self.input_features=input_features
		self.layers=layers
		
		#Collection of activation functions
		self.act_dict={"relu":lambda x:(x>0.0)*x,"sigmoid":lambda x:1.0/(1.0+np.exp(-x)),"tanh":np.tanh}
	def assign_activation(self,string_in, activation, l=None , j=None):

		substr="_"+str(l)+"_"+str(j)
		
		if activation=="tanh":
			return "var l"+ substr+" = "+ "TANH( " + string_in + " )"
		elif activation=="sigmoid":
			tmp = "var z" + substr+ " = " + string_in + "\n"
			return tmp+"var l"+substr+" = DIVIDE( 1,0; 1,0+EXP(- z"  + substr + " ) )\n"
		elif activation=="relu":

			tmp= "var z" + substr+ " = " + string_in + "\n"
			return tmp + "var l" + substr + " = IF(z"+substr+" > 0,0; z"+substr+"; 0,0)\n"
		else:
			return "var l"+substr+" = " +string_in
		
	def generate_dax(self):
		input_features=self.input_features
		layers=self.layers
		assert(layers[0]["W"].shape[0] == len(input_features))
		for i in range(1,len(layers)):
			assert(layers[i]["W"].shape[0] == layers[i-1]["W"].shape[1])
			assert(layers[i-1]["b"].shape[1]==layers[i]["W"].shape[0])
		assert(layers[len(layers)-1]["b"].shape[1] == 1)
		assert(layers[len(layers)-1]["W"].shape[1] == 1)
		
		str1 =""
		for i in range(0,len(input_features)):
			str1+="var l_0_"+str(i)+" = " + input_features[i]+ "\n"
		str1+="\n\n"
		
		for i in range(0,len(layers)):
			l=(i+1)
			for j in range(0,layers[i]["W"].shape[1]):
			
				tmp=""
				
				for k in range(0,layers[i]["W"].shape[0]-1):
					if k==0:
						tmp+="{0:04e} * ".format(layers[i]["W"][k,j]).replace(".",",") + "l_"+str(l-1)+"_"+str(k)
					else:
						tmp+="{0:04e} * ".format(np.abs(layers[i]["W"][k,j])).replace(".",",") + "l_"+str(l-1)+"_"+str(k)
					if layers[i]["W"][k+1,j] < 0.0:
						tmp+= " - "
					else:
						tmp+= " + "
				
						
				k=layers[i]["W"].shape[0]-1
				tmp+="{0:04e} * ".format(np.abs(layers[i]["W"][k,j])).replace(".",",") + "l_"+str(l-1)+"_"+str(k)		
				
				if layers[i]["b"][0,j] < 0:
					tmp += " - {0:04e}".format(np.abs(layers[i]["b"][0,j])).replace(".",",")
				else:
					tmp += " + {0:04e}".format(np.abs(layers[i]["b"][0,j])).replace(".",",")
				
				
				str1+= self.assign_activation(tmp,layers[i]["activation"],l,j) + "\n"
			if i != len(layers)-1:
				str1+="\n\n"
				
		str1+= "\nreturn l_"+str(len(layers))+"_0"
		return str1
		
		

	def calculate(self,input_features_numeric):
	
		layers=self.layers
		if len(input_features_numeric.shape)==1:
			input_features_numeric=input_features_numeric.reshape(1,-1)
		assert(input_features_numeric.shape[1]==len(self.input_features))
		
		a=input_features_numeric
		for i in range(0,len(layers)):
			#Get Activation Function for current layer
			fn= self.act_dict.get(layers[i]["activation"],lambda x: x)
			
			#Calculate output activations.
			z=np.dot(a,layers[i]["W"])+layers[i]["b"]
			a=fn( z )
		return a


	
	
	

