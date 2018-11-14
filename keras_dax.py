import numpy as np

class KDAX:

	def __init__(self,input_features,model):
		self.input_features=input_features
		self.model=model
		
		
		#Collection of activation functions
		self.act_dict={"relu":lambda x:(x>0.0)*x,"sigmoid":lambda x:1.0/(1.0+np.exp(-x)),"tanh":np.tanh}
		
	def generate_timeseries(self,input_features,steps,interval=1,unit="DAY",dates="'Calendar'[Date]",reverse=False, bidirectional=False):
		
		iff = []
		for i in range(0,steps):
			tmp=[]
			if not bidirectional and not reverse:
				for j in range(0,len(input_features)):
					tmp.append("CALCULATE("+input_features[j]+"; DATEADD("+dates+";"+str(-(steps-1-i)*interval)+"; "+unit + " ))")
			
			iff.append(tmp)
		self.input_features=iff
		
	def assign_activation(self,string_in, activation, l=None , j=None,t=None,rnn=False):

		if rnn:
			substr="_"+str(l)+"_"+str(t)+"_"+str(j)
		else:
			substr="_"+str(l)+"_"+str(j)
		
		if activation=="tanh":
			return "var a"+ substr+" = "+ "TANH( " + string_in + " )"
		elif activation=="sigmoid":
			tmp = "var z" + substr+ " = " + string_in + "\n"
			return tmp+"var a"+substr+" = DIVIDE( 1,0; 1,0+EXP(- z"  + substr + " ) )\n"
		elif activation=="relu":

			tmp= "var z" + substr+ " = " + string_in + "\n"
			return tmp + "var a" + substr + " = IF(z"+substr+" > 0,0; z"+substr+"; 0,0)\n"
		else:
			return "var a"+substr+" = " +string_in
		
	def generate_input(self):
	
		tmp=""
		if isinstance(self.input_features[0], (list,)):
			for i in range(0,len(self.input_features)):
				for j in range(0,len(self.input_features[0])):
					substr="_0_"+str(i)+"_"+str(j)
					tmp+="var a"+substr+" = " + self.input_features[i][j] + "\n"
		else:
			for i in range(0,len(self.input_features)):
				substr="_0_"+str(i)
				tmp+="var a"+substr+" = " + self.input_features[i] + "\n"
		return tmp
		
	def generate_srnn(self,layer,i,return_sequences=False):
	
		ws=layer.get_weights()
		strr=""
		for t in range(0,len(self.input_features)):
			strr+=self.generate_rnn_step(layer,i,t)
			
		if not return_sequences:
			t=len(self.input_features)-1
			for j in range(0,ws[0].shape[1]):
				strr+="var a_"+str(i)+"_"+str(j)+ " = a_"+str(i)+"_"+str(t)+"_"+str(j)+"\n"
			
		strr+="\n"
		
		return strr
	def generate_dense(self,layer,i):
		
		l=i
		
		W=layer.get_weights()[0]
		b=layer.get_weights()[1].reshape(1,-1)
		act=layer.get_config()["activation"]
		str1=""
		for j in range(0,W.shape[1]):
		
			tmp=" "
			for k in range(0,W.shape[0]-1):
				if k==0:
					tmp+="{0:06e} * ".format(W[k,j]).replace(".",",") + "a_"+str(l-1)+"_"+str(k)
				else:
					tmp+="{0:06e} * ".format(np.abs(W[k,j])).replace(".",",") + "a_"+str(l-1)+"_"+str(k)
					
				if W[k+1,j] < 0.0:
					tmp+= " - "
				else:
					tmp+= " + "
			
					
			k=W.shape[0]-1
			if k==0:
				tmp+="{0:06e} * ".format(W[k,j]).replace(".",",") + "a_"+str(l-1)+"_"+str(k)
			else:
				tmp+="{0:06e} * ".format(np.abs(W[k,j])).replace(".",",") + "a_"+str(l-1)+"_"+str(k)		
			
			if b[0,j] < 0:
				tmp += " - {0:06e}".format(np.abs(b[0,j])).replace(".",",")
			else:
				tmp += " + {0:06e}".format(np.abs(b[0,j])).replace(".",",")
			
			
			str1+= self.assign_activation(tmp,act,l,j) + "\n"

		return str1
	
	def generate_rnn_step(self,layer,i,t):
		l=i
		substr2=str(t-1)
		if t==0:
			substr2="i"
			
		W=layer.get_weights()[0]
		Wr=layer.get_weights()[1]
		b=layer.get_weights()[2].reshape(1,-1)
		act=layer.get_config()["activation"]
		str1=""
		
		if t==0:
			for j in range(0,W.shape[1]):
				substr=str(l)+"_i_"+str(j)
				str1 += "var a_"+substr+" = 0,0\n"
		for j in range(0,W.shape[1]):
			tmp=""
			for k in range(0,W.shape[0]-1):
				if k==0:
					tmp+="{0:06e} * ".format(W[k,j]).replace(".",",") + "a_"+str(l-1)+"_"+str(t)+"_"+str(k)
				else:
					tmp+="{0:06e} * ".format(np.abs(W[k,j])).replace(".",",") + "a_"+str(l-1)+"_"+str(t)+"_"+str(k)
				if W[k+1,j] < 0.0:
					tmp+= " - "
				else:
					tmp+= " + "
			
					
			k=W.shape[0]-1
			if k==0:
				tmp+="{0:06e} * ".format(W[k,j]).replace(".",",") + "a_"+str(l-1)+"_"+str(t)+"_"+str(k)
			else:
				tmp+="{0:06e} * ".format(np.abs(W[k,j])).replace(".",",") + "a_"+str(l-1)+"_"+str(t)+"_"+str(k)	
				
			tmp += " + ( "
			for k in range(0,Wr.shape[0]-1):
				if k==0:
					tmp+="{0:06e} * ".format(Wr[k,j]).replace(".",",") + "a_"+str(l)+"_"+substr2+"_"+str(k)
				else:
					tmp+="{0:06e} * ".format(np.abs(Wr[k,j])).replace(".",",") + "a_"+str(l)+"_"+substr2+"_"+str(k)
				if Wr[k+1,j] < 0.0:
					tmp+= " - "
				else:
					tmp+= " + "
			
					
			k=Wr.shape[0]-1
			if k==0:
				tmp+="{0:06e} * ".format(Wr[k,j]).replace(".",",") + "a_"+str(l)+"_"+substr2+"_"+str(k)
			else:
				tmp+="{0:06e} * ".format(np.abs(Wr[k,j])).replace(".",",") + "a_"+str(l)+"_"+substr2+"_"+str(k)
			tmp+=" )"
			
			
			if b[0,j] < 0:
				tmp += " - {0:06e}".format(np.abs(b[0,j])).replace(".",",")
			else:
				tmp += " + {0:06e}".format(np.abs(b[0,j])).replace(".",",")
			
			
			str1+= self.assign_activation(tmp,act,l,j,t=t,rnn=True) + "\n"

		return str1
		
	def generate_dax(self):
		
		val=""
		for i in range(0,len(self.model.layers)):
			if "input" in self.model.layers[i].get_config()["name"]:
				val+=self.generate_input()+"\n"
			elif "simple_rnn" in self.model.layers[i].get_config()["name"]:
				val+=self.generate_srnn(self.model.layers[i],i,return_sequences=self.model.layers[i].get_config()["return_sequences"])
			elif "dense" in self.model.layers[i].get_config()["name"]:
				val+=self.generate_dense(self.model.layers[i],i)
		val+="return a_"+str(len(self.model.layers)-1)+"_0"
		return val
	
	
	

