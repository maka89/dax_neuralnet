import numpy as np
from keras.layers import SimpleRNN,Dense,Input
from keras.models import Model
from keras_dax import KDAX


np.random.seed(0)
n_ts=3
X=np.random.randn(10,n_ts,3)
Y=np.random.randn(10,1)

inp=Input((n_ts,3))
srnn1= SimpleRNN(5,activation='relu',return_sequences=True)
srnn2= SimpleRNN(5,activation='tanh',return_sequences=False)
d_1=Dense(3,activation='tanh')
d_2=Dense(1,activation='sigmoid')

h=srnn1(inp)
h=srnn2(h)
h=d_1(h)
out=d_2(h)

model=Model(inp,out)

model.compile(loss='mse',optimizer='adam')

model.fit(X,Y,batch_size=10,epochs=1)

model.summary()

for layer in model.layers:
	print(layer.get_weights(),layer.get_config())
	

iff=[]
for i in range(0,n_ts):
	iff.append(["[x"+str(i)+"]","[y"+str(i)+"]","[z"+str(i)+"]"])

iff=["[x1]","[x2]","[x3]"]

rn=KDAX(iff,model)
rn.generate_timeseries(iff,3,interval=1,unit="DAY")

print(rn.generate_dax())


	
tmp=np.array([[0.1,	0.3,	-0.3],[-0.3,	1,	-0.5],[-1,	-0.7,	0.01],[0.3,	0.76,	-0.003],[0.1,	0.15,	-1.5],[-0.7,	0.1,	-0.6],[-0.3,	1.76,	0.02]])

X=np.concatenate(([tmp[-3::,:]],[tmp[-4:-1,:]],[tmp[-5:-2,:]],[tmp[-6:-3,:]],[tmp[-7:-4,:]]),0)
X=X[::-1,:,:]
#X=np.concatenate((X,np.array([[1.0,0.1,-1.0]])),0)
def forward(x,model):
	#x=x[:,::-1,:]
	ws=model.layers[1].get_weights()
	a=np.zeros((x.shape[0],x.shape[1],ws[0].shape[1]))
	
	a[:,0,:] = np.tanh(np.dot(x[:,0,:],ws[0])+ws[2])
	for i in range(1,x.shape[1]):
		a[:,i,:] =np.tanh(np.dot(x[:,i,:],ws[0])+np.dot(a[:,i-1,:],ws[1])+ws[2])
	
	a=a[:,-1,:]
	
	ws=model.layers[2].get_weights()
	a=np.tanh(np.dot(a,ws[0])+ws[1])
	ws=model.layers[3].get_weights()
	a=np.dot(a,ws[0])+ws[1]
	return a
#print(forward(X,model))
print(model.predict(X))
