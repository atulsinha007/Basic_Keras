from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing
import numpy
from keras import optimizers
import pandas as pd
def normalize(x):
	maxx = numpy.max(x, axis = 0)
	minn = numpy.min(x, axis = 0)
	for row in x:
		for i in range(row.size):
			row[i] = (row[i] - minn[i])
			if maxx[i] == minn[i]:
				continue
			row[i] = row[i]/(maxx[i] - minn[i])
		
	return x

def detail_about_model(model):
    print("\nSUMMARY OF THE MODEL:")
    summary = model.summary()
    weights_hidden_layer_1 = model.layers[0].get_weights()[0]
    weights_hidden_layer_2 = model.layers[1].get_weights()[0]
    weights_output_layer = model.layers[2].get_weights()[0]
    bias_1 = model.layers[0].get_weights()[1]
    bias_2 = model.layers[1].get_weights()[1]
    bias_3 = model.layers[2].get_weights()[1]
    print(summary)
    print("\n\nWeights of Hidden layer 1:")
    print(weights_hidden_layer_1)
    print("\n\nWeights of Hidden layer 2:")
    print(weights_hidden_layer_2)
    print("\nbias1:")
    print(bias_1)
    print("\nbias2:")
    print(bias_2)
    print("\nWeights of Output layer:")
    print(weights_output_layer)
    print("\nbias:")
    print(bias_2)

dataset = pd.read_excel('test_mod.xlsx')
dataset = dataset.values
X = dataset[:,0:6]
Y = dataset[:,6]
maxx = numpy.max(Y,axis=0)
minn = numpy.min(Y,axis=0)
print(X)
X = normalize(X)
print(X)
Y = normalize(Y.reshape(-1,1))
model = Sequential()
model.add(Dense(3, input_dim=6, activation='tanh'))
model.add(Dense(3, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
sgd = optimizers.sgd(lr=0.005)
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
summary = model.summary()
print(summary)
model.fit(X, Y, epochs=15000)
detail_about_model(model)
predictions = model.predict(X)
denormalised_pred = []
for y in predictions.flatten():
	denormalised_pred.append(minn + y*(maxx-minn))

print("Predictions")
print(numpy.array(denormalised_pred).reshape(-1,1))