import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import csv
numpy.random.seed(7)

xtrain = []
ytrain = []

with open("data.csv") as csvfile:
	reader = csv.DictReader(csvfile)
	for row in reader:
		r = []
		r.append(float(row['mse']))
		r.append(float(row['psnr']))
		r.append(float(row['histogram_compare'])) 
		xtrain.append(r)
		ytrain.append(int(row['class'])) 

X=numpy.array(xtrain, dtype=object)
Y=numpy.array(ytrain, dtype=object)

model = Sequential()
model.add(Dense(12, input_dim=3, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs=150, batch_size=5) 
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))




xtest = []
ytest = []

with open("test.csv") as csvfile:
	reader = csv.DictReader(csvfile)
	for row in reader:
		r = []
		r.append(float(row['mse']))
		r.append(float(row['psnr']))
		r.append(float(row['histogram_compare'])) 
		xtest.append(r)
		ytest.append(int(row['class'])) 

X=numpy.array(xtest, dtype=object)
Y=numpy.array(ytest, dtype=object)

scores = model.evaluate(X, Y, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))