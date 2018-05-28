import numpy
import csv
from sklearn import tree
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


def decisionTree(xtrain,ytrain,xtest,ytest):
	model = tree.DecisionTreeClassifier(criterion='gini')
	model.fit(xtrain, ytrain)
	model.score(xtrain,ytrain)
	pred = model.predict(xtest)
	print("Decision Tree Accuracy: %.2f%%" % (accuracy_score(ytest,pred)*100))
	# print confusion_matrix(ytest,pred)

def kNN(xtrain,ytrain,xtest,ytest):
	model = KNeighborsClassifier(n_neighbors=6)
	model.fit(xtrain, ytrain)
	model.score(xtrain,ytrain)
	pred = model.predict(xtest)
	ct=0
	for i in pred:
		if i==0:
			ct+=1
	# print ct,len(pred)
	
	print("KNN Accuracy: %.2f%%" % (accuracy_score(ytest,pred)*100))
	# print confusion_matrix(ytest,pred)

def logisticRegression(xtrain,ytrain,xtest,ytest):
	model = LogisticRegression()
	model.fit(xtrain, ytrain)
	model.score(xtrain, ytrain)
	pred = model.predict(xtest)
	ct=0
	for i in pred:
		if i==0:
			ct+=1
	# print ct,len(pred)
	
	print("Logistic Regression Accuracy: %.2f%%" % (accuracy_score(ytest,pred)*100))
	# print confusion_matrix(ytest,pred)


def naiveBaiyes(xtrain,ytrain,xtest,ytest):
	model = GaussianNB()
	model.fit(xtrain, ytrain)
	model.score(xtrain,ytrain)
	pred = model.predict(xtest)
	ct=0
	for i in pred:
		if i==0:
			ct+=1
	# print ct,len(pred)
	
	print("Naive Baiyes Accuracy: %.2f%%" % (accuracy_score(ytest,pred)*100))
	# print confusion_matrix(ytest,pred)


def randomForest(xtrain,ytrain,xtest,ytest):
	model= RandomForestClassifier()
	model.fit(xtrain, ytrain)
	model.score(xtrain,ytrain)
	pred = model.predict(xtest)
	ct=0
	for i in pred:
		if i==0:
			ct+=1
	# print ct,len(pred)
	
	print("Random Forest Accuracy: %.2f%%" % (accuracy_score(ytest,pred)*100))
	# print confusion_matrix(ytest,pred)

def svm(xtrain,ytrain,xtest,ytest):
	model = SVC()
	model.fit(xtrain, ytrain)
	model.score(xtrain,ytrain)
	pred = model.predict(xtest)
	
	print("SVM Accuracy: %.2f%%" % (accuracy_score(ytest,pred)*100))
	print confusion_matrix(ytest,pred)


def neuralNetwork(xtrain,ytrain,xtest,ytest):
	x_train=numpy.array(xtrain, dtype=object)
	y_train=numpy.array(ytrain, dtype=object)
	model = Sequential()
	model.add(Dense(12, input_dim=len(xtrain[0]), activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(x_train,y_train, epochs=10, batch_size=15) 
	x_test = numpy.array(xtest , dtype = object)
	y_test = numpy.array(ytest , dtype = object)
	scores = model.evaluate(x_test, y_test, verbose=0)
	print("Neural Network Accuracy: %.2f%%" % (scores[1]*100))


xtrain = []
ytrain = []

with open("data.csv") as csvfile:
	reader = csv.DictReader(csvfile)
	for row in reader:
		r = []
		r.append(float(row['mse']))
		r.append(float(row['psnr']))
		r.append(float(row['ssim']))
		r.append(float(row['histogram_compare'])) 
		r.append(float(row['entropy'])) 
		r.append(float(row['avgObjArea'])) 
		r.append(float(row['displacedObjects'])) 
		xtrain.append(r)
		ytrain.append(int(row['class'])) 


xtest = []
ytest = []
with open("testVideo.csv") as csvfile:
	reader = csv.DictReader(csvfile)
	for row in reader:
		r = []
		r.append(float(row['mse']))
		r.append(float(row['psnr']))
		r.append(float(row['ssim']))
		r.append(float(row['histogram_compare'])) 
		r.append(float(row['entropy'])) 
		r.append(float(row['avgObjArea'])) 
		r.append(float(row['displacedObjects'])) 
		xtest.append(r)
		ytest.append(int(row['class'])) 

neuralNetwork(xtrain,ytrain,xtest,ytest)
decisionTree(xtrain,ytrain,xtest,ytest)
kNN(xtrain,ytrain,xtest,ytest)
logisticRegression(xtrain,ytrain,xtest,ytest)
naiveBaiyes(xtrain,ytrain,xtest,ytest)
randomForest(xtrain,ytrain,xtest,ytest)
svm(xtrain,ytrain,xtest,ytest)

