import cv2
from skimage.measure import compare_mse, compare_nrmse, compare_ssim, compare_psnr
import csv
import os
import math
from django.conf import settings
import imutils
import numpy
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
from sklearn.preprocessing import normalize
# 1. Mse
def getMse(img1,img2):
	return compare_mse(img1,img2)

# 2. PSNR
def getPSNR(img1,img2):
	return compare_psnr(img1,img2)

# 3. Histogram
def getHistCompare(img1,img2):
	hist1 = cv2.calcHist([img1],[0],None,[256],[0,256])
	hist2 = cv2.calcHist([img2],[0],None,[256],[0,256])
	#Histogram Bhattacharya Distance
	return cv2.compareHist(hist1,hist2,3)

# 4. SSIM
def getSSIM(img1,img2):
	grayA = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	grayB = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	(score, diff) = compare_ssim(grayA, grayB, full=True)
	diff = (diff * 255).astype("uint8")
	return score,diff

# 5. Entropy 
def getEntropy(img1,img2):
	im = cv2.absdiff(img1,img2)
	return sum(sum(sum(im)))

# 6. Average Object Area
# 7. Number of objects displaced
def getAvgObjArea(img):
	thresh = cv2.threshold(img, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	numofcnts = 0
	area = 0.0
	for cnt in cnts:
		area+=cv2.contourArea(cnt)
		numofcnts+=1
	if numofcnts ==0:
		return 0,0
	return numofcnts,area/numofcnts

def getFeatures(im1,im2):
	img1 = cv2.imread(im1)
	img2 = cv2.imread(im2)
	# mse =  getMse(img1,img2)
	# psnr = getPSNR(mse)
	# histogram_compare = getHistCompare(img1,img2)
	ssim,diffImg = getSSIM(img1,img2)
	features ={}
	features['mse']=getMse(img1,img2)
	features['psnr']=getPSNR(img1,img2)
	features['histogram_compare'] = getHistCompare(img1,img2)
	features['ssim']=ssim
	features['entropy']=getEntropy(img1,img2)
	features['avgObjArea'],features['displacedObjects'] = getAvgObjArea(diffImg)
	return features


def kNN(xtrain,ytrain,xtest):
	model = KNeighborsClassifier(n_neighbors=6)
	model.fit(xtrain, ytrain)
	model.score(xtrain,ytrain)
	pred = model.predict(xtest)
	ct=0
	for i in pred:
		if i==1:
			return 1
	return 0

def decisionTree(xtrain,ytrain,xtest):
	model = tree.DecisionTreeClassifier(criterion='gini')
	model.fit(xtrain, ytrain)
	model.score(xtrain,ytrain)
	pred = model.predict(xtest)
	ct=0
	for i in pred:
		if i==1:
			return 1
	return 0
	
def logisticRegression(xtrain,ytrain,xtest):
	model = LogisticRegression()
	model.fit(xtrain, ytrain)
	model.score(xtrain, ytrain)
	pred = model.predict(xtest)
	ct=0
	for i in pred:
		if i==1:
			return 1
	return 0



def naiveBaiyes(xtrain,ytrain,xtest):
	model = GaussianNB()
	model.fit(xtrain, ytrain)
	model.score(xtrain,ytrain)
	pred = model.predict(xtest)
	ct=0
	for i in pred:
		if i==1:
			return 1
	return 0

def randomForest(xtrain,ytrain,xtest):
	model= RandomForestClassifier()
	model.fit(xtrain, ytrain)
	model.score(xtrain,ytrain)
	pred = model.predict(xtest)
	ct=0
	for i in pred:
		if i==1:
			return 1
	return 0	

def svm(xtrain,ytrain,xtest):
	model = SVC()
	model.fit(xtrain, ytrain)
	model.score(xtrain,ytrain)
	pred = model.predict(xtest)
	ct=0
	for i in pred:
		if i==1:
			return 1
	return 0



def neuralNetwork(xtrain,ytrain,xtest):
	x_train=numpy.array(xtrain, dtype=object)
	y_train=numpy.array(ytrain, dtype=object)
	model = Sequential()
	model.add(Dense(12, input_dim=len(xtrain[0]), activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(x_train,y_train, epochs=10, batch_size=15) 
	x_test = numpy.array(xtest , dtype = object)
	# y_test = numpy.array(ytest , dtype = object)
	# scores = model.evaluate(x_test, y_test, verbose=0)
	pred = model.predict_classes(x_test)
	for i in pred:
		if i==1:
			return 1
	return 0


def getTrainingFiles(features):
	xtrain = []
	ytrain = []
	BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'algorithms','data.csv')
	with open(BASE_DIR) as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			r = []
			if 'mse' in features:
				r.append(float(row['mse']))
			if 'psnr' in features:
				r.append(float(row['psnr']))
			if 'histogram_compare' in features:
				r.append(float(row['histogram_compare']))
			if 'ssim' in features:
				r.append(float(row['ssim']))
			if 'entropy' in features:
				r.append(float(row['entropy']))
			if 'avgObjArea' in features:
				r.append(float(row['avgObjArea']))
			if 'displacedObjects' in features:
				r.append(float(row['displacedObjects']))
			xtrain.append(r)
			ytrain.append(int(row['class']))

	xtrain = numpy.array(xtrain,dtype=object)	
	xtrain = normalize(xtrain, norm='l2') 
	return xtrain,ytrain


def getResults(algortihm,features):
	folderPath = os.path.join(settings.MEDIA_ROOT,'frames')
	count = len(os.walk(folderPath).next()[2])
	i=0
	testFile = []
	while i<count-1:
		im1 = os.path.join(folderPath,'frame%d.png'%i)
		im2 = os.path.join(folderPath,'frame%d.png'%(i+1))
		featureSet= getFeatures(im1,im2)
		feature = []
		if 'mse' in features:
			feature.append(featureSet['mse'])
		if 'psnr' in features:
			feature.append(featureSet['psnr'])
		if 'histogram_compare' in features:
			feature.append(featureSet['histogram_compare'])
		if 'ssim' in features:
			feature.append(featureSet['ssim'])
		if 'entropy' in features:
			feature.append(featureSet['entropy'])
		if 'avgObjArea' in features:
			feature.append(featureSet['avgObjArea'])		
		if 'displacedObjects' in features:
			feature.append(featureSet['displacedObjects'])
		testFile.append(feature)
		i+=1
	xtrain,ytrain=getTrainingFiles(features)
	testFile = numpy.array(testFile,dtype=object)	
	testFile = normalize(testFile, norm='l2') 
	print numpy.any(numpy.isnan(xtrain))
	print numpy.any(numpy.isnan(testFile))
	xtrain[numpy.isnan(xtrain)] = numpy.median(xtrain[~numpy.isnan(xtrain)])
	testFile[numpy.isnan(testFile)] = numpy.median(testFile[~numpy.isnan(testFile)])
	
	if algortihm == 'knn':
		return kNN(xtrain,ytrain,testFile)
	elif algortihm == 'decisionTree':
		return decisionTree(xtrain,ytrain,testFile)
	elif algortihm =='logReg':
		return logisticRegression(xtrain,ytrain,testFile)
	elif algortihm == 'naiveBayes':
		return naiveBaiyes(xtrain,ytrain,testFile)
	elif algortihm == 'randomForest':
		return randomForest(xtrain,ytrain,testFile)
	elif algortihm == 'svm':
		return svm(xtrain,ytrain,testFile)
	else:
		return neuralNetwork(xtrain,ytrain,testFile)


	
