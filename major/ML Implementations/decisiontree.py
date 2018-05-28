import csv
from sklearn import tree
from sklearn.metrics import accuracy_score

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




model = tree.DecisionTreeClassifier(criterion='gini')
model.fit(xtrain, ytrain)
model.score(xtrain,ytrain)

pred = model.predict(xtest)

print("Accuracy: %.2f%%" % (accuracy_score(ytest,pred)*100))
