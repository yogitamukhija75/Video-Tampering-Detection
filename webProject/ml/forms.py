from django import forms

class UploadFileForm(forms.Form):
	CHOICES=[('ml_approach','Machine Learning Approach'),('watermarking','Watermarking Approach')]
	Approach = forms.ChoiceField(choices=CHOICES, widget=forms.RadioSelect())
	file = forms.FileField()

class mlForm(forms.Form):
	algoChoices = [
                ('knn','KNN'),
                ('decisionTree','Decision Tree'),
                ('logReg','Logistic Regression'),
                ('naiveBayes','Naive Bayes'),
                ('randomForest','Random Forest'),
                ('svm','SVM'),
                ('neuralNet','neuralNetwork')]
	
        featureChoices = [
                ('mse','Mean Square Error'),
                ('psnr','Peak Signal to Noise Ratio'),
                ('histogram_compare','Histogram Compare'),
                ('ssim','Structural Similarity Index'),
                ('entropy','Entropy'),
                ('displacedObjects','Number of Objects Displaced'),
                ('avgObjArea','Average Object Area')]
	
        features = forms.MultipleChoiceField(choices = featureChoices,widget=forms.CheckboxSelectMultiple())  
	algorithms = forms.ChoiceField(choices=algoChoices,widget=forms.RadioSelect())
	
class TamperingForm(forms.Form):
	start = forms.IntegerField(min_value=0,max_value=100000)
	end  = forms.IntegerField(min_value=0,max_value=100000)
