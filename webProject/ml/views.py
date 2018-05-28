from __future__ import unicode_literals
from django.http import HttpResponseRedirect
from django.shortcuts import render
from django.conf import settings
from django.core.files.base import ContentFile
import os
from algorithms.FrameManipulation import extract,tamperVideo,getNumberofFrames
from algorithms.ModelTraining import getResults
from algorithms.watermarking import addWaterMark,checkAuthenticity
from .forms import UploadFileForm,mlForm,TamperingForm

def home(request):
	form=UploadFileForm(request.POST or None)
	if request.method == 'POST':
		form = UploadFileForm(request.POST, request.FILES)
		if form.is_valid():
			path = os.path.join(settings.MEDIA_ROOT,"test.mp4")
			lines = form.cleaned_data['file'].readlines()
			with open(path,'w') as destination:
				for line in lines:
					destination.write(line)
			request.session['category'] = form.cleaned_data['Approach']	
			if form.cleaned_data['Approach']=='watermarking':
				addWaterMark()
			else:
				extract()
			return HttpResponseRedirect('tamper')
	else:
		form = UploadFileForm()
	return render(request, 'base.html', {'form': form})


def tamper(request):
	val = getNumberofFrames()
	form  = TamperingForm(request.POST or None,val)
	cat = request.session.get('category')
	context = {
		'form' : form,
		'count': val,
	}
	if form.is_valid():
		stValue = form.cleaned_data['start']
		enValue = form.cleaned_data['end']
		if stValue==0 and enValue==0:
			if cat=='ml_approach' :
				return HttpResponseRedirect('mlApproach')
			else:
				result = checkAuthenticity()
				if result==1:
					ans = 'Video is Tampered'
				else:
					ans = 'Video is not Tampered'
				context = {
					'result' : ans,
				} 
				return render(request,'finalPage.html',context)
		val = tamperVideo(stValue,enValue)
		context = {
			'form' : form,
			'count' : val,
		}
	return render(request,'tampering.html',context)




def mlApproach(request):
	form = mlForm(request.POST or None)	
	context = {
		'form' : form,
	}
	if form.is_valid():
		result = getResults(form.cleaned_data['algorithms'],form.cleaned_data['features'])	
		ans=''
		if result==1:
			ans = 'Video is Tampered'
		else:
			ans = 'Video is not Tampered'
		context = {
			'result' : ans,
		}
		return render(request,'finalPage.html',context)
	return render(request,'mlPage.html',context)
