import cv2
from skimage.measure import compare_mse, compare_nrmse, compare_ssim, compare_psnr
import csv
import os
from random import randint
import math
import imutils
"""
In Code References
https://stackoverflow.com/questions/45945258/compare-frame-of-video-with-another-image-python
http://scikit-image.org/docs/stable/api/skimage.measure.html
https://docs.opencv.org/3.2.0/d8/dc8/tutorial_histogram_comparison.html
"""
count = len(os.walk('frames').next()[2])

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



def getFeatures(im1,im2,isTampered):
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
	features['class']=isTampered
	return features


with open('testVideo.csv', 'w') as csvfile:
	fieldnames = ['mse','psnr','histogram_compare','ssim','entropy','avgObjArea','displacedObjects','class']	
	writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
	writer.writeheader()	
#	Non Tampered Frames
	# i=0
	# while i<count-1:
	# 	im1 = 'frames/frame%d.jpg' %i
	# 	im2 = 'frames/frame%d.jpg' %(i+1) 
	# 	features = getFeatures(im1,im2,0)
	# 	# print features
	# 	writer.writerow(features)
	# 	i+=1

	i=0
	while i<5:
		im1 = 'frames/frame%d.jpg' %i
		im2 = 'frames/frame%d.jpg' %(i+1) 
		features = getFeatures(im1,im2,0)
		# print features
		writer.writerow(features)
		i+=1
	im1 = 'frames/frame5.jpg'
	im2 = 'frames/frame15.jpg' 
	features = getFeatures(im1,im2,1)
	writer.writerow(features)
	i=15
	while i<count-1:
		im1 = 'frames/frame%d.jpg' %i
		im2 = 'frames/frame%d.jpg' %(i+1) 
		features = getFeatures(im1,im2,0)
		# print features
		writer.writerow(features)
		i+=1
# #   Tampered Frames
# 	i=0
# 	while i<count-5:
# 		j = randint(i+2,count-1)
# 		im1 = 'frames/frame%d.jpg' %i
# 		im2 = 'frames/frame%d.jpg' %j 
# 		features = getFeatures(im1,im2,1)
# 		# print features
# 		writer.writerow(features)
# 		i+=1


