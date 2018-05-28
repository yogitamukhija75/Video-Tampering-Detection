import cv2
import sys
import os
from django.conf import settings
import shutil
from FrameManipulation import extract,getNumberofFrames
def getHashCol(height,width,length,row):
	return (row*abs(height-width)+length-row-1)%width


def encode_image(img, msg):
	length = len(msg)
	encoded = img.copy()
	height, width = img.shape[:2]
	r,g,b = img[0,0]	
	encoded[0,0]=(length,g,b)
	row=1
	for m in msg:
		col = getHashCol(height,width,length,row)
		r,g,b = img[row,col]
		encoded[row,col]=(ord(m),g,b)
		row+=1
	return encoded

def decode_image(img):
	height, width = img.shape[:2]
	index = 0
	r,g,b = img[0,0]
	length = r
	row=1
	msg = ''
	for i in range(length):
		col = getHashCol(height,width,length,row)
		r,g,b = img[row,col]
		msg +=chr(r)
		row+=1
	return msg

def addWaterMark():
	count = extract()
	v = len("{0:b}".format(count))
	i = 0
	folderPath = os.path.join(settings.MEDIA_ROOT,'frames')
	while i < count:
		imPath = os.path.join(folderPath,'frame%d.png'%i)
		img = cv2.imread(imPath)
		value = "{0:b}".format(i+1)
		if len(value)<v:
			value = (v-len(value))*'0'+value
		enc = encode_image(img,str(value))
		cv2.imwrite(imPath,enc)
		i+=1

def checkAuthenticity():
	count = getNumberofFrames()
	folderPath = os.path.join(settings.MEDIA_ROOT,'frames')
	for i in range(count):
		imPath = os.path.join(folderPath,'frame%d.png'%i)
		img = cv2.imread(imPath)
		value = decode_image(img)
		value = int(value,2)
		print value
		if value is not i+1:
			return 1
	return 0

