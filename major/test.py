import cv2
import numpy as np
import sys
import os
import shutil
from PIL import Image

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

	count = len(os.walk('frames').next()[2])
	i=0
	v = len("{0:b}".format(count))
	while i<count:
		imPath = os.path.join('frames/frame%d.jpg'%i)
		img = cv2.imread(imPath)
		value = "{0:b}".format(i+1)
		if len(value)<v:
			value = (v-len(value))*'0'+value
		imPath = os.path.join('ff/frame%d.png'%i)
		enc = encode_image(img,str(value))
		cv2.imwrite(imPath,enc)
		enc = cv2.imread(imPath)
		arr =  decode_image(enc) 
		print int(arr,2)
		i+=1

addWaterMark()