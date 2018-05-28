import cv2
# from skimage.measure import compare_mse, compare_nrmse, compare_ssim, compare_psnr
# from PIL import Image
import sys
import csv

vidcap = cv2.VideoCapture('test.mp4')
count=0
while True:
	success,image = vidcap.read()
	if success == False:
		break;
	cv2.imwrite("frames/frame%d.jpg" % count, image) 
	count+=1
print count