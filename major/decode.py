from PIL import Image
import os


def decode_image(img):
    """
    check the red portion of an image (r, g, b) tuple for
    hidden message characters (ASCII values)
    """
    width, height = img.size
    msg = ""
    index = 0
    for row in range(height):
        for col in range(width):
            try:
                r, g, b = img.getpixel((col, row))
            except ValueError:
                # need to add transparency a for some .png files
                r, g, b, a = img.getpixel((col, row))		
            # first pixel r value is length of message
            if row == 0 and col == 0:
                length = r
            elif index <= length:
                msg += chr(r)
            index += 1
    return msg


count = len(os.walk('watermark_frames').next()[2])
# total_frames = len(os.walk('png_frames').next()[2])
i = 0
flag = 0
tamper = 0

while i < count:
	if os.path.isfile('watermark_frames/frame%d.png' %i):
		im = Image.open('watermark_frames/frame%d.png' %i)
	
		hidden_text = int(decode_image(im))
		prev = hidden_text
		print("Hidden text:\n{}".format(hidden_text))
		# flag = 1;
		#index+=1
	i += 1

# if prev != 0: #cut from start
# 	tamper = 1
# while i < total_frames and tamper == 0:
# 	if os.path.isfile('test_frames/frame%d.jpg' %i):
# 		im = Image.open('test_frames/frame%d.jpg' %i)
	
# 		hidden_text = int(decode_image(im))
# 		diff = hidden_text - prev
# 		prev = hidden_text
# 		if diff != 1:
# 			tamper = 1
# 			break
# 		print("Hidden text:\n{}".format(hidden_text))
# 	i += 1

# if hidden_text != total_frames: #cut from end
# 	tamper = 1
# print tamper

