from PIL import Image
import os

def encode_image(img, msg):
    """
    use the red portion of an image (r, g, b) tuple to
    hide the msg string characters as ASCII values
    red value of the first pixel is used for length of string
    """

    length = len(msg)
    # limit length of message to 255
    if length > 255:
        print("text too long! (don't exeed 255 characters)")
        return False
    if img.mode != 'RGB':
        print("image mode needs to be RGB")
        return False
    # use a copy of image to hide the text in
    encoded = img.copy()
    width, height = img.size
    index = 0
    for row in range(height):
        for col in range(width):
            r, g, b = img.getpixel((col, row))
            # first value is length of msg
            if row == 0 and col == 0 and index < length:
                asc = length
            elif index <= length:
                c = msg[index -1]
                asc = ord(c)
            else:
                asc = r
            encoded.putpixel((col, row), (asc, g , b))
            index += 1
    return encoded



count = len(os.walk('frames').next()[2])

i=0
while i<count-1:
	im = Image.open('frames/frame%d.jpg' %i)
	# im.save('png_frames/frame%d.png' %i)	
	# original_image_file = 'png_frames/frame%d.png' %i
	# img = Image.open(original_image_file)
	#print(img, img.mode)  # test
	#encoded_image_file = "enc_" + original_image_file
	secret_msg = str(i)
	img_encoded = encode_image(im, secret_msg)
	if img_encoded:
		# save the image with the hidden text
	        img_encoded.save("watermark_frames/frame%d.png" % i)
	        #print("{} saved!".format(encoded_image_file))

	i+=1





