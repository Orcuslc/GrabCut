# test for opencv
import cv2
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image

BLUE = [255,0,0]        # rectangle color
RED = [0,0,255]         # PR BG
GREEN = [0,255,0]       # PR FG
BLACK = [0,0,0]         # sure BG
WHITE = [255,255,255]   # sure FG

DRAW_BG = {'color' : BLACK, 'val' : 0}
DRAW_FG = {'color' : WHITE, 'val' : 1}
DRAW_PR_FG = {'color' : GREEN, 'val' : 3}
DRAW_PR_BG = {'color' : RED, 'val' : 2}

rect = (0,0,1,1)
drawing = False         # flag for drawing curves
rectangle = False       # flag for drawing rect
rect_over = False       # flag to check if rect drawn
rect_or_mask = 100      # flag for selecting rect or mask mode
value = DRAW_FG         # drawing initialized to FG
thickness = 3           # brush thickness

def onmouse(event, x, y, flags, param):
	global img,img2,drawing,value,mask,rectangle,rect,rect_or_mask,ix,iy,rect_over

	# Draw Rectangle
	if event == cv2.EVENT_RBUTTONDOWN:
		rectangle = True
		ix,iy = x,y

	elif event == cv2.EVENT_MOUSEMOVE:
	    if rectangle == True:
	    	img = img2.copy()
	    	cv2.rectangle(img,(ix,iy),(x,y),BLUE,2)
	    	rect = (min(ix,x),min(iy,y),abs(ix-x),abs(iy-y))
	    	rect_or_mask = 0

	elif event == cv2.EVENT_RBUTTONUP:
		rectangle = False
		rect_over = True
		cv2.rectangle(img,(ix,iy),(x,y),BLUE,2)
		rect = (min(ix,x),min(iy,y),abs(ix-x),abs(iy-y))
		rect_or_mask = 0
		print(" Now press the key 'n' a few times until no further change \n")

	# draw touchup curves

	# if event == cv2.EVENT_LBUTTONDOWN:
	# 	if rect_over == False:
	# 		print("first draw rectangle \n")
	# 	else:
	# 		drawing = True
	# 		cv2.circle(img,(x,y),thickness,value['color'],-1)
	# 		cv2.circle(mask,(x,y),thickness,value['val'],-1)

	# elif event == cv2.EVENT_MOUSEMOVE:
	# 	if drawing == True:
	# 		cv2.circle(img,(x,y),thickness,value['color'],-1)
	# 		cv2.circle(mask,(x,y),thickness,value['val'],-1)

	# elif event == cv2.EVENT_LBUTTONUP:
	# 	if drawing == True:
	# 		drawing = False
	# 		cv2.circle(img,(x,y),thickness,value['color'],-1)
	# 		cv2.circle(mask,(x,y),thickness,value['val'],-1)

if __name__ == '__main__':
	img = cv2.imread('E:\\Chuan\\Pictures\\ad.jpg', cv2.IMREAD_COLOR)
	img2 = img.copy()
	output = np.zeros(img.shape,np.uint8)   

	cv2.namedWindow('output')
	cv2.namedWindow('input')
	a = cv2.setMouseCallback('input',onmouse)
	cv2.moveWindow('input',img.shape[1]+10,90)

	while(1):
		cv2.imshow('output', output)
		cv2.imshow('input', img)

		k = 0xFF & cv2.waitKey(1)
		if k == 27:
			break
	cv2.destroyAllWindows()