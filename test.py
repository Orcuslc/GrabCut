# test for opencv
import cv2
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image

if __name__ == '__main__':
	img = cv2.imread('E:\\Chuan\\Pictures\\a.jpg', cv2.IMREAD_COLOR)
	cv2.imshow("img", img)
	k = cv2.waitKey(0)
	if k == 27:
		cv2.destroyAllWindows()
		