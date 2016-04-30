# GrabCut 
# Copyleft 2016 Orcuslc

import cv2
import numpy as np 
import matplotlib.pyplot as plt 

def get_size(img):
	return list(img.shape)[:2]

class Grab_Cut_Client:
	'''
	The main class for GrabCut Algorithm.
	'''
	def __init__(self, img, k = 5):
		'''
		Init;
		img : the img to cut;
		k : the number of characters in GMM, with a default value of 5;
		'''
		self.foreground = []
		self.background = []
		# self.T is the trimap consists of B, F, U
		self.B = []
		self.F = []
		self.U = []
		self.row, self.col = get_size(img)
		'''
		The image should be flatten to a one-dimension array
		'''
		self.img = self._img_flat(img)

	def _img_flat(self, img):
		


	def E(self, alpha, k, theta, z):
		'''
		The Gibbs energy
		'''

	def init_trimap(self, places, *args):
		'''
		Need to complete later.
		Now set by input a list of four points on the corner:
		[[i1, j1], [i2, j2], [i3, j3], [i4, j4]]
		The pixels in the rectangle are set to U,
		with pixels outside are set to B.
		'''
		a = places[0]
		b = places[1]
		c = places[2]
		d = places[4]
		for i in range(self.row):
			for j in range(self.col):
				if i 

