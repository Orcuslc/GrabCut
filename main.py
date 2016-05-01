# GrabCut 
# Copyleft 2016 Orcuslc

import cv2
import numpy as np 
import matplotlib.pyplot as plt 

def get_size(img):
	return list(img.shape)[:2]

def flat(img):
	return img.reshape([1, img.size])[0]

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
		'''
		self.T is the trimap consists of B, F, U
		'''
		self._B = []
		self._F = []
		self._U = []
		'''
		The image should be flatten to a one-dimension array
		'''
		self.row, self.col = get_size(img)
		self.size = img.size
		self.img = flat(img)
		'''
		The Learning Parameters
		'''
		self.alpha = [0 for i in range(self.size)]
		self.k = k

	def _handle_place(self, place):
		'''
		Transform the place in [x, y] shape into the index
		of self.img
		'''
		return self.col*place[0]+place[1]

	def E(self, alpha, k, theta, z):
		'''
		The Gibbs energy
		'''

	def init_trimap(self, places, *args):
		'''
		Need to complete later.
		
		Now the init area is determined by the points
		on the corner.

		....(i1, j1), ...(i2, j2), ...
		...
		...
		....(i3, j3), ...(i4, j4), ...

		The pixels in the rectangle are set to U,
		with pixels outside are set to B.
		'''
		left = self._handle_place(places[0])
		length = places[1][1] - places[0][1]
		height = places[2][0] - places[0][0]
		self._U = [left + k*self.col + j for k in range(height+1) for j in range(length+1)]
		self._B = [i for i in range(self.size) if i not in self._U]
		'''
		Initialise alpha = 0 for n in B and alpha = 1 for n in U
		'''
		for index in self._U:
			self.alpha[index] = 1
		


	def test(self):
		print(self._U)
		print(self._B)

if __name__ == '__main__':
	img = np.array([i for i in range(100)]).reshape([10, 10])
	G = Grab_Cut_Client(img)
	G.init_trimap([[1,1], [1,7], [3,1], [3,7]])
	G.test()