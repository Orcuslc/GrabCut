# GrabCut 
# Copyleft 2016 Orcuslc

import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import random
from kmeans import kmeans

def get_size(img):
	return list(img.shape)[:2]

def flat(img):
	return img.reshape([1, img.size])[0]

class GMM:
	'''The GMM algorithm'''
	'''Each point in the image belongs to a GMM, and because each pixel owns
		three channels: RGB, so each component owns three means, 9 covs and a weight.'''
	
	def __init__(self, k):
		'''k is the number of components of GMM'''
		self.k = k
		self.weights = np.asarray([0 for i in range(k)]) # Weight of each component
		self.means = np.asarray([[0, 0, 0] for i in range(k)]) # Means of each component
		self.covs = np.asarray([[0, 0, 0], [0, 0, 0], [0, 0, 0]] for i in range(k)) # Covs of each component
		self.cov_inv = np.asarray(np.linalg.inv(cov) for cov in self.covs) # Calculate the inverse of each cov matrix in advance
		self.cov_det = np.asarray(np.linalg.det(cov) for cov in self.covs) # Calculate the det of each cov matrix;

	def _prob_pixel_component(self, pixel, ci):
		'''Calculate the probability of each pixel belonging to the ci_th component of GMM'''
		'''Using the formula of multivariate normal distribution'''
		inv = self.cov_inv[ci]
		det = self.cov_det[ci]
		t = pixel - self.means[ci]
		return 1/(((2*np.pi)**1.5)*(det**0.5)) * np.exp(-0.5*np.dot(t, inv, np.transpose(t)))

	def prob_pixel_GMM(self, pixel):	
		'''Calculate the probability of each pixel belonging to this GMM, which is the sum of 
			the prob. of the pixel belonging to each component * the weight of the component'''
		'''Also the first term of Gibbs Energy(inversed;)'''
		return sum([self._prob_pixel_component(pixel, ci) * self.weights[ci] for ci in range(self.k)])

	def most_likely_pixel_component(self, pixel):
		'''Calculate the most likely component that the pixel belongs to'''
		prob = np.asarray([self._prob_pixel_component(pixel, ci) * self.weights[ci] for ci in range(self.k)])
		return prob.argmax()

	def learning():

	def 



class GrabClient:
	'''The engine of grabcut'''
	def __init__(self, img):
		self.img = img

	def init_mask(self, mask):
		


































# class Grab_Cut_Client:
# 	'''
# 	The main class for GrabCut Algorithm.
# 	'''
# 	def __init__(self, img, k = 5):
# 		'''
# 		Init;
# 		img : the img to cut;
# 		k : the number of characters in GMM, with a default value of 5;
# 		'''
# 		self.foreground = []
# 		self.background = []
# 		'''
# 		self.T is the trimap consists of B, F, U
# 		'''
# 		self._B = []
# 		self._F = []
# 		self._U = []
# 		'''
# 		The image should be flatten to a one-dimension array
# 		'''
# 		self.row, self.col = get_size(img)
# 		self.size = img.size
# 		self.img = flat(img)
# 		'''
# 		The Learning Parameters
# 		'''
# 		self.alpha = [0 for i in range(self.size)]
# 		self.k = k
# 		self.sigma = []
# 		self.pi = []
# 		self.mu = []

# 	def _handle_place(self, place):
# 		'''
# 		Transform the place in [x, y] shape into the index of self.img
# 		'''
# 		return self.col*place[0]+place[1]

# 	# def E(self, alpha, k, theta, z):
# 	# 	'''
# 	# 	The Gibbs energy
# 	# 	'''

# 	def init_trimap(self, places, *args):
# 		'''
# 		Need to complete later.
		
# 		Now the init area is determined by the points
# 		on the corner.

# 		....(i1, j1), ...(i2, j2), ...
# 		...
# 		...
# 		....(i3, j3), ...(i4, j4), ...

# 		The pixels in the rectangle are set to U,
# 		with pixels outside are set to B.
# 		'''
# 		left = self._handle_place(places[0])
# 		length = places[1][1] - places[0][1]
# 		height = places[2][0] - places[0][0]
# 		self._U = [left + k*self.col + j for k in range(height+1) for j in range(length+1)]
# 		self._B = [i for i in range(self.size) if i not in self._U]
# 		'''
# 		Initialise alpha = 0 for n in B and alpha = 1 for n in U
# 		'''
# 		for index in self._U:
# 			self.alpha[index] = 1
# 		'''
# 		Initialise the two distributions with GMM respectively;
# 		We may use kmeans to devide U and B into different GMM models.
# 		The number of models is k;
# 		A more concrete explanation:
# 		http://blog.csdn.net/zouxy09/article/details/8534954
# 		'''
# 		# self.mu = [[random.random() for i in range(k)] for j in range(2)]
# 		# self.sigma = [np.array([random.random() for i in range(k*k)]).reshape([k, k]) for j in range(2)]
# 		# self.pi = [np.array([random.random() for i in range(k)]) for j in range(2)]
# 		# self.pi = [x/sum(x) for x in self.pi]
# 		U = self.img[self._U]
# 		B = self.img[self._B]
# 		k1 = kmeans(U, self.k)
# 		k1.run()
# 		k2 = kmeans(B, self.k)
# 		k2.run()
# 		self._U_components = k1.output()
# 		self._B_components = k2.output()
# 		self.pi = []


# 	def iter(self):


# 	def test(self):
# 		print(self._U)
# 		print(self._B)

# if __name__ == '__main__':
# 	img = np.array([i for i in range(100)]).reshape([10, 10])
# 	G = Grab_Cut_Client(img)
# 	G.init_trimap([[1,1], [1,7], [3,1], [3,7]])
# 	G.test()