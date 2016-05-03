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
	
	def __init__(self, k = 5):
		'''k is the number of components of GMM'''
		self.k = k
		self.weights = np.asarray([0. for i in range(k)]) # Weight of each component
		self.means = np.asarray([[0., 0., 0.] for i in range(k)]) # Means of each component
		self.covs = np.asarray([[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]] for i in range(k)]) # Covs of each component
		# self.cov_inv = np.asarray([np.linalg.inv(cov) for cov in self.covs]) # Calculate the inverse of each cov matrix in advance
		# self.cov_det = np.asarray([np.linalg.det(cov) for cov in self.covs]) # Calculate the det of each cov matrix;
		self.pixel_counts = np.asarray([0. for i in range(k)]) # Count of pixels in each components
		self.pixel_total_count = self.pixel_counts.sum() # The total number of pixels in the GMM
		# The following two parameters are assistant parameters for counting pixels and calc. pars.
		self._sums = np.asarray([[0., 0., 0.] for i in range(k)])
		self._prods = np.asarray([[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]] for i in range(k)])

	def _prob_pixel_component(self, pixel, ci):
		'''Calculate the probability of each pixel belonging to the ci_th component of GMM'''
		'''Using the formula of multivariate normal distribution'''
		# print(self.cov_inv)
		inv = self.cov_inv[ci]
		det = self.cov_det[ci]
		t = pixel - self.means[ci]
		nt = np.asarray([t])
		expo = np.dot(nt, inv)
		expo = np.dot(expo, np.transpose(nt))
		return (1/(det**0.5) * np.exp(-0.5*expo))[0]

	def prob_pixel_GMM(self, pixel):	
		'''Calculate the probability of each pixel belonging to this GMM, which is the sum of 
			the prob. of the pixel belonging to each component * the weight of the component'''
		'''Also the first term of Gibbs Energy(inversed;)'''
		return sum([self._prob_pixel_component(pixel, ci) * self.weights[ci] for ci in range(self.k)])

	def most_likely_pixel_component(self, pixel):
		'''Calculate the most likely component that the pixel belongs to'''
		g = np.vectorize(self._prob_pixel_component)
		prob = np.asarray([g(pixel, ci) * self.weights[ci] for ci in range(self.k)])
		return prob.argmax(0)

	def vec_pix_comp(self, pixel):
		f = np.vectorize(self.most_likely_pixel_component)
		return f(pixel)

	def add_pixel(self, pixel, ci):
		'''Add a pixel to the ci_th component of GMM, and refresh the parameters'''
		# print(np.asarray(pixel))
		tp = pixel.copy()
		tp.shape = (tp.size, 1)
		op = pixel.copy()
		op.shape = (1, op.size)
		self._sums[ci] += pixel
		self._prods[ci] += np.dot(tp, op)
		self.pixel_counts[ci] += 1
		self.pixel_total_count += 1

	def learning(self):
		variance = 0.01
		'''Learn the parameters with the data given; Also the 2th step in 'Iterative Minimization'.'''
		self.weights = np.asarray([self.pixel_counts[i]/self.pixel_total_count for i in range(self.k)]) # The weight of each comp. is the pixels in the comp. / total pixels.
		self.means = np.asarray([self._sums[i]/self.pixel_counts[i] for i in range(self.k)]) # The mean of each comp. is the sum of pixels of the comp. / the number of pixels in the comp.
		nm = np.asarray([[i] for i in self.means])
		self.covs = np.asarray([self._prods[i]/self.pixel_counts[i] - np.dot(np.transpose(nm[i]), nm[i]) for i in range(self.k)]) # The cov of each comp.
		self.cov_det = np.asarray([np.linalg.det(cov) for cov in self.covs])
		'''Avoid Singular Matrix'''
		for i in range(self.k):
			if self.cov_det[i] <= 0:
				self.covs[i] += np.diag([variance for i in range(3)])
				self.cov_det[i] = 0.000000001
		self.cov_inv = np.asarray([np.linalg.inv(cov) for cov in self.covs])
		# print(self.means)


class GCClient:
	'''The engine of grabcut'''
	def __init__(self, img, k):
		self.k = k # The number of components in each GMM model

		self.real_img = img
		self.img = np.asarray(img, dtype = np.float32)
		self.rows, self.cols = get_size(img)
		self.gamma = 50
		self.beta = 0

		self._BLUE = [255,0,0]        # rectangle color
		self._RED = [0,0,255]         # PR BG
		self._GREEN = [0,255,0]       # PR FG
		self._BLACK = [0,0,0]         # sure BG
		self._WHITE = [255,255,255]   # sure FG

		# setting up flags
		self._rect = [0, 0, 1, 1]
		self._drawing = False         # flag for drawing curves
		self._rectangle = False       # flag for drawing rect
		self._rect_over = False       # flag to check if rect drawn
		# se;f._rect_or_mask = 100      # flag for selecting rect or mask mode
		# self._value = DRAW_FG         # drawing initialized to FG
		self._thickness = 3           # brush thickness
		
		self._GC_BGD = 0	#{'color' : BLACK, 'val' : 0}
		self._GC_FGD = 1	#{'color' : WHITE, 'val' : 1}
		self._GC_PR_BGD = 2	#{'color' : GREEN, 'val' : 3}
		self._GC_PR_FGD = 3	#{'color' : RED, 'val' : 2}


	def calc_beta(self):
		'''Calculate Beta -- The Exp Term of Smooth Parameter in Gibbs Energy'''
		'''beta = 1/(2*average(sqrt(||pixel[i] - pixel[j]||)))'''
		'''Beta is used to adjust the difference of two nearby pixels in high or low contrast rate'''
		self._left_diff = self.img[:, 1:] - self.img[:, :-1] # Left-difference
		self._upleft_diff = self.img[1:, 1:] - self.img[:-1, :-1] # Up-Left difference
		self._up_diff = self.img[1:, :] - self.img[:-1, :] # Up-difference
		self._upright_diff = self.img[1:, :-1] - self.img[:-1, 1:] # Up-Right difference
		beta = (self._left_diff*self._left_diff).sum() + (self._upleft_diff*self._upleft_diff).sum() \
			+ (self._up_diff*self._up_diff).sum() + (self._upright_diff*self._upright_diff).sum() # According to the formula
		self.beta = 1/(2*beta/(4*self.cols*self.rows - 3*self.cols - 3*self.rows + 2)) # According to the paper

	def calc_nearby_weight(self):
		'''Calculate the weight of the edge of each pixel with its nearby pixel, as each pixel is regarded
			as a vertex of the graph'''
		'''The weight of each direction is saved in a image the same size of the original image'''
		'''weight = gamma*exp(-beta*(diff*diff))'''
		self.left_weight = np.zeros([self.rows, self.cols])
		self.upleft_weight = np.zeros([self.rows, self.cols])
		self.up_weight = np.zeros([self.rows, self.cols])
		self.upright_weight = np.zeros([self.rows, self.cols])
		# Use the formula to calculate the weight
		self.left_weight[:, 1:] = self.gamma*np.exp(-self.beta*(self._left_diff*self._left_diff))
		self.upleft_weight[1:, 1:] = self.gamma*exp(-self.beta*(self._upleft_diff*self._upleft_diff))
		self.up_weight[1:, :] = self.gamma*exp(-self.beta*(self._up_diff*self._up_diff))
		self.upright_weight = self.gamma*exp(-self.beta*(self._upright_diff*self._upright_diff))

	
	'''The following function is derived from the sample of opencv sources'''
	def init_mask(self, event, x, y, flags, param):
		'''Init the mask with interactive movements'''
		'''Notice: the elements in the mask should be within the follows:
			"GC_BGD":The pixel belongs to background;
			"GC_FGD":The pixel belongs to foreground;
			"GC_PR_BGD":The pixel MAY belongs to background;
			"GC_PR_FGD":The pixel MAY belongs to foreground;'''

		# Draw Rectangle
		if event == cv2.EVENT_RBUTTONDOWN:
			self._rectangle = True
			self._ix,self._iy = x,y

		# elif event == cv2.EVENT_MOUSEMOVE:
		#     if self._rectangle == True:
		#     	cv2.rectangle(self.img,(self._ix,self._iy),(x,y),self._BLUE,2)
		    	# self._rect = [min(self._ix,x),min(self._iy,y),abs(self._ix-x),abs(self._iy-y)]

		elif event == cv2.EVENT_RBUTTONUP:
			self._rectangle = False
			self._rect_over = True
			cv2.rectangle(self.img,(self._ix,self._iy),(x,y),self._BLUE,2)
			self._rect = [min(self._ix,x),min(self._iy,y),abs(self._ix-x),abs(self._iy-y)]
			# print(" Now press the key 'n' a few times until no further change \n")

		self._mask = np.zeros([self.rows, self.cols], dtype = np.uint8) # Init the mask
		self._mask[:, :] = self._GC_BGD
		self._mask[self._rect[0]:self._rect[0]+self._rect[2], self._rect[1]:self._rect[1]+self._rect[3]] = self._GC_PR_FGD

	def init_with_kmeans(self):
		print(self.cols*self.rows)
		print(len(list(np.where(self._mask == 0))[1]))
		'''Initialise the BGDGMM and FGDGMM, which are respectively background-model and foreground-model,
			using kmeans algorithm'''
		max_iter = 10 # Max-iteration count for Kmeans
		'''In the following two indexings, the np.logical_or is needed in place of or'''
		self._bgd = np.where(np.logical_or(self._mask == self._GC_BGD, self._mask == self._GC_PR_BGD)) # Find the places where pixels in the mask MAY belong to BGD.
		self._fgd = np.where(np.logical_or(self._mask == self._GC_FGD, self._mask == self._GC_PR_FGD)) # Find the places where pixels in the mask MAY belong to FGD.
		self._BGDpixels = self.img[self._bgd]
		self._FGDpixels = self.img[self._fgd]
		KMB = kmeans(self._BGDpixels, n = self.k) # The Background Model by kmeans
		KMF = kmeans(self._FGDpixels, n = self.k) # The Foreground Model by kmeans
		KMB.run()
		KMF.run()
		self._BGD_by_components = KMB.output()
		self._FGD_by_components = KMF.output()
		self.BGD_GMM = GMM() # The GMM Model for BGD
		self.FGD_GMM = GMM() # The GMM Model for FGD
		'''Add the pixels by components to GMM'''
		for ci in range(self.k):
			# print(self._BGD_by_components[ci])
			for pixel in self._BGD_by_components[ci]:
				# pixel = np.asarray([j for j in pixel], dtype = np.float32)
				# print(pixel)
				# print('\n')
				self.BGD_GMM.add_pixel(pixel, ci)
			for pixel in self._FGD_by_components[ci]:
				self.FGD_GMM.add_pixel(pixel, ci)
		self.BGD_GMM.learning()
		self.FGD_GMM.learning()

	'''The first step of the iteration in the paper: Assign components of GMMs to pixels,
		(the kn in the paper), which is saved in self.components_index'''
	def assign_GMM_components(self):
		self.components_index = np.zeros([self.rows, self.cols], dtype = np.uint)
		self.components_index[self._bgd] = [i[0] for i in self.BGD_GMM.vec_pix_comp(self.img[self._bgd])]
		self.components_index[self._fgd] = [i[0] for i in self.FGD_GMM.vec_pix_comp(self.img[self._fgd])]

	'''The second step in the iteration: Learn the parameters from GMM models'''
	def learn_GMM_parameters(self):
		for ci in range(self.k):
			# The places where the pixel belongs to the ci_th model and background model.
			bgd_ci = np.where(np.logical_and(self.components_index == ci, np.logical_or(self._mask == self._GC_BGD, self._mask == self._GC_PR_BGD)))
			fgd_ci = np.where(np.logical_and(self.components_index == ci, np.logical_or(self._mask == self._GC_FGD, self._mask == self._GC_PR_FGD)))
			for pixel in self.img[bgd_ci]:
				self.BGD_GMM.add_pixel(pixel, ci)
			for pixel in self.img[fgd_ci]:
				self.FGD_GMM.add_pixel(pixel, ci)
		self.BGD_GMM.learning()
		self.FGD_GMM.learning()

	def 




if __name__ == '__main__':
	img = cv2.imread('E:\\Chuan\\Pictures\\ab.jpg', cv2.IMREAD_COLOR)
	output = np.zeros(img.shape,np.uint8)

	GC = GCClient(img, k = 5)

	cv2.namedWindow('output')
	cv2.namedWindow('input')
	a = cv2.setMouseCallback('input',GC.init_mask)
	cv2.moveWindow('input',img.shape[1]+10,90)

	while(1):
		cv2.imshow('output', output)
		cv2.imshow('input', np.asarray(GC.img, dtype = np.uint8))

		k = 0xFF & cv2.waitKey(1)

		if k == 27:
			break
		elif k == ord('n'):
			GC.init_with_kmeans()
			GC.assign_GMM_components()
			GC.learn_GMM_parameters()

	cv2.destroyAllWindows()
