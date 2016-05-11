import numpy as np

class a(object):
	"""docstring for a"""
	def __init__(self, arg):
		super(a, self).__init__()
		self.arg = arg
		
	def add(self, b):
		return self.arg + b

	def vec_add(self, b):
		f = np.vectorize(self.add)
		return f(b)

if __name__ == '__main__':
	b = a(1)
	c = b.vec_add(np.asarray([1,1,1,1]))
	print(c)