#!/usr/bin/env python3

import numpy as np

class Nmf:
	"""
	This class changes a passed in 2d nested list into a NumPy matrix datatype, and holds the self.difCost() function for calculating the error between the current weights and features matrices and the dataset matrix.
	
	This class should be extended with added features to operate with specific datasets in specific applications.
	"""
	def __init__(self, dataset):
		"""
		Instances of the Nmf class are initialized with a passed in 2d nested list dataset.
		"""
		self.data = np.array(dataset)
		
	def dif_cost(self, dmatrix, wfmatrix):
		"""
		Calculates the total error between the weights * features matrix and the dataset matrix.
		
		Loops over every value in the two equal sized matrices and sums the squares of the differences between each value.
		"""
		dif = 0.0
		#loop over every row and column in these equal sized matrices
		for i in range(np.shape(dmatrix)[0]):
			for j in range(np.shape(dmatrix)[1]):
				#add the square of their difference to the error sum, note NumPy matrix syntax
				dif += (dmatrix[i, j] - wfmatrix[i, j]) ** 2
		
		#total error
		return dif
		
	def factorize(self, fnum=10, maxiter=50):
		"""
		Factorization driver function.
		
		NMF iteratively factors the dataset matrix into a weights matrix and a features matrix using "multiplicative update rules". It returns them as a (weights, features) tuple of NumPy matrix objects.
		"""
		ic = np.shape(self.data)[0]
		fc = np.shape(self.data)[1]
		
		#initialize weights and features matrices with random values
		weights = []
		for i in range(ic):
			weights.append([])
			for j in range(fnum):
				weights[i].append(np.random.random())
		weights = np.array(weights)
		
		features = []
		for i in range(fnum):
			features.append([])
			for j in range(fc):
				features[i].append(np.random.random())
		features = np.array(features)
		
		#main loop
		for i in range(maxiter):
			wf = np.matmul(weights, features)
			
			#calculate current error between dataset matrix and wf matrix
			cost = self.dif_cost(self.data, wf)
            
			#if i is divisble by 10 print this error value
			if i % 10 == 0:
				print(cost)
				
			#terminate if matrix has been fully factorized
			if cost == 0:
				break
				
			#update features matrix
			hn = np.matmul((np.transpose(weights)), self.data)
			hd = np.matmul(np.matmul((np.transpose(weights)), weights), features)
			features = np.array(np.array(features) * np.array(hn) / np.array(hd))
			
			#update weights matrix
			wn = np.matmul(self.data, (np.transpose(features)))
			wd = np.matmul(np.matmul(weights, features), (np.transpose(features)))
			weights = np.array(np.array(weights) * np.array(wn) / np.array(wd))
			
		#After maxiter iterations or full factorization return the weights and features matrix objects.
		return weights, features
