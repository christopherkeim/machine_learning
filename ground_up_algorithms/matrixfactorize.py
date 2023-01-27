#!/usr/bin/env python3

import numpy as np
import random

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
		
	def difCost(self, matrix, wfmatrix):
		"""
		Calculates the total error between the weights * features matrix and the dataset matrix.
		
		Loops over every value in the two equal sized matrices and sums the squares of the differences between each value.
		"""
		dif = 0.0
		#loop over every row and column in these equal sized matrices
		for i in range(len(np.shape(dmatrix)[0])):
			for j in range(np.shape(dmatrix)[1]):
				#add the square of their difference to the error sum, note NumPy matrix syntax
				dif += (dmatrix[i, j] - wfmatrix[i, j]) ** 2
		
		#total error
		return dif
		
	def factorize(self, dataset=self.data, fnum=10, maxiter=50):
		"""
		Factorization driver function.
		
		NMF iteratively factors the dataset matrix into a weights matrix and a features matrix using "multiplicative update rules". It returns them as a (weights, features) tuple of NumPy matrix objects.
		"""
		ic = np.shape(dataset)[0]
		fc = np.shape(dataset)[1]
		
		#initialize weights and features matrices with random values
		weights = []
		for i in range(ic):
			weights.append([])
			for j in range(fnum):
				weights[i].append(random.random())
		weights = np.array(weights)
		
		features = []
		for i in range(fnum):
			features[i] = []
			for j in range(fc):
				features[i][j] = random.random()
		features = np.array(features)
		
		#main loop
		for i in range(maxiter):
			wf = weights * features
			
			#calculate current error between dataset matrix and wf matrix
			cost = self.difCost(dataset, wf)
            
			#if i is divisble by 10 print this error value
			if i % 10 == 0:
				print(cost)
				
			#terminate if matrix has been fully factorized
			if cost == 0:
				break
				
			#update features matrix
			hn = (np.transpose(weights) * dataset)
			hd = (np.transpose(weights) * weights * features)
			features = np.matrix(np.array(features) * np.array(hn) / np.array(hd))
			
			#update weights matrix
			wn = (dataset * np.transpose(features))
			wd = (weights * features * np.transpose(features))
			weights = np.matrix(np.array(weights) * np.array(wn) / np.array(wd))
			
		#after maxiter iterations or full factorization return the weights and features matrix objects 
		return weights, features
