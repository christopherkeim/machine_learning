#!/usr/bin/env python3.11

from math import log
from PIL import Image, ImageDraw

class TreeNode:
	"""
	Decision nodes in our tree.
	
	A TreeNode instance can either be a "branch" or a "leaf"; a branch holds 2 other TreeNode instances in its self.trueBranch and self.falseBranch variables, and a leaf is an endpoint that holds only a results dictionary.
	"""
	def __init__(self, columnId=-1, value=None, trueBranch=None, falseBranch=None, results=None):
		self.columnId = columnId
		self.value = value
		self.trueBranch = trueBranch
		self.falseBranch = falseBranch
		self.results = results
		
class DecisionTree:
	"""
	This class is instantiated with a 2d nested list matrix and holds a TreeNode datatype instance variable with the automatically trained model root node, along with all of the training, classification/regression, and display functionality of the CART algorithm. 
	
	The algorithm assumes the target output variable you are trying to model is the last index within each sample. [more]
	"""
	def __init__(self, samples, mode='c', outputVarIndex=-1):
		"""
		This class is initialized with a 2d nested list matrix.
		
		Specify either a classification tree or regression tree, declare if target output variable is not in default last column index.
		"""
		if outputVarIndex != -1:
			lastColMatrix = self.transformMatrix(samples, outputVarIndex)
			samples = lastColMatrix
			
		if mode == 'c':
		    
			self.tree = self.train(samples, scoreFxn=self.entropy)
			
		elif mode == 'r':
		
			self.tree = self.train(samples, scoreFxn=self.variance)
			
		else:
			
			self.tree = self.train(samples, scoreFxn=self.entropy)
			print("mode parameter can only be 'c' for classification or 'r' for regression - set to 'c' as default.")
			
	def transformMatrix(self, samples, varIndex):
		"""
		Utility function that returns 2d nested list matrix with target output variable in last column.
		"""
		tmatrix = samples[:]
		for i in range(len(tmatrix)):
			val = tmatrix[i].pop(varIndex)
			tmatrix[i].append(val)
		return tmatrix
		
	def divideSet(self, samples, varIndex, value):
		"""
		Takes a 2d nested list of samples, input variable index, and value and divides this group of samples on that variable's value. 
		
		Samples with that value are placed into the True group and samples with all other values are placed in the False group.
		
		Can handle numerical or categorical input variable values.
		"""
		#determines whether a sample is in True or False group
		def splitFunction(sample, varIndex, value):
			if isinstance(value, int) or isinstance(value, float):
				if sample[varIndex] >= value:
					return True
					
				else:
					return False
					
			else:
				if sample[varIndex] == value:
					return True
					
				else:
					return False
		#divide the samples into a True group and False group
		groupTrue = []
		groupFalse = []
		for s in samples:
			if splitFunction(s, varIndex, value):
				groupTrue.append(s)
			else:
				groupFalse.append(s)
		#return them as a tuple holding two 2d nested list matrices
		return (groupTrue, groupFalse)
			
	def outputValueCounts(self, samples):
		"""Takes a 2d nested list as input, counts each sample's output variable value, and returns a dictionary of each unique output variable value along with its count. 
		
		Assumes output variable is last index in each sample.
		"""
		valCounts = {}
		for s in samples:
			#assume target output
			#variable is last index
			value = s[len(s)-1]
			if value not in valCounts:
				valCounts[value] = 0
			valCounts[value] += 1
		return valCounts
		
	def entropy(self, samples):
		"""
		Takes a 2d nested list of samples, extracts a frequency count of the output variable values, them calculates the entropy for this group
of samples as entropy = E pi * log2(pi)   [from i=1 to n]
		"""
		#log2 function
		def log2(x):
			return log(x) / log(2)
			
		#store the frequency counts of the output variable values in this group of samples in a dictionary
		counts = self.outputValueCounts(samples)
		
		#calculate the entropy
		entropy = 0.0
		
		for value in counts:
			p = float(counts[value] / len(samples))
			entropy -= p * log2(p)
			
		return entropy
		
	def variance(self, samples):
		"""
		Extracts the output variable values from a group of samples into a list, assuming they are the last index of each sample, and calculates the variance of these values.
		"""
		if len(samples) == 0:
			return 0
			
		#extract output values, assume last index
		values = []
		for i in range(len(samples)):
			values.append(float(samples[i][len(samples[i]) -1]))
			
		#calculate the mean
		mean = sum(values) / len(values)
		
		#calculate the variance
		varSum = 0.0
		for v in values:
			varSum += (v - mean)**2
		variance = varSum / len(values)
		
		return variance
		
	def train(self, samples, scoreFxn):
		"""
		Learning algorithm for CART which recursively creates the decision tree model from the 2d nested list matrix.
		"""
		#recursion stop condition
		if len(samples) == 0:
			return TreeNode()
			
		#calculate current node's entropy or variance; if root node this is entropy/variance of the entire dataset
		currentScore = scoreFxn(samples)
		
		#variables to track best information gain, best (variable, value) pair, and best sample groups resulting from divide
		bestGain = 0.0
		bestVar = None
		bestGroups = None
		
		#loop through each variable in dataset, excluding last column
		colExclude = len(samples[0]) - 1
		for variable in range(0, colExclude):
			#extract the list of unique values for this variable
			varVals = {}
			for s in samples:
				varVals[s[variable]] =  1
				
			#loop through each possible value of this variable, divide the samples into 2 groups on this value, 
			#calculate the information gain, and store the best gain, (variable, value), and (groupTrue, groupFalse)
			for value in varVals.keys():
				#split the node's samples on this value
				group1, group2 = self.divideSet(samples, variable, value)
				
				#calculate the information gain for this split
				p = float(len(group1)/len(samples))
				gain = currentScore - p * scoreFxn(group1) - (1 - p) * scoreFxn(group2)
				
				#store the best outcomes
				if gain > bestGain and len(group1) > 0 and len(group2) > 0:
					bestGain = gain
					bestVar = (variable, value)
					bestGroups = (group1, group2)
					
		#recursion point: if the entropy/variance can still be lowered, create 2 new branches with train()
		if bestGain > 0:
			trueBranch = self.train(bestGroups[0], scoreFxn)
			falseBranch = self.train(bestGroups[1], scoreFxn)
			return TreeNode(columnId=bestVar[0],value=bestVar[1],trueBranch=trueBranch,falseBranch=falseBranch)
			
		#if the entropy/variance is 0, this is a leaf
		else:
			return TreeNode(results=self.outputValueCounts(samples))
			
	def prune(self, tree, mingain):
		"""
		Prunes the tree by checking pairs of leaf nodes that have a common parent to see if merging them would increase the entropy by less than a specified mingain threshold.
		
		Can handle classification trees or regression trees.
		"""
		#recursively traverse down the tree structure to the branch nodes that only have leaf nodes as children
		if tree.trueBranch.results == None:
			self.prune(tree.trueBranch, mingain)
			
		if tree.falseBranch.results == None:
			self.prune(tree.falseBranch, mingain)
			
		#if both subbranches are now leaves see if they should be merged
		if tree.trueBranch.results != None and tree.falseBranch.results != None:
			tb, fb = [],[]
			
			#build a combined dataset of output values; note this logic works for both categorical and numerical
			#output variables
			for value, count in tree.trueBranch.results.items():
				for i in range(count):
					tb.append([value])
					
			for value, count in tree.falseBranch.results.items():
				for i in range(count):
					fb.append([value])
					
			#test the increase in entropy or variance
			if isinstance(fb[0][0], int) or isinstance(fb[0][0], float):
				delta = self.variance(tb+fb) - ((self.variance(tb) + self.variance(fb)) / 2)
				
			else:
				delta = self.entropy(tb+fb) - ((self.entropy(tb) + self.entropy(fb)) / 2)
			if delta < mingain:
				#merge the leaves
				tree.trueBranch = None
				tree.falseBranch = None
				tree.results = self.outputValueCounts(tb+fb)
				
	def classify(self, sample, tree):
		"""
		Takes a new sample vector , (minus an output variable index) and returns its predicted output value.
		
		Recursively traverses down the decision tree checking its input variable values against each decision node.
		"""
		#recursion endpoint: return the results stored in this leaf node
		if tree.results != None:
			return tree.results
			
		#search the tree top down checking your vector against each decision node
		else:
			value = sample[tree.columnId]
			
			#if data for this variable is missing, follow both the trueBranch and falseBranch
			if value == None:
				trueResults = self.classify(sample, tree=tree.trueBranch)
				falseResults = self.classify(sample, tree.falseBranch)
				
				#frequency counts
				trueCount = sum(trueResults.values())
				falseCount = sum(falseResults.values())
				
				#true and false weight calculations
				tw = float(trueCount / (trueCount + falseCount))
				fw = float(falseCount / (trueCount + falseCount))
				
				#results
				results = {}
				for key, val in trueResults.items():
					results[key] = val * tw
					
				for key, val in falseResults.items():
					if key in results:
						results[key] += val * fw
						
					else:
						results[key] = val * fw
						
				return results
				
			#else, if there is data for this variable in the sample evaluate its value against this decision node, 
			#then call self.classify() on either its trueBranch or falseBranch
			else:
				if isinstance(value, int) or isinstance(value, float):
					if value >= tree.value:
						branch = tree.trueBranch
					else:
						branch = tree.falseBranch
					
				else:
					if value == tree.value:
						branch = tree.trueBranch
					else:
						branch = tree.falseBranch
					
				return self.classify(sample, tree=branch)
				
				
	def getWidth(self, tree):
				"""
				Calculates the total width of a TreeNode recursively, where the total width is the combined widths of its branches, or 1 if it is a leaf.
				"""
				if tree.trueBranch == None and tree.falseBranch == None:
					return 1
					
				return self.getWidth(tree.trueBranch) + self.getWidth(tree.falseBranch)
				
	def getDepth(self, tree):
				"""
				Calculates the total height of a TreeNode recursively, called "depth".
				
				The total depth of a branch is 1 + the total depth of its longest child branch.
				"""
				if tree.trueBranch == None and tree.falseBranch == None:
					return 0
					
				return max(self.getDepth(tree.trueBranch), self.getDepth(tree.falseBranch)) + 1
				
	def drawTree(self, tree, varNames=None, jpeg='tree.jpg'):
				"""
				Draws an image of the decision tree and saves it as a .jpg file.
				
				Calculates total width and height of image, sets up canvas, passes canvas, tree, and list of variable names to self.drawNode(), then saves image.
				"""
				#calculate width and height of image
				w = self.getWidth(tree) * 100
				h = self.getDepth(tree) * 100 + 120
				
				#set up canvas
				img = Image.new('RGB', (w,h), (255, 255, 255))
				draw = ImageDraw.Draw(img)
				
				#call self.drawNode()
				self.drawNode(draw, tree, w/2, 20, varNames)
				
				#save the image
				img.save(jpeg, 'JPEG')
				
	def drawNode(self, draw, tree, x, y, varNames):
				"""
				Main computational function for decision tree visualization.
				
				Recursively draws decision tree from root node down, one TreeNode at a time. Draws true branch on right and false branch on left.
				"""
				#branch TreeNodes
				if tree.results == None:
					#calculate the widths of this node's true and false branches
					w1 = self.getWidth(tree.falseBranch) * 100
					w2 = self.getWidth(tree.trueBranch) * 100
					
					#determine total width required by node
					left = x - (w1 + w2) / 2
					right = x + (w1 + w2) / 2
					
					#draw the variable and value decision condition string for thisTreeNode
					if isinstance(tree.value, int) or isinstance(tree.value, float):
						if varNames != None:
							draw.text((x-20, y-10), varNames[tree.columnId] + ' > ' + str(tree.value), (0,0,0))
							
						else:
							draw.text((x - 20, y - 10), 'Variable ' + str(tree.columnId) + ' > ' + str(tree.value), (0,0,0))
							
					else:
						if varNames != None:
							draw.text((x-20, y-10), varNames[tree.columnId] + ' = ' + str(tree.value), (0,0,0))
							
						else:
							draw.text((x-20, y-10), 'Variable ' + str(tree.columnId) + ' = ' + str(tree.value), (0,0,0))
							
					#draw links to the branches
					draw.line((x, y, left + w1/2, y + 100), fill=(0,0,0))
					draw.line((x, y, right - w2/2, y + 100), fill=(0,0,0))
					
					#recursion point: draw true and false nodes
					self.drawNode(draw, tree.falseBranch, left + w1/2, y + 100, varNames)
					self.drawNode(draw, tree.trueBranch, right - w2/2, y + 100, varNames)
				else:
					text = ''
					for value, count in tree.results.items():
						text += '%s:	%d\n' % (str(value), count)
						
					draw.text((x - 20, y), text, (0,0,0))


#Test option executed on run.
def main():
	test = input("Would you like to create a test instance of the DecisionTree class? (y/n) ")
	if test == 'y' or test == 'Y':
		test_dt()
	else:
		print("Enjoy the code! If you'd like to run a quick test like this, just re-run the script.")
		
def test_dt():
	import numpy as np

	l = []
	for i in range(100):
		l.append([])
		for j in range(5):
			l[i].append(np.random.randint(1,10))

	n = DecisionTree(l)
	n.drawTree(n.tree)
	print("You can open the file named 'tree.jpg' to view the instantiated DecisionTree's structure.")
	print(type(n))
main()
