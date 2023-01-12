#!/usr/bin/env python3.11

from math import tanh
from pysqlite2 import dbapi2 as sqlite

class NeuralNetwork:
	"""Docstring comment"""
	
	#initialize the class with the name
	#of the database file which will
	#store the neural net
	def __init__(self, dbname):
		self.con = sqlite.connect(dbname)
		
	def __del__(self):
		"""Close method for SQLite db"""
		
		self.con.close()
		
	def dtanh(self, y):
		"""the derivative of the tanh(x) function 
		
		used to calculate each
	neuron's activation
	"""
		return 1.0 - y*y
		
	def makeTables(self):
		"""creates SQLite tables used to 
	  store our hidden neuron IDs and
	  the synapses btwn input neurons
	  - hidden neurons and hidden 
	  neurons - output neurons
	  """
	
		self.con.execute("create table hidNeuronList (create_key)")
		self.con.execute("create table synInputHidden (fromId, toId, strength)")
		self.con.execute("create table synHiddenOutput (fromId, toId, strength)")
		self.con.commit()
		
	
	def getStrength(self, fromId, toId, synLayer):
		"""accesses SQLite db and
	  retrieves the current weight 
	  value of given synapse;
	  if there is no synapse for 2 
	  neurons passed in it returns 
	  -0.2 for synLayer = 0 and 
	  0 for synLayer = 1
	  """	
		
		#synLayer 0 is the synInput
		#Hidden table
		if synLayer == 0:
			table = 'synInputHidden'
			
		#synLayer1 is the synHidden
		#Output table
		else:
			table = 'synHiddenOutput'
			
		#extract the synapse weight
		result = self.con.execute("select strength from %s where fromId=%d and toId=%d" % (table, fromId, toId)).fetchone()
		
		#if the synapse doesn't exist
		# yet return default values 
		#set for that layer
		if  result == None:
			
			if synLayer == 0:
				return -0.2
				
			if synLayer == 1:
				return 0
				
		#return the synapse weight
		return result[0]
		
	def setStrength(self, fromId, toId, synLayer, strength):
		"""Sets strength for given synapse in SQLite database.
		
		Accesses the SQLite db and determines whether a synapse already exists - if it does it updates its weight value, if it does not it creates the synapse with the input weight value
		"""
		
		if synLayer == 0:
			table = 'synInputHidden'
			
		#synLayer1 is the synHiddenOut
		#put table
		else:
			table = 'synHiddenOutput'
			
		#retrieve the synapse if it exists
		#in the SQLite db
		result = self.con.execute("select rowid from %s where fromId=%d and toId =%d" % (table,fromId,toId)).fetchone()
		
		#if this synapse does not exist
		#create it 
		if result == None:
			self.con.execute("insert into %s (fromId, toId, strength) values (%d, %d, %f)" % (table, fromId, toId, strength))
			
		#otherwise update synapse 
		#with new weight value
		else:
			rowid = result[0]
			self.con.execute("update %s set strength=%f where rowid=%d" % (table, strength, rowid))
			
	def generateHiddenNode(self, inputIds, outputIds):
		"""
		Creates a new hidden neuron
		
		With default weighted synapses to each input neuron and each output neuron, every time its passed a combination of inpit variables it has not seen before
		"""
		
		#limit hidden neuron size to
		#3 input vars or less
		if len(inputIds) > 3:
			return None
				
		#the create_key for each
		#hidden neuron is each 
		#inputId sorted least to 
		#greatest then joined with a
		#"_"
		strWordIds = []
		for i in range(len(inputIds)):
			strWordIds[i] = str(inputIds[i])
				
		create_key = "_".join(strWordIds.sort())
			
		#check to see if there is 
		#already a hidden neuron for
		#combination
		result = self.con.execute("select rowid from hidNeuronList where create_key='%s'" % (create_key)).fetchone()
			
		#if we haven't, create the 
		#new hidden neuron and its
		#synapses 
		if result == None:
			cur = self.con.execute("insert into hidNeuronList (create_key) values ('%s')" % (create_key))
				
			hiddenId = cur.lastrowid
				
			#create synapses between
			#each input neuron and
			#the hidden neuron in 
			#synInputHidden, with 
			#default weights of 
			#strength = 1.0
			#/len(inputIds)
			for inputId in inputIds:
				self.setStrength(inputId, hiddenId, 0, 1.0/len(inputIds))
				
			#create synapses between
			#the hidden neuron and 
			#each outputNeuron in 
			#the synHiddenOutput 
			#table with default weights
			#of strength = 0.1
			for outputId in outputIds:
				self.setStrength(hiddenId, urlId, 1, 0.1)
					
			#Save the new hidden
			#neuron with its Ni + No
			#new synapses to the
			#SQLite db
			self.con.commit()
				
	def getAllHiddenIds(self, inputIds, outputIds):
		"""
		Finds and returns a list of all the hidden neurons that are relevant to a specific input variable query.
    
		The criteria is that the hidden neuron must be connected to one of the inpit variables or to one of the output variables. 
		"""
		hiddenIds = {}
				
		#iterate through every
		#inputId in the input 
		#variables and find every
		#hidden neuron connected
		#to it
		for inputId in inputIds:
			cur = self.con.execute("select toId from synInputHidden where fromId=%d" % (inputId))
			for tupRow in cur:
				hiddenIds[tupRow[0]] = 1
						
		#iterate through each
		#outputId for this query 
		#and find every hidden
		#neuron connected to it
		for outputId in outputIds:
      cur = self.con.execute("select fromId from synHiddenOutput where toId=%d" % (outputId))
			for tupRow in cur:
				hiddenIds[tupRow[0]] = 1
					
		#return a list of unique hidden
		#Ids
		return hiddenIds.keys()
			
	def setUpNetwork(self, inputIds, outputIds):
		"""
		Takes the list of inputIds and outputIds and constructs the relevant subnetwork for this query.
		"""
		#input neuron list, hidden
		#neuron list, and output
		#neuron list
		self.inputIds = inputIds
		self.hiddenIds = hiddenIds
		self.outputIds = outputIds
					
		#initialize activation
		#value lists for input, 
		#hidden, output layers
		#with placeholder 
		#values of 1.0
		self.outputsI = [1.0] * len(self.inputIds)
		self.outputsH = [1.0] * len(self.hiddenIds)
		self.outputsO = [1.0] * len(self.outputIds)
					
		#extract relevant 
		#synapse weight values
		# each input neuron 
		#and hidden neuron 
		#from the 
		#synInputHidden table
		#and store them in a 
		#input x hidden neuron
		#matrix
		self.synInWeights = []
		for i in range(len(self.inputIds)):
			self.synInWeights[i] = []
			for j in range(len(self.hiddenIds)):
				self.synInWeights[i][j] = self.getStrength(self.inputIds[i], self.hiddenIds[j], 0)
					
		#extract the relevant 
		#synapse weight values
		#between each hidden
		#neuron and output
		#neuron from the 
		#synHiddenOutput 
		#table and store them
		#in a hidden neuron x
		#output neuron matrix
		self.synOutWeights =[]
		for i in range(len(self.hiddenIds)):
			self.synOutWeights[i] = []
			for j in range(len(self.outputIds)):
				self.synOutWeights[i][j] = self.getStrength(self.hiddrnIds[i], self.outputIds[j], 1)
		
	def feedForward(self):
    """Feedforward algorithm.
				
		Takes a list of input variables and pushes them through the instantiated subnetwork, calculating and saving the outputs of every neuron in the network in the instance variables and returning the self.outputsO list
		"""
		#input neuron activations
		for i in range(len(self.inputIds)):
			self.outputsI[i] = 1.0
					
		#hidden neuron activations
		#: each hidden neuron has
		#Ni = len(self.inputIds) 
		#inputs, and its activation
		#output is the sum of 
		#each input neuron's
		#output multiplied by 
		#its synapse weight, 
		#passed into the tanh(x) 
		#function
		for i in range(len(self.hiddenIds)):
			sum =0.0
			for j in range(len(self.inputIds)):
				sum += self.outputsI[j] * self.synInWeights[j][i]
						
			self.outputsH[i] = tanh(sum)
				
		#output neuron activations
		#: each output neuron has
		#Nh = len(self.hiddenIds)
		# inputs, and its activat
		#ion output is calculated
		#by taking the sum of each
		#hidden neuron's output
		#multiplied by its synapse 
		#weight, passed into the
		#tanh(x) function
		for i in range(len(self.outputIds)):
			sum = 0.0
			for j in range(len(self.hiddenIds)):
				sum += self.outputsH[j] * self.synOutWeights[j][i]
			self.outputsO[i] = tanh(sum)
					
		#return the output list
		#vector for the output
		#neurons
		return self.outputsO[:]
				
	def getResult(self, inputIds, outputIds):
    """
		Instantiates a subnetwork and runs feedForward() for a given query of input variables
		"""
		self.setUpNetwork(inputIds, outputIds)
		return self.feedForward()
				
	def backPropagate(self, targets, learningRate=0.5):
    """
		Backpropagation algorithm.
		
		Minimal comments rn.
		"""
		#error for each output 
		#neuron
		outputDeltas = [0.0] * len(self.outputIds)
				
		for i in range(len(self.outputIds)):
			outputError = targets[i] - self.outputsO[i]
			outputDeltas[i] = self.dtanh(self.outputsO[i]) * outputError
				
      #error for each hidden
			#neuron
				
			hiddenDeltas = [0.0] * len(self.hiddenIds)
				
			for i in range(len(self.hiddenIds)):
			  hiddenError = 0.0
					
				for j in range(len(self.outputIds)):
					hiddenError += self.synOutWeights[i][j] * outputDeltas[j]
						
				hiddenDeltas[i] = self.dtanh(self.outputsH[i]) * hiddenError
					
			#update the
			#self.synOutWeights 
			#synapse weights matrix
			for i in range(len(self.hiddenIds)):
				for j in range(len(self.outputIds)):
          change = self.outputsH[i] * outputDeltas[j]
						
					self.synOutWeights[i][j] += change * learningRate
						
			#update the self.synIn
			#Weights synapse matric
			for i in range(len(self.inputIds)):
				for j in range(len(self.hiddenIds)):
				  change = self.outputsI[i] * hiddenDeltas[j]
						
					self.synInWeights[i][j] += change * learningRate
						
	def trainQuery(self, inputIds, outputIds, selectedOutput):
    """
		Runs a full training session.
				
		For a given vector of input variables.
    """
		#generate new hidden
		#neuron if needed
		self.generateHiddenNode(inputIds, outputIds)
				
		#instantiate the relevant
		#subnetwork
		self.setUpNetwork(inputIds, outputIds)
				
		#feed forward algorithm
		#saves outputs in 
		#instance variables
		self.feedForward()
				
		#generate targets list of
		#correct outputs for each
		#output variable - 0's for
		#each nonselected output
		#and 1 for selected output
		targets = [0.0] * len(outputIds)
		targets[outputIds.index(selectedOutput)] = 1
				
		#back propagation to train
		#neurons in subnetwork
		self.backPropagate(targets)
				
		#save new adjusted
		#synapse weights held in
		#instance variables to db
		self.updateDatabase()
				
	def updateDatabase(self):
    """
		Saves data stored in instance variables into the SQLite database.
				
		More doc to come.
		"""
		#push new/adjusted
		#synapse weights held in
		#self.synInWeights matrix
		#to the synInputHidden
		#table
		for i in range(len(self.inputIds)):
			for j in range(len(self.hiddenIds)):
				self.setStrength(self.inputIds[i], self.hiddenIds[j], 0, self.synInWeights[i][j])
						
		#push new/adjusted
		#synapse weights in 
		#self.synOutWeights 
		#matrix into the synHidden
		#Output table
		for i in range(len(self.hiddenIds)):
			for j in range(len(self.outputIds)):
				self.setStrength(self.hiddenIds[i], self.outputIds[j], 1, self.synOutWeights[i][j])
						
		#save the database
		self.con.commit()
