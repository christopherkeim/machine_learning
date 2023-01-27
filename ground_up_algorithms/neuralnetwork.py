#!/usr/bin/env python3

from math import tanh
from pysqlite2 import dbapi2 as sqlite

class NeuralNetwork:
    """
    A lightweight multilayered perceptron neural network implementation.
    """
    
    def __init__(self, dbname):
        """
        Initialize the class with the name of the database file which will store the neural net.
        """
        self.con = sqlite.connect(dbname)
        
    def __del__(self):
        """
        Close method for our SQLite database.
        """
        self.con.close()
        
    def dtanh(self, y):
        """
        The derivative of the tanh(x) function.
        
        Used to calculate each neuron's activation.
        """
        return 1.0 -y*y
        
    def makeTables(self):
        """
        Creates the SQLite tables that will store our hidden neuron IDs, the synapes between input
        neurons -- hidden neurons, and the synapses between hidden neurons -- output neurons.
        """
        self.con.execute("create table hidNeuronList (create_key)")
		self.con.execute("create table synInputHidden (fromId, toId, strength)")
		self.con.execute("create table synHiddenOutput (fromId, toId, strength)")
		self.con.commit()
		
	def getStrength(self, fromId, toId, synLayer):
	    """
	    Accesses the SQLite database and retrieves the current weight value of the given
	    synapse. 
	    
	    If there is no synapse for 2 neurons passed in it returns -0.2 for synLayer = 0 and
	    0 for synLayer = 1.
	    """
	    #synLayer 0 is the synInputHidden table
	    if synLayer == 0:
	        table = 'synInputHidden'
	        
	    #synlayer 1 is the synHiddenOutput table
	    else:
	        table = 'synHiddenOutput'
            
        #Extract the synapse weight.
        result = self.con.execute("select strength from %s where fromId=%d and toId=%d" % (table, fromId, toId)).fetchone()
        
        #If the synapse does not exist yet return default values for that layer
        if result == None:
            
            #The default initial weight value for synInputHidden is -0.2
            if synLayer == 0:
                return -0.2
            
            #The default initial weight value for synHiddenOutput is 0
            if synLayer == 1:
                return 0 
            
        #Return the synapse weight
        return result[0]
    
    def setStrength(self, fromId, toId, synLayer, strength):
        """
        Sets the weight value for a given syanpse in the SQLite databse.
        
        Accesses the database and determines if a given synapse already exists. If it does, it updates its weight value. If not it creates
        the synapse with the input weight value.
        """
        if synLayer == 0:
            table = 'synInputHidden"
            
        else:
            table = 'synHiddenOutput'
            
        #Retrieve the synapse if it exists.
        result = self.con.execute("select rowid from %s where fromId=%d and toId=%d" % (table,fromId,toId)).fetchone()
        
        #If the synpase does not exist then create it.
        if result == None:
            self.con.execute("insert into %s (fromId, toId, strength) values (%d, %d, %f)" % (table, fromId, toId, strength))
            
        #Otherwise, update the synpase with its new weight value. 
        else:
            rowid = result[0]
            self.con.execute("update %s set strength=%f where rowid=%d" % (table, strength, rowid))
            
    def generateHiddenNode(self, inputIds, outputIds):
        """
        Creates a new hidden neuron in the SQLite database.
        
        Everytime it is passed a set of input variables it has never seen before it creates default weighted synapses between input, hidden,
        and output neuron.
        """
        #Limit hidden neuron size to 3 input variables or less.
        if len(inputIds) > 3:
            return None
        
        #the create_key for each hidden neuron is each inputId sorted least to greatest then joined with a "_"
        strInputIds = []
        for i in range(len(inputIds)):
            strInputIds[i] = str(inputIds[i])
            
        create_key = "_".join(strInputIds.sort())
        
        #Check to see if there already a hidden neuron for this this combination
        result = self.con.execute("select rowid from hidNeuronList where create_key='%s'" % (create_key)).fetchone()
        
        #If there isn't, create the new hidden neuron and its synapses. 
        if result == None:
            cur = self.con.execute("insert into hidNeuronList (create_key) values ('%s')" % (create_key))
            hiddenId = cur.lastrowid
            
            #Create synapses between each input neuron and the hidden neuron in the synInputHidden table with default weights of
            #strength = 1.0 / len(inputIds)
            for inputId in inputIds:
                self.setStrength(inputId, hiddenId, 0, 1.0/len(inputIds))
                
            #Create synapses between the hidden neuron and each output neuron in the synHiddenOutput table with default weights
            #of strength = 0.1
            for outputId in outputIds:
                self.setStrenght(hiddenId, outputId, 0.1)
                
            #Save the new hidden neuron with its Ni + No new synapses in the SQLite database. 
            self.con.commit()
            
    def getAllHiddenIds(self, inputIds, outputIds):
        """
        Finds and returns a list of all of the hidden neurons that are relevant to a given input variable query.
        
        The criteria is that the hidden neuron must be connected to one of the input variables or to one of the output variables.
        """
        hiddenIds = {}
        
        #Iterate through every inputId in the input variables and find every hidden neuron connected to each one.
        for inputId in inputIds:
            cur = self.con.execute("select toId from synInputHidden where fromId=%d" % (inputId))
            for tupRow in cur:
                hiddenIds[tupRow[0]] = 1
                
        #Iterate through ever outputId for this query and find every hidden neuron connected to each one. 
        for outputId in outputIds:
            cur = self.con.execute("select fromId from synHiddenOutput where toId=%d" % (outputId))
            for tupRow in cur:
                hiddenIds[tupRow[0]] = 1
                
        #Return a list of unique hiddenIds
        return hiddenIds.keys()
    
    def setUpNetwork(self, inputIds, outputIds):
        """
        Takes the list of inputIds and outputIds and constructs the relevant subnetwork for this query.
        """
        
        #Input neuron list, hidden neuron list, and output neuron list instance variables. 
        self.inputIds = inputIds
		self.hiddenIds = hiddenIds
		self.outputIds = outputIds
        
        #Initialize activation values for input, hidden, and output layers with placeholder values of 1.0.
        self.outputsI = [1.0] * len(self.inputIds)
		self.outputsH = [1.0] * len(self.hiddenIds)
		self.outputsO = [1.0] * len(self.outputIds)
        
        #Extract the relevant synapse weight values for each input neuron and hidden neuron from the synInput
        #Hidden table and store them in an input neuron x hidden neuron matrix. 
        self.synInWeights = []
        for i in range(len(self.inputIds)):
            self.synInWeights.append([])
            for j in range(len(self.hiddenIds)):
                self.synInWeights[i].append(self.getStrenth(self.inputIds[i]. self.hiddenIds[j], 0))
                
        #Extract the relevant synapse weight values for each hidden neuron and output neuron from the synHidden
        #Output table and store them in a hidden neuron x output neuron matrix. 
        self.synOutWeights = []
        for i in range(len(self.hiddenIds)):
            self.synOutWeights.append([])
            for j in range(len(self.outputIds)):
                self.synOutWeights[i].append(self.getStrength(self.hiddenIds[i], self.outputIds[j], 1))
                
    def feedForward(self):
        """
        The feed forward algorgithm.
        
        Takes a list of input variables and pushes them through the instantiated subnetwork, calculating and saving the outputs
        of each neuron in instance variables and returning the self.outputsO list.
        """
        
        #Input neuron activations.
        for i in range(len(self.inputIds)):
            self.outputsI[i] = 1.0
            
        #Hidden neuron activations. Each hidden neuron has Ni = len(self.inputIds) inputs, and its activation
        #output is the sum of each input neuron's output multiplied by its synapse weight, passed into the
        #tanh(x) function. 
        for i in range(len(self.hiddenIds)):
            sum = 0.0
            for j in range(len(self.inputIds)):
                sum += self.outputsI[j] * self.synInWeights[j][i]
                
            self.outputsH[i] = tanh(sum)
            
        #Output neuron activations. Each output neuron has Nh = len(self.hiddenIds) inputs, and its activation
        #output is the sum of each hidden neuron's output multiplied by its synapse weight, passed into the 
        #tanh(x) function.
        for i in range(len(self.outputIds)):
            sum = 0.0
            for j in range(len(self.hiddenIds)):
                sum += self.outputsH[j] * self.synOutWeights[j][i]
                
            self.outputsO[i] = tanh(sum)
            
        #Return the outputsO list of output neuron activations.
        return self.outputsO[:]
    
    def getResult(self, inputIds, outputIds):
        """
        Instantiates a subnetwork and runs feedForward() for a given query of input variables.
        """
        self.setUpNetwork(inputIds, outputIds)
        return self.feedForward()
    
    def backPropogate(self, targets, learningRate=0.5):
        """
        Back propogation learning algorithm.
        
        Minimal comments right now.
        """
        
        #Error for output neurons. 
        outputDeltas = [0.0] * len(self.outputIds)
				
		for i in range(len(self.outputIds)):
		    outputError = targets[i] - self.outputsO[i]
		    outputDeltas[i] = self.dtanh(self.outputsO[i]) * outputError
            
        #Error for each hidden neuron. 
        hiddenDeltas = [0.0] * len(self.hiddenIds)
				
		for i in range(len(self.hiddenIds)):
	        hiddenError = 0.0
					
			for j in range(len(self.outputIds)):
			    hiddenError += self.synOutWeights[i][j] * outputDeltas[j]		
			
            hiddenDeltas[i] = self.dtanh(self.outputsH[i]) * hiddenError
                
        #Update the self.synOutWeights synapse matrix.
        for i in range(len(self.hiddenIds)):
            for j in range(len(self.outputIds)):
                change = self.outputsH[i] * outputDeltas[j]
                #Add this change to the synapse weighted by the learning rate.
                self.synOutWeights[i][j] += change * learningRate
        
        #Update the self.synInWeights synapse matrix.
        for i in range(len(self.inputIds)):
            for j in range(len(self.hiddenIds)):
                change = self.outputIds[i] * hiddenDeltas[j]
                #Add this change to the synapse weighted by the learning rate.
                self.synInWeights[i][j] += change * learningRate
                
    def trainQuery(self, inputIds, outputIds, selectedOutput):
        """
        Runs a full training session for a given input variable query.
        """
        
        #Generate a new hidden neuron if necessary.
        self.generateHiddenNode(inputIds, outputIds)
        
        #Instantiate the relevant subnetwork.
        self.setUpNetwork(inputIds, outputIds)
        
        #Feed forward algorithm saves outputs of each neuron in instance variables.
        self.feedForward()
        
        #Generates target list of correct outputs for each output neuron - 0's for incorrect answers, 
        #and 1 for selected output.
        targets = [0.0] * len(outputIds)
        targets[outpudIds.index(selectedOutput)] = 1
        
        #Back propogation to train neurons in the instantiated subnetwork. 
        self.backPropogate(targets)
        
        #Save new adjusted synapse weights held in instance variables into the SQLite database.
        self.updateDatabase()
        
    def updateDatabase(self):
        """
        Pushes data stored in instance variables into the SQLite databse.
        """
        
        #Update synInputHidden table
        for i in range(len(self.inputIds)):
		    for j in range(len(self.hiddenIds)):
			    self.setStrength(self.inputIds[i], self.hiddenIds[j], 0, self.synInWeights[i][j]) 
        
        #Update synHiddenOutput table
        for i in range(len(self.hiddenIds)):
			for j in range(len(self.outputIds)):
				self.setStrength(self.hiddenIds[i], self.outputIds[j], 1, self.synOutWeights[i][j])
        
       
        
        
                
       
