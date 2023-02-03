#!/usr/bin/env python3

from math import tanh
import sqlite3
from sqlite3 import Error

class NeuralNetwork:
    """
    A lightweight multilayered perceptron neural network implementation.

    """

    def __init__(self, dbname):
        """
        Initialize the class with the name of the database file which will store the neural net.
        """
        self.con = None
        try:
            self.con = sqlite3.connect(dbname)
            print("Connection to SQLite DB successful.")
        except Error as e:
            print(f"The error '{e}' occurred")
        
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
        
    def make_tables(self):
        """
        Creates the SQLite tables that will store our hidden neuron IDs, the synapes between input
        neurons -- hidden neurons, and the synapses between hidden neurons -- output neurons.
        """
        cursor = self.con.cursor()
        try:
            cursor.execute("CREATE TABLE IF NOT EXISTS hidNeuronList (create_key)")
            cursor.execute("CREATE TABLE IF NOT EXISTS synInputHidden (fromId, toId, strength)")
            cursor.execute("CREATE TABLE IF NOT EXISTS synHiddenOutput (fromId, toId, strength)")
            self.con.commit()
            print("Tables hidNeuronList, synInputHidden, and synHiddenOutput successfully created.")
        except Error as e:
            print(f"The error '{e}' occurred")
		
    def get_strength(self, from_id, to_id, syn_layer):
        """
        Accesses the SQLite database and retrieves the current weight value of the given
        synapse. 

        If there is no synapse for 2 neurons passed in it returns -0.2 for syn_layer = 0 and
        0 for syn_layer = 1.
        """
	    #syn_layer 0 is the synInputHidden table
        if syn_layer == 0:
            table = 'synInputHidden'
	        
	    #syn_layer 1 is the synHiddenOutput table
        else:
            table = 'synHiddenOutput'
            
        #Extract the synapse weight.
        cursor = self.con.cursor()
        result = cursor.execute("SELECT strength FROM %s WHERE fromId=%d AND toId=%d" % (table, from_id, to_id)).fetchone()
        
        #If the synapse does not exist yet return default values for that layer
        if result == None:
            
            #The default initial weight value for synInputHidden is -0.2
            if syn_layer == 0:
                return -0.2
            
            #The default initial weight value for synHiddenOutput is 0
            if syn_layer == 1:
                return 0 
            
        #Return the synapse weight
        return result[0]
    
    def set_strength(self, from_id, to_id, syn_layer, strength):
        """
        Sets the weight value for a given syanpse in the SQLite databse.
        
        Accesses the database and determines if a given synapse already exists. If it does, it updates its weight value. If not it creates
        the synapse with the input weight value.
        """
        if syn_layer == 0:
            table = 'synInputHidden'
        else:
            table = 'synHiddenOutput'
            
        #Retrieve the synapse if it exists.
        cursor = self.con.cursor()
        result = cursor.execute("SELECT rowid FROM %s WHERE fromId=%d AND toId=%d" % (table, from_id, to_id)).fetchone()
        
        #If the synpase does not exist then create it.
        if result == None:
            cursor.execute("INSERT into %s (fromId, toId, strength) VALUES (%d, %d, %f)" % (table, from_id, to_id, strength))
            
        #Otherwise, update the synpase with its new weight value. 
        else:
            rowid = result[0]
            cursor.execute("UPDATE %s SET strength=%f WHERE rowid=%d" % (table, strength, rowid))
            
    def generate_hidden_node(self, input_ids, output_ids):
        """
        Creates a new hidden neuron in the SQLite database.
        
        Everytime it is passed a set of input variables it has never seen before it creates default weighted synapses between input, hidden,
        and output neuron.
        """
        #Limit hidden neuron size to 3 input variables or less.
        if len(input_ids) > 3:
            return None
        
        #the create_key for each hidden neuron is each inputId sorted least to greatest then joined with a "_"
        str_input_ids = []
        for i in range(len(input_ids)):
            str_input_ids.append(str(input_ids[i]))
            
        create_key = "_".join(str_input_ids.sort())
        
        #Check to see if there already a hidden neuron for this this combination
        cursor = self.con.cursor()
        result = cursor.execute("SELECT rowid FROM hidNeuronList WHERE create_key='%s'" % (create_key)).fetchone()
        
        #If there isn't, create the new hidden neuron and its synapses. 
        if result == None:
            cur = cursor.execute("INSERT INTO hidNeuronList (create_key) VALUES ('%s')" % (create_key))
            hidden_id = cur.lastrowid
            
            #Create synapses between each input neuron and the hidden neuron in the synInputHidden table with default weights of
            #strength = 1.0 / len(inputIds)
            for input_id in input_ids:
                self.set_strength(input_id, hidden_id, 0, 1.0/len(input_ids))
                
            #Create synapses between the hidden neuron and each output neuron in the synHiddenOutput table with default weights
            #of strength = 0.1
            for output_id in output_ids:
                self.set_strength(hidden_id, output_id, 0.1)
                
            #Save the new hidden neuron with its Ni + No new synapses in the SQLite database. 
            self.con.commit()
            
    def get_all_hidden_ids(self, input_ids, output_ids):
        """
        Finds and returns a list of all of the hidden neurons that are relevant to a given input variable query.
        
        The criteria is that the hidden neuron must be connected to one of the input variables or to one of the output variables.
        """
        hidden_ids = {}
        
        #Iterate through every inputId in the input variables and find every hidden neuron connected to each one.
        cursor = self.con.cursor()
        for input_id in input_ids:
            cur = cursor.execute("SELECT toId FROM synInputHidden WHERE fromId=%d" % (input_id))
            for tup_row in cur:
                hidden_ids[tup_row[0]] = 1
                
        #Iterate through ever outputId for this query and find every hidden neuron connected to each one. 
        for output_id in output_ids:
            cur = cursor.execute("SELECT fromId FROM synHiddenOutput WHERE toId=%d" % (output_id))
            for tup_row in cur:
                hidden_ids[tup_row[0]] = 1
                
        #Return a list of unique hiddenIds
        return list(hidden_ids.keys())
    
    def set_up_network(self, input_ids, output_ids):
        """
        Takes the list of inputIds and outputIds and constructs the relevant subnetwork for this query.
        """
        
        #Input neuron list, hidden neuron list, and output neuron list instance variables. 
        self.input_ids = input_ids
        self.hidden_ids = self.get_all_hidden_ids(input_ids, output_ids)
        self.output_ids = output_ids
        
        #Initialize activation values for input, hidden, and output layers with placeholder values of 1.0.
        self.outputs_i = [1.0] * len(self.input_ids)
        self.outputs_h = [1.0] * len(self.hidden_ids)
        self.outputs_o = [1.0] * len(self.output_ids)
        
        #Extract the relevant synapse weight values for each input neuron and hidden neuron from the synInput
        #Hidden table and store them in an input neuron x hidden neuron matrix. 
        self.syn_in_weights = []
        for i in range(len(self.input_ids)):
            self.syn_in_weights.append([])
            for j in range(len(self.hidden_ids)):
                self.syn_in_weights[i].append(self.get_strenth(self.input_ids[i]. self.hidden_ids[j], 0))
                
        #Extract the relevant synapse weight values for each hidden neuron and output neuron from the synHidden
        #Output table and store them in a hidden neuron x output neuron matrix. 
        self.syn_out_weights = []
        for i in range(len(self.hidden_ids)):
            self.syn_out_weights.append([])
            for j in range(len(self.output_ids)):
                self.syn_out_weights[i].append(self.get_strength(self.hidden_ids[i], self.output_ids[j], 1))
                
    def feed_forward(self):
        """
        The feed forward algorgithm.
        
        Takes a list of input variables and pushes them through the instantiated subnetwork, calculating and saving the outputs
        of each neuron in instance variables and returning the self.outputsO list.
        """
        
        #Input neuron activations.
        for i in range(len(self.input_ids)):
            self.outputs_i[i] = 1.0
            
        #Hidden neuron activations. Each hidden neuron has Ni = len(self.inputIds) inputs, and its activation
        #output is the sum of each input neuron's output multiplied by its synapse weight, passed into the
        #tanh(x) function. 
        for i in range(len(self.hidden_ids)):
            sum = 0.0
            for j in range(len(self.input_ids)):
                sum += self.outputs_i[j] * self.syn_in_weights[j][i]
                
            self.outputs_h[i] = tanh(sum)
            
        #Output neuron activations. Each output neuron has Nh = len(self.hiddenIds) inputs, and its activation
        #output is the sum of each hidden neuron's output multiplied by its synapse weight, passed into the 
        #tanh(x) function.
        for i in range(len(self.output_ids)):
            sum = 0.0
            for j in range(len(self.hidden_ids)):
                sum += self.outputs_h[j] * self.syn_out_weights[j][i]
                
            self.outputs_o[i] = tanh(sum)
            
        #Return the outputsO list of output neuron activations.
        return self.outputs_o[:]
    
    def get_result(self, input_ids, output_ids):
        """
        Instantiates a subnetwork and runs feedForward() for a given query of input variables.
        """
        self.set_up_network(input_ids, output_ids)
        return self.feed_forward()
    
    def back_propogate(self, targets, learning_rate=0.5):
        """
        Back propogation learning algorithm.
        Minimal comments right now.
        """
        #Error for output neurons. 
        output_deltas = [0.0] * len(self.output_ids)
				
        for i in range(len(self.output_ids)):
	        output_error = targets[i] - self.outputs_o[i]
	        output_deltas[i] = self.dtanh(self.outputs_o[i])*output_error
            
        #Error for each hidden neuron. 
        hidden_deltas = [0.0] * len(self.hidden_ids)

        for i in range(len(self.hidden_ids)):
            hidden_error = 0.0

            for j in range(len(self.output_ids)):
                hidden_error += self.syn_out_weights[i][j] * output_deltas[j]		
			
            hidden_deltas[i] = self.dtanh(self.outputs_h[i]) * hidden_error
                
        #Update the self.synOutWeights synapse matrix.
        for i in range(len(self.hidden_ids)):
            for j in range(len(self.output_ids)):
                change = self.outputs_h[i] * output_deltas[j]
                #Add this change to the synapse weighted by the learning rate.
                self.syn_out_weights[i][j] += change * learning_rate
        
        #Update the self.synInWeights synapse matrix.
        for i in range(len(self.input_ids)):
            for j in range(len(self.hidden_ids)):
                change = self.output_ids[i] * hidden_deltas[j]
                #Add this change to the synapse weighted by the learning rate.
                self.syn_in_weights[i][j] += change * learning_rate

    def train_query(self, input_ids, output_ids, selected_output):
        """
        Runs a full training session for a given input variable query.
        """
        
        #Generate a new hidden neuron if necessary.
        self.generate_hidden_node(input_ids, output_ids)
        
        #Instantiate the relevant subnetwork.
        self.set_up_network(input_ids, output_ids)
        
        #Feed forward algorithm saves outputs of each neuron in instance variables.
        self.feed_forward()
        
        #Generates target list of correct outputs for each output neuron - 0's for incorrect answers, 
        #and 1 for selected output.
        targets = [0.0] * len(output_ids)
        targets[output_ids.index(selected_output)] = 1
        
        #Back propogation to train neurons in the instantiated subnetwork. 
        self.back_propogate(targets)
        
        #Save new adjusted synapse weights held in instance variables into the SQLite database.
        self.update_database()
                
    def update_database(self):
        """
        Pushes data stored in instance variables into the SQLite databse.
        """
        
        #Update synInputHidden table
        for i in range(len(self.input_ids)):
            for j in range(len(self.hidden_ids)):
                self.set_strength(self.input_ids[i], self.hidden_ids[j], 0, self.syn_in_weights[i][j]) 
        
        #Update synHiddenOutput table
        for i in range(len(self.hidden_ids)):
            for j in range(len(self.output_ids)):
                self.set_strength(self.hidden_ids[i], self.output_ids[j], 1, self.syn_out_weights[i][j])
        
#Test option executed on run.
def main():
	test = input("Would you like to create a test instance of the NeuralNetwork class? (y/n) ")
	if test == 'y' or test == 'Y':
		db_name = input("Please type any filename for the SQLite database you'd like to use: ")
		test_nn(db_name)
	else:
		print("Enjoy the code! If you'd like to run a quick test like this, just re-run the script.")
		
def test_nn(db_name):
	n = NeuralNetwork(db_name)
	n.make_tables()
	return n
main()
