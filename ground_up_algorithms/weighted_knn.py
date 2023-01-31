#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt

import math
import random

from optimization import Optimize

class WeightedKnn:
    """
    The WeightedKnn class.
    """
    def __init__(self, samples):
        """
        The WeightedKnn class takes a 2d nested list matrix as on initialization, transforms it into the object list data structure,
        runs optimization and scales the data in order to build the weighted kNN model dataset which is used to make predictions.
        """
        self.data = self.matrix_to_objlist(samples)
        self.scales = self.optimize_scales()
        #Overwrite the data instance variable with scaled weighted kNN model dataset
        scaled_data = self.rescale_data(self.data, self.scales)
        self.data = scaled_data
        
    def matrix_to_objlist(self, samples):
        """
        Utility function that transforms a 2d nested list into the object list data structure used by this algorithm. 
        """
        obj_list = []
        for s in samples:
            input_vals = s[0:len(s)-1]
            entry = {'input': input_vals, 'output': s[len(s)-1]}
            obj_list.append(entry)
        return obj_list
    
    def euclidean(self, sample1, sample2):
        """
        This is the distance metric used in our weighted kNN calculations, where d = ((Δx)**2 + Δy**2 + Δz**2 + ...))**0.5.
        """
        sum_sq = 0.0
        for i in range(len(sample1)):
            sum_sq += (sample1[i] - sample2[i]) ** 2
        distance = sum_sq ** 0.5
        return distance
    
    def gaussian_weight(self, distance_score, sigma=10.0):
        """
        Takes a Euclidean distance value as an input and returns its Gaussian or bell curve weight value.
        """
        return math.e ** ((-distance_score ** 2) / (2 * sigma ** 2))
    
    def get_distances(self, target_sample):
        """
        Calculates the Euclidean distances between an input target sample and every sample in the dataset.
        
        Stores them in a list of (distance, index) tuples and returns this list sorted from least to greatest.
        """
        distance_list = []
        for i in range(len(self.data)):
            comp_sample = self.data[i]['input']
            dist = (self.euclidean(target_sample, comp_sample), i)
            distance_list.append(dist)
        distance_list.sort()
        return distance_list
    
    def predict(self, target_sample, data, k=5):
        """
        The prediction method for WeightedKnn.
        
        Takes a target sample, dataset, and k-value as inputs and calculates a weighted kNN output value prediction for this target sample.
        """
        #Calculate the Euclidean distances between this target sample and every sample in the model's dataset. 
        d_list = self.get_distances(target_sample)
        #Initialize avg and total_weight variables
        avg = 0.0
        total_weight = 0.0
        #Calculate the weighted average of the output values for our sample's k-nearest neighbors
        for i in range(k):
            distance = d_list[i][0]
            index = d_list[i][1]
            weight = self.gaussian_weight(distance)
            avg += weight * data[index]['output']
            total_weight += weight
        pred = avg / total_weight
        return pred
    
    def divide_data(self, data, test=0.05):
        """
        Divides a dataset into a training set and a test set, given a ratio that you specify, and returns them as a (train_set, test_set) tuple.
        """
        train_set, test_set = [], []
        for sample in data:
            if random.random() < test:
                test_set.append(sample)
                
            else:
                train_set.append(sample)
                
        return train_set, test_set
    
    def test_algorithm(self, alg_f, train_set, test_set):
        """
        Runs a single cross-validation test on a passed in algorithm function, training set, and test set.
        
        Returns its average error = Σ (prediciton - actual) ** 2 / n test samples
        """
        error = 0.0
        for sample in test_set:
            pred = alg_f(sample['input'], data=train_set)
            error += (pred - sample['output']) ** 2
        mean_error = error / len(test_set)
        return mean_error
    
    def cross_validate(self, alg_f, data, trials=100, test=0.05):
        """
        Runs cross-validation a specified number of trials with a specified test:train ratio, adding all the results to get a grand average error value. 
        """
        error = 0.0
        #Main loop
        for i in range(trials):
            train_set, test_set = self.divide_data(data, test)
            error += self.test_algorithm(alg_f, train_set, test_set)
        mean_error = error / trials
        return mean_error
    
    def rescale_data(self, data, scales):
        """
        Takes an object list structured dataset and a list of scaling constants as inputs and returns a new dataset with every sample's input variable values
        scaled by these constants.
        """
        scaled_data = []
        for sample in data:
            scaled_vals = []
            for i in range(len(scales)):
                scaled_vals.append(scales[i] * sample['input'][i])
            scaled_data.append({'input': scaled_vals, 'output': sample['output']})
        return scaled_data
    
    def cost_function(self, scales):
        """
        The cost function that will be passed into our optimization function.
        
        Takes a list of scaling constants as input, constructs a scaled dataset with them using self.rescale_data(), then runs cross-validation with
        this dataset and returns its grand average error. 
        """
        sdata = self.rescale_data(self.data, scales)
        return self.cross_validate(alg_f=self.predict, data=sdata, trials=10)
    
    def optimize_scales(self, low=0, high=20, mode='g'):
        """
        Runs either genetic optimization or annealing optimization, using cross-validation as a cost function, and returns the best set of scaling constants
        in a list.
        """
        weight_domain = [(low, high)] * len(self.data[0]['input'])
        if mode == 'g':
            return Optimize.genetic_optimize(weight_domain, self.cost_function, pop_size=5, maxiter=20)
        
        elif mode == 'a':
            return Optimize.annealing_optimize(weight_domain, self.cost_function)
        
        else:
            print("mode parameter can only be 'g' for genetic optimization or 'a' for annealing optmization. mode set to 'g' as default.")
            return Optimize.genetic_optimize(weight_domain, self.cost_function, pop_size=5, maxiter=20)
        
    def probability(self, target_sample, min, max, k=5):
        """
        Takes a target sample, min output value, max output value, and k-value as inputs and calculates the probability 
        that the target sample's output value falls within this range.
        
        p = Σ Wi kNN within range / Σ Wi kNN total samples
        """
        #Get the Euclidean distances between the target sample and every other sample in the dataset.
        dlist = self.get_distances(target_sample)
        #Distance weight sums of kNN within range and kNN total samples
        nweight = 0.0
        tweight = 0.0
        #Run the probability calculation for this sample's k-nearest neighbors.
        for i in range(k):
            dist = dlist[i][0]
            index = dlist[i][1]
            weight = self.gaussian_weight(dist)
            value = self.data[index]['output']
            #Is this out value within our range?
            if value > min and value <= max:
                nweight += weight
            tweight += weight
        if tweight == 0:
            return 0
        #Probability is kNN weights in range divided by all kNN weights.
        return nweight / tweight
    
    def probability_graph(self, target_sample, max, k=5, ss=5.0):
        """
        Takes a target sample, maximum output value, k-value, and an ss parameter specifying the extent of smoothing as inputs. 
        
        Graphs the probability density function for this sample's output variable values. 
        """
        #Slice the output variable's value range into 0.1 length intervals and store these in an array using NumPy's arange function. 
        x = np.arange(0.0, max, 0.1)
        #Calculate the probability for every interval in this range
        probs = []
        for val in x:
            probs.append(self.probability(target_sample, val, val+0.1, k))
        #Create the smoothed probability density curve.
        smoothed_probs = []
        for i in range(len(probs)):
            sp = 0.0
            for j in range(0, len(probs)):
                dist = abs(i - j) * 0.1
                weight = self.gaussian_weight(dist, sigma=ss)
                sp += weight * probs[j]
            smoothed_probs.append(sp)
        smoothed_probs = np.array(smoothed_probs)
        #Plot and show the graph.
        plt.plot(x, smoothed_probs)
        plt.title('Probability Density Graph')
        plt.xlabel('Output Variable Value')
        plt.ylabel('Probability')
        plt.show()
        
        
