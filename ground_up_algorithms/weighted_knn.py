#!/usr/bin/env python3
import numpy as np
import matplotlib.pylot as plt

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
    
    def predict(self, target_sample, k=5, data=self.data, weight_f=self.gaussian):
        """
        The prediction method for WeightedKnn.
        
        Takes a target sample and k-value as inputs and calculates a weighted kNN output value prediction for this target sample.
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
            weight = weight_f(distance)
            avg += weight * data[index]['output']
            total_weight =+ weight
        pred = avg / total_weight
        return pred
    
    def divide_data(self, data=self.data, test=0.05):
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
            pred = alg_f(data=train_set, sample['input'])
            error += (pred - sample['result']) ** 2 
        mean_error = error / len(test_set)
        return mean_error
    
    def cross_validate(self, alg_f=self.predict, data=self.data, trials=100, test=0.05):
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
    
    def rescale_data(self, data, samples):
        """
        
        
