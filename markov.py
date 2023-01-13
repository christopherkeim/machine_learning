#!/usr/bin/env python3

import random

class MarkovChain:
    """Markov Chain simulator"""
    
    def __init__(self, states):
        """
        Initialize each instance with a list of states for the system.
        """
        self.states = states
        
    def setValues(self, initialStatesVec, transitionProbsMatrix):
        """
        Takes a vector and a 2d matrix.
        """
        self.stateValues = dict(zip(states, initialStatesVec))
        self.transitionProbabilities = {}
        for i in range(len(states)):
            self.transitionProbabilities[states[i]] = dict(zip(states, transitionProbsMatrix[i]))
            
    def simulate(self, t=1):
        """
        Simulation driver.
        """
        #Main loop iterates a full simulation t times, where t is usually a unit of time (secs, mins, hrs, months, years).
        for m in range(t):
            
            #Extract the current number of nodes in each state for this iteration's calculations.
            currentPopulations = self.stateValues.values()
            
            #Calculations are run for each state of the system.
            for x in range(len(states)):
                
                #Pull the current state population, for this state.
                nodes = currentPopulations
                
                #Run probability calculations for each person/node currently in this state, updating the
                #self.stateValues instance dictionary as you go
                for i in range(nodes):
                    
                    #Test the probability for transitioning into a new state, for each state, in a random order
                    #each time, until you get a 'hit' for a transition (each node must transition because Ep=1)
                    hit = False
                    while hit == False:
                        choices = self.randOrder(states)
                        for c in choices:
                            if random.random() < self.transitionProbabilities[states[x]][c]:
                                if c == states[x]:
                                    hit = True
                                    break
                                else:
                                    self.stateValues[states[x]] -= 1
                                    self.stateValues[c] += 1 
                                    hit = True
                                    break
       
        #Return the stateValues dictionary.
        return self.stateValues
                
        
    def randOrder(self, list):
        """
        Utility function.
        """
        cList = list[:]
        rOrder = []
        for i in range(len(list)):
            choice = round(random.randint(0, len(list)-(-i))
            rOrder[i] = cList[choice]
            del cList[choice]
        return rOrder
                           
                
       
