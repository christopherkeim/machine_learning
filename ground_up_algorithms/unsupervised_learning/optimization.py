#!/usr/bin/env python3.11

import math
import random

class Optimize:
	"""
	The Optimize class holds two static methods: genetic_optimize and annealing_optimize.
	"""
	
	@staticmethod
	def genetic_optimize(domain, cost_function, pop_size=50, step=1, mut_probability=0.2, elitism=0.2, maxiter=100):
		"""
		Genetic optimization randomly generates an initial 'population' of solutions of size popSize, then in each iteration scores each solution with the cost function, takes the top 20% (percentage elitism) of solutions and creates the next generation by randomly mutating and breeding them.
		
		The process is repeated until the maxiter number of generations is reached.
		"""
		#build initial random population of solutions
		population = []
		for i in range(pop_size):
			sol = []
			for j in range(len(domain)):
				sol.append(random.randint(domain[j][0], domain[j][1]))
			population.append(sol)
		
		#calculate the number of 'winners' from each generation
		top_elite = int(elitism*pop_size)
	
		#Mutation operation
		def mutate(sol):
			"""Mutation operation."""
			index = random.randint(0, len(domain) - 1)
			if random.random() < 0.5 and sol[index] > domain[index][0]:
				return sol[0:index] + [sol[index] - step] + sol[index + 1:]
			
			elif sol[index] < domain[index][1]:
				return sol[0:index] + [sol[index] + step] + sol[index + 1:]
			
		def crossover(sol1, sol2):
			"""Crossover operation"""
			index = random.randint(1, len(domain) -2)
			return sol1[0:index] + sol2[index:]
		
		#main loop
		for i in range(maxiter):
			gen_scores = []
			for v in population:
				gen_scores.append((cost_function(v), v))
			gen_scores.sort()
		
			#now take the ranked list for operations
			ranked = []
			for (s, v) in gen_scores:
				ranked.append(v)
			
			#overwrite population, starting with pure elites 
			population = ranked[0:top_elite]
		
			#mutate and breed elites
			while len(population) < pop_size:
				#mutation
				if random.random() < mut_probability:
					c = random.randint(0, top_elite)
					population.append(mutate(ranked[c]))
				
				#crossover
				else:
					c1 = random.randint(0, top_elite)
					c2 = random.randint(0, top_elite)
					population.append(crossover(ranked[c1], ranked[c2]))
				
			#print current best solution's score
			print(gen_scores[0][0])
		
		#return the best solution at end of final generation
		return gen_scores[0][1]
	
	@staticmethod
	def annealingOptimize(domain, cost_function, T=10000, cool=0.95, step=1):
		"""
		Simulated annealijg optimization first generates a random solution, and in each iteration randomly chooses one number in the solution and changes it in a random direction.
	
		If the new cost is lower, the new solution becomes the current solution. But it the cost is higher it can still become the current solution with a certain probability. Process ends when temperature ≈ 0.
		"""
		#create a random solution
		sol = []
		for i in range(len(domain)):
			sol.append(random.randint(domain[i][0], domain[i][1]))
		
		#main loop
		while T > 0.1:
			#choose a random index in the solution
			rand_index = random.randint(0, len(domain) - 1)
		
			#randomly choose a direction to change it
			change = step * (-1)**int(round(random.random()))
		
			#create a new solution with one element changed
			solb = sol[:]
			solb[rand_index] += change
			if solb[rand_index] < domain[rand_index][0]:
				solb[rand_index] = domain[rand_index][0]
			elif solb[rand_index] > domain[rand_index][1]:
				solb[rand_index] = domain[rand_index][1]
			
			#calculate current solution cost and new solution cost
			sol_cost = cost_function(sol)
			solb_cost = cost_function(solb)
		
			#if solb has a lower cost, or if it makes the probability cutoff,
			# make it the current solution
			if solb_cost < sol_cost:
				sol = solb
			
			else:
				probability = pow(math.e, (-solb_cost - sol_cost)/T)
				if random.random() < probability:
					sol = solb
			#decrease the temperature T
			T = T * cool
		
		return sol

