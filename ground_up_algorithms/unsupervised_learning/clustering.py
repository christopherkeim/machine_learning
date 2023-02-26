#!usr/bin/env python3
import numpy as np

class Cluster:

    @staticmethod
    def kcluster(data, distance_f, k=4):
        """
        The K-means clustering algorithm.
        
        This implementation takes a 2-d nested list matrix dataset, as well as a similarity or distance function, and the number of groups
        k as inputs and outputs a matrix holding each calculated cluster.
        """
        #Determine the minimum and maximum values for each variable in our dataset.
        ranges = [(min([sample[i] for sample in data]), max([sample[i] for sample in data])) for i in range(len(data[0]))]

        #Create k randomly placed centroids.
        clusters = []
        for i in range(k):
            clusters.append([])
            for j in range(len(data[0])):
                clusters[i].append(np.random.random())

        #Initialize the last_matches variable.
        last_matches = None
        
        #Main loop.
        for m in range(100):
            print(f"Iteration '{m}'")
            best_matches = []
            for i in range(k):
                best_matches.append([])
            
            #Compare each data sample with each centroid and determine which centroid it is closest to.
            for i in range(len(data)):
                row = data[i]
                best_match = 0
                for j in range(k):
                    d = distance_f(clusters[j], row)
                    if d < distance_f(clusters[best_match], row):
                        best_match = j
                best_matches[best_match].append(i)

            #If the clusters or groups are the same as last time then our process is complete.
            if best_matches == last_matches:
                break

            #Update last_matches
            last_matches = best_matches

            #Update or move the centroids' data values to the averages of their members
            for i in range(k):
                avgs = [0.0] * len(data[0])
                if len(best_matches[i]) > 0:
                    for row_id in best_matches[i]:
                        for c in range(len(data[row_id])):
                            avgs[c] += data[row_id][c]
                    for j in range(len(avgs)):
                        avgs[j] /= len(best_matches[i])
                    clusters[i] = avgs

        return best_matches

#Test option executed on run.
def main():
	test = input("Would you like to run a test of the K-means clustering function? (y/n) ")
	if test == 'y' or test == 'Y':
		test_kmc()
	else:
		print("Enjoy the code! If you'd like to do a quick test like this, just re-run the script.")
		
def test_kmc():
    data = []
    for i in range(100):
        data.append([])
        for j in range(5):
            data[i].append(np.random.randint(1,10))

    a = Cluster.kcluster(data, distance_f=euclidean, k=4)
    print("Here are our clusters, each holding the indices of samples from the dataset: ", "\n")
    for i in range(len(a)):
        print(a[i])
	
main()
