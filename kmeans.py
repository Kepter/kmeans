##  This code is a modified version of sample code provided by Dr Yanjie Fu from the University of Central Florida ##

import math
import random
import time

######################################################################
# This section contains functions for loading CSV (comma separated values)
# files and convert them to a dataset of instances.
# Each instance is a tuple of attributes. The entire dataset is a list
# of tuples.
######################################################################

# Loads a CSV files into a list of tuples.
# Ignores the first row of the file (header).
# Numeric attributes are converted to floats, nominal attributes
# are represented with strings.
# Parameters:
#   fileName: name of the CSV file to be read
# Returns: a list of tuples
def loadCSV(fileName):
    fileHandler = open(fileName, "rt")
    lines = fileHandler.readlines()
    fileHandler.close()
    del lines[0] # remove the header
    dataset = []
    for line in lines:
        instance = lineToTuple(line)
        dataset.append(instance)
    return dataset

def loadData(fileName):
    fileHandler = open(fileName, "rt")
    lines = fileHandler.readlines()
    fileHandler.close()
    dataset = []
    for line in lines:
        instance = lineToTuple(line)
        if (len(instance) != 5):
            continue
        reordered = (instance[4], instance[0], instance[1], instance[2], instance[3])
        dataset.append(reordered)
    return dataset



# Converts a comma separated string into a tuple
# Parameters
#   line: a string
# Returns: a tuple
def lineToTuple(line):
    # remove leading/trailing witespace and newlines
    cleanLine = line.strip()
    # get rid of quotes
    cleanLine = cleanLine.replace('"', '')
    # separate the fields
    lineList = cleanLine.split(",")
    # convert strings into numbers
    stringsToNumbers(lineList)
    lineTuple = tuple(lineList)
    return lineTuple

# Destructively converts all the string elements representing numbers
# to floating point numbers.
# Parameters:
#   myList: a list of strings
# Returns None
def stringsToNumbers(myList):
    for i in range(len(myList)):
        if (isValidNumberString(myList[i])):
            myList[i] = float(myList[i])

# Checks if a given string can be safely converted into a positive float.
# Parameters:
#   s: the string to be checked
# Returns: True if the string represents a positive float, False otherwise
def isValidNumberString(s):
  if len(s) == 0:
    return False
  if  len(s) > 1 and s[0] == "-":
      s = s[1:]
  for c in s:
    if c not in "0123456789.":
      return False
  return True


######################################################################
# This section contains functions for clustering a dataset
# using the k-means algorithm.
######################################################################

def squared_distance(instance1, instance2):
    if instance1 == None or instance2 == None:
        return float("inf")
    sumOfSquares = 0
    for i in range(1, len(instance1)):
        sumOfSquares += (instance1[i] - instance2[i])**2
    return sumOfSquares

def manhattan_distance(instance1, instance2):
    if instance1 == None or instance2 == None:
        return float("inf")
    dist = 0
    for i in range(1, len(instance1)):
        dist += abs(instance1[i] - instance2[i])
    #print("Mdist: ", instance1, "  ", instance2, "\n = ", dist)
    return dist

def euclid_distance(instance1, instance2):
    if instance1 == None or instance2 == None:
        return float("inf")
    dist = 0
    for i in range(1, len(instance1)):
        dist += (instance1[i] - instance2[i])**2
    #print("Edist: ", instance1, "  ", instance2, "\n = ", math.sqrt(dist))
    return math.sqrt(dist)

def cosine_distance(instance1, instance2):
    if instance1 == None or instance2 == None:
        return float("inf")
    dot = 0
    mag1 = 0
    mag2 = 0
    for i in range(1, len(instance1)):
        dot += instance1[i] * instance2[i]
        mag1 += instance1[i]**2
        mag2 += instance2[i]**2
    return 1 - (dot / (math.sqrt(mag1) * math.sqrt(mag2)))

def generalized_jacard_distance(instance1, instance2):
    if instance1 == None or instance2 == None:
        return float("inf")
    maxsum = 0
    minsum = 0
    for i in range(1, len(instance1)):
        maxsum += max(instance1[i], instance2[i])
        minsum += min(instance1[1], instance2[i])

    return 1 - (minsum/maxsum)

def centroids_distance(centroids1, centroids2):
    sum = 0
    for i in range(min(len(centroids1), len(centroids2))):
        if (centroids1[i] == None or centroids2[i] == None):
            continue
        sum += squared_distance(centroids1[i], centroids2[i])
    return sum

def distance(instance1, instance2):
    return euclid_distance(instance1, instance2)

def meanInstance(name, instanceList):
    numInstances = len(instanceList)
    if (numInstances == 0):
        return
    numAttributes = len(instanceList[0])
    means = [name] + [0] * (numAttributes-1)
    for instance in instanceList:
        for i in range(1, numAttributes):
            means[i] += instance[i]
    for i in range(1, numAttributes):
        means[i] /= float(numInstances)
    return tuple(means)

def assign(instance, centroids):
    minDistance = distance(instance, centroids[0])
    minDistanceIndex = 0
    for i in range(1, len(centroids)):
        d = distance(instance, centroids[i])
        if (d < minDistance):
            minDistance = d
            minDistanceIndex = i
    return minDistanceIndex

def createEmptyListOfLists(numSubLists):
    myList = []
    for i in range(numSubLists):
        myList.append([])
    return myList

def assignAll(instances, centroids):
    clusters = createEmptyListOfLists(len(centroids))
    for instance in instances:
        clusterIndex = assign(instance, centroids)
        clusters[clusterIndex].append(instance)
    return clusters

def computeCentroids(clusters):
    centroids = []
    for i in range(len(clusters)):
        name = "centroid" + str(i)
        centroid = meanInstance(name, clusters[i])
        centroids.append(centroid)
    return centroids

def kmeans(instances, k, initCentroids=None, maxItr=-1):
    result = {}
    if (initCentroids == None or len(initCentroids) < k):
        # randomly select k initial centroids
        random.seed(time.time())
        centroids = random.sample(instances, k)
    else:
        centroids = initCentroids
    prevCentroids = []
    #print("Intial: ", centroids)

    iteration = 0
    clusters = assignAll(instances, centroids)
    withinss = computeWithinss(clusters, centroids)
    prev_withinss = 1e9
    start_time = time.time()
    while(prev_withinss > withinss):
        #(centroids_distance(centroids, prevCentroids) > 0.00001 or iteration == 0):
        iteration += 1
        #print(centroids, prevCentroids, centroids_distance(centroids, prevCentroids))
        clusters = assignAll(instances, centroids)
        prevCentroids = centroids
        centroids = computeCentroids(clusters)
        prev_withinss = withinss
        withinss = computeWithinss(clusters, centroids)
        #print("P: ", prev_withinss, "N: ", withinss)

        if (maxItr > 0 and iteration >= maxItr):
            break
    
    result["runtime"] = time.time() - start_time
    result["clusters"] = clusters
    result["centroids"] = centroids
    result["withinss"] = withinss
    result["iterations"] = iteration
    return result

def computeWithinss(clusters, centroids):
    result = 0
    for i in range(len(centroids)):
        centroid = centroids[i]
        cluster = clusters[i]
        for instance in cluster:
            result += squared_distance(centroid, instance)
    return result

# Repeats k-means clustering n times, and returns the clustering
# with the smallest withinss
def repeatedKMeans(instances, k, n):
    bestClustering = {}
    bestClustering["withinss"] = float("inf")

    running_time = 0
    running_itr = 0
    for i in range(1, n+1):
        #print("k-means trial %d," % i)
        trialClustering = kmeans(instances, k)
        running_time += trialClustering["runtime"]
        running_itr += trialClustering["iterations"]
        #print("withinss: %.1f" % trialClustering["withinss"])
        if trialClustering["withinss"] < bestClustering["withinss"]:
            bestClustering = trialClustering
            minWithinssTrial = i
    #print("Trial with minimum withinss:", minWithinssTrial)
    bestClustering["avg_runtime"] = running_time/(n+1)
    bestClustering["avg_iterations"] = running_itr/(n+1)
    return bestClustering

def calc_accuracy(clusters, centroids):
    running_acc = 0

    for i in range(len(centroids)):
        if centroids[i] == None or len(clusters[i]) == 0:
            continue

        centroid = centroids[i]
        cluster = clusters[i]

        label_cnt = {}
        for instance in cluster:
            if instance[0] in label_cnt:
                label_cnt[instance[0]] += 1
            else:
                label_cnt[instance[0]] = 1

        max_cnt = -1
        max_label = ""
        for label in label_cnt:
            if label_cnt[label] > max_cnt:
                max_cnt = label_cnt[label]
                max_label = label
        
        correct = 0
        total = 0
        for instance in cluster:
            total += 1
            if instance[0] == max_label:
                correct += 1

        running_acc += correct/total

    return running_acc/len(centroids)


######################################################################
# Test code
######################################################################

### Testing with small, team games dataset ###
# Perform 1 iteration using manhattan distance
dataset = loadCSV("./data.csv")
distance = manhattan_distance
clustering = kmeans(dataset, 2,initCentroids=[('C', 4, 6), ('C', 5, 4)], maxItr=1)
print("Clusters after 1 iteration: ", clustering)

# Fully cluster data using manhattan distance
clustering = kmeans(dataset, 2,initCentroids=[('C', 4, 6), ('C', 5, 4)])
print("\nFinal Clusters: ", clustering)

# Perform 1 iteration using euclidean distance
print("\n-------- euclidean distance ---------")
dataset = loadCSV("./data.csv")
distance = euclid_distance
clustering = kmeans(dataset, 2,initCentroids=[('C', 4, 6), ('C', 5, 4)], maxItr=1)
print("\nClusters after 1 iteration: ", clustering)

# Fully cluster data using euclidean distance
clustering = kmeans(dataset, 2,initCentroids=[('C', 4, 6), ('C', 5, 4)])
print("\nFinal Clusters: ", clustering)

# Perform 1 iteration using manhattan distance
print("\n-------- manhattan distance ---------")
dataset = loadCSV("./data.csv")
distance = manhattan_distance
clustering = kmeans(dataset, 2,initCentroids=[('C', 3, 3), ('C', 8, 3)], maxItr=1)
print("\nClusters after 1 iteration: ", clustering)

# Fully cluster data using manhattan distance
clustering = kmeans(dataset, 2,initCentroids=[('C', 3, 3), ('C', 8, 3)])
print("\nFinal Clusters: ", clustering)

# Perform 1 iteration using euclidean distance
print("\n-------- manhattan distance ---------")
dataset = loadCSV("./data.csv")
distance = manhattan_distance
clustering = kmeans(dataset, 2,initCentroids=[('C', 3, 2), ('C', 4, 8)], maxItr=1)
print("\nClusters after 1 iteration: ", clustering)

# Fully cluster data using euclidean distance
clustering = kmeans(dataset, 2,initCentroids=[('C', 3, 2), ('C', 4, 8)])
print("\nFinal Clusters: ", clustering)

### Testing with Iris data set ###

# Test euclidean, consine, and jarad distance 150 times each and compare results
dataset = loadData("./iris.data")
print ("\n\n--------- IRIS DATASET ----------\n")

distance = euclid_distance
euclid_clustering = repeatedKMeans(dataset, 3, 150)
print("Euclidean distance SSE:\n", euclid_clustering["withinss"])
print("Euclidean distance Accuracy:\n", calc_accuracy(euclid_clustering["clusters"], euclid_clustering["centroids"]))
print("Euclidean distance Iterations:\n", euclid_clustering["iterations"])
print("Euclidean distance Runtime:\n", euclid_clustering["runtime"])
print("Euclidean distance Avg Iterations:\n", euclid_clustering["avg_iterations"])
print("Euclidean distance Avg Runtime:\n", euclid_clustering["avg_runtime"], "\n")

distance = cosine_distance
cosine_clustering = repeatedKMeans(dataset, 3, 150)
print("Cosine distance SSE:\n", cosine_clustering["withinss"])
print("Cosine distance Accuracy:\n", calc_accuracy(cosine_clustering["clusters"], cosine_clustering["centroids"]))
print("Cosine distance Iterations:\n", cosine_clustering["iterations"])
print("Cosine distance Runtime:\n", cosine_clustering["runtime"])
print("Cosine distance Avg Iterations:\n", cosine_clustering["avg_iterations"])
print("Cosine distance Avg Runtime:\n", cosine_clustering["avg_runtime"], "\n")

distance = generalized_jacard_distance
jacard_clustering = repeatedKMeans(dataset, 3, 150)
print("Jarcrad distance SSE:\n", jacard_clustering["withinss"])
print("Jarcrad distance Accuracy:\n", calc_accuracy(jacard_clustering["clusters"], jacard_clustering["centroids"]))
print("Jarcrad distance Iterations:\n", jacard_clustering["iterations"])
print("Jarcrad distance Runtime:\n", jacard_clustering["runtime"])
print("Jarcrad distance Avg Iterations:\n", jacard_clustering["avg_iterations"])
print("Jarcrad distance Avg Runtime:\n", jacard_clustering["avg_runtime"], "\n")
