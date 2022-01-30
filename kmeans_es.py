import os
import numpy as np
from numpy.core.defchararray import add
from numpy.matrixlib.defmatrix import matrix
from sklearn.datasets import load_iris
import random
import math
import pandas as pd
import sys


number_of_centroids = 2
len_array_data = 9
bounds_list = []
def load_iris_dataset():
    data = pd.read_csv("iris.data", header=None)
    data = np.array(data.iloc[:, 0:4])
    print(data[0])
    print((data.shape))
    return data

def load_breast_cancer_dataset():
    data = pd.read_csv("breast-cancer-wisconsin.data", header=None)
    data = np.array(data.iloc[:, 1:10])
    for i in range(len(data)):
        print(i)
        for j in range(len(data[i])):
            if data[i][j] == '?':
                data[i][j] = 1
            data[i][j] = int(data[i][j])
    print(data[0])
    print((data.shape))
    return data

def load_wine_dataset():
    data = pd.read_csv("wine.data", header=None)
    data = np.array(data.iloc[:, 1:14])
    print(data[0])
    print((data.shape))
    return data

class genome:
    def __init__(self, centroids, sigma):
        self.centroids = centroids
        self.sigma = sigma
    
    def get_sigma(self):
        return self.sigma
    
    def get_centroids(self):
        return self.centroids

    def set_sigma(self, sigma):
        self.sigma = sigma
    
    def set_centroid(self, centroid, index):
        self.centroids[index] = centroid

class cluster:
    average_shift = 0
    def __init__(self, centroids, sigma, list_index_points):
        self.genome = genome(centroids, sigma)
        self.list_index_points = list_index_points

    def get_genome(self):
        return self.genome
    
    def get_list_index_points(self):
        return self.list_index_points
    
    def shift_to_new_centroid(self, list_points, index):
        
        shift_aux = 0
        for i in range(len(list_points)):
            for j in range(len(list_points[i])):
                shift_aux = shift_aux + (list_points[i][j] - self.genome.get_centroids()[index][j]) * (list_points[i][j] - self.genome.get_centroids()[index][j])
        return shift_aux

    def average_shift_to_new_centroids(self, data):
        shift = 0
        for i in range(number_of_centroids):
            if(len(self.list_index_points[i])) <= 5:
                shift = shift + 2
                list_aux = []
                for j in range(len_array_data):
                    list_aux.append(random.uniform(bounds_list[j][0], bounds_list[j][1]))
                self.genome.set_centroid(np.array(list_aux), i)
            else:
                list_points = []
                for j in range(len(self.list_index_points[i])):
                    list_points.append(data[self.list_index_points[i][j]])
                shift_aux = self.shift_to_new_centroid(list_points, i)
                shift = shift + shift_aux
        return shift


def define_bounds(data):
    for i in range(data.shape[1]):
        tmp = data[:,i]
        tmp_list = []
        max_elem = np.amax(tmp)
        min_elem = np.amin(tmp)
        tmp_list.append(min_elem)
        tmp_list.append(max_elem)
        bounds_list.append(tmp_list)

def normalize_data(data):
    norm_data = data
    for i in range(data.shape[1]):
        tmp = data[:,i]
        max_elem = np.amax(tmp)
        min_elem = np.amin(tmp)
        for j in range(data.shape[0]):
            norm_data[j][i] = float(
                data[j][i] - min_elem) / (max_elem - min_elem)
    return norm_data

def create_centroids_of_population(x):
    population = []
    for i in range(30):
        list_aux = []
        matrix_aux = []
        for a in range(number_of_centroids):
            for j in range(len_array_data):
                #list_aux.append(kmeans.cluster_centers_[a][j])
                list_aux.append(np.random.uniform(bounds_list[j][0], bounds_list[j][1]))
            matrix_aux.append(np.array(list_aux))
            list_aux.clear()
        population.append(np.array(matrix_aux))
    population = np.array(population)
    return population

def find_index_to_min_dist_centroid(point_data, centroids):
    index = -1
    dist_min = 100000000
    for i in range(len(centroids)):
        dist = 0
        for j in range(len(point_data)):
            dist = dist + (point_data[j] - centroids[i][j]) * (point_data[j] - centroids[i][j])
        dist = math.sqrt(dist)
        if dist < dist_min:
            dist_min = dist
            index = i
    return index

def create_clusters_to_genome(genome, x):
    matrix_index = []
    for a in range(number_of_centroids):
        index = -1
        list_index = []
        for i in range(len(x)):
            index = find_index_to_min_dist_centroid(x[i], genome.get_centroids())
            if index == a:
                list_index.append(i)
        matrix_index.append(np.array(list_index))
        list_index.clear()
    clusters_to_genome = (cluster(genome.get_centroids(), genome.get_sigma() , matrix_index))
    #matrix_index.clear()
    return clusters_to_genome

def create_clusters_to_each_genome_initial(centroids, x):
    clusters_to_each_genome = []
    for j in range(len(centroids)):
        matrix_index = []
        for a in range(number_of_centroids):
            index = -1
            list_index = []
            for i in range(len(x)):
                index = find_index_to_min_dist_centroid(x[i], centroids[j])
                if index == a:
                    list_index.append(i)
            matrix_index.append(np.array(list_index))
            list_index.clear()
        list_sigma = []
        for w in range(len_array_data):
            list_sigma.append(np.random.uniform(bounds_list[w][0], bounds_list[w][1]))
        list_sigma = np.array(list_sigma)
        clusters_to_each_genome.append(cluster(centroids[j], list_sigma, matrix_index))
        #matrix_index.clear()
    return clusters_to_each_genome
#overall_normal = np.random.normal(0, 1, 4)

def kmeans(data, clusters_to_one_genome):
    for a in range(10):
        new_centroids = []
        for i in range(number_of_centroids):
            list_points = []
            for j in range(len(clusters_to_one_genome.get_list_index_points()[i])):
                list_points.append(data[clusters_to_one_genome.get_list_index_points()[i][j]])
            new_centroid = []
            #print(list_points)
            for j in range(len_array_data):
                aux = 0
                for v in range(len(list_points)):
                    aux += list_points[v][j]
                if len(list_points) != 0:
                    aux = aux / len(list_points)
                new_centroid.append(aux)
            new_centroids.append(new_centroid)
        centroids_old = clusters_to_one_genome.get_genome().get_centroids()
        array_equal = 0
        for i in range(number_of_centroids):
            if np.array_equal(centroids_old[i], new_centroids[i]) == True:
                array_equal += 1
            clusters_to_one_genome.get_genome().set_centroid(new_centroids[i], i)
        if array_equal == number_of_centroids:
            break
        clusters_to_one_genome = create_clusters_to_genome(clusters_to_one_genome.get_genome(), data)
    return clusters_to_one_genome   

def do_mutation(genome_to_mutate):
    #list_of_new_genomes_mutated = []
    global overall_normal
    sigma = genome_to_mutate.get_sigma()
    for i in range(len(sigma)):
        sigma[i] = sigma[i] * math.exp(1/ math.sqrt(len_array_data) * np.random.normal(0.0, 1.0))
        if(sigma[i] > bounds_list[i][1] / 2):
            sigma[i] = bounds_list[i][1] / 2
       

    centroids = genome_to_mutate.get_centroids()
    for i in range(len(centroids)):
        for j in range(len(centroids[i])):
            new_centroid_point = None
            while new_centroid_point is None or new_centroid_point < bounds_list[j][0] or new_centroid_point > bounds_list[j][1]:
                new_centroid_point = centroids[i][j] + sigma[j] * np.random.normal(0.0, 1.0)
            centroids[i][j] = new_centroid_point
    genome_new = genome(centroids, sigma)

    return genome_new

def do_recombination(genome1, genome2):
    sigma = np.add(genome1.get_sigma(), genome2.get_sigma()) / 2
    centroids_new = []
    for i in range(len(genome1.get_centroids())):
        aux = np.add(genome1.get_centroids()[i], genome2.get_centroids()[i]) / 2
        centroids_new.append(aux)
    genome_new = genome(centroids_new, sigma)
    return genome_new
#2433905
number = 1
def print_best_genome(clusters_to_each_genome):
    global number
    print("DESLOCAMENTO MEDIO")
    print(clusters_to_each_genome[0].average_shift)
    print("SIGMA")
    print(clusters_to_each_genome[0].get_genome().get_sigma())
    print("PONTOS DO CLUSTERS")
    print(clusters_to_each_genome[0].get_list_index_points())
    print("PONTOS DO CENTROIDS")
    print(clusters_to_each_genome[0].get_genome().get_centroids())
    print("TERMINOU A EPOCH ", number)
    number += 1
    print("\n")

def calculate_intra_cluster_distance(best_cluster, data):
    intra_cluster_dist = 0
    for i in range(number_of_centroids):
        for j in range(len(best_cluster.get_list_index_points()[i])):
            for k in range(len(data[best_cluster.get_list_index_points()[i][j]])):
                intra_cluster_dist += math.pow((data[best_cluster.get_list_index_points()[i][j]][k]) - best_cluster.get_genome().get_centroids()[i][k],2)
    intra_cluster_dist = intra_cluster_dist / number_of_centroids
    return intra_cluster_dist
        

def calculate_inter_cluster_distance(best_cluster):
    dists_list = []
    for i in range(number_of_centroids):
        for j in range(number_of_centroids):
            if i != j:
                aux = 0
                for k in range(len(best_cluster.get_genome().get_centroids()[i])):
                    aux+= math.pow((best_cluster.get_genome().get_centroids()[i][k] - best_cluster.get_genome().get_centroids()[j][k]), 2)
                dists_list.append(aux)
    dists_list = np.array(dists_list)
    dist = np.min(dists_list)
    return dist

def calculate_quantization_error(best_cluster, data):
    quantization_error = 0
    for i in range(number_of_centroids):
        aux = 0
        for j in range(len(best_cluster.get_list_index_points()[i])):
            for k in range(len(data[best_cluster.get_list_index_points()[i][j]])):
                aux += math.pow((data[best_cluster.get_list_index_points()[i][j]][k]) - best_cluster.get_genome().get_centroids()[i][k],2)
        aux = aux / len(best_cluster.get_list_index_points()[i])
        quantization_error += aux
    quantization_error = quantization_error / number_of_centroids
    return quantization_error


def execute_es_algorithm(number_of_dataset, kmeans_flag):
    global len_array_data
    global number_of_centroids
    x= load_iris_dataset()
    number_of_centroids = 3
    len_array_data = 4
    #loading the iris dataset
    #iris = load_iris()
    if number_of_dataset == 1:
        x = load_breast_cancer_dataset() #array of the data
        number_of_centroids = 2
        len_array_data = 9
    if number_of_dataset == 2:
        x = load_wine_dataset()
        number_of_centroids = 3
        len_array_data = 13
    #y = iris.target #array of labels (i.e answers) of each data entry
    #print(x)
    #x = normalize_data(x)
    define_bounds(x)
    print(bounds_list)
    number_of_epochs = 200

    centroids_of_population = create_centroids_of_population(x)
    clusters_to_each_genome = None

    clusters_to_each_genome = create_clusters_to_each_genome_initial(centroids_of_population, x)

    if kmeans_flag == 1:
        for i in range(len(clusters_to_each_genome)):
            clusters_to_each_genome[i] = kmeans(x ,clusters_to_each_genome[i])
    for i in range(len(clusters_to_each_genome)):
        clusters_to_each_genome[i].average_shift = clusters_to_each_genome[i].average_shift_to_new_centroids(x)
    best_avarage_shift_to_new_centroids = 100000

    best_antes = best_avarage_shift_to_new_centroids
    clusters_to_each_genome = sorted(clusters_to_each_genome, key=lambda x: x.average_shift, reverse= False)
    antes_centrois =clusters_to_each_genome[0].get_genome().get_centroids()
    repeted = 0

    for _ in range(number_of_epochs):
            
        
        
        if best_avarage_shift_to_new_centroids < 0.02:
            break
        cluster_child_only = []
        len_cluster_before_tranformation =  len(clusters_to_each_genome)
        for i in range(len_cluster_before_tranformation):
            for j in range(6):
                genome_aux = do_mutation(clusters_to_each_genome[i].get_genome())
                clusters_aux = create_clusters_to_genome(genome_aux, x)
                cluster_child_only.append(clusters_aux)
            if i == len_cluster_before_tranformation - 1:
                break
            genome_aux = do_recombination(clusters_to_each_genome[i].get_genome(), clusters_to_each_genome[i + 1].get_genome())
            clusters_aux = create_clusters_to_genome(genome_aux, x)
            cluster_child_only.append(clusters_aux)

        for i in range(len(cluster_child_only)):
            cluster_child_only[i].average_shift = cluster_child_only[i].average_shift_to_new_centroids(x)
        for i in range(len(cluster_child_only)):
            clusters_to_each_genome.append(cluster_child_only[i])
        
        
        clusters_sorted = sorted(clusters_to_each_genome, key=lambda x: x.average_shift, reverse= False)
        #for i in range(15):
        #   print(clusters_sorted[i].average_shift)
        del(clusters_to_each_genome[:])
        for i in range(30):
            clusters_to_each_genome.append(clusters_sorted[i])
        print_best_genome(clusters_to_each_genome)
        best_avarage_shift_to_new_centroids = clusters_to_each_genome[0].average_shift
        if(best_antes == best_avarage_shift_to_new_centroids):
            repeted += 1
            if repeted == 40:
                break
        else:
            repeted = 0
        
        best_antes = best_avarage_shift_to_new_centroids
        antes_centrois = clusters_to_each_genome[0].get_genome().get_centroids()
    with open("results.txt", "a") as writer:
        writer.write("intra = " + str(best_avarage_shift_to_new_centroids/ number_of_centroids) + "\n")
        writer.write("inter = " + str(calculate_inter_cluster_distance(clusters_to_each_genome[0])) + "\n")
        writer.write("qe = " + str(calculate_quantization_error(clusters_to_each_genome[0], x)) + "\n" )
        writer.write("sse = " + str(best_avarage_shift_to_new_centroids) + "\n")
        writer.write("\n")
"""


"""