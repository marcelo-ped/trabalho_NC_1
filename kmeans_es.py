import os
import numpy as np
import random
import math
import pandas as pd
import sys
import time

number_of_centroids = 2
len_array_data = 9
bounds_list = []
def load_iris_dataset():
    data = pd.read_csv("iris.data", header=None)
    output_str = data.iloc[:, 4]
    #data = np.array(data.iloc[:, 0:4])
    output = []
    for i in output_str:
        if i == "Iris-setosa":
            output.append(0)
        elif i == "Iris-versicolor":
            output.append(1)
        else:
            output.append(2)
    output = np.array(output)
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for i in range(37):
        x_train.append(data.iloc[i, 0:4])
        y_train.append(output[i])
    for i in range(37, 50):
        x_test.append(data.iloc[i, 0:4])
        y_test.append(output[i])
    for i in range(50, 87):
        x_train.append(data.iloc[i, 0:4])
        y_train.append(output[i])
    for i in range(87, 100):
        x_test.append(data.iloc[i, 0:4])
        y_test.append(output[i])
    for i in range(100, 138):
        x_train.append(data.iloc[i, 0:4])
        y_train.append(output[i])
    for i in range(138, 150):
        x_test.append(data.iloc[i, 0:4])
        y_test.append(output[i])
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    perm = np.random.permutation(len(x_train))
    x_train = x_train[perm[:]]
    y_train = y_train[perm[:]]
    perm = np.random.permutation(len(x_test))
    y_test = y_test[perm[:]]
    x_test = x_test[perm[:]]
    n_inputs = 4
    n_classes = 3
    return x_train, y_train, x_test, y_test, n_inputs, n_classes

def load_cancer_dataset():
    print("ENTREI CANCER DATA")
    input_data = pd.read_csv("breast-cancer-wisconsin.data", header=None, skiprows=[23, 40, 139, 145, 158, 164, 235, 249, 275, 292, 294, 297, 315, 321, 411, 617])
    benign_data = []
    malignant_data = []
    for i in range(input_data.shape[0]):
        if input_data.iloc[i, 10] == 2:
            benign_data.append(input_data.iloc[i, 1 : -1])
        else:
            malignant_data.append(input_data.iloc[i, 1 : -1])
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for i in range(int(0.75 * len(benign_data))):
        x_train.append(benign_data[i])
        y_train.append(1)
    for i in range(int(0.75 * len(malignant_data))):
        x_train.append(malignant_data[i])
        y_train.append(0)
    for i in range(int(0.75 * len(benign_data)) + 1 , len(benign_data)):
        x_test.append(benign_data[i])
        y_test.append(1)
    for i in range(int(0.75 * len(malignant_data)) + 1 , len(malignant_data)):
        x_test.append(malignant_data[i])
        y_test.append(0)
    x_train = np.asarray(x_train).astype('float32')
    x_test = np.asarray(x_test).astype('float32')
    y_train = np.asarray(y_train).astype('float32')
    y_test = np.asarray(y_test).astype('float32')
    perm = np.random.permutation(len(x_train))
    x_train = x_train[perm[:]]
    y_train = y_train[perm[:]]
    perm = np.random.permutation(len(x_test))
    y_test = y_test[perm[:]]
    x_test = x_test[perm[:]]
    n_inputs = 10
    n_classes = 2
    return x_train, y_train, x_test, y_test, n_inputs, n_classes   


def load_wine_dataset():
    input_data = pd.read_csv("wine.data", header=None)
    data = (input_data.iloc[:, 1:14])
    data_output = (input_data.iloc[:, 0])
    output = []
    for i in data_output:
        output.append(i - 1)
    output = np.array(output)
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for i in range(44):
        x_train.append(data.iloc[i, 0:13])
        y_train.append(output[i])
    for i in range(44, 59):
        x_test.append(data.iloc[i, 0:13])
        y_test.append(output[i])
    print(y_test[-1])
    for i in range(59, 112):
        x_train.append(data.iloc[i, 0:13])
        y_train.append(output[i])
    print(y_train[43])
    for i in range(112, 130):
        x_test.append(data.iloc[i, 0:13])
        y_test.append(output[i])
    for i in range(130, 166):
        x_train.append(data.iloc[i, 0:13])
        y_train.append(output[i])
    for i in range(166, 178):
        x_test.append(data.iloc[i, 0:13])
        y_test.append(output[i])
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    perm = np.random.permutation(len(x_train))
    x_train = x_train[perm[:]]
    y_train = y_train[perm[:]]
    perm = np.random.permutation(len(x_test))
    y_test = y_test[perm[:]]
    x_test = x_test[perm[:]]
    n_inputs = 13
    n_classes = 3
    return x_train, y_train, x_test, y_test, n_inputs, n_classes


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
    
    def SSE_cluster(self, list_points, index):
        
        shift_aux = 0
        for i in range(len(list_points)):
            for j in range(len(list_points[i])):
                shift_aux = shift_aux + (list_points[i][j] - self.genome.get_centroids()[index][j]) * (list_points[i][j] - self.genome.get_centroids()[index][j])
        return shift_aux

    def calculate_SSE_clusters(self, data):
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
                shift_aux = self.SSE_cluster(list_points, i)
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

def calculate_accuracy(cluster, data_output):
    global number_of_centroids
    general_list = []
    for i in range(number_of_centroids):
        list_points = []
        list_index = cluster.list_index_points
        #print(list_index)
        for j in range(len(list_index[i])):
            list_points.append(data_output[list_index[i][j]])
        general_list.append(list_points)
    right_points = 0
    for i in range(number_of_centroids):
        
        index = -1
        max_cont = 0
        for j in range(number_of_centroids):
            cont = 0
            for k in range(len(general_list[j])):
                if general_list[j][k] == i:
                    cont += 1
            if max_cont < cont :
                index = j
                max_cont = cont
        print(index)
        for k in range(len(general_list[index])):
            if general_list[index][k] == i:
                right_points += 1
        

    print(right_points/len(data_output))

def calculate_accuracy_1(cluster, data_output):
    global number_of_centroids
    general_list = []
    for i in range(number_of_centroids):
        list_points = []
        list_index = cluster.list_index_points
        #print(list_index)
        for j in range(len(list_index[i])):
            list_points.append(data_output[list_index[i][j]])
        general_list.append(list_points)
    right_points = 0
    list_classes = []
    for i in range(number_of_centroids):
        list_classes.append(i)

    for i in range(number_of_centroids):
        
        index = -1
        max_cont = 0
        for j in list_classes:
            cont = 0
            for k in range(len(general_list[j])):
                if general_list[j][k] == i:
                    cont += 1
            if max_cont < cont :
                index = j
                max_cont = cont
        #print(index)
        if index == -1:
            index = list_classes[0]
        for k in range(len(general_list[index])):
            if general_list[index][k] == i:
                right_points += 1
        list_classes.remove(index)
        #print(list_classes)
        

    print(right_points/len(data_output))
    return right_points/len(data_output)


def execute_es_algorithm(number_of_dataset, kmeans_flag):
    global len_array_data
    global number_of_centroids
    global number
    x_train, y_train, x_test, y_test, n_inputs, n_classes  = load_iris_dataset()
    number_of_centroids = 3
    len_array_data = 4
    #loading the iris dataset
    #iris = load_iris()
    if number_of_dataset == 1:
        x_train, y_train, x_test, y_test, n_inputs, n_classes  = load_cancer_dataset() #array of the data
        number_of_centroids = 2
        len_array_data = 9
    if number_of_dataset == 2:
        x_train, y_train, x_test, y_test, n_inputs, n_classes = load_wine_dataset()
        number_of_centroids = 3
        len_array_data = 13
    #y = iris.target #array of labels (i.e answers) of each data entry
    #print(x)
    #x = normalize_data(x)
    define_bounds(x_train)
    print(bounds_list)
    number_of_epochs = 200
    start_time = time.time()

    centroids_of_population = create_centroids_of_population(x_train)
    clusters_to_each_genome = None

    clusters_to_each_genome = create_clusters_to_each_genome_initial(centroids_of_population, x_train)

    if kmeans_flag == 1:
        for i in range(len(clusters_to_each_genome)):
            clusters_to_each_genome[i] = kmeans(x_train ,clusters_to_each_genome[i])
    for i in range(len(clusters_to_each_genome)):
        clusters_to_each_genome[i].average_shift = clusters_to_each_genome[i].calculate_SSE_clusters(x_train)
    best_avarage_SSE_clusters = 100000

    best_antes = best_avarage_SSE_clusters
    clusters_to_each_genome = sorted(clusters_to_each_genome, key=lambda x: x.average_shift, reverse= False)
    antes_centrois =clusters_to_each_genome[0].get_genome().get_centroids()
    repeted = 0

    for _ in range(number_of_epochs):
            
        
        
        if best_avarage_SSE_clusters < 0.02:
            break
        cluster_child_only = []
        len_cluster_before_tranformation =  len(clusters_to_each_genome)
        for i in range(len_cluster_before_tranformation):
            for j in range(6):
                genome_aux = do_mutation(clusters_to_each_genome[i].get_genome())
                clusters_aux = create_clusters_to_genome(genome_aux, x_train)
                cluster_child_only.append(clusters_aux)
            if i == len_cluster_before_tranformation - 1:
                break
            genome_aux = do_recombination(clusters_to_each_genome[i].get_genome(), clusters_to_each_genome[i + 1].get_genome())
            clusters_aux = create_clusters_to_genome(genome_aux, x_train)
            cluster_child_only.append(clusters_aux)

        for i in range(len(cluster_child_only)):
            cluster_child_only[i].average_shift = cluster_child_only[i].calculate_SSE_clusters(x_train)
        for i in range(len(cluster_child_only)):
            clusters_to_each_genome.append(cluster_child_only[i])
        
        
        clusters_sorted = sorted(clusters_to_each_genome, key=lambda x: x.average_shift, reverse= False)
        #for i in range(15):
        #   print(clusters_sorted[i].average_shift)
        del(clusters_to_each_genome[:])
        for i in range(30):
            clusters_to_each_genome.append(clusters_sorted[i])
        print_best_genome(clusters_to_each_genome)
        best_avarage_SSE_clusters = clusters_to_each_genome[0].average_shift
        if(best_antes == best_avarage_SSE_clusters):
            repeted += 1
            if repeted == 40:
                break
        else:
            repeted = 0
        
        best_antes = best_avarage_SSE_clusters
        antes_centrois = clusters_to_each_genome[0].get_genome().get_centroids()
    end_time = (time.time() - start_time)
    acc_train = calculate_accuracy_1(clusters_to_each_genome[0], y_train)
    inter_cluster_train = calculate_inter_cluster_distance(clusters_to_each_genome[0])
    sse_train = best_avarage_SSE_clusters
    
    genome_aux = clusters_to_each_genome[0].get_genome()
    clusters_aux = create_clusters_to_genome(genome_aux, x_test)
    sse_test =  clusters_aux.calculate_SSE_clusters(x_test)
    inter_cluster_test = calculate_inter_cluster_distance(clusters_aux)
    acc_test = calculate_accuracy_1(clusters_aux, y_test)
    number = 1
    del(clusters_to_each_genome[:])
    return acc_train, sse_train, inter_cluster_train, acc_test, sse_test, inter_cluster_test, end_time
    
    


    
"""
    with open("results.txt", "a") as writer:
        writer.write("inter = " + str(calculate_inter_cluster_distance(clusters_to_each_genome[0])) + "\n")
        writer.write("sse = " + str(best_avarage_SSE_clusters) + "\n")
        writer.write("\n")
"""
