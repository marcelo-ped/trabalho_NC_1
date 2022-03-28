import os
import numpy as np
import random
import math
import pandas as pd
import sys
import time


number_of_centroids = 3
len_array_data = 13
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
    n_inputs = 9
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
    def __init__(self, centroids):
        self.centroids = centroids
    
    def get_centroids(self):
        return self.centroids
    
    def set_centroid(self, centroid, index):
        self.centroids[index] = centroid
    

class cluster:
    average_shift = 0
    def __init__(self, centroids, list_index_points):
        self.genome = genome(centroids)
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
        #shift_aux  = (shift_aux) / len(list_points)
        return shift_aux

    def calculate_SSE_clusters(self, data):
        shift = 0
        #print(len(self.list_index_points))
        for i in range(number_of_centroids):
            if(len(self.list_index_points[i])) <= 5:
                shift = shift + 2
                list_aux = []
                for j in range(len_array_data):
                    list_aux.append(random.uniform(bounds_list[j][0], bounds_list[j][1]))
                self.genome.set_centroid(np.array(list_aux), i)
                #self.genome.set_sigma(np.random.uniform(0.0, 7.0, 4)) 
            else:
                list_points = []
                for j in range(len(self.list_index_points[i])):
                    list_points.append(data[self.list_index_points[i][j]])
                shift_aux = self.SSE_cluster(list_points, i)
                shift = shift + shift_aux
        #shift = shift / number_of_centroids
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

def create_centroids_of_population():
    population = []
    for i in range(200):
        list_aux = []
        matrix_aux = []
        for a in range(number_of_centroids):
            for j in range(len_array_data):
                list_aux.append(random.uniform(bounds_list[j][0], bounds_list[j][1]))
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
    clusters_to_genome = (cluster(genome.get_centroids(), matrix_index))
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
        clusters_to_each_genome.append(cluster(centroids[j], matrix_index))
        #matrix_index.clear()
    return clusters_to_each_genome

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

def de_procedure(genomes_to_do_mutatation, old_genome, best_genome):
    dice = np.random.uniform(0.0, 1.0)
    #mutation part
    aux_centroids_1 = np.random.uniform(0.0, 1) * np.add(best_genome.get_centroids() ,-1 * np.array(genomes_to_do_mutatation[1].get_centroids()))
    aux_centroids_2 = 0.6 * np.add(best_genome.get_centroids() ,-1 * np.array(genomes_to_do_mutatation[2].get_centroids()))
    aux_centroids_3 = np.add(aux_centroids_1, -1 * np.array(aux_centroids_2))
    candidate_centroids_to_new_genome = np.add(genomes_to_do_mutatation[0].get_centroids() , np.array(aux_centroids_3) )
    centroids_new_genome = []
    #crossover part
    for i in range(number_of_centroids - 1):
        dice = np.random.uniform(0.0, 1.0)
        if dice < 0.66:
            centroids_new_genome.append(candidate_centroids_to_new_genome[i])
        else:
            centroids_new_genome.append(old_genome.get_centroids()[i])
    centroids_new_genome.append(candidate_centroids_to_new_genome[number_of_centroids - 1])
    genome_new = genome(centroids_new_genome)
    return genome_new


number = 1
        

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


def print_best_genome(clusters_to_each_genome):
    global number
    print("DESLOCAMENTO MEDIO")
    print(clusters_to_each_genome[0].average_shift)
    #print("SIGMA")
    #print(clusters_to_each_genome[0].get_genome().get_sigma())
    print("PONTOS DO CLUSTERS")
    print(clusters_to_each_genome[0].get_list_index_points())
    print("PONTOS DO CENTROIDS")
    print(clusters_to_each_genome[0].get_genome().get_centroids())
    print("TERMINOU A EPOCH ", number)
    number += 1
    print("\n")

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



def execute_de_algorithm(number_of_dataset, kmeans_flag):
    global len_array_data
    global number_of_centroids
    global number
    #loading the iris dataset
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
        x_train, y_train, x_test, y_test, n_inputs, n_classes  = load_wine_dataset()
        number_of_centroids = 3
        len_array_data = 13
    #print(x)
    #x = normalize_data(x)
    define_bounds(x_train)
    number_of_epochs = 200
    start_time = time.time()

    centroids_of_population = create_centroids_of_population()
    clusters_to_each_genome = None

    clusters_to_each_genome = create_clusters_to_each_genome_initial(centroids_of_population, x_train)
    if kmeans_flag == 1:
        for i in range(len(clusters_to_each_genome)):
            clusters_to_each_genome[i] = kmeans(x_train ,clusters_to_each_genome[i])

    best_avarage_SSE_clusters = 100000
    best_antes = 0
    repeted = 0
    for _ in range(number_of_epochs):
            
        len_cluster_before_tranformation =  len(clusters_to_each_genome)
        for i in range(len_cluster_before_tranformation):
            #for j in range(6):
            index_set = set()
            while(len(index_set) < 3):
                index_aux = np.random.randint(0, len_cluster_before_tranformation - 1)
                if index_aux != i:
                    index_set.add(index_aux)
            index_list = list(index_set)
            genomes_to_calculate_mutation = []
            for v in index_list:
                genomes_to_calculate_mutation.append(clusters_to_each_genome[v].get_genome())
            genome_aux = de_procedure(genomes_to_calculate_mutation, clusters_to_each_genome[i].get_genome(), clusters_to_each_genome[0].get_genome())
            clusters_aux = create_clusters_to_genome(genome_aux, x_train)
            #clusters_to_each_genome.append(clusters_aux)
            index_set.clear()
            index_list.clear()
            clusters_aux.average_shift = clusters_aux.calculate_SSE_clusters(x_train)

            clusters_to_each_genome[i].average_shift = clusters_to_each_genome[i].calculate_SSE_clusters(x_train)
            #if i < 15:
            #    print(clusters_to_each_genome[i].average_shift)
            if clusters_to_each_genome[i].average_shift > clusters_aux.average_shift:
                clusters_to_each_genome[i] = clusters_aux
        
        clusters_to_each_genome = sorted(clusters_to_each_genome, key=lambda x: x.average_shift, reverse= False)

        print_best_genome(clusters_to_each_genome)
        best_avarage_SSE_clusters = clusters_to_each_genome[0].average_shift
        if(best_antes == best_avarage_SSE_clusters):
            repeted += 1
            if repeted == 40:
                break
        else:
            repeted = 0
        best_antes = best_avarage_SSE_clusters
    end_time = (time.time() - start_time)
    """
    with open("results.txt", "a") as writer:
        writer.write("inter = " + str(calculate_inter_cluster_distance(clusters_to_each_genome[0])) + "\n")
        writer.write("sse = " + str(best_avarage_SSE_clusters) + "\n")
        writer.write("\n")
    """
    inter_cluster_train = calculate_inter_cluster_distance(clusters_to_each_genome[0])
    sse_train = best_avarage_SSE_clusters
    acc_train = calculate_accuracy_1(clusters_to_each_genome[0], y_train)
    
    genome_aux = clusters_to_each_genome[0].get_genome()
    clusters_aux = create_clusters_to_genome(genome_aux, x_test)
    clusters_aux.average_shift = clusters_aux.calculate_SSE_clusters(x_test)
    sse_test =  clusters_aux.average_shift
    inter_cluster_test = calculate_inter_cluster_distance(clusters_aux)
    acc_test = calculate_accuracy_1(clusters_aux, y_test)
    number = 1
    del(clusters_to_each_genome[:])
    return acc_train, sse_train, inter_cluster_train, acc_test, sse_test, inter_cluster_test, end_time
    