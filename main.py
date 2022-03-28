
import os
from kmeans_de import execute_de_algorithm
from kmeans_es import execute_es_algorithm
import argparse
import numpy as np


parser = argparse.ArgumentParser(description='Execute DE and ES in your hybrid versions with K-means or not')
parser.add_argument('-d', '--dataset', action= 'store', dest = 'dataset' ,help='type 0 for iris dataset, type 1 for breast cancer dataset and type 2 for wine dataset', required=True)
parser.add_argument('-p' , '--hybrid',action= 'store', dest = 'hybrid', help='type 0 for execute DE and ES in your pure versions , type 1 for execute DE and ES in your hybrid versions with K-means', required=True)
parser.add_argument('-a', '--algorithm', action= 'store', dest = 'algorithm', help='type 0 for execute ES algoritm, type 1 for execute DE algoritm , type 2 for execute both algoritms', required=True)
args = parser.parse_args()
acc_train = 0 
sse_train = 0 
inter_cluster_train = 0 
acc_test =  0 
sse_test = 0
inter_cluster_test = 0
end_time = 0
if args.algorithm == str(0):
        
    acc_train, sse_train, inter_cluster_train, acc_test, sse_test, inter_cluster_test, end_time = execute_es_algorithm( int(args.dataset), int(args.hybrid))
        

elif args.algorithm == str(1):
    
    
    acc_train, sse_train, inter_cluster_train, acc_test, sse_test, inter_cluster_test, end_time = execute_de_algorithm(int(args.dataset), int(args.hybrid))
        
else:
    execute_es_algorithm( int(args.dataset), int(args.hybrid))
    execute_de_algorithm(int(args.dataset), int(args.hybrid))

print("Acc train " + str(acc_train) + "\n")
print("SSE train " + str(sse_train) + "\n")
print("Inter cluster train " + str(inter_cluster_train) + "\n")
print("Acc test " + str(acc_test) + "\n")
print("SSE test " + str(sse_test) + "\n")
print("Inter cluster test " + str(inter_cluster_test) + "\n")

