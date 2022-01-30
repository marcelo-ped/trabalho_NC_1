import os
from kmeans_de import execute_de_algorithm
from kmeans_es import execute_es_algorithm
import argparse



parser = argparse.ArgumentParser(description='Execute DE and ES in your hybrid versions with K-means or not')
parser.add_argument('-d', '--dataset', action= 'store', dest = 'dataset' ,help='type 0 for iris dataset, type 1 for breast cancer dataset and type 2 for wine dataset', required=True)
parser.add_argument('-p' , '--hybrid',action= 'store', dest = 'hybrid', help='type 0 for execute DE and ES in your pure versions , type 1 for execute DE and ES in your hybrid versions with K-means', required=True)
parser.add_argument('-a', '--algorithm', action= 'store', dest = 'algorithm', help='type 0 for execute ES algoritm, type 1 for execute DE algoritm , type 2 for execute both algoritms', required=True)
args = parser.parse_args()
print(args)
if args.algorithm == str(0):
    execute_es_algorithm( int(args.dataset), int(args.hybrid))
elif args.algorithm == str(1):
    execute_de_algorithm(int(args.dataset), int(args.hybrid))
else:
    execute_es_algorithm( int(args.dataset), int(args.hybrid))
    execute_de_algorithm(int(args.dataset), int(args.hybrid))
