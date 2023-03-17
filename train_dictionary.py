import argparse
import yaml
from Classes import *
import pickle
import numpy as np

# Initialize argument parser
parser = argparse.ArgumentParser(description='Process training data.')

# Add arguments
parser.add_argument('--training_images', type=str, required=True, help='Path to training images YAML file.')
parser.add_argument('--training_params', type=str, required=True, help='Path to training parameters YAML file.')
parser.add_argument('--output_dictionary_path', type=str, required=True, help='Path to output dictionary pickle file.')


# Parse arguments
args = parser.parse_args()

with open(args.training_images, 'r') as file:
    paths_yaml = yaml.load(file, Loader=yaml.FullLoader)
paths = paths_yaml['paths']
print(f"Your training image paths are: {paths}")

with open(args.training_params, 'r') as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
print(f"Your training parameters are: {params}")


num_samples = params['N']
L = params['L']
K = params['K']
I = params['I']
patch_shape = np.array(params['Patch_Shape'])

sam = Sampler(paths = paths, num_samples = num_samples, patch_shape=patch_shape)
print('got sampler')

learner = DictionaryLearner(L=L, K=K, sampler=sam, algo='OMP')
print('got learner')

D = learner.sparse_dictionary_learning(iters = I, output = True)
print('learned the dictionary')

with open(args.output_dictionary_path, 'wb') as f:
    pickle.dump(D, f)

print("!!!!!!!!! DONE !!!!!!!!!")
