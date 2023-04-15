import argparse
import yaml
from Classes import *
import pickle
import numpy as np
import cv2

# Initialize argument parser
parser = argparse.ArgumentParser(description='Process training data.')

# Add arguments
#parser.add_argument('--training_images', type=str, required=True, help='Path to training images YAML file.')
#parser.add_argument('--training_params', type=str, required=True, help='Path to training parameters YAML file.')
parser.add_argument('--input_dictionary_path', type=str, required=True, help='Path to output dictionary pickle file.')
parser.add_argument('--reconstruction_params', type=str, required=True, help='Path to reconstruction parameters YAML file.')
parser.add_argument('--input_image_path', type=str, required=True, help='Path to the input image.')
parser.add_argument('--output_image_path', type=str, required=True, help='Path to the output image.')

# Parse arguments
args = parser.parse_args()

# Get image path
path = args.input_image_path
print(f"Image path: {path}")
img_orig = load_image(path)
#print(img_orig)

# Get dictionary
with open(args.input_dictionary_path, 'rb') as f:
    D = pickle.load(f)
print('Got dictionary')

# Get parameters
with open(args.reconstruction_params, 'r') as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
print(f"Your training parameters are: {params}")

num_samples = params['N']
L = params['L']
K = params['K']
I = params['I']
patch_shape = np.array(params['Patch_Shape'])
partial = params['Partial']

# Get output path
new_path = args.output_image_path
print(f"Output path: {new_path}")

sam = Sampler(patch_shape=patch_shape)
learner = DictionaryLearner(sampler=sam, Dictionary=D, algo = 'OMP')
print('Got learner')

if partial:
    (img, error) = learner.SPIR(path, percent=.01)
    print(f"Reconstruction finished with Error Est. = {error}")

    cv2.imwrite(new_path, img)

else:
    img = learner.image_reconstruction(path)
    print(f"Reconstruction finished.")
    cv2.imwrite(new_path, img)

print(f'Saved reconstructed image to {new_path}')



