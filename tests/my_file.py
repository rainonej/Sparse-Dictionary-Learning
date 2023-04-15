import argparse

# Initialize argument parser
parser = argparse.ArgumentParser(description='Process training data.')

# Add arguments
parser.add_argument('--training_images', type=str, required=True, help='Path to training images YAML file.')
parser.add_argument('--training_params', type=str, required=True, help='Path to training parameters YAML file.')
parser.add_argument('--output_dictionary_path', type=str, required=True, help='Path to output dictionary pickle file.')

# Parse arguments
args = parser.parse_args()
import yaml
with open(args.training_params, 'r') as file:
    my_data = yaml.load(file, Loader=yaml.FullLoader)
import numpy as np
print(type(np.array(my_data['shape'])))

# Access arguments
print(args.training_images)
print(my_data)
print(args.output_dictionary_path)

#%%
