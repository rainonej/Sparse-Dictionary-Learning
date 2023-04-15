# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from sparse_dic_learning.Classes import *
import pickle
import os
import argparse

import utils

def main():
    params, paths, output_image_path = load_params()

    reconstruct_image(output_image_path= output_image_path,
                     paths = paths,
                     D =  load_dictionary(),
                     patch_shape = params["Patch_Shape"], 
                     partial =params["Partial"])

def load_params():
    config = utils.load_config()
    params = config["params"]
    paths = list(config["training_image_paths"].values())
    reconstruct_image_path = config["IMAGE_RESCONSTRUCTION_DIR"]
   
    return params, paths, reconstruct_image_path

def load_dictionary():
    output_dictionary = utils.load_config()["OUTPUT_DICTIONARY_PATH"]

    with open(output_dictionary, 'rb') as f:
        D = pickle.load(f)
    print('Got dictionary')

    return D

def reconstruct_image(output_image_path, paths, D, patch_shape, partial):

    input_image_path = paths[]
    sam = Sampler(patch_shape=patch_shape)
    learner = DictionaryLearner(sampler=sam, Dictionary=D, algo = 'OMP')
    print('Got learner')

    if partial == "True":
        img, error = learner.SPIR(input_image_path, percent=.01)
        print(f"Reconstruction finished with Error Est. = {error}")

        return img

    else:
        img = learner.image_reconstruction(input_image_path)
        print(f"Reconstruction finished.")

        return img

def save_images(image_file_path, image):
    cv2.imwrite(image_file_path, image)
    print("Saved reconstructed image")

def parse_args():

    parser = argparse.ArgumentParser(description='Process training data.')
    parser.add_argument('--output-image-path', type=str, required=True, help='Path to the input image.')
    args = parser.parse_args()

    return args


if __name__ == "__main__":

   args = parse_args()

   main(output_image_path = args.output_image_path)