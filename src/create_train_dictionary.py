# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


from sparse_dic_learning.Classes import *
import pickle
import os

import utils

def main():

    params, paths = load_params()

    print(params)
    print(paths)

    D=get_dictionary(paths=paths, 
                    num_samples= params["N"], 
                    L = params["L"], 
                    K = params["K"], 
                    I = params["I"], 
                    patch_shape = np.array(params['Patch_Shape']))

    save_dictionary(D)

def load_params():
    config = utils.load_config()
    params = config["params"]
    paths = list(config["training_image_paths"].values())
   
    return params, paths

def get_dictionary(paths, num_samples, L, K, I, patch_shape):
    sam = Sampler(paths = paths, num_samples = num_samples, patch_shape=patch_shape)
    print('got sampler')

    learner = DictionaryLearner(L=L, K=K, sampler=sam, algo='OMP')
    print('got learner')

    D = learner.sparse_dictionary_learning(iters = I, output = True)
    print('learned the dictionary')

    return D

def save_dictionary(D):
    output_dictionary = utils.load_config()["OUTPUT_DICTIONARY_PATH"]
    with open(output_dictionary, 'wb') as f:
        pickle.dump(D, f)
    
    print("saved dictionary")

if __name__ == "__main__":
    
   params, paths= load_params()
   print(paths)
   print(params)

   main()