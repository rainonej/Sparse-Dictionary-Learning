"""
This Classes.py file contains all of the same code as in Classes.ipybn.
The difference is that this is more streamlined and easier to read if you know exactly what you are looking for.
The Classes.ipynb file is easier to read as a story, explaining why I did what I did, and providing the results of simple experiments.
"""

####### Load libraries

import numpy as np
import random
import cv2
from tqdm.notebook import tqdm
import time
import pandas as pd
import matplotlib.pyplot as plt

# Define some helper functions

def get_ith_patch(large_shape, patch_shape, i):
    # How many patches can we fit in this large array?
    patch_index_shape = np.array(large_shape) - np.array(patch_shape) + 1

    # What are the coordinates of the starting pixel of the ith patch?
    patch_index = np.unravel_index(i, patch_index_shape)

    # Get the indices for the pixels in the ith patch
    patch_indices = tuple(slice(start, start + size) for start, size in zip(patch_index, patch_shape))

    return patch_indices

def get_num_patches(large_shape, patch_shape):
    patch_index_shape = np.array(large_shape) - np.array(patch_shape) + 1
    return patch_index_shape.prod()

def load_image(path):
    img = cv2.imread(path)
    if (img[:,:,0] == img[:,:,1]).all():
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img

####### Sampler Class

class Sampler:

    def __init__(self, paths = [], num_samples = 300, patch_shape = np.array([8,8,3])):
        #self.seed = 1 # A random seed which will be updated after every use. It is there to ensure reproducibility
        self.paths = paths # A list of the paths where original images/signals are stored
        self.add_filter(None)
        self.num_samples = num_samples
        self.patch_shape = patch_shape

    def set_patch_shape(self, patch_shape):
        self.patch_shape = patch_shape

    def set_num_samples(self, num_samples):
        self.num_samples = num_samples

    def add_path(self, path):
        self.paths.append(path)

    def set_paths(self, paths):
        self.paths = paths

    def add_filter(self, filter, std = 10):

        if filter == None:
            foo = lambda x: x
            self.filter = foo

        elif filter == 'noise':

            def filter(sample):

                noise = np.zeros_like(sample)
                cv2.randn(noise, 0, std)
                sample = np.clip(sample + noise, 0, 255).astype(np.uint8)

                return sample
            self.filter = filter

        elif filter == 'RB0':
            # A filter to remove the green channel

            def filter(sample):
                new_sample = sample.copy()
                new_sample[:,:,2] = 0
                return new_sample
            self.filter = filter


    def sample(self, N = None):
        """
        A function which returns N samples. It returns a pair of matrices (Y_corrupted, Y_original).

        :param N:
        :param two_copies:

        :return:
        """
        def get_sample(img1, img2):
            """
            Helper function to get a single patch sample from an image.
            Args:
                img: np.array, representing an image
                add_noise: bool, whether to add Gaussian noise to the patch
            Returns:
                sample: np.array, of shape (d,), representing the sampled patch
            """

            # Get a random patch from the image
            patch = get_ith_patch(large_shape, self.patch_shape, random.randint(0, num_patches-1))

            return (img1[patch].flatten(), img2[patch].flatten())

        # Get the number of samples
        if N == None:
            N = self.num_samples

        # Get the product of the patch dimensions
        d = self.patch_shape.prod()

        # Initialize an array to hold the sampled patches
        Y = np.zeros([d, N], dtype=np.uint8)
        Y_orig = np.zeros([d, N], dtype=np.uint8)

        # Divide the number of samples requested evenly amongst each original full image
        j = 0
        r = N % len(self.paths)

        for i, path in enumerate(self.paths):

            # Read in the image, apply the filter, and if nessisary preserve the original copy
            img2 = load_image(path)
            img1 = self.filter(img2)

            # Get the shape of the image
            large_shape = img1.shape

            # Get the number of patches that can be extracted from the image
            num_patches = get_num_patches(large_shape, self.patch_shape)

            # Sample patches from the image
            for _ in range(N // len(self.paths)):
                (corrupted_sample, original_sample) = get_sample(img1, img2)
                Y[:, j] = corrupted_sample
                Y_orig[:, j] = original_sample
                j += 1

            # If there are any remaining samples, sample them from the current image
            if i < r:
                (corrupted_sample, original_sample) = get_sample(img1, img2)
                Y[:, j] = corrupted_sample
                Y_orig[:, j] = original_sample
                j += 1

        return (Y, Y_orig)

####### Optimization Algorithms

def find_sparse_rep_MP(Y, D, L):
    """
    :param Y: This is a (d x N) matrix representing the N different d-dimensional given signals.
    :param D: This is a (d x K) matrix representing the dictionary of K different atoms, where the atoms are d-dimensional vectors. Each column vector must have already been normalized.
    :param L: This is an integer satisfying 0 < L <= K representing the maximum number of atoms which can be used in a sparse representation.

    Runs in O( N K L ) time.

    Note that we need N > K > d >= L

    :return: A, the (K x N) matrix of the N different K-dimensional sparse representations of the columns of Y.
    """

    (d, N) = Y.shape
    (d1, K) = D.shape
    assert d == d1, f"The dimensions dont add up: Y.shape = {Y.shape} and D.shape = {D.shape}" # Make sure the dimensions match up

    A = np.zeros((K, N)) # Get our Sparse Representation Matrix

    for j in range(N):  # Iterate over all of the N given signal vectors Y[:,j].

        alpha = np.zeros(K)  # Initialize the sparse representation vector (will be a column vector in A)
        r_vec = Y[:, j] - np.dot(D, alpha)  # Initialize the "residual" vector

        for i in range(L): # Repeat until we utilize L atoms, or no more are needed
            position_coeff_error = []

            for k in range(K): # Find the best atom, D[:,k]

                atom = D[:, k]

                # Project the residual vector r_vec down to the linear subspace defined by the atom
                coeff = np.inner(r_vec, atom)
                error = np.linalg.norm(r_vec - coeff * atom)
                position_coeff_error.append((k, coeff, error))

            position, coeff, error = min(position_coeff_error, key=lambda x: x[2]) # Find the atom whose linear subspace is closest to the residual vector r_vec

            if np.abs(coeff) < 1e-6: #If the coefficient is too small, we don't add it and instead end the iteration
                break

            else:
                alpha[position] += coeff # Update the sparse representation vector, alpha
                r_vec -= coeff * D[:, position] # Update the residual vector, r_vec

        A[:, j] = alpha # Insert the sparse representation vector alpha into the matrix A

    return A

def find_sparse_rep_OMP(Y, D, L):
    """
    Find the sparse representation of the given signals Y over the dictionary D using the Orthogonal Matching Pursuit
    algorithm.
    Runs in O( N K L^2 ) time.

    :param Y: A (d x N) matrix representing the N different d-dimensional given signals.
    :param D: A (d x K) matrix representing the dictionary of K different atoms, where the atoms are d-dimensional
    vectors. Each column vector must have already been normalized.
    :param L: An integer representing the maximum number of atoms which can be used in a sparse representation.

    Note that we need N > K > d >= L

    :return: A, a (K x N) matrix of the N different K-dimensional sparse representations of the columns of Y.
    """

    # Get the shapes of the input matrices
    (d, N) = Y.shape
    (d1, K) = D.shape

    # Ensure that the dimensions match up
    assert d == d1

    # Initialize the sparse representation matrix A
    A = np.zeros((K, N))

    # Iterate over all of the N given signal vectors Y[:,j].
    for j in range(N):

        # Initialize the set of indices for the selected atoms
        idx_set = set()

        # Repeat until we utilize L atoms, or no more are needed
        for i in range(L):

            # Find the remaining unused atoms
            remaining_atoms = set(range(K)).difference(idx_set)

            # Initialize a list to store the coefficients and errors for each candidate atom
            position_coeff_error = []

            # Iterate over the remaining unused atoms and calculate the projection error for each
            for k in remaining_atoms:
                # Create the subspace basis from the remaining unused atoms plus the current candidate atom
                subspace_basis = D[:, list(idx_set) + [k]]

                # Solve for the coefficients of the projection
                coeff = np.linalg.lstsq(subspace_basis, Y[:, j], rcond=None)[0]

                # Calculate the projected vector
                projected = np.dot(subspace_basis, coeff)

                # Calculate the error between the original signal and the projection
                error = np.linalg.norm(Y[:, j] - projected)

                # Store the position, coefficients, and error in the list
                position_coeff_error.append((k, coeff, error))

            # Select the candidate atom with the minimum projection error
            position, coeff, error = min(position_coeff_error, key=lambda x: x[2])

            # Add the selected atom to the set of indices for the selected atoms
            idx_set.add(position)

        # Create the final subspace basis from the selected atoms
        subspace_basis = D[:, list(idx_set)]

        # Solve for the coefficients of the sparse representation using the selected atoms
        coeff = np.linalg.lstsq(subspace_basis, Y[:, j], rcond=None)[0]

        # Initialize the sparse representation vector alpha
        alpha = np.zeros(K)

        # Insert the computed coefficients into the sparse representation vector alpha
        for position, index in enumerate(idx_set):
            alpha[index] = coeff[position]

        # Insert the sparse representation vector alpha into the matrix A
        A[:, j] = alpha

        # Return the matrix of sparse representations
    return A

def update_dictionary_kSVD(Y, D, A):
    """
    Update the dictionary using the k-SVD algorithm. This

    :param Y: This is the (d x N) matrix representing the N different d-dimensional given signals.
    :param D: This is the (d x K) matrix representing the dictionary of K different atoms, where the atoms are d-dimensional
    vectors. Each column vector must have already been normalized.
    :param A: This is the (K x N) matrix of the N different K-dimensional sparse representations of the columns of Y.

    :return: (D, A), where D is updated and optimized to the given  a (K x N) matrix of the N different K-dimensional sparse representations of the columns of Y.
    """

    # Get the shapes of the input matrices
    (d, N) = Y.shape
    (d1, K) = D.shape
    (K1, N1) = A.shape

    # Ensure that the dimensions match up
    assert d == d1
    assert K == K1
    assert N == N1

    # Iterate over every atom in the dictionary
    unused_atoms = []
    for k in range(K):
        # Find the signal vectors, Y[:,j], whose sparse representation, A[:,j], have a non-zero entry in the k^th position. That is, they use the k^th atom.
        non_zero_indices = np.nonzero(A[k, :])[0]

        if len(non_zero_indices) == 0:
            unused_atoms.append(k)

        else:
            # Get the k^th "error matrix"
            E = Y - np.dot(D, A) + np.outer(D[:, k], A[k, :])

            # Restrict the matrix to only those non-zero values. The resulting matrix should be KxL
            E = E[:, non_zero_indices]

            # Do the SVD (Singular Value Decomposition) step to the KxL matrix E
            U, S, V = np.linalg.svd(E, full_matrices=False)

            #print(f'For the k={k} atom, E={E}, and non_zero_indices = {non_zero_indices}')

            # Update the k^th atom, D[:, k], and the k^th coefficients in the sparse representation, A[k, :].
            # Note: The k-SVD algorithm also converges when run in parallel, only updating the matrix D at the end. However running the algorithm in series, updating the atoms and coefficients after each step, produces more robust results and typically requires more than four times as long to converge.
            D[:, k] = U[:, 0]
            A[k, non_zero_indices] = S[0] * V[0, :]

    # Replace the unused atoms with the worst represented sample vectors
    E = Y - np.dot(D, A)
    errors = []
    for j in range(N):
        errors.append((j, np.linalg.norm(E[:,j])))
    errors.sort(key = lambda x: -x[1])

    num_unused = len(unused_atoms)
    for i in range(num_unused):
        D[:, unused_atoms[i]] = Y[:, errors[i][0]]

    return (D, A)

####### Dictionary Learner Class

class DictionaryLearner:

    def __init__(self, L=5, K=100, sampler = None, algo = None, Dictionary = None, previous_iters = 0):

        assert L<K, f"The total number of atoms, K={K}, must be greater than the maximum number of allowed atoms per sparse representation, L = {L}"

        self.L = L # The maximum number of atoms a sparse representation can use
        self.K = K # The size of the dictionary
        self.Dictionary = Dictionary # The initial guesses for the dictionary
        self.sampler = sampler
        self.select_algorithm(algo)
        self.errors = []
        self.dictionary_learning_time = 0
        self.iters = 0

        self.update_dictionary_kSVD = update_dictionary_kSVD
        self.update_step()

    def set_sampler(self, sampler):
        self.sampler = sampler

    def set_initial_dictionary(self, D):
        self.Dictionary = D

    def select_algorithm(self, algo):
        if algo == 'MP':
            self.sparse_rep = find_sparse_rep_MP
        elif algo == 'OMP':
            self.sparse_rep = sparse_rep = find_sparse_rep_OMP

        else:
            self.sparse_rep = None

    def update_step(self, inner_loop = 1, use_orig = False):
        def update_dictionary(Y, Y_orig, D, A):
            for i in range(inner_loop):
                if use_orig:
                    (D, A) = self.update_dictionary_kSVD(Y_orig, D, A)
                else:
                    (D,A) = self.update_dictionary_kSVD(Y, D, A)
            return (D,A)
        self.update_dictionary = update_dictionary

    def sparse_dictionary_learning(self, iters=10, output = True):
        """
        This algorithm finds a (d x K) matrix D (the dictionary) and a (K x N) matrix A (the sparse representation) which minimise the L2 distance between Y and D A, ie, minimise ||Y - D A ||, subject to the constraint that each column of A has at most L non-zero elements.

        :param Y: This is the (d x N) matrix representing the N different d-dimensional given signals.
        :param K: An integer representing the size of the dictionary.
        :param L: An integer representing the maximum number of "atoms", D[:, k], in the dictionary that each sparse representation vector, A[:, i], can use.

        Note: This algorithm is written under the assumption that: 0 < L < d < K < N

        :param D_initial: This is the initial guess for the (d x N) matrix D. If not None, the columns of this matrix must be unit length.
        :param algo: This is a string defining the sparse representation algorithm. Either algo = 'OMP' for Orhtogonal Matching Pursuit, or algo = 'MP' for Matching Pursuit.
        :param iters: The number of iterations this will run for
        :param with_errors: A boolean which determines if the output includes the list of the error values at each step of the iteration.
        :param samples: This tells us the number of random samples to take from the training data Y at each step

        :return: (D, A, errors)
            D: This is the (d x K) matrix representing the dictionary of K different atoms, where the atoms are d-dimensional
        vectors.
            A: This is the (K x N) matrix of the N different K-dimensional sparse representations of the columns of Y.
            errors: This is an optional output. It is the list of the error values at each step of the iteration.
        """

        # Make sure we have the proper stuff defined
        assert self.sparse_rep != None

        # Get the internal variables for ease
        K = self.K
        L = self.L
        sampler = self.sampler
        D = self.Dictionary
        sparse_rep = self.sparse_rep
        update_dictionary = self.update_dictionary

        # Get Initial Dictionary if there is none
        if D is None:
            (Y, _) = sampler.sample()
            N = len(Y[0, :])
            D = Y[:, random.sample(range(N), k=K)]
            D = D / np.linalg.norm(D, axis=0)

        # Start the timer
        start_time = time.time()

        for step in tqdm(range(iters)):

            # Get the batch of random samples
            (Y, Y_orig) = sampler.sample()

            # Find the Sparse Representations
            A = sparse_rep(Y, D, L)

            # Record the error
            error = np.linalg.norm(Y - np.dot(D, A))
            self.errors.append(error)

            # Update the Dictionary
            (D, A) = update_dictionary(Y, Y_orig, D, A)


        # Calculate and Update the Learning Time
        run_time = time.time() - start_time
        self.dictionary_learning_time += run_time

        # Record the error one last time
        A = sparse_rep(Y, D, L)
        error = np.linalg.norm(Y - np.dot(D, A))
        self.errors.append(error)

        # Update Dictionary
        self.Dictionary = D

        # Update the Iters count
        self.iters += iters

        if output:
            return D

    def image_reconstruction(self, path):

        # Get internal stuff for ease of use
        D = self.Dictionary
        patch_shape = self.sampler.patch_shape
        patch_size = patch_shape[0]
        L = self.L

        img = load_image(path)
        large_shape = img.shape
        num_rows, num_cols = large_shape[:2]
        num_patches_rows = num_rows - patch_size + 1
        num_patches_cols = num_cols - patch_size + 1

        # Initialize the reconstructed image
        recon_img = np.zeros(img.shape, dtype=np.float32)
        count = np.zeros(img.shape, dtype=np.float32)

        # Initialize the progress bar
        pbar = tqdm(total=num_patches_rows*num_patches_cols)

        # Loop over all patches in the image
        for i in range(num_patches_rows):
            for j in range(num_patches_cols):
                # Extract the patch from the image
                patch = img[i:i+patch_size, j:j+patch_size, ...]

                # Compute the sparse coding of the patch
                sparse_patch_code = self.sparse_rep(patch.flatten().reshape(-1,1), D, L)
                recon_patch = np.dot(D, sparse_patch_code)
                recon_patch = recon_patch.reshape(patch_shape)

                # Add the reconstructed patch to the reconstructed image
                recon_img[i:i+patch_size, j:j+patch_size, ...] += recon_patch
                count[i:i+patch_size, j:j+patch_size, ...] += 1

                # Increment the counter variable and update the progress bar
                pbar.update(1)

        # Close the progress bar
        pbar.close()

        # Average the pixel values at each pixel to get the final reconstructed image
        recon_img /= count

        # Convert the reconstructed image to uint8
        recon_img = np.clip(recon_img, 0, 255).astype(np.uint8)

        return recon_img

    def SPIR(self, path, percent = .2, min_count = 1, apply_filter = False):

        # Start the timer
        start_time = time.time()

        # Get internal stuff for ease of use
        D = self.Dictionary
        patch_shape = self.sampler.patch_shape
        patch_size = patch_shape[0]
        L = self.L

        img_orig = load_image(path)
        if apply_filter:
            img = self.sampler.filter(img_orig)
        else:
            img = img_orig
        large_shape = img.shape
        num_rows, num_cols = large_shape[:2]
        num_patches_rows = num_rows - patch_size + 1
        num_patches_cols = num_cols - patch_size + 1
        num_patches = num_patches_cols*num_patches_rows

        #print(f'The image shape is {img.shape} with ')

        # Initialize the reconstructed image
        recon_img = np.zeros(img.shape, dtype=np.float32)
        count = np.zeros(img.shape, dtype=np.float32)

        # Initialize the progress bar
        total = int(percent*num_patches)
        pbar = tqdm(total=total)

        for i in range(total):

            index = random.randint(0,num_patches-1)
            row_idx, col_idx  = (index // num_patches_cols, index % num_patches_cols)

            # Extract the patch from the image
            patch = img[row_idx:row_idx+patch_size, col_idx:col_idx+patch_size, ...]

            # Compute the sparse coding of the patch
            sparse_patch_code = self.sparse_rep(patch.flatten().reshape(-1,1), D, L)
            recon_patch = np.dot(D, sparse_patch_code)
            recon_patch = recon_patch.reshape(patch_shape)

            # Add the reconstructed patch to the reconstructed image
            recon_img[row_idx:row_idx+patch_size, col_idx:col_idx+patch_size, ...] += recon_patch
            count[row_idx:row_idx+patch_size, col_idx:col_idx+patch_size, ...] += 1

            # Increment the counter variable and update the progress bar
            pbar.update(1)

        # Calculate and Update the Reconstruction Time
        run_time = time.time() - start_time
        self.reconstruction_time = run_time

        # Compute Error Old way

        temp_recon = recon_img.copy().flatten()
        temp_count = count.copy().flatten()
        temp_img = img.copy().flatten()
        temp_indices = np.where(temp_count>min_count)[0]


        M = len(temp_indices)
        error = np.linalg.norm(temp_img[temp_indices] -  (temp_recon[temp_indices]/temp_count[temp_indices]))/np.sqrt(M)

        recon_img = recon_img // count
        recon_img = np.clip(recon_img, 0, 255).astype(np.uint8)


        # Testing the New Function
        self.update_EVALUATION_METRICS(temp_indices, img_orig, recon_img, img_corrupted=img, )

        return (recon_img, error)

    def SPIR_test(self, path, apply_filter = False, min_count = 1, replacement = False):
        """
        This is just a test to see how quickly the different error metrics converge when using SPIR without replacement.
        """

        # Get internal stuff for ease of use
        D = self.Dictionary
        patch_shape = self.sampler.patch_shape
        patch_size = patch_shape[0]
        L = self.L

        img_orig = load_image(path)
        if apply_filter:
            img = self.sampler.filter(img_orig)
        else:
            img = img_orig
        large_shape = img.shape
        num_rows, num_cols = large_shape[:2]
        num_patches_rows = num_rows - patch_size + 1
        num_patches_cols = num_cols - patch_size + 1
        num_patches = num_patches_cols*num_patches_rows

        #print(f'The image shape is {img.shape} with ')

        # Initialize the reconstructed image
        recon_img = np.zeros(img.shape, dtype=np.float32)
        count = np.zeros(img.shape, dtype=np.float32)

        # Initialize the progress bar
        patch_order = list(range(num_patches))
        random.shuffle(patch_order)
        if replacement:
            patch_order = random.choices(patch_order, k=num_patches)
        pbar = tqdm(total=num_patches)

        # Get the DataFrame that we store the evaluation metrics in
        df = pd.DataFrame(columns = ['MSE', 'RMSE', 'PSNR', 'SSIM', 'Luminance Dist.', 'Contrast Dist.', 'Structural Comp.'])

        for i, index in enumerate(patch_order):

            row_idx, col_idx  = (index // num_patches_cols, index % num_patches_cols)

            # Extract the patch from the image
            patch = img[row_idx:row_idx+patch_size, col_idx:col_idx+patch_size, ...]

            # Compute the sparse coding of the patch
            sparse_patch_code = self.sparse_rep(patch.flatten().reshape(-1,1), D, L)
            recon_patch = np.dot(D, sparse_patch_code)
            recon_patch = recon_patch.reshape(patch_shape)

            # Add the reconstructed patch to the reconstructed image
            recon_img[row_idx:row_idx+patch_size, col_idx:col_idx+patch_size, ...] += recon_patch
            count[row_idx:row_idx+patch_size, col_idx:col_idx+patch_size, ...] += 1

            # Increment the counter variable and update the progress bar
            pbar.update(1)

            # Compute the Error
            if ((i % 100) == 0):

                # Initialize the data storage
                data = {}

                # Find all pixels that have been visited
                temp_count = count.copy().flatten().astype(int)
                temp_indicies = np.where(temp_count>=min_count)[0]
                M = len(temp_indicies)

                if M>0:

                    # Get the correct original image vector
                    vec_orig = img_orig.copy().flatten().astype(int)
                    vec_orig = vec_orig[temp_indicies]

                    # Get the correct reconstructed image vector
                    vec_recon = recon_img.copy().flatten().astype(int)
                    vec_recon = vec_recon[temp_indicies]/temp_count[temp_indicies]


                    # Compute RMSE
                    MSE = ((vec_orig - vec_recon)**2).sum() / M
                    data['MSE'] = MSE
                    RMSE = np.sqrt(MSE)
                    data['RMSE'] = RMSE

                    # Compute PSNR
                    peak_signal = vec_orig.max()
                    PSNR = 10 * (2 * np.log10(peak_signal) - np.log10(MSE))
                    data['PSNR'] = PSNR

                    ### Compute SSIM

                    mean_orig = vec_orig.mean()
                    mean_recon = vec_recon.mean()
                    var_orig = vec_orig.var()
                    var_recon = vec_recon.var()
                    sigma_xy = ((vec_orig - mean_orig) * (vec_recon - mean_recon) ).sum() / M

                    # Choice of Constants
                    #C_l, C_c, C_s = 1,1,1
                    # In  this paper, the authors suggest these constants : Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.:‘Image qualityassessment: from error visibility to structural similarity’,IEEE Trans.Image Process., 2004,13, (4), pp. 600–612
                    const_k1 = .01
                    const_k2 = .03
                    const_L = 255
                    C_l = (const_k1 * const_L)**2
                    C_c = (const_k2 * const_L)**2
                    C_s = C_c/2

                    luminance_dist = ( 2 * mean_recon * mean_orig + C_l ) / ( mean_recon**2 + mean_orig**2 + C_l)
                    contrast_dist = (2 * np.sqrt(var_recon * var_orig) + C_c) / (var_orig + var_recon + C_c)
                    structural_comp = (sigma_xy + C_s) / (np.sqrt( var_recon * var_orig) + C_s)
                    SSIM = luminance_dist * contrast_dist * structural_comp
                    data['Luminance Dist.'] = luminance_dist
                    data['Contrast Dist.'] = contrast_dist
                    data['Structural Comp.'] = structural_comp
                    data['SSIM'] = SSIM


                    # Record the data
                    df.loc[i/num_patches] = data

        return df


    def update_EVALUATION_METRICS(self, temp_indicies, img_orig, img_recon, img_corrupted = False, img_orig_path = None, img_recon_path = None, img_corrupted_path = None):
        """
        This will update the pandas DataFrame stored in EVALUATION_METRICS.csv.
        :param img_orig:
        :param img_recon:
        :param img_corrupted:
        :return:
        """

        # Change the type
        img_orig = img_orig.astype(int)
        img_recon = img_recon.astype(int)
        img_corrupted = img_corrupted.astype(int)


        # Get the Data
        data = {}

        # Get the Run Time data
        run_time = self.dictionary_learning_time + self.reconstruction_time
        run_time_learning = self.dictionary_learning_time
        run_time_recon = self.reconstruction_time
        data['Run Time'] = run_time
        data['Run Time (learning)'] = run_time_learning
        data['Run Time (reconstruction)'] = run_time_recon

        # Get the Path info
        data['Original Path'] = img_orig_path
        data['Reconstructed Path'] = img_recon_path
        data['Corrupted Path'] = img_corrupted_path

        # Get the Parameter info
        data['param: K'] = int(self.K)
        data['param: N'] = int(self.sampler.num_samples)
        data['param: L'] = int(self.L)
        data['param: Iters'] = int(self.iters)

        ### Compute the Errors

        # Get the image vectors
        vec_orig = img_orig.copy().flatten()
        vec_recon = img_recon.copy().flatten()
        vec_corrupt = img_corrupted.copy().flatten()

        # Compute RMSE
        M = len(temp_indicies)
        MSE = ((vec_orig[temp_indicies].astype(int) - vec_recon[temp_indicies].astype(int))**2).sum() / M
        RMSE = np.sqrt(MSE)
        data['RMSE'] = RMSE

        # Compute PSNR
        peak_signal = vec_orig[temp_indicies].max()
        PSNR = 10 * (2 * np.log10(peak_signal) - np.log10(MSE))
        data['PSNR'] = PSNR

        # Compute RMSE Gain
        MSE_corrupt = ((vec_orig.astype(int) - vec_corrupt.astype(int))**2).sum() / len(vec_orig)
        RMSE_corrupt = np.sqrt(MSE_corrupt)
        RMSE_Gain = RMSE - RMSE_corrupt
        data['RMSE Gain'] = RMSE_Gain

        # Compute PNSR Gain
        PSNR_Corrupt = 10 * (2 * np.log10(peak_signal) - np.log10(MSE_corrupt))
        PSNR_Gain = PSNR - PSNR_Corrupt
        data['PSNR Gain'] = PSNR_Gain

        ### Compute SSIM

        mean_orig = vec_orig[temp_indicies].mean()
        mean_recon = vec_recon[temp_indicies].mean()
        var_orig = vec_orig[temp_indicies].var()
        var_recon = vec_recon[temp_indicies].var()
        sigma_xy = ((vec_orig[temp_indicies] - mean_orig) * (vec_recon[temp_indicies] - mean_recon) ).sum() / M

        # Choice of Constants
        #C_l, C_c, C_s = 1,1,1
        # In  this paper, the authors suggest these constants : Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.:‘Image qualityassessment: from error visibility to structural similarity’,IEEE Trans.Image Process., 2004,13, (4), pp. 600–612
        const_k1 = .01
        const_k2 = .03
        const_L = 255
        C_l = (const_k1 * const_L)**2
        C_c = (const_k2 * const_L)**2
        C_s = C_c/2

        luminance_dist = ( 2 * mean_recon * mean_orig + C_l ) / ( mean_recon**2 + mean_orig**2 + C_l)
        contrast_dist = (2 * np.sqrt(var_recon * var_orig) + C_c) / (var_orig + var_recon + C_c)
        structural_comp = (sigma_xy + C_s) / (np.sqrt( var_recon * var_orig) + C_s)
        SSIM = luminance_dist * contrast_dist * structural_comp
        data['Luminance Dist.'] = luminance_dist
        data['Contrast Dist.'] = contrast_dist
        data['Structural Comp.'] = structural_comp
        data['SSIM'] = SSIM



        # Load, Update, and Save the DataFrame
        df = pd.read_csv('EVALUATION_METRICS.csv', index_col=0)
        index = df.index.max()+1
        df.loc[index] = data
        df.to_csv('EVALUATION_METRICS.csv')




