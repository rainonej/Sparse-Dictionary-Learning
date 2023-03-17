# Sparse-Dictionary-Learning
 
This repo demonstrates the capabilities of Sparse Dictionary Learning 
on image reconstruction and denoising. 

### Mathematical Overview

Each image is broken up into "patches", usually 8x8 block. 
This are the fundamental objects we work with.
Each patch can be represented by a single vector 
$v\in \mathbb{R}^d$ where $d=(8\times 8)=64$. (Here we are assuming our image is in gray scale, 
however this works perfectly fine in color where $d = (8\times 8\times 3)$.)

Select a set $D:=\{ u_i \}_{i=1}^K \subset \mathbb{R}^d$ of unit vectors, called "atoms". 
It is important that $K>d$, so the set may be spanning (almost always is spanning) 
but is never a basis. The set $D$ is called a "dictionary" and typically is viewed 
as a $d\times K$ matrix. $\alpha$ 

Each patch $8\times 8$ patch has a vector representation $\vec{y} \in \mathbb{R}^d$. 
This vector $\vec{y}$ has a "sparse representation" $\vec{x}$. 

$$ \vec{y} \approx \vec{x} := D \vec{\alpha}$$

The best representation of $\vec{y}$ is the vector $\vec{\alpha}$ which minimizes

$$ ||\vec{y} - D \vec{\alpha} ||_2$$

Of course since $D$ spans $\mathbb{R}^d$, this is an overdetermined problem. 
We add a constraint that the $L^0$-norm of $\vec{\alpha}$ is bounded by a 
constant $L$. 
This gives the optimization problem 

$$ ||\vec{y}_i - D \vec{\alpha}_i ||_2 \quad s.t. ||\vec{\alpha}_i||_0\leq L$$

or rather: Find the $D$ and $A=[ \vec{\alpha}_1, \dots, \vec{\alpha}_N ]$ so that 

$$ || Y - D A ||_2$$ 

is minimized for $Y = [\vec{y}_1, \dots, \vec{y}_N]$,  subject to the contraint

$$ ||\vec{\alpha}||_0\leq L.$$


### Algorithms

Analytically finding optimal sparse representation requires preforming ${K \choose L}$ 
computations, which is unreasonable. In stead, we use an approximate form. 
There are many algorithms that do this, but we use Orthogonal Matching Pursuit and Matching Pursuit. 

Finding the optimal dictionary is equally difficult. We are using that k-SVD algorithm, 
first proposed in this paper: https://legacy.sites.fas.harvard.edu/~cs278/papers/ksvd.pdf. 
We are slightly modifying the algorithm by replacing unused atoms as soon as they are skipped over. 
We are replacing them with the worst respresented signal. This method was examined and 
determined to be optimimal in the following paper: https://cs.unibuc.ro//~pirofti/papers/Irofti16_AtomReplacement.pdf.

### Error Determination

Because the algorithms are designed to minimize the $L^2$-distance between the signal and its 
sparse representation, that is the error metric we will be using (after normalizing by an appropriate constant). 
This is also the metric used in most papers on the subject (including those linked in this ReadMe).

### Stochastic Partial Image Reconstruction

Reconstructing an image is extremely costly, since we have to preform approximatly 
1 sparse coding operation per pixel in the image. However, every patch contains the same 
amount of information. In fact, when the patches are randomly sampled, 
the first 20% of the patches contain 90-95% of the information. This was confirmed experimentally 
in this Classes.ipynb notebook. We present the results below. 

![Graph](SPIR_graph.jpg)

This leads to the creation of the Stochastic Partial Image Reconstruction (SPIR) algorithm
which randomly samples 20% of the patches in order to reconstruct the image. 
It then gives an accurate (slightly higher) estimate of the error which a true image
reconstruction would have. 

# Usage

First you must train the dictionary. To do that, follow the following steps. 

#### Dictionary Training

Have a parameters file saved as a .yaml file. 
You can edit these values as you see fit. 
You will be asked to enter the path to this fill after --training_parameters.
```azure
# This is the parameters.yaml file
N : 300 # The batch size when random sampling your training data.
K : 150 # Number of atoms in the dictionary.
I : 10 # Number of iterations to run the k-SVD algorithm to train your dictionary. Values are typically between 0 and 20.
L : 10 # Number of atoms to use in sparse coding. Inscreased values increase preformance at the cost of run time.
Patch_Shape : [8,8] # This size of the patches in pixels. If the image has color, set it equal to [8,8,3].
Partial : False # If False, you will get the full image reconstruction algorithm. If True, you will get the Stochastic Partial Image Reconstruction (SPIR) algorithm. This will take about 20% of the time as the full agorithm, be about 90-90% as accurate, but will most likely have dead pixels on the sides.
```

Have a image_paths.yaml file which contains the paths to the training images.
You will be asked to enter the path to this file after --image_paths
```azure
# This is the image_paths.yaml file. It contains a list of all the paths that the training images will be pulled from.
paths:
    - Compressed Images/cheese_board.jpg
    - octopus_test_image.jpg
```

Know the path you want you dictionary to end up at. This will be pickled, 
so it is suggested that the file path end in '.pkl'. 
You will be asked to enter this path after --output_dictionary_path

Execute the code
```azure
train_dictionary.py --training_images image_paths.yaml --training_params params.yaml --output_dictionary_path test_dictionary.pkl
```

Your dictionary is now stored as a pickle file in the file path you entered.

#### Image Reconstruction

Recall the .pkl path that your dictionary is stored in. You can use the one you just 
created, or use a different one. You will be asked to enter this path after
--input_dictionary_path

Have a parameters.yaml file. This is the EXACT same file as before.
You will be asked to enter the path to this fill after --reconstruction_parameters.

```azure
# This is the parameters.yaml file
N : 300 # The batch size when random sampling your training data.
K : 150 # Number of atoms in the dictionary.
I : 10 # Number of iterations to run the k-SVD algorithm to train your dictionary. Values are typically between 0 and 20.
L : 10 # Number of atoms to use in sparse coding. Inscreased values increase preformance at the cost of run time.
Patch_Shape : [8,8] # This size of the patches in pixels. If the image has color, set it equal to [8,8,3].
Partial : False # If False, you will get the full image reconstruction algorithm. If True, you will get the Stochastic Partial Image Reconstruction (SPIR) algorithm. This will take about 20% of the time as the full agorithm, be about 90-90% as accurate, but will most likely have dead pixels on the sides.
```

Know the path of the input image. This will be the image you want reconstructed. 
You will be asked to enter it after --input_image_path.

Know the path where you want your reconstructed image to go.
You will be asked to enter it after --output_image_path.

Run the code
```azure
image_reconstruction.py --input_dictionary_path test_dictionary.pkl --reconstruction_params params.yaml --input_image_path small_gray_dinner.jpg --output_image_path test_image.jpg
```
#### Example 
There is an example of how to follow these instruction in Example.ipynb

## Tests and Test Results
There are plenty of tests, test results, and images in the notebook Test.ipynb

## Process
The process of defining all of the Classes and code is recorded in detail in 
Classes.ipynb

## Code
All of the actual code is stored in Classes.py.
Note that this contains the exact same information as in Classes.ipynb. 
The only difference is that the note book will be easier to read if you want the 
full story, and the python file will be easier to read if you know what you're looking for. 

## Example Results: Comparison

Here are the results of a few tests. Note that these tests are purely to get an idea of what the algorithms are capable of. A full battery of tests will be preformed in another notebook.

ALl tests are run by compressing and reconstructing a single image. The algorithm is trained and tested on the same image.

I = number of iterations of the dictionary learning algorithm (k-SVD)
K = number of atoms
L = max number of atoms allowed in a sparse representation
N = number of random samples provided (batch size) at each step

### Dictionary Learning Iterations
How many iterations (I = num_iters) of the dictionary learning (k-SVD) algorithm do we need to run? Note that at I=0 our atoms are initialized as random patches of our image.

| I = 0 | I = 1 | I = 10 | Original |
|-------|-------|--------|----------|
| <img alt="Recon 0" height="300" src="recon0_small_gray_dinner.png" width="400"/> | <img alt="Recon 1" height="300" src="recon_small_gray_dinner.png" width="400"/> | <img alt="Recon 10" height="300" src="recon10_small_gray_dinner.png" width="400"/> | <img alt="Original" height="300" src="small_gray_dinner.png" width="400"/> |

Notice how there isn't much of a difference.

### Batch Size
How many samples should I provide? Both experiments were preformed with I=10

 N=100  | N=500                                          | Original |
-------|------------------------------------------------|----------|
| <img alt="Recon 1" height="300" src="recon10_small_gray_dinner.png" width="400"/> | <img alt="Recon 10" height="300" src="recon10K500_small_gray_dinner.png" width="400"/> | <img alt="Original" height="300" src="small_gray_dinner.png" width="400"/> |

Again, not much of a change.

### Atoms in the Sparse Representation
How many atoms should I use in the sparse representation?

 L=5 | L=20                                                                                   | Original |
-------|----------------------------------------------------------------------------------------|----------|
| <img alt="Recon 1" height="300" src="recon10K500_small_gray_dinner.png" width="400"/> | <img alt="Recon 10" height="300" src="reconI10L20_small_gray_dinner.png" width="400"/> | <img alt="Original" height="300" src="small_gray_dinner.png" width="400"/> |

We finally have a dramatic improvement. It seems that of all the parameters, L is by far the most important.
