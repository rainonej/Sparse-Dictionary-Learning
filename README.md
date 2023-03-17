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

This leads to the creation of the Stochastic Partial Image Reconstruction (SPIR) algorithm
which randomly samples 20% of the patches in order to reconstruct the image. 
It then gives an accurate (slightly higher) estimate of the error which a true image
reconstruction would have. 





Rarity Enhancement (not using yet)
https://arxiv.org/ftp/arxiv/papers/1305/1305.0871.pdf

Online Coding (not using yet)
https://www.di.ens.fr/~fbach/mairal_icml09.pdf

<img alt="alt text" height="300" src="small_gray_dinner.png" width="400"/>
<img alt="alt text" height="300" src="small_gray_dinner.png" width="400"/>

| Title 1 | Title 2 |
|---------|---------|
| <img alt="Caption 1" height="300" src="small_gray_dinner.png" width="400"/> | <img alt="Caption 2" height="300" src="small_gray_dinner.png" width="400"/> |

| Title 1 | Title 2                                     |
|---------|---------------------------------------------|
| <img alt="Caption 1" height="300" src="small_gray_dinner.png" width="400"/> | <img alt="Caption 2" height="300" src="small_gray_dinner.png" width="400"/> |

Caption 1: This is the caption for the first image.

Caption 2: This is the caption for the second image.


## Cost Function
$$R := \bigcup_{|I|\leq L} span\{\vec{v}_{I_1}, \dots, \vec{v}_{I_L}\}$$

$x+y$



#%% md
## Comparison

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
