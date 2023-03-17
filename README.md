# Sparse-Dictionary-Learning
 
This repo demonstrates the capabilities of Sparse Dictionary Learning 
on image reconstruction and denoising. 

### Mathematical Overview

Each image is broken up into "patches", usually 8x8 block. 
This are the fundamental objects we work with.
Each patch can be represented by a vector $v$ 
$$x+y$$

The algorithms used:

k-SVD
https://legacy.sites.fas.harvard.edu/~cs278/papers/ksvd.pdf

Atom Replacement
https://cs.unibuc.ro//~pirofti/papers/Irofti16_AtomReplacement.pdf

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


