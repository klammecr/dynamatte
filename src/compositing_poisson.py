# Third Party
import numpy as np
import scipy.sparse as spmtx
import scipy
import cv2

"""
Poisson image blending using linear system of equations.

Author: Christopher Klammer
Affiliation: Carnegie Mellon University
Email: cklammer@andrew.cmu.edu
Inspred from: https://github.com/willemmanuel/poisson-image-editing
"""

"""
Overview:
    1. Preprocess source, target, and mask
        - Mask, source and source image must match target
        - If the mask extends to the border, add
    2. Generate sparse matrix (quicker to solve)
    3. Blend the color channels one at a time
""" 
# Third Party
import numpy as np
from scipy.sparse import linalg as linalg
from scipy.sparse import lil_matrix as lil_matrix

def is_valid_idx(pt, mask):
    return pt[0] >= 0 and pt[1] >= 0 and pt[0] < mask.shape[0] and pt[1] < mask.shape[1]

# See if we are on the edge of the mask
def edge(index, mask):
    if is_valid_idx(index, mask) and mask[index]:
        for pt in get_surrounding(index):
            if is_valid_idx(pt, mask) and not mask[pt]:
                return True
    return False

# Apply the Laplacian operator at a given index
def lapl_at_index(source, index):
    i,j = index
    val = 0
    if i+1 < source.shape[0]:
        val += source[i,j] - (1 * source[i+1, j])
    if i-1 >= 0:
        val += source[i,j] - (1 * source[i-1, j])
    if j+1 < source.shape[1]:
        val += source[i,j] - (1 * source[i, j+1])
    if j-1 >= 0:
        val += source[i,j] - (1 * source[i, j-1])
    return val

# Find the indicies of omega, or where the mask is 1
def mask_indicies(mask):
    nonzero = np.nonzero(mask)
    return list(zip(nonzero[0], nonzero[1]))

# Get indicies above, below, to the left and right
def get_surrounding(index):
    i,j = index
    return [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]

# Create the A sparse matrix
def compute_A_b(points, source, target, mask, alpha):
    # N = number of points in mask
    N = len(points)
    A = lil_matrix((N,N))
    b = np.zeros(N)

    # Set up row for each point in mask
    for i,index in enumerate(points):
        # For four neighbors
        A[i,i] = 4

        # Default is you will have the laplacian
        b[i] = lapl_at_index(source, index)

        # If on boundry, add in target intensity
        # Creates constraint lapl source = target at boundary
        for pt in get_surrounding(index):
            if pt in points:
                j = points.index(pt)
                A[i,j] = -1

            # Edge of the mask
            if edge(index, mask):
                if is_valid_idx(pt, mask) and not mask[pt]:
                    # Blend in background
                    #b[i] += target[index]
                    b[i] = alpha[index] * b[i] + (1 - alpha[index]) * lapl_at_index(target, index)

    return A, b

def poisson_blend(source, target, mask, alpha):
    # A matrix will be constraints on gradients
    indicies = mask_indicies(mask)
    A, b = compute_A_b(indicies, source, target, mask, alpha)

    # Solve for x, unknown intensities
    x = linalg.cg(A, b)
    # Copy target photo, make sure as int
    composite = np.copy(target).astype(int)
    # Place new intensity on target at given index
    for i,index in enumerate(indicies):
        composite[index] = x[0][i]
    return composite

# Naive blend, puts the source region directly on the target.
# Useful for testing
def preview(source, target, mask):
    return (target * (1.0 - mask)) + (source * (mask))

"""Poisson image editing.
"""

import numpy as np
import cv2
import scipy.sparse
from scipy.sparse.linalg import spsolve

from os import path

def laplacian_matrix(n, m):
    """Generate the Poisson matrix. 
    Refer to: 
    https://en.wikipedia.org/wiki/Discrete_Poisson_equation
    Note: it's the transpose of the wiki's matrix 
    """
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(4)
    mat_D.setdiag(-1, 1)
        
    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()
    
    mat_A.setdiag(-1, 1*m)
    mat_A.setdiag(-1, -1*m)
    
    return mat_A


def poisson_edit(source, target, mask, offset):
    """The poisson blending function. 
    Refer to: 
    Perez et. al., "Poisson Image Editing", 2003.
    """

    # Assume: 
    # target is not smaller than source.
    # shape of mask is same as shape of target.
    y_max, x_max = target.shape[:-1]
    y_min, x_min = 0, 0

    x_range = x_max - x_min
    y_range = y_max - y_min
        
    M = np.float32([[1,0,offset[0]],[0,1,offset[1]]])
    source = cv2.warpAffine(source,M,(x_range,y_range))
        
    mask = mask[y_min:y_max, x_min:x_max]    
    mask[mask != 0] = 1
    #mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
    
    mat_A = laplacian_matrix(y_range, x_range)

    # for \Delta g
    laplacian = mat_A.tocsc()

    # set the region outside the mask to identity    
    for y in range(1, y_range - 1):
        for x in range(1, x_range - 1):
            if mask[y, x] == 0:
                k = x + y * x_range
                mat_A[k, k] = 1
                mat_A[k, k + 1] = 0
                mat_A[k, k - 1] = 0
                mat_A[k, k + x_range] = 0
                mat_A[k, k - x_range] = 0

    # corners
    # mask[0, 0]
    # mask[0, y_range-1]
    # mask[x_range-1, 0]
    # mask[x_range-1, y_range-1]

    mat_A = mat_A.tocsc()

    mask_flat = mask.flatten()    
    for channel in range(source.shape[2]):
        source_flat = source[y_min:y_max, x_min:x_max, channel].flatten()
        target_flat = target[y_min:y_max, x_min:x_max, channel].flatten()        

        #concat = source_flat*mask_flat + target_flat*(1-mask_flat)
        
        # inside the mask:
        # \Delta f = div v = \Delta g       
        alpha = 1
        mat_b = laplacian.dot(source_flat)*alpha

        # outside the mask:
        # f = t
        mat_b[mask_flat==0] = target_flat[mask_flat==0]
        
        x = spsolve(mat_A, mat_b)
        #print(x.shape)
        x = x.reshape((y_range, x_range))
        #print(x.shape)
        x[x > 255] = 255
        x[x < 0] = 0
        x = x.astype('uint8')
        #x = cv2.normalize(x, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        #print(x.shape)

        target[y_min:y_max, x_min:x_max, channel] = x

    return target

def main():    
    scr_dir = '.'
    out_dir = scr_dir
    source = cv2.imread(path.join(scr_dir, "source.jpg")) 
    target = cv2.imread(path.join(scr_dir, "target.jpg"))    
    mask = cv2.imread(path.join(scr_dir, "mask.jpg"), 
                      cv2.IMREAD_GRAYSCALE) 
    offset = (0,0)
    result = poisson_edit(source, target, mask, offset)

    cv2.imwrite(path.join(out_dir, "possion1.png"), result)
    

if __name__ == '__main__':
    main()