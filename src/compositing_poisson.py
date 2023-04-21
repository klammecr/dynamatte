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
def poisson_sparse_matrix(points):
    # N = number of points in mask
    N = len(points)
    A = lil_matrix((N,N))
    # Set up row for each point in mask
    for i,index in enumerate(points):
        # Should have 4's diagonal
        A[i,i] = 4
        # Get all surrounding points
        for x in get_surrounding(index):
            # If a surrounding point is in the mask, add -1 to index's
            # row at correct position
            if x in points:
                j = points.index(x)
                A[i,j] = -1
    return A

def poisson_blend(source, target, mask):
    indicies = mask_indicies(mask)
    N = len(indicies)

    # A matrix will be constraints on gradients
    A = poisson_sparse_matrix(indicies)
    b = np.zeros(N)

    for i, idx in enumerate(indicies):
        # Pixel values, find gradients
        b[i] = lapl_at_index(source, idx)

        # If on boundry, add in target intensity
        # Creates constraint lapl source = target at boundary
        if mask[idx]:
            for pt in get_surrounding(idx):
                if edge(pt, mask):
                        # Blend in background
                        b[i] += target[pt]

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