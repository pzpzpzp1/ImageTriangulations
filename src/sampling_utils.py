"""
Sampling utility functions
"""
import numpy as np
from typing import Tuple, Optional, Dict, Any
from mesh import Mesh
from math_utils import get_barycentric_sampling_weights

def load_v_sampling_matrix():
    """
    Constructs tensors representing how triangle vertex perturbations map to edge perturbations. Accounts for all velocities of a triangle's vertices    
    Returns constant tensors
    * cached_zf: forward. Triangle is 0,1,2. 
    * cached_zb: backward. Triangle is 2,0,1. 

    Size of output:
    1 placeholder for number of samples per edge, 
    3 edges per triangle, 
    6 perturbbations dofs for 3 vertices, 3 = 3(verts_per_tri) x 2(xy) because row major. NOT 2 x 3
    2 xy coords for output response. corresponds to edge normals
    """

    if not hasattr(load_v_sampling_matrix, 'cached_zf') or not hasattr(load_v_sampling_matrix, 'cached_zb'):
        load_v_sampling_matrix.cached_zf = np.zeros((1, 3, 6, 2))
        load_v_sampling_matrix.cached_zf[0, 0, 0, 0] = 1  # vertex 1 is part of edge 3 and 1.
        load_v_sampling_matrix.cached_zf[0, 0, 1, 1] = 1  # vertex 1 is part of edge 3 and 1.
        load_v_sampling_matrix.cached_zf[0, 1, 2, 0] = 1  # vertex 2 is part of edge 1 and 2.
        load_v_sampling_matrix.cached_zf[0, 1, 3, 1] = 1  # vertex 2 is part of edge 1 and 2.
        load_v_sampling_matrix.cached_zf[0, 2, 4, 0] = 1  # vertex 3 is part of edge 2 and 3.
        load_v_sampling_matrix.cached_zf[0, 2, 5, 1] = 1  # vertex 3 is part of edge 2 and 3.
        
        load_v_sampling_matrix.cached_zb = np.zeros((1, 3, 6, 2))
        load_v_sampling_matrix.cached_zb[0, 2, 0, 0] = 1  # vertex 1 is part of edge 3 and 1.
        load_v_sampling_matrix.cached_zb[0, 2, 1, 1] = 1  # vertex 1 is part of edge 3 and 1.
        load_v_sampling_matrix.cached_zb[0, 0, 2, 0] = 1  # vertex 2 is part of edge 1 and 2.
        load_v_sampling_matrix.cached_zb[0, 0, 3, 1] = 1  # vertex 2 is part of edge 1 and 2.
        load_v_sampling_matrix.cached_zb[0, 1, 4, 0] = 1  # vertex 3 is part of edge 2 and 3.
        load_v_sampling_matrix.cached_zb[0, 1, 5, 1] = 1  # vertex 3 is part of edge 2 and 3.
    
    return load_v_sampling_matrix.cached_zf, load_v_sampling_matrix.cached_zb


def sample_v_dot_n_dl(mesh: Mesh, n: int) -> np.ndarray:
    """
    vn will be (nT n 3 6). 3 for number of edges. 6 for number of velocity values per triangle.
    """
    T = mesh.T
    nT = mesh.nT
    ws = np.linspace(1, 0, n)
    TEN = mesh.triangle_edge_normals
    
    # load cached constant v sampling indexing tensor. 
    vsamplerf, vsamplerb = load_v_sampling_matrix()
    ws.reshape(-1,1,1,1)
    vsamples = vsamplerf * ws.reshape(-1,1,1,1) + vsamplerb * np.flip(ws).reshape(-1, 1, 1, 1)
    
    # (vsamples): 1  n 3 6 2
    # (TEN):      nT 1 3 1 2
    vsamples = vsamples.reshape(1, n, 3, 6, 2)
    TEN = TEN.reshape(nT, 1, 3, 1, 2)
    # (vn): nT n 3 6
    vn = np.sum(vsamples * TEN, axis=4)
    
    # multiply with dl to get integrands
    edge_lengths = mesh.triangle_edge_lengths.reshape(nT, 1, 3, 1)
    vn_dl = vn * edge_lengths / n
    
    return vn_dl

def get_tri_edge_sample_points(mesh: Mesh, n: int) -> np.ndarray:
    """
    N samples per edge per triangle
    edgeSampleedge_sample_pointsPoints: [nT, n, 3(edges), 2(xy)]
    """
    X = mesh.X
    T = mesh.T
    nT = mesh.nT
    
    edge_ws = np.linspace(0, 1, n).reshape(1,n,1,1)
    v123 = X[T, :].reshape(nT, 1, 3, 2)
    v231 = X[T[:, [1, 2, 0]], :].reshape(nT, 1, 3, 2)
    edge_sample_points = v123 * (1 - edge_ws) + v231 * edge_ws
    
    return edge_sample_points
