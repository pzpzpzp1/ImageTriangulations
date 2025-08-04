"""
Mathematical utilities
"""
import numpy as np
from typing import Tuple, Optional

def get_triangle_areas(X: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Compute oriented areas of triangle mesh in 2D.
    ALL AREA COMPUTATIONS MUST GO THROUGH THIS IN ORDER TO GUARANTEE CODEBASE AS A WHOLE HAS CONSISTENT AREA SIGN. 

    Args:
        X: Vertex coordinates (n_vertices, 2)
        T: Triangle connectivity (n_triangles, 3)
        
    Returns:
        Triangle areas (n_triangles,)
    """
    v1 = X[T[:, 0], :]  # First vertex of each triangle
    v2 = X[T[:, 1], :]  # Second vertex of each triangle
    v3 = X[T[:, 2], :]  # Third vertex of each triangle
    
    # Compute cross product for 2D area calculation
    e12 = v1 - v2
    e23 = v2 - v3
    cross_z = e12[:, 0] * e23[:, 1] - e12[:, 1] * e23[:, 0]
    tri_areas = cross_z / 2.0    
    return tri_areas

def get_barycentric_sampling_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate barycentric coordinates for sampling within triangles.
    Args:
        n: Number of samples per dimension        
    Returns:
        ws: Barycentric weights (n_samples, 3)
        interior_inds: Boolean mask for interior samples. constant number of layers inwards
    """
    assert n >= 5, "n must be at least 5"
    eps = 1e-10  

    # Create grid of u, v coordinates
    u = np.linspace(0, 1, n)
    v = np.linspace(0, 1, n)
    U, V = np.meshgrid(u, v, indexing='ij')
    W = 1 - U - V
    
    # Keep only points inside the triangle (W >= -eps)
    keep = W >= -eps
    
    # Extract valid barycentric coordinates
    ws = np.column_stack([U[keep], V[keep], W[keep]])
    
    # Identify interior points (not too close to edges)
    # Fixed number of layers away from boundary
    layers_deep = 3
    threshold = (layers_deep - 1) / (n - 1) + eps
    interior_inds = np.all(ws >= threshold, axis=1)
    
    """
        # Visualization snippet - uncomment to plot
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        ax.scatter(ws[:, 0], ws[:, 1], c='lightblue', alpha=0.6, s=10, label='All samples')
        
        # Highlight interior points
        ax.scatter(ws[interior_inds, 0], ws[interior_inds, 1], c='red', s=15, label='Interior samples')
        
        # Add triangle boundary
        triangle = np.array([[0, 0], [1, 0], [0, 1], [0, 0]])
        ax.plot(triangle[:, 0], triangle[:, 1], 'k-', linewidth=2, label='Triangle boundary')
        
        ax.set_xlabel('u (barycentric coordinate)')
        ax.set_ylabel('v (barycentric coordinate)')
        ax.set_title(f'Barycentric Sampling Weights (n={n})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        plt.show()
        plt.savefig('a.png')
    """

    return ws, interior_inds

def initial_grid_mesh(width: int, height: int, initial_horizontal_sampling: int, 
                     rand_slant: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate initial grid mesh.
    
    Args:
        width: Image width
        height: Image height
        initial_horizontal_sampling: Number of horizontal samples
        rand_slant: Whether to randomize triangle orientations
        
    Returns:
        X: Vertex coordinates (n_vertices, 2)
        T: Triangle connectivity (n_triangles, 3)
    """
    eps = 0.00001
    
    x_vals = np.linspace(0, width, initial_horizontal_sampling)
    x_vals[0] = eps
    x_vals[-1] = width - eps
    dx = x_vals[1] - x_vals[0]
    
    initial_vertical_sampling = int(np.ceil(height / dx)) # choose dy to make grid elements close to square
    y_vals = np.linspace(0, height, initial_vertical_sampling)
    y_vals[0] = eps
    y_vals[-1] = height - eps
    
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
    n_rows, n_cols = Y_grid.shape
    
    # Create index mapping 
    inds = np.arange(0, n_rows * n_cols).reshape(n_rows, n_cols)
    
    # grid cell corners
    TL = inds[:-1, :-1]  # Top-left
    TR = inds[:-1, 1:]   # Top-right
    BL = inds[1:, :-1]   # Bottom-left
    BR = inds[1:, 1:]    # Bottom-right
    
    if rand_slant:
        # Randomly choose diagonal orientation for each grid cell
        slant_direction = np.random.rand(*BR.shape) > 0.5
        
        # Triangles with one orientation
        mask1 = slant_direction
        T1 = np.column_stack([
            TL[mask1].flatten(), TR[mask1].flatten(), BL[mask1].flatten(),
            BL[mask1].flatten(), TR[mask1].flatten(), BR[mask1].flatten()
        ]).reshape(-1, 3)
        
        # Triangles with opposite orientation  
        mask2 = ~slant_direction
        T2 = np.column_stack([
            TL[mask2].flatten(), BR[mask2].flatten(), BL[mask2].flatten(),
            BR[mask2].flatten(), TL[mask2].flatten(), TR[mask2].flatten()
        ]).reshape(-1, 3)
        
        T = np.vstack([T1, T2])
    else:
        # Uniform diagonal orientation
        T = np.column_stack([
            TL.flatten(), TR.flatten(), BL.flatten(),
            BL.flatten(), TR.flatten(), BR.flatten()
        ]).reshape(-1, 3)
    
    # Convert to vertex coordinates
    X = np.column_stack([X_grid.flatten(), Y_grid.flatten()])
    
    return X, T


def initial_hex_lattice_mesh(width: int, height: int, initial_horizontal_sampling: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate initial hexagonal lattice mesh.
    
    Args:
        width: Image width
        height: Image height
        initial_horizontal_sampling: Number of horizontal samples
        
    Returns:
        X: Vertex coordinates (n_vertices, 2)
        T: Triangle connectivity (n_triangles, 3)
    """
    assert initial_horizontal_sampling >= 3, "initial_horizontal_sampling must be at least 3"
    eps = 0.00001
    
    # Create horizontal sampling points
    x_vals = np.linspace(0, width, initial_horizontal_sampling)
    x_vals[0] = eps
    x_vals[-1] = width - eps
    dx = x_vals[1] - x_vals[0]
    
    # Calculate vertical sampling
    initial_vertical_sampling = int(np.ceil(height / dx / np.sin(np.pi / 3)))
    y_vals = np.linspace(0, height, initial_vertical_sampling)
    y_vals[0] = eps
    y_vals[-1] = height - eps
    
    # Create mesh grid and offset alternating rows
    X_grid, Y_grid = np.meshgrid(x_vals, np.flip(y_vals))
    X_grid[1::2, :] += dx / 2  # Offset every other row
    
    # Create index mapping
    n_rows, n_cols = Y_grid.shape
    inds = np.arange(0, n_rows * n_cols).reshape(n_rows, n_cols)
    
    # Generate triangles for hexagonal pattern
    # First set of triangles (odd rows)
    TL = inds[0:-1:2, 0:-1]  # Top-left
    TR = inds[0:-1:2, 1:]    # Top-right
    BL = inds[1::2, 0:-1]    # Bottom-left
    BR = inds[1::2, 1:]      # Bottom-right
    
    T1 = np.column_stack([
        np.column_stack([TL.flatten(), TR.flatten(), BL.flatten()]),
        np.column_stack([BL.flatten(), TR.flatten(), BR.flatten()])
    ]).reshape(-1, 3)
    
    # Second set of triangles (even rows)
    if n_rows > 2:
        TL2 = inds[1:-1:2, 0:-1]  # Top-left for even rows
        TR2 = inds[1:-1:2, 1:]    # Top-right for even rows
        BL2 = inds[2::2, 0:-1]    # Bottom-left for even rows
        BR2 = inds[2::2, 1:]      # Bottom-right for even rows
        
        T2 = np.column_stack([
            np.column_stack([TL2.flatten(), TR2.flatten(), BR2.flatten()]),
            np.column_stack([BL2.flatten(), TL2.flatten(), BR2.flatten()])
        ]).reshape(-1, 3)
        
        T_main = np.vstack([T1, T2])
    else:
        T_main = T1
    
    # Add padding triangles on the left edge
    pad_X = np.full_like(Y_grid[1::2, 0], eps)
    pad_Y = Y_grid[1::2, 0]
    pad_inds = np.arange(len(pad_X)) + n_rows * n_cols
    
    pad_U_ind = inds[0::2, 0]
    pad_D_ind = inds[2::2, 0] if len(inds[2::2, 0]) > 0 else []
    pad_R_ind = inds[1::2, 0]
    
    # Create padding triangles
    pad_T1 = np.column_stack([pad_inds, pad_U_ind[:len(pad_inds)], pad_R_ind[:len(pad_inds)]])
    
    if len(pad_D_ind) > 0:
        pad_T2 = np.column_stack([pad_inds[:len(pad_D_ind)], pad_R_ind[:len(pad_D_ind)], pad_D_ind])
        pad_T = np.vstack([pad_T1, pad_T2])
    else:
        pad_T = pad_T1
    
    # Combine all triangles
    T = np.vstack([T_main, pad_T])
    T = T[:, [0, 2, 1]]
    
    # Combine all vertices
    X = np.vstack([
        np.column_stack([X_grid.flatten(), Y_grid.flatten()]),
        np.column_stack([pad_X, pad_Y])
    ])
    
    # Clamp vertices to image bounds
    X[X[:, 0] > width, 0] = width - eps
    
    return X, T

def clip_verts(X: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Clip vertices to image boundaries.
    
    Args:
        X: Vertex coordinates (n_vertices, 2)
        width: Image width
        height: Image height
        
    Returns:
        Clipped vertex coordinates
    """
    eps = .00001;
    X_clipped = X.copy()
    X_clipped[:, 0] = np.clip(X_clipped[:, 0], eps, width-eps)
    X_clipped[:, 1] = np.clip(X_clipped[:, 1], eps, height-eps)
    return X_clipped


def sample_image(img: np.ndarray, sample_points: np.ndarray) -> np.ndarray:
    """
    takes in rgb image and sample locations. returns sampled values.
    samplePoints must be size (n,m,...,2)
    sampleVals will be size (n,m,...,3)
    """
    nd = sample_points.ndim
    assert sample_points.shape[nd-1] == 2

    # saliency map doesnt have rgb. need to expand last dim.
    if img.ndim == 2: 
        img = img[:,:,np.newaxis]
    
    # For non-patterned generic debugging: sample_points = np.random.randn(*sample_points.shape)
    samplexvals = sample_points[...,0]
    sampleyvals = sample_points[...,1]

    xind = np.ceil(samplexvals).astype(int)-1
    yind = np.ceil(sampleyvals).astype(int)-1
    
    # Clip sampling to ensure in bounds
    xind = np.minimum(xind, img.shape[1]-1)
    xind = np.maximum(xind, 0)
    yind = np.minimum(yind, img.shape[0]-1)
    yind = np.maximum(yind, 0)
    
    sample_vals = img[yind, xind, :]
    return sample_vals


def get_sample_points_from_barycentric_weights(ws: np.ndarray, X: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Convert barycentric coordinates to world coordinate samples for all triangles.
    
    Args:
        ws: Barycentric weights (n_samples, 3)
        X: Vertex coordinates (n_vertices, 2)
        T: Triangle connectivity (n_triangles, 3)
    Returns:
        Sample points (n_samples, n_triangles, 2)
    """
    tri_verts = X[T]  # (n_triangles, 3, 2)
    sample_points = np.einsum('ij,kjl->ikl', ws, tri_verts)  # (n_samples, n_triangles, 2)
    
    return sample_points

def vecnorm(x: np.ndarray, ord: int = 2, axis: int = -1) -> np.ndarray:
    """
    Vector norm    
    Args:
        x: Input array
        ord: Order of the norm
        axis: Axis along which to compute norm
    Returns:
        Vector norms
    """
    return np.linalg.norm(x, ord=ord, axis=axis, keepdims=True)
