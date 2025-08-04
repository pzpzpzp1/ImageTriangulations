"""
Approximator classes for non-conforming constant and linear per triangle approimator methods.
"""
import numpy as np
from typing import Tuple, Dict, Any, Optional
from mesh import Mesh
from math_utils import (
    get_barycentric_sampling_weights, 
    get_sample_points_from_barycentric_weights,
    sample_image,
    vecnorm
)
from sampling_utils import (
    sample_v_dot_n_dl,
    get_tri_edge_sample_points
)

def move_triangle_vert_values_to_vertex_values(mesh: Mesh, tri_vert_values: np.ndarray) -> np.ndarray:
    """
    Move values from triangle vertices to mesh vertices by accumulation.    
    Args:
        mesh: Mesh object
        tri_vert_values: Values at triangle vertices (nT, 3, ...)
        
    Returns:
        Values at mesh vertices (nX, ...)
    """
    nX = mesh.nX
    nT = mesh.nT
    maxdim = tri_vert_values.ndim
    extra_dims = tri_vert_values.shape[2:] # shape of fields to aggregate
    
    # flatten field dimensions
    flattened = tri_vert_values.reshape(nT, 3, -1)
    vert_vals = np.zeros((nX, flattened.shape[2]))
    
    # Accumulate values for each field dimension. note: fields are not meant to be very large so forlooping is ok
    for i in range(flattened.shape[2]):
        # accumulate triangule values to vertices
        triangle_values = flattened[:, :, i].flatten()
        vertex_indices = mesh.T.flatten()
        vert_vals[:, i] = np.bincount(vertex_indices, weights=triangle_values, minlength=nX)
    
    # reshape to original dims
    return vert_vals.reshape(nX, *extra_dims)

def slip_conditions(mesh: Mesh, grad: np.ndarray) -> np.ndarray:
    """
    Apply slip boundary conditions by zeroing gradient components that violate boundaries.    
    Args:
        mesh: Mesh object
        grad: Gradient array (nX, 2)
    Returns:
        Modified gradient with boundary conditions applied
    """
    grad_modified = grad.copy()
    grad_modified[mesh.is_x_vert, 1] = 0  # Zero y-component for x-boundary vertices
    grad_modified[mesh.is_y_vert, 0] = 0  # Zero x-component for y-boundary vertices
    return grad_modified

def constant_get_colors_from_samples(sample_vals: np.ndarray, interior_inds: np.ndarray, 
                                   saliency_samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute colors per triangle from sampled values.
    Returns two versions: using all samples, and using only interior samples.
    
    Args:
        sample_vals: Sampled color values (n_samples, nT, 3)
        interior_inds: Boolean mask for interior samples (n_samples,)
        saliency_samples: Saliency weights at sample points (n_samples, nT, 1)
        
    Returns:
        colors: Colors using all samples (nT, 3)
        colors_alt: Colors using only interior samples (nT, 3)
    """
    # Compute weighted mean using all samples
    normalizer = np.sum(saliency_samples, axis=0)  # (nT,1)
    colors = np.sum(sample_vals * saliency_samples, axis=0) / normalizer # (nT, 3)
    
    # Compute weighted mean using only interior samples
    normalizer_alt = np.sum(saliency_samples[interior_inds, :], axis=0)  # (nT,1)
    colors_alt = np.sum(sample_vals[interior_inds, :, :] * saliency_samples[interior_inds, :, :], axis=0) / normalizer_alt  # (nT, 3)
    
    return colors, colors_alt


class ConstantApproximator:
    """Constant (degree 0) approximation method."""
    def compute_energy(self, img: np.ndarray, mesh: Mesh, integral_1d_samples: int, 
                      salmap: Optional[np.ndarray],
                      return_gradient: bool,
                      ):
        """
        Compute energy for constant approximation.
        
        Args:
            img: Input image (height, width, 3)
            mesh: Mesh object
            integral_1d_samples: Number of samples for integration along edges
            salmap: Saliency map (height, width)
            return_gradient: Whether to return the gradient
            
        Returns:
            extra: Extra computation results
            energy: Total energy
            colors: Triangle vertex colors (nT, 3)
            gradient: Gradient (nX, 2)
        """
        if salmap is None:
            salmap = np.ones(img.shape[:2])
        
        extra = {}
        X, T = mesh.X, mesh.T
        nT = T.shape[0]
        
        # Generate sample locations in barycentric coords
        ws, interior_inds = get_barycentric_sampling_weights(integral_1d_samples)
        sample_points = get_sample_points_from_barycentric_weights(ws, X, T)
        n = ws.shape[0]
        dA = mesh.tri_areas / n
        
        # Perform sampling of image and saliency map
        f_triangle = sample_image(img, sample_points).astype(float)
        sal_triangle = sample_image(salmap, sample_points).astype(float)
        
        # Get colors per triangle. In constant case this is a saliency weighted color average.
        colors, colors_alt = constant_get_colors_from_samples(
            f_triangle, interior_inds, sal_triangle
        )
        extra['colors_alt'] = colors_alt # average computed using only interior samples is better for final render. less flickering
        
        # Compute energy integrated over triangles
        color_diff = f_triangle - colors[np.newaxis, :, :]  # (n, nT, 3)
        extra['per_triangle_rgb_error'] = np.sum(sal_triangle * color_diff**2, axis=0) * dA[:,np.newaxis]
        energy = np.sum(extra['per_triangle_rgb_error'])
        
        if not return_gradient:
            return extra, energy, colors
        else:
            # compute velocity-normal dot products
            vndl = sample_v_dot_n_dl(mesh, integral_1d_samples)
            
            # Reynold's thm in this case results in triangle boundary integrals. Sample image and saliency on edges.
            edge_sample_points = get_tri_edge_sample_points(mesh, integral_1d_samples)
            f_tri_edges = sample_image(img, edge_sample_points).astype(float)
            sal_tri_edges = sample_image(salmap, edge_sample_points).astype(float)
            
            # Create various (int)egral components needed to compute gradient
            # f - function, s - saliency, v - velocity, n - normal
            int_svn_dl = np.einsum('tnxs,tnxb->ts', vndl, sal_tri_edges) # (nT, 6)
            int_s_dA = np.sum(sal_triangle, axis=0) * dA.reshape(nT,1)  # (nT, 1)
            int_fs_dA = colors * int_s_dA  # (nT, 3)
            int_fs_dA2 = int_fs_dA**2  # (nT, 3)            
            int_sfvn_dl = np.einsum('tnxs,tnxz,tnxy->tsy', vndl, sal_tri_edges, f_tri_edges) # nT, 6, 3
            
            # Build gradient on triangle vertices. Can be vectorized this way per triangle. 
            # gradPrep: nT(tris per mesh), 3(verts per tri), 2(xy coords), 3(rgb channels)
            # chain rule results in two components
            A = 2 * int_s_dA[:, np.newaxis] * int_fs_dA[:, np.newaxis, :] * int_sfvn_dl
            B = -int_fs_dA2[:, np.newaxis, :] * int_svn_dl[:, :, np.newaxis]
            grad_prep = (-(A + B) / (int_s_dA[:, np.newaxis]**2)).reshape(nT, 3,2, 3) # 6 got split into 3x2. REMEMBER: python is ROW MAJOR. IT IS NOT 2x3!
            extra['grad_prep'] = grad_prep # nT, (3(verts) x 2(xy)), 3(rgb)
            
            # Accumulate per triangle gradient values to vertices. correct by linearity of derivative.
            vert_grad = move_triangle_vert_values_to_vertex_values(mesh, extra['grad_prep'])
            
            # Sum over RGB channels
            gradient = np.sum(vert_grad, axis=2)
            
            # Apply boundary conditions
            gradient = slip_conditions(mesh, gradient)
        
        return extra, energy, colors, gradient

class LinearApproximator:
    """Linear (degree 1) approximation method."""
    
    def compute_energy(self, img: np.ndarray, mesh: Mesh, integral_1d_samples: int,
                      salmap: Optional[np.ndarray] = None) -> Tuple[Dict[str, Any], float, np.ndarray, Optional[np.ndarray]]:
        raise NotImplementedError()

class Approximator:
    """
    Factory class for creating approximators of different degrees.
    Bit overengineered because higher degree approximators were left unimplemented. But this leaves room for them, and establishes the interface.
    """
    
    def __init__(self, degree: int):
        """
        Initialize approximator with given polynomial degree.
        
        Args:
            degree: Polynomial degree (0 for constant, 1 for linear, n for nodal polynomial bases)
        """
        self.degree = degree
        
        if degree == 0:
            self.apx = ConstantApproximator()
        elif degree == 1:
            self.apx = LinearApproximator()
        else:
            raise ValueError(f"Unknown approximator degree: {degree}")
    
    def compute_energy(self, img: np.ndarray, mesh: Mesh, integral_1d_samples: int,
                      salmap: Optional[np.ndarray], return_gradient: bool) -> Tuple[Dict[str, Any], float, np.ndarray, Optional[np.ndarray]]:
        """
        Compute energy using the selected approximation method.
        
        Args:
            img: Input image
            mesh: Mesh object
            integral_1d_samples: Number of samples for integration  
            salmap: Saliency map
            
        Returns:
            Tuple of (extra, energy, colors, gradient)
        """
        return self.apx.compute_energy(img, mesh, integral_1d_samples, salmap, return_gradient)
