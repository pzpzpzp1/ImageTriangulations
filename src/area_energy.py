"""
Triangle saliency weighted area energy and gradient.
"""
import numpy as np
from typing import Tuple, Optional
from mesh import Mesh
from math_utils import (
    get_barycentric_sampling_weights,
    get_sample_points_from_barycentric_weights, 
    sample_image
)
from approximator import move_triangle_vert_values_to_vertex_values, slip_conditions


def get_area_energy(mesh: Mesh, salmap: Optional[np.ndarray], return_gradient: bool) -> Tuple[float, np.ndarray]:
    """
    Triangle saliency weighted area energy and gradient.
    
    The energy is the negative log of triangle areas, encouraging larger triangles. Infinite barrier from collapse.
    Differs from mesh._compute_area_gradient due to saliency map
    
    Args:
        mesh: Mesh object containing vertices and triangles
        salmap: Optional saliency map (height, width)
        return_gradient: Whether to return the gradient
    Returns:
        energy: Area energy (scalar)
        gradient: Gradient with respect to vertex positions (nX, 2)
    """
    nT = mesh.nT
    energy = np.sum(-np.log(mesh.tri_areas))
    
    if not return_gradient:
        return energy
    else:
        # Compute gradient preparation: -dA/dt / A for each triangle
        # mesh.dA_dt has shape (nT, 6) where 6 = 2(xy) * 3(vertices)
        grad_prep = (-mesh.dA_dt.reshape(nT, 2, 3) / mesh.tri_areas[:, np.newaxis, np.newaxis]).transpose(0, 2, 1) # (nT, 3, 2)
        
        if salmap is not None:
            # Weight triangle gradient by saliency map
            # Area doesn't have kinks like image colors do so interior_inds isnt necessary.
            ws, interior_inds = get_barycentric_sampling_weights(5)
            sample_points = get_sample_points_from_barycentric_weights(ws, mesh.X, mesh.T)
            sal_triangle = sample_image(salmap, sample_points)
            sal_grad_prep = grad_prep / sal_triangle.sum(axis=0)[:,np.newaxis]
        else:
            sal_grad_prep = grad_prep
        
        # Accumulate per-triangle-vertex gradients to vertices
        vert_grad = move_triangle_vert_values_to_vertex_values(mesh, sal_grad_prep)
        
        # Apply boundary conditions
        gradient = slip_conditions(mesh, vert_grad)
        
        return energy, gradient
