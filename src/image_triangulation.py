"""
Main image triangulation algorithm.
"""
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Optional
from PIL import Image

from enums import SaliencyStrategy, OptStrategy, DtStrategy, SubdivisionStrategy, InitialMesh
from mesh import Mesh, mesh_from_xt
from approximator import Approximator
from area_energy import get_area_energy
from saliency import compute_saliency_map
from optimization import get_adadelta_desc_dir
from subdivision import (
    get_tris_to_collapse, collapse_sliver_triangles,
    get_edge_split_score, draw_edges_to_split, subdivide_mesh_edges
)
from rendering import render, save_still
from math_utils import (
    initial_hex_lattice_mesh, 
    initial_grid_mesh,
    clip_verts, 
    get_triangle_areas,
    vecnorm
)
from video_renderer import VideoRecorder
class TimeData:
    """Container for timing and performance data."""    
    def __init__(self):
        self.energy_comp_time = []
        self.n_Xs = []
        self.n_Ts = []
        self.subdiv_time = []
        self.total_time = 0.0

def image_triangulation(
    fname: str,
    sal_strat: SaliencyStrategy = SaliencyStrategy.MANUAL,
    boost_factor: float = 10.0,
    initial_horizontal_sampling: int = 15,
    perturb_init: bool = False,
    degree: int = 1,
    force_gray: bool = False,
    max_iters: int = 500,
    save_out: bool = False,
    output_dir: str = "outputcache",
    opt_strat: OptStrategy = OptStrategy.RMS_PROP,
    demanded_energy_density_drop: float = 5.0,
    window_size: int = 10,
    dt_strat: DtStrategy = DtStrategy.CONSTRAINED,
    integral_1d_samples: int = 15,
    integral_1d_samples_subdiv: int = 50,
    edge_split_resolution: int = 10,
    n_edges_2_subdivide: int = 20,
    subdiv_max: int = 10,
    subdivision_damper: float = 5.0,
    sub_strat: SubdivisionStrategy = SubdivisionStrategy.EDGE,
    i_mesh: InitialMesh = InitialMesh.HEXAGONAL
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, TimeData]:
    """
    Main image triangulation algorithm.
    
    Args:
        fname: Input image file path
        sal_strat: Saliency computation strategy
        boost_factor: Saliency amplification factor
        initial_horizontal_sampling: Initial mesh resolution
        perturb_init: Whether to perturb initial vertices
        degree: Approximation degree (0=constant, 1=linear)
        force_gray: Convert image to grayscale
        max_iters: Maximum optimization iterations
        save_out: Whether to save output
        output_dir: Output directory
        opt_strat: Optimization strategy
        demanded_energy_density_drop: Energy convergence threshold
        window_size: Convergence window size
        dt_strat: Time step strategy
        integral_1d_samples: Integration samples for energy computation
        integral_1d_samples_subdiv: Integration samples for subdivision
        edge_split_resolution: Edge splitting resolution
        n_edges_2_subdivide: Number of edges to subdivide per iteration
        subdiv_max: Maximum subdivision iterations
        subdivision_damper: Subdivision threshold damping factor
        sub_strat: Subdivision strategy
        i_mesh: Initial mesh type
    Returns:
        X: Final vertex coordinates
        T: Final triangle connectivity
        colors: Final triangle/vertex colors
        timedata: Performance timing data
    """
    # Reset persistent fig and any matplotlib persistent states.
    from rendering import _persistent_fig
    _persistent_fig.reset()
    plt.close('all')
    
    # Load and preprocess image
    img = np.array(Image.open(fname))
    if img.ndim == 2:  # Convert grayscale to RGB
        img = np.stack([img, img, img], axis=2)    
    height, width = img.shape[:2]
    total_area = width * height
    
    if force_gray:
        # Convert to grayscale using luminance weights but keep 3 channels
        gray_img = (0.298936021293775 * img[:, :, 0] + 
                   0.587043074451121 * img[:, :, 1] + 
                   0.114020904255103 * img[:, :, 2])
        gray_img = gray_img.astype(np.uint8)[:, :, np.newaxis]
        img = np.repeat(gray_img, 3, axis=2)
        # Debug viz the grayscale image
        # Image.fromarray(img).save('a.png');
    
    # Compute saliency map
    base_name = os.path.splitext(os.path.basename(fname))[0]
    saliency_path = os.path.join("manualsaliency", f"{base_name}.png")
    salmap = compute_saliency_map(img, sal_strat, boost_factor, saliency_path)
    
    # initialize tracking data
    timedata = TimeData()
    
    # Initialize triangulation
    if i_mesh == InitialMesh.HEXAGONAL:
        X, T = initial_hex_lattice_mesh(width, height, initial_horizontal_sampling)
    elif i_mesh == InitialMesh.GRID:
        X, T = initial_grid_mesh(width, height, initial_horizontal_sampling, rand_slant=False)
    else:
        raise ValueError(f"Unknown initial mesh type: {i_mesh}")
    mesh = mesh_from_xt(X, T)
    
    # Break symmetries
    if perturb_init:
        perturbation_scale = np.sqrt(2*mesh.tri_areas).mean()/15
        perturbation = np.random.randn(*X[mesh.is_interior, :].shape) * perturbation_scale
        X[mesh.is_interior, :] += perturbation
        mesh = mesh_from_xt(X, T)

    # Debug visualization
    # mesh.render_to()
    
    # Initialize approximator
    approx = Approximator(degree) # Approximator for vertex gradient flow
    subdiv_approx = Approximator(0)  # Constant approximator for subdivision
    
    # Compute initial energy to find balance between area and image losses
    extra, energy, colors = approx.compute_energy(img, mesh, integral_1d_samples, salmap, return_gradient=False)
    area_energy0 = get_area_energy(mesh, salmap, return_gradient=False)
    area_factor = abs(energy / (area_energy0 * 2))
    
    print(f"Initial energy: {energy:.6f}, Area energy: {area_energy0:.6f}, Area factor: {area_factor:.6f}")
    
    # Initialize video recording
    video_recorder = None
    if save_out:
        # Check if output directory already exists. Then SKIP
        if os.path.exists(output_dir): 
            print(f'SKIPPED: {output_dir}')
            return X, T, colors, timedata
        
        # Create video recorder
        video_recorder = VideoRecorder(f"{output_dir}/video.mp4")
        video_recorder.open_video()
        
    # Display initial state
    render(img, mesh, colors, approx, None, salmap)
    
    # allocate tracking data
    subdiv_iters = np.zeros(subdiv_max, dtype=int)
    dts = np.zeros(max_iters)
    energies = np.zeros(max_iters)
    grad_norms = np.zeros(max_iters)
    
    # Optimization loop. Vary continuous and discrete parameters to minimize energy
    dt = 0.4 # initial dt. only matters for the 'None' dtStrategy, which is not recommended
    subdiv_count = 1
    for i in range(max_iters):
        if save_out:
            plt.title(f'(iter:{i}) (subdiviters:{subdiv_count-1}) (nX:{mesh.nX}) (nT:{mesh.nT})')
            # debug: plt.savefig('a.png',dpi=400)
            video_recorder.write_frame(plt.gcf())
        # Update mesh
        mesh = mesh_from_xt(X, T)
        
        energy_comp_start = time.time()
        
        # Compute energy and gradient for continuous update
        extra, approx_energy, colors, grad = approx.compute_energy(img, mesh, integral_1d_samples, salmap, return_gradient=True)
        area_energy, area_gradient = get_area_energy(mesh, salmap, return_gradient=True)
        
        total_energy = area_energy * area_factor + approx_energy
        total_grad = area_gradient * area_factor + grad
        
        energies[i] = total_energy
        grad_norms[i] = np.linalg.norm(total_grad, 'fro')
        
        timedata.energy_comp_time.append(time.time() - energy_comp_start)
        timedata.n_Xs.append(X.shape[0])
        timedata.n_Ts.append(T.shape[0])
        
        print(f"Iter {i:3d}: Energy = {total_energy:.6f}, Grad norm = {grad_norms[i]:.6f}, "
              f"Vertices = {X.shape[0]}, Triangles = {T.shape[0]}")
        
        # Get descent direction from gradient and optimizer strat
        if opt_strat == OptStrategy.ADA_DELTA:
            desc_dir, _ = get_adadelta_desc_dir(total_grad, 1)
        elif opt_strat == OptStrategy.RMS_PROP:
            desc_dir, _ = get_adadelta_desc_dir(total_grad, 2)
        elif opt_strat == OptStrategy.NONE:
            desc_dir = -total_grad
        else:
            raise ValueError(f"Unknown optimization strategy: {opt_strat}")
        
        # Render current state. Could render desc_dir too but it's too noisy visually so that's turned off to None.
        render(img, mesh, extra['colors_alt'], approx, desc_dir=None, salmap=salmap)
        
        # check convergence and either subdivide or stop
        # if energy hasn't dropped significantly since window iterations ago, then energy is 'flat'
        if (i > window_size and 
            energies[i - window_size] - energies[i] < demanded_energy_density_drop * total_area):
            
            # Save still image on convergence
            if video_recorder is not None:
                save_still(output_dir, 
                           i, subdiv_count, 
                           img, mesh, colors, extra['colors_alt'], 
                           approx, desc_dir, salmap)
                pass
            
            # Topological edits - discrete optimization
            if subdiv_count <= subdiv_max:
                print(f"Energy plateau detected. Performing subdivision {subdiv_count}/{subdiv_max}")
                
                subdiv_time_start = time.time()
                
                # Handle bad sliver triangles
                bad_tri_inds = get_tris_to_collapse(mesh, extra['per_triangle_rgb_error'], img)
                if len(bad_tri_inds) != 0:
                    print(f"Collapsing {len(bad_tri_inds)} sliver triangles")
                    mesh, num_success = collapse_sliver_triangles(mesh, bad_tri_inds)
                    print(f"Collapsed {num_success} sliver triangles")
                    # Update X, T from modified mesh
                    X, T = mesh.X, mesh.T
                
                # Perform subdivision
                subdiv_iters[subdiv_count - 1] = i
                subdiv_count += 1
                
                # Choose subdivision approximator
                if sub_strat == SubdivisionStrategy.EDGE:
                    # Split mesh via edge cutting
                    print("Performing edge-based subdivision")
                    score = get_edge_split_score(mesh, img, subdiv_approx, integral_1d_samples_subdiv, salmap)
                    edge_inds = draw_edges_to_split(n_edges_2_subdivide, score)
                    X, T = subdivide_mesh_edges(mesh, edge_inds, edge_split_resolution, img)

                elif sub_strat == SubdivisionStrategy.LOOP:
                    raise NotImplementedError("Not implemented yet.")
                
                # Update demanded energy drop for next convergence check
                demanded_energy_density_drop /= subdivision_damper
                timedata.subdiv_time.append(time.time() - subdiv_time_start)
                
                print(f"Subdivision {subdiv_count-1} completed: {X.shape[0]} vertices, {T.shape[0]} triangles")
                continue # do not do continuous step if already did discrete step
            else:
                print("Optimization finished!")
                break
        
        # Find good time step
        if dt_strat == DtStrategy.CONSTRAINED:
            # start with onepix strat
            dt = 1.0 / np.max(vecnorm(desc_dir, axis=1))
        
            # then line search to ensure no triangle inversions
            attempt = 0
            while any(get_triangle_areas(X + dt * desc_dir, T) < 0):
                dt = dt / 2
                attempt += 1
                if attempt >= 20:
                    raise Exception(f'Couldnt find dt small enough to prevent triangle area inversion. {dt=}')                
        elif dt_strat == DtStrategy.ONE_PIX:
            dt = 1.0 / np.max(vecnorm(desc_dir, axis=1))
        elif dt_strat == DtStrategy.NONE:
            if i == 0:
                print("WARNING: Constant dt is NOT recommended. Mesh inversions imminent.")
        
        dts[i] = dt
        
        # Update vertex positions
        X = X + dt * desc_dir
        X = clip_verts(X, width, height)
    
    # Final render
    # render(img, mesh, colors, approx, None, None)
    render(img, mesh, extra['colors_alt'], approx, None, salmap)
    
    # Close video recording and save final data
    if video_recorder is not None:
        video_recorder.close_video()
        
    # Plot and save loss trajectories and other parameter data
    fig,(ax1,ax2,ax3) = plt.subplots(3,1)
    ax1.plot(energies[:i])
    [ax1.axvline(val) for val in subdiv_iters[:subdiv_count-1]]
    ax1.set_title('Loss per iteration\nVertical lines indicate topology change')
    ax1.set_ylabel('Loss')

    ax2.plot(timedata.n_Xs)
    [ax2.axvline(val) for val in subdiv_iters[:subdiv_count-1]]
    ax2.set_ylabel('#Vertices')

    ax3.plot(timedata.n_Ts)
    [ax3.axvline(val) for val in subdiv_iters[:subdiv_count-1]]
    ax3.set_ylabel('#Triangles')

    ax3.set_xlabel('continuous iteration steps')
    plt.tight_layout() 
    fig.savefig(f"{output_dir}/loss_per_iter.png", dpi=400)

    # Visualize losses per time
    fig,(ax1) = plt.subplots(1,1)
    cumul_time = np.cumsum(timedata.energy_comp_time[:i])
    for sd_ind, sd_iter in enumerate(subdiv_iters[:subdiv_count-1]):
        ax1.axvline(cumul_time[sd_iter],color='g')
        ax1.axvline(cumul_time[sd_iter]+timedata.subdiv_time[sd_ind],color='r')
        cumul_time[sd_iter:] += timedata.subdiv_time[sd_ind]
    ax1.plot(cumul_time, energies[:i])
    ax1.set_title('Loss per time\nGreen-red lines indicate start-end topology change')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Wall-time (s)')
    plt.tight_layout() 
    fig.savefig(f"{output_dir}/loss_per_time.png", dpi=400)

    # Save misc data 
    timing_data_path = os.path.join(output_dir, 'timedata.npz')
    np.savez(timing_data_path, 
                total_time=timedata.total_time,
                energy_comp_time=timedata.energy_comp_time,
                n_Xs=timedata.n_Xs,
                n_Ts=timedata.n_Ts,
                subdiv_time=timedata.subdiv_time)
    
    return X, T, colors, timedata
