"""
Rendering functions for visualization and output.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional, Tuple
from mesh import Mesh
from approximator import Approximator

class PersistentFigure:
    """
    Manages persistent figure state
    Global persistent figures can be tricky to manage but for debugging purposes having a persistent figure within reach is super convenient.
    The convenience of easy state-visualization without passing figure and axes handles in and out of every function outweights the difficulty of managing a global state.
    As long as there aren't too many of these.
    """
    
    def __init__(self):
        self.fig = None
        self.axes = None
        self.initialized = False
    
    def reset(self):
        self.initialized = False
        self.fig = None
        self.axes = None

    def get_figure(self) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
        """Get or create the persistent figure."""
        if not self.initialized or self.fig is None:
            self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 6))
            self.fig.patch.set_facecolor('white')
            self.initialized = True
        
        return self.fig, self.axes

# Global persistent figure instance
_persistent_fig = PersistentFigure()

def render(img: np.ndarray, mesh: Mesh, colors: np.ndarray, approx: Approximator,
          desc_dir: Optional[np.ndarray] = None, salmap: Optional[np.ndarray] = None):
    """
    Main rendering function - displays original image and triangulation side by side.
    Args:
        img: Original image (height, width, 3)
        mesh: Mesh object
        colors: Triangle or vertex colors
        approx: Approximator object
        desc_dir: Optional descent direction for gradient visualization
        salmap: Optional saliency map for overlay
    """
    fig, (ax1, ax2) = _persistent_fig.get_figure()
    
    # reset prev axes
    ax1.clear()
    ax2.clear()
    
    height, width = img.shape[:2]
    
    # Left subplot: Original image
    ax1.imshow(img)
    ax1.set_title('Original Image')
    ax1.axis('off')
    ax1.set_xlim(0, width)
    ax1.set_ylim(height, 0)  # Flip y-axis to match image coordinates
    ax1.set_aspect('equal', adjustable='datalim')
    
    # Add saliency overlay if provided
    if salmap is not None:
        # Normalize saliency map
        salmap_norm = salmap / np.max(salmap) if np.max(salmap) > 0 else salmap
        im_overlay = ax1.imshow(salmap_norm, alpha=0.3, cmap='hot')
    # mesh overlay on ax1
    render_mesh_edges(ax1, mesh, edge_color='cyan', edge_alpha=0.3, linewidth=0.3)
    
    # Right subplot: Triangulation
    # ax2.imshow(img, alpha=0.7)  # optional: transparent background image
    render_mesh_triangulation(ax2, mesh, colors, approx)
    
    # Add gradient arrows if provided
    if desc_dir is not None:
        render_gradient_arrows(ax2, mesh.X, desc_dir)
    
    ax2.set_title(f'({mesh.nT} triangles) ({mesh.nX} vertices)')
    ax2.axis('off')
    ax2.set_xlim(0, width)
    ax2.set_ylim(height, 0)
    ax2.set_aspect('equal', adjustable='datalim')

    # unverbose specific matplotlib rendering warning. doesnt work.
    # import warnings
    # warnings.filterwarnings("ignore", message="Ignoring fixed y limits to fulfill fixed data aspect with adjustable data limits.")

    plt.tight_layout()
    plt.draw()
    # debug: fig.savefig('a.png', dpi=400)
    pass


def render_mesh_triangulation(ax: plt.Axes, mesh: Mesh, colors: np.ndarray, approx: Approximator):
    """
    Render the triangulated mesh with colors.
    
    Args:
        ax: Matplotlib axis
        mesh: Mesh object
        colors: Triangle or vertex colors
        approx: Approximator object (determines color format)
    """
    X, T = mesh.X, mesh.T
    
    if approx.degree == 0:
        # Constant approximation - colors per triangle
        render_constant_triangulation(ax, mesh, colors)
    else:
        raise NotImplementedError("Linear approximation not fully supported")

def render_constant_triangulation(ax: plt.Axes, mesh: Mesh, colors: np.ndarray):
    """
    Render triangulation with constant colors per triangle.
    
    Args:
        ax: Matplotlib axis
        X: Vertex coordinates (nX, 2)
        T: Triangle connectivity (nT, 3)
        colors: Triangle colors (nT, 3)
    """
    # from matplotlib.collections import TriMesh
    X, T = mesh.X, mesh.T
    # Ensure colors are in [0, 1] range
    colors_norm = np.clip(colors / 255.0 if np.max(colors) > 1 else colors, 0, 1)
    
    # Create triangular mesh collection using PolyCollection
    from matplotlib.collections import PolyCollection
    
    # Create triangle vertices for each triangle
    triangles = X[T]  # (nT, 3, 2)
    
    # Create PolyCollection
    pc = PolyCollection(triangles, facecolors=colors_norm, edgecolors='none', alpha=1)
    ax.add_collection(pc)
    
    # Add mesh edges
    # render_mesh_edges(ax, mesh, edge_color='cyan', edge_alpha=0.3, linewidth=0.1)

def render_mesh_edges(ax: plt.Axes, mesh: Mesh, 
                     edge_color: str = 'cyan', edge_alpha: float = 0.8, 
                     linewidth: float = 1.5):
    """
    Render mesh edges.
    
    Args:
        ax: Matplotlib axis
        X: Vertex coordinates (nX, 2)
        T: Triangle connectivity (nT, 3)
        edge_color: Edge color
        edge_alpha: Edge transparency
        linewidth: Edge line width
    """
    from matplotlib.collections import LineCollection
    X, T = mesh.X, mesh.T
    edges = mesh.edges

    edge_Xs = [(X[i], X[j]) for i, j in edges]
    lc = LineCollection(edge_Xs, colors=edge_color, linewidths=linewidth)
    ax.add_collection(lc)
    
def render_gradient_arrows(ax: plt.Axes, X: np.ndarray, desc_dir: np.ndarray, 
                         color: str = 'red', alpha: float = 0.7):
    """
    Render gradient descent direction as arrows.
    
    Args:
        ax: Matplotlib axis
        X: Vertex coordinates (nX, 2)
        desc_dir: Descent direction (nX, 2)
        color: Arrow color
        alpha: Arrow transparency
    """
    ax.quiver(X[::step, 0], X[::step, 1], 
              desc_dir[::step, 0], -desc_dir[::step, 1],
              color=color, alpha=alpha, scale_units='xy', width=0.003)

def save_still(output_dir, 
                i, subdiv_count, 
                img, mesh, colors, colors_alt, 
                approx, desc_dir, salmap):
    """
    Helper function to render current state to pngs.
    Saves two types. 
     - first is a still shot of pre-rendered current optimization state.
     - second is just the image triangulation as a png. no metadata. This is the version you put on the cover.
    """
    # render the current optimization state
    stillname_A = f"{output_dir}/still_A{subdiv_count}.png"
    plt.title(f'(iter:{i}) (subdiviters:{subdiv_count-1}) (nX:{mesh.nX}) (nT:{mesh.nT})')
    plt.gcf().savefig(stillname_A,dpi=400)
    print(f"Saved to {stillname_A}")

    # render the final mosaic in current state
    stillname_B = f"{output_dir}/still_B{subdiv_count}.png"
    height, width = img.shape[:2]
    fig,ax = plt.subplots()
    ax.imshow(img)
    ax.axis('off')
    render_mesh_triangulation(ax, mesh, colors_alt, approx)
    fig.savefig(stillname_B,bbox_inches='tight',dpi=400,pad_inches=0)    
    print(f"Saved to {stillname_B}")
    plt.close(fig)

    pass
