"""
Saliency map computation and handling.
"""
import numpy as np
import os
from typing import Optional
from enums import SaliencyStrategy

def compute_saliency_map(img: np.ndarray, strategy: SaliencyStrategy, 
                        boost_factor: float = 10.0, 
                        saliency_map_path: str = None) -> np.ndarray:
    """
    Compute saliency map using manual strategy only.
    
    Args:
        img: Input image (height, width, 3)
        strategy: Saliency computation strategy (only MANUAL supported)
        boost_factor: Amplification factor for saliency
        saliency_map_path: Path to pre-computed manual saliency map file, or where it should be saved
        
    Returns:
        Saliency map (height, width)
    """
    height, width = img.shape[:2]
    
    if strategy != SaliencyStrategy.MANUAL:
        raise ValueError(f"Only MANUAL saliency strategy is supported, got: {strategy}")
    
    # Manual saliency - look for pre-computed manual saliency map or create interactively
    if os.path.exists(saliency_map_path):
        from PIL import Image
        map_img = np.array(Image.open(saliency_map_path)).astype(float)
        
        assert map_img.ndim == 2, "saliency boost map should be grayscale"
            
        if np.max(map_img) > 0:
            map_img = map_img / np.max(map_img)
            
        salmap = np.ones((height, width)) + map_img * boost_factor
    else:
        # Create manual saliency interactively. Returns binary image.
        salmap = get_boost_map_interactive(img)
        
        # Save the raw saliency map so it's loadable later
        from PIL import Image
        os.makedirs(os.path.dirname(saliency_map_path), exist_ok=True)
        Image.fromarray((salmap*255).astype(np.uint8)).save(saliency_map_path)
        
        if np.linalg.norm(salmap, 'fro') != 0:
            salmap = salmap / np.max(salmap)
        
        salmap = np.ones((height, width)) + salmap * boost_factor
        
    return salmap


def get_boost_map_interactive(img: np.ndarray) -> np.ndarray:
    """
    Create manual saliency map through interactive input.
    Binary output map.
    
    Args:
        img: Input image
        
    Returns:
        Boost map from user interaction
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    print(f"Begin manual saliency map construction!")
    fig, ax = plt.subplots()
    plt.suptitle('Manual Saliency Map Creation')
    fig.suptitle('Click and drag to circle important regions.\nClose out to skip manual saliency.')
    ax.imshow(img)
    ax.set_aspect('equal')
    ax.axis('on')
        
    # Track circles
    circles = []
    current_circle = None
    drawing = False

    def on_press(event):
        nonlocal current_circle, drawing
        if event.inaxes == ax and event.button == 1:  # Left mouse button
            drawing = True
            # Start new circle at mouse position
            current_circle = {'center': [event.xdata, event.ydata], 'radius': 1}
    
    def on_motion(event):
        nonlocal current_circle, drawing
        if drawing and current_circle and event.inaxes == ax:
            # Update circle radius based on distance from center
            dx = event.xdata - current_circle['center'][0]
            dy = event.ydata - current_circle['center'][1]
            current_circle['radius'] = max(1, np.sqrt(dx**2 + dy**2))
            
            # Update display
            ax.clear()
            ax.imshow(img)
            
            # Draw all completed circles
            for circle in circles:
                circle_patch = plt.Circle(circle['center'], circle['radius'], 
                                        fill=False, color='red', linewidth=2)
                ax.add_patch(circle_patch)
            
            # Draw current circle being drawn
            if current_circle:
                circle_patch = plt.Circle(current_circle['center'], current_circle['radius'],
                                        fill=False, color='yellow', linewidth=2, linestyle='--')
                ax.add_patch(circle_patch)
            
            ax.set_xlim(0, img.shape[1])
            ax.set_ylim(img.shape[0], 0)
            plt.draw()
    
    def on_release(event):
        nonlocal current_circle, drawing
        if drawing and current_circle and event.inaxes == ax:
            drawing = False
            # Finalize circle
            circles.append(current_circle.copy())
            current_circle = None
            print(f"Added circle at ({circles[-1]['center'][0]:.1f}, {circles[-1]['center'][1]:.1f}) "
                  f"with radius {circles[-1]['radius']:.1f}")
    
    def on_done(event):
        plt.close(fig)
    
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)
    
    ax_button = plt.axes([0.81, 0.01, 0.1, 0.04]) # left bottom width height
    button = Button(ax_button, 'Done')
    button.on_clicked(on_done)
    
    plt.tight_layout()
    # Safe to ignore error message sometimes appears here: SystemError: NULL object passed to Py_BuildValue.
    plt.show() 
    # INTERACTIVE SALIENCY COMPUTATION DOESNT WORK WELL IN DEBUG MODE!! GETS SKIPPED BY UNBLOCKABLE SHOW().
    
    # Create boost map from circles
    boost = np.zeros(img.shape[:2])
    height, width = img.shape[:2]
    
    # Create coordinate grids
    x_vals = np.arange(width) + 0.5  # Pixel centers
    y_vals = np.arange(height) + 0.5
    X, Y = np.meshgrid(x_vals, y_vals)
    
    print(f"Creating boost map from {len(circles)} circles...")
    
    for i, circle in enumerate(circles):
        center_x, center_y = circle['center']
        radius = circle['radius']
        
        # Check which pixels are inside this circle
        distances_sq = (X - center_x)**2 + (Y - center_y)**2
        inside_circle = distances_sq < radius**2
        
        # Add boost to pixels inside circle
        boost[inside_circle] += 1.0
    
    print(f"Boost map created with {np.sum(boost > 0)} boosted pixels")
    
    return boost
