#!/usr/bin/env python3
"""
Batch processing script
Main entry point for running image triangulation algorithm on a collection of images.
For more specific usage, go directly to image_triangulation method.
Run this file directly: python batchrun1.py
"""
import os
import sys
import glob
from pathlib import Path

sys.path.append('src')

from image_triangulation import image_triangulation
from enums import (
    SaliencyStrategy,
    OptStrategy, 
    DtStrategy,
    SubdivisionStrategy,
    InitialMesh
)

# ============================================================================
# CONFIGS - Edit below as needed
# ============================================================================

# Image processing parameters
IMAGES_FOLDER = "images"                    # Input images directory
OUTPUT_FOLDER = "output"                    # Output results directory
WHITELIST = None                       # Images to process (None for all)
# WHITELIST = ['apple'] # Modify this to run specific images

# Algorithm parameters
BOOST_FACTOR = 10.0                         # Saliency map boost factor
INITIAL_HORIZONTAL_SAMPLINGS = [25]         # Initial mesh resolutions to try
DEGREES = [0]                               # Approximation degrees (0=constant, 1=linear (unavailable))
MAX_ITERS = 500                             # Maximum optimization iterations
SAVE_OUT = True                             # Save output results

# Integration and subdivision parameters
INTEGRAL_1D_SAMPLES = 15                    # Integration samples for energy
INTEGRAL_1D_SAMPLES_SUBDIV = 50             # Integration samples for subdivision
EDGE_SPLIT_RESOLUTION = 10                  # Edge splitting - resolution to evaluate split location
N_EDGES_2_SUBDIVIDE = 20                    # Target edges to subdivide per iteration
SUBDIV_MAX = 9                              # Maximum subdivision iterations
SUBDIVISION_DAMPER = 5.0                    # Subdivision threshold damping - energy drop rate always slows as optimization progresses. constant energy slope requirement is not reasonable. this factor prevents repeated topology edit spamming.
DEMANDED_ENERGY_DENSITY_DROP = 5.0          # Energy drop threshold - if doesn't decrease enough in a window, conclude: continous part is converged
WINDOW_SIZE = 20                            # Convergence detection window

# Saliency strategies to try
SAL_STRATS = [SaliencyStrategy.MANUAL]      # Saliency computation methods

def batchrun1():
    """
    Batch process multiple images with image triangulation.
    Basically just a big nested loop over configuration defined at the top of this file.
    """
    
    # Find all jpg images in the folder
    image_pattern = os.path.join(IMAGES_FOLDER, "*.jpg")
    image_files = glob.glob(image_pattern)
    
    if not image_files:
        print(f"No .jpg files found in {IMAGES_FOLDER}")
        return
    
    print(f"Found {len(image_files)} images in {IMAGES_FOLDER}")
    print(f"Processing configurations:")
    print(f"  - Initial samplings: {INITIAL_HORIZONTAL_SAMPLINGS}")
    print(f"  - Degrees: {DEGREES}")
    print(f"  - Whitelist: {WHITELIST}")
    
    # Create output directory
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    processed_count = 0
    skipped_count = 0
    for image_file in image_files:
        # Extract image name without extension
        image_path = Path(image_file)
        name = image_path.stem
        
        # Check whitelist
        if WHITELIST and name not in WHITELIST:
            print(f"Skipping {name} (not in whitelist)")
            skipped_count += 1
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing image: {name}")
        print(f"{'='*60}")
        
        # Process all parameter combinations
        for initial_horizontal_sampling in INITIAL_HORIZONTAL_SAMPLINGS:
            for degree in DEGREES:
                for sal_idx, sal_strat in enumerate(SAL_STRATS):
                    
                    # output directory
                    output_dir = os.path.join(
                        OUTPUT_FOLDER,
                        f"{name}_init_{initial_horizontal_sampling}_deg_{degree}_sal_{sal_idx+1}"
                    )
                    
                    print(f"\nConfiguration:")
                    print(f"  - Initial sampling: {initial_horizontal_sampling}")
                    print(f"  - Degree: {degree}")
                    print(f"  - Saliency: {sal_strat.value}")
                    print(f"  - Output: {output_dir}")
                    
                    try:
                        # Run image triangulation
                        X, T, colors, timedata = image_triangulation(
                            fname=image_file,
                            sal_strat=sal_strat,
                            boost_factor=BOOST_FACTOR,
                            initial_horizontal_sampling=initial_horizontal_sampling,
                            perturb_init=False,
                            degree=degree,
                            force_gray=False,
                            max_iters=MAX_ITERS,
                            save_out=SAVE_OUT,
                            output_dir=output_dir,
                            opt_strat=OptStrategy.RMS_PROP,
                            demanded_energy_density_drop=DEMANDED_ENERGY_DENSITY_DROP,
                            window_size=WINDOW_SIZE,
                            dt_strat=DtStrategy.CONSTRAINED,
                            integral_1d_samples=INTEGRAL_1D_SAMPLES,
                            integral_1d_samples_subdiv=INTEGRAL_1D_SAMPLES_SUBDIV,
                            edge_split_resolution=EDGE_SPLIT_RESOLUTION,
                            n_edges_2_subdivide=N_EDGES_2_SUBDIVIDE,
                            subdiv_max=SUBDIV_MAX,
                            subdivision_damper=SUBDIVISION_DAMPER,
                            sub_strat=SubdivisionStrategy.EDGE,
                            i_mesh=InitialMesh.HEXAGONAL
                        )
                        
                        print(f"SUCCESS: Generated {T.shape[0]} triangles in {timedata.total_time:.2f}s")
                        processed_count += 1
                        
                    except Exception as e:
                        print(f"ERROR processing {name}: {str(e)}")
                        continue
    
    print(f"\n{'='*60}")
    print(f"Batch processing completed!")
    print(f"  - Processed: {processed_count} configurations")
    print(f"  - Skipped: {skipped_count} images")
    print(f"{'='*60}")

if __name__ == "__main__":
    if not os.path.exists(IMAGES_FOLDER):
        print(f"Creating example images directory: {IMAGES_FOLDER}")
        os.makedirs(IMAGES_FOLDER)
        print("Please add .jpg image files to the 'images' directory and run again.")
    else:
        batchrun1()
