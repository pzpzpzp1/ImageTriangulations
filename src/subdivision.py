"""
Mesh subdivision functions.
"""
import numpy as np
from typing import Tuple, List, Optional
from mesh import Mesh, mesh_from_xt
from math_utils import sample_image, get_triangle_areas, vecnorm
from approximator import Approximator

def get_tris_to_collapse(mesh: Mesh, per_tri_error: np.ndarray, img: np.ndarray) -> np.ndarray:
    """
    Returns triangle indices of low-quality and high-energy triangles so they can be handled.
    """
    T = mesh.T
    X = mesh.X
    total_area = img.shape[0] * img.shape[1]
    
    # compute which triangles are bad
    a = mesh.triangle_edge_lengths[:, 0]
    b = mesh.triangle_edge_lengths[:, 1]
    c = mesh.triangle_edge_lengths[:, 2]
    s = (a + b + c) / 2
    in_radii = np.sqrt(s * (s - a) * (s - b) * (s - c)) / s
    
    # triangle quality metric: inradius to max edge len
    tri_qual = in_radii / np.max(mesh.triangle_edge_lengths, axis=1)
    areas = mesh.tri_areas
    # cost is image error dependent
    tri_costs = np.sqrt(np.sum(per_tri_error, axis=1)) / mesh.tri_areas
    
    # 'bad' means low quality, sliver, low relative area
    is_bad_tri_shape = (tri_qual < 0.1) & (areas < 0.0009 * total_area)
    still_bad_tris = np.where(is_bad_tri_shape)[0]
    tri_costs = tri_costs[is_bad_tri_shape]

    # keep 50%tile worst 'cost' triangles. these are sliver triangles with high image energy
    if len(tri_costs) > 0:
        bad_tri_inds = still_bad_tris[tri_costs > np.percentile(tri_costs, 50)]
    else:
        bad_tri_inds = np.array([], dtype=int)
    
    return bad_tri_inds

def collapse_sliver_triangles(mesh: Mesh, bad_tri_inds: np.ndarray) -> Tuple[Mesh, np.ndarray]:
    """
    Collapse sliver triangles by flipping long edge
    
    Args:
        mesh: Input mesh
        bad_tri_inds: Indices of triangles to collapse
        
    Returns:
        new_mesh: Modified mesh
        num_success: number of edges flipped
    """
    MIN_AREA_TOLERATED_DURING_COLLAPSE = 5
    debug_viz = False # create testable data and visualize 
    num_success = 0
    X, T = mesh.X.copy(), mesh.T.copy()
    Xnew, Tnew = X.copy(), T.copy()

    if debug_viz:
        # generic selection of triangles to 'collapse'
        bad_tri_inds = np.random.choice(mesh.nT, 50)

    # get longest edges per sliver triangle that needs flipping
    TEL = mesh.triangle_edge_lengths[bad_tri_inds,:]
    TE = mesh.triangles_to_edges[bad_tri_inds,:]
    edges_to_flip = [TE[i,np.argmax(TEL[i,:])] for i in range(len(bad_tri_inds))]

    if debug_viz:
        # forced selection of boundary edges to test that code path
        edges_to_flip = np.random.choice(np.where(mesh.is_boundary_edge)[0],5).tolist()

    # find non-overlapping triangle butterflies for non-conflicting edge flips
    non_overlapping_edges_to_flip = mesh.get_non_butterfly_overlapping_edges(edges_to_flip)

    if debug_viz:
        # non_overlapping_edges_to_flip = np.random.choice(mesh.nE, 10)
        import matplotlib.pyplot as plt
        fig, ax = mesh.render_to('a.png', return_figax = True)
        for eind in non_overlapping_edges_to_flip:
            evs = mesh.edges[eind,:]
            verts = mesh.X[evs,:]
            ax.plot(verts[:,0], verts[:,1],'r-', linewidth=1)
        plt.savefig('a.png',dpi=400)

    # perform flip
    for edge in non_overlapping_edges_to_flip:
        butterflyTinds = mesh.edges_to_triangles[edge,:]
        if not mesh.is_boundary_edge[edge]:
            verts = mesh.T[butterflyTinds,:]
            v1s = verts[0]
            v2s = verts[1]
            farvert1 = list(set(verts[0,:]) - set(mesh.edges[edge,:]))[0]
            farvert2 = list(set(verts[1,:]) - set(mesh.edges[edge,:]))[0]
            v1s = np.roll(v1s, 2-list(v1s).index(farvert1)) # permute until farvert1 is last vert
            v2s = np.roll(v2s, 2-list(v2s).index(farvert2)) # permute until farvert2 is last vert
            newTris = np.array([[v1s[0], farvert2, farvert1], [v2s[0], farvert1, farvert2]])
            # replace old triangles with new flipped triangles if no triangle inversion
            if all(get_triangle_areas(Xnew, newTris) > MIN_AREA_TOLERATED_DURING_COLLAPSE):
                Tnew[butterflyTinds,:] = newTris
                num_success+=1
        else:
            tind = butterflyTinds[0]
            verts = mesh.T[tind,:]
            farvert = list(set(verts) - set(mesh.edges[edge,:]))[0]
            verts = np.roll(verts, 2-list(verts).index(farvert)) # permute until farvert is last vert
            
            # a,b,c oriented so c is the far vert, (a,b) is the long edge to split
            a,b,c = verts
            newX = (mesh.X[a,:]+mesh.X[b,:])/2
            d = Xnew.shape[0] # new vert index. updates with Xnew 
            Xnew = np.row_stack((Xnew, newX)) # add new vertex to bottom
            
            # replace one triangle and add one new triangle
            Tnew[tind,:] = [a,d,c] # replace one of the old tris with new tri from split
            new_tri = np.array([c,d,b]) # add next tri 

            # note this subdivision can never invert triangles so we don't have to check
            Tnew = np.row_stack((Tnew, new_tri))
            num_success+=1
    
    if debug_viz:
        new_mesh = mesh_from_xt(Xnew, Tnew)
        import matplotlib.pyplot as plt
        fig, ax = new_mesh.render_to('b.png', return_figax = True)
        for eind in non_overlapping_edges_to_flip:
            evs = mesh.edges[eind,:]
            verts = mesh.X[evs,:]
            ax.plot(verts[:,0], verts[:,1],'r-', linewidth=0.5, alpha = 0.5)
        plt.savefig('b.png',dpi=400)

    if num_success>0:
        return mesh_from_xt(Xnew, Tnew), num_success
    
    return mesh, num_success

def build_edge_split_mesh(mesh: Mesh) -> Mesh:
    # No accounting for boundary vs interior edge. 
    # Means boundary-adjacent triangles get doubled
    nE = mesh.nE; nT = mesh.nT; nX = mesh.nX; T = mesh.T; X = mesh.X;
    v1,v2,v3,v4,v5 = mesh.get_v12345()

    Xnew = np.row_stack((X, (X[mesh.edges[:,0],:]+X[mesh.edges[:,1],:])/2))
    # build triangulation
    t1s = np.column_stack((v1, v2, v5))
    t2s = np.column_stack((v1, v5, v3))
    t3s = np.column_stack((v5, v2, v4))
    t4s = np.column_stack((v3, v5, v4))
    Tnew = np.row_stack((t1s,t2s,t3s,t4s)) # 4 x nE x 3
    
    # reorient inverted triangles
    flippedTris = get_triangle_areas(Xnew, Tnew) < 0;
    flips = Tnew[flippedTris,:]
    Tnew[flippedTris,:] = flips[:,[0, 2, 1]];

    esMesh = mesh_from_xt(Xnew, Tnew, 1);
    # esMesh.render_to('a.png')
    return esMesh

def get_edge_split_score(mesh: Mesh, img: np.ndarray, approx: Approximator,
                        integral_1d_samples: int, salmap: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute edge split scores based on energy reduction potential.
    Args:
        mesh: Current mesh
        img: Original image
        approx: Approximator for energy computation
        integral_1d_samples: Integration samples
        salmap: Optional saliency map
        
    Returns:
        Split scores per edge (nE,)
    """
    nE = mesh.nE
    scores = np.zeros(nE)

    esMesh = build_edge_split_mesh(mesh);
    
    # Get current energy per triangle
    extra, _, _ = approx.compute_energy(img, mesh, integral_1d_samples, salmap, return_gradient=False)
    esExtra, _, _ = approx.compute_energy(img, esMesh, integral_1d_samples, salmap, return_gradient=False)
    current_errors = extra['per_triangle_rgb_error']
    
    # assemble the edge division comparisions
    brokenEdgeEnergies = esExtra['per_triangle_rgb_error'].reshape(4,-1,3).sum(axis=0)
    originalEdgeEnergies = extra['per_triangle_rgb_error'][mesh.edges_to_triangles,:].sum(axis=1)
    energyDrop = (originalEdgeEnergies - brokenEdgeEnergies).sum(axis=1)
    # boundary triangles were double counted. actual differential is needs to be halved.
    score = energyDrop; score[mesh.is_boundary_edge] = score[mesh.is_boundary_edge]/2; 

    # error per triangle is computed including shared edges TWICE. 
    # this means it's possible for the subdivision energy to be greater than the original. 
    # this error goes away as integral samples increases, and you can clip any remaining outliers to 0.
    score[score<0]=0;

    return score

def draw_edges_to_split(N: int, scores: np.ndarray) -> np.ndarray:
    """
    Select edges to split based on scores.
    
    Args:
        N: Number of edges to select
        scores: Edge scores
        
    Returns:
        Indices of edges to split
    """
    nE = scores.shape[0]
    eps = 1e-16
    if N >= nE:
        return np.arange(nE)
    
    # Draw according to pdf with replacement. 
    # This should help not let the edges to divide get too clustered.
    if sum(scores) == 0:
        probs = None
    else:
        probs = scores; probs = probs/sum(probs); 
    edge_inds = np.unique(np.random.choice(nE, int(np.floor(N/10)), replace=True, p=probs))
    
    # draw the remainder from N greedily
    score2 = scores + eps;
    score2[edge_inds]=0;
    perm = np.argsort(score2)
    perm = np.flip(perm) # descending list
    edgelist = np.arange(nE);
    priorityEdgeList = edgelist[perm];
    
    remainingN = N - edge_inds.shape[0]
    edge_inds = np.unique(np.concatenate((priorityEdgeList[:remainingN], edge_inds)))
    
    # return edgeInds sorted by score
    perm = np.argsort(scores[edge_inds]); perm = np.flip(perm); # descending
    edge_inds = edge_inds[perm];
    return edge_inds

def subdivide_mesh_edges(mesh: Mesh, 
                         edgeInds: np.ndarray, 
                         n1D: int, 
                         img: np.ndarray = None
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split mesh at edges. Optionally try to determine optimal split location.
    """
    # part 1: find subset of edge inds that don't conflict with each other. prioritize edges at top of list.
    edgeInds = np.array(mesh.get_non_butterfly_overlapping_edges(edgeInds))
    ne = edgeInds.shape[0]
    
    # part 1.5: figure out on each edge where to split it. edge midpoint might not be ideal.
    nE = mesh.nE; nT = mesh.nT; nX = mesh.nX;
    T = mesh.T; X = mesh.X;
    if img is not None:
        # Sample image along edge and find biggist 'jump' in value. choose to split there
        edgeWs = np.linspace(1/4,3/4,n1D);
                
        dw2 = (edgeWs[1]-edgeWs[0])/2
        X1 = mesh.X[mesh.edges[edgeInds,0]].reshape(ne,1,2)
        X2 = mesh.X[mesh.edges[edgeInds,1]].reshape(ne,1,2)
        Xsamples = X1*(1-edgeWs[np.newaxis,:,np.newaxis]) + X2*edgeWs[np.newaxis,:,np.newaxis]
        fsamples = sample_image(img,Xsamples).astype(float); # [ne n1D 3]
        jumps = np.linalg.norm(fsamples[:,1:n1D,:] - fsamples[:,0:n1D-1,:],ord=2,axis=2,keepdims=False); # [ne, n1D-1]
        inds = np.argmax(jumps, axis=1)
        ws = 1-(edgeWs[inds] + dw2)
    else:
        ws = .5*np.ones(ne)

    v1,v2,v3,v4,v5 = mesh.get_v12345(edgeInds)

    split_verts = ws[:,np.newaxis]*X[mesh.edges[edgeInds,0],:] + (1-ws[:,np.newaxis])*X[mesh.edges[edgeInds,1],:]
    Xnew = np.row_stack((X, split_verts))
    dedupBoundary = ~mesh.is_boundary_edge[edgeInds]
    isCoveredTriangles = np.zeros(nT,dtype=bool); isCoveredTriangles[np.unique(mesh.edges_to_triangles[edgeInds,:])] = True

    # my lord numpy has incredibly verbose concatenation notation. 
    keptTris = T[~isCoveredTriangles,:]
    t1s = np.column_stack((v1, v2, v5))
    t2s = np.column_stack((v1, v5, v3))
    t3s = np.column_stack((v5[dedupBoundary], v2[dedupBoundary], v4[dedupBoundary]))
    t4s = np.column_stack((v3[dedupBoundary], v5[dedupBoundary], v4[dedupBoundary]))
    Tnew = np.row_stack((keptTris, t1s,t2s,t3s,t4s)) # (4 x nE) x 3

    assert np.unique(np.sort(Tnew, axis=1), axis=0).shape == Tnew.shape # no dupe triangles

    # debug viz
    # newmesh = mesh_from_xt(Xnew,Tnew)
    # mesh.render_to('a.png')
    # newmesh.render_to('b.png')

    return Xnew, Tnew
