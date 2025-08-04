"""
Mesh data structure and operations
"""
import numpy as np
from typing import Optional
from math_utils import get_triangle_areas, vecnorm

class Mesh:
    """
    2D Triangle Mesh data structure containing vertices, triangles, and derived properties.
    """    
    def __init__(self, X: np.ndarray, T: np.ndarray, is_triangle_soup: bool = False):
        """
        Initialize mesh from vertices and triangles.        
        Args:
            X: Vertex coordinates (n_vertices, 2)
            T: Triangle connectivity (n_triangles, 3)
            is_triangle_soup: Whether triangles form a soup (no connectivity)
        """
        self.X = X
        self.T = T
        self.nX = X.shape[0]
        self.nT = T.shape[0]
        
        # triangle areas
        self.signed_tri_areas = get_triangle_areas(X, T)
        self.tri_areas = self.signed_tri_areas # signed is more robust
        
        # initial mesh should have positive area
        assert np.all(self.tri_areas > 0), "All triangles must have positive area"
        
        # get unique edges
        all_edges = np.vstack([
            T[:, [0, 1]],
            T[:, [1, 2]], 
            T[:, [2, 0]]
        ])
        sorted_edges = np.sort(all_edges, axis=1)
        self.edges, ia, ic = np.unique(sorted_edges, axis=0, return_inverse=True, return_index=True)
        self.nE = self.edges.shape[0]
        
        # If this is done right: self.edges[self.triangles_to_edges[i,:],:] will have 3 unique numbers representing vertices of i'th triangle
        self.triangles_to_edges = ic.reshape(3, self.nT).T 
        
        # triangle edge normals
        v1 = X[T[:, 0], :]
        v2 = X[T[:, 1], :]
        v3 = X[T[:, 2], :]
        e1 = v2 - v1
        e2 = v3 - v2
        e3 = v1 - v3
        # Compute normals for each edge of each triangle
        self.triangle_edge_normals = np.zeros((self.nT, 3, 2))
        self.triangle_edge_normals[:, 0, 0] = -e1[:, 1]
        self.triangle_edge_normals[:, 0, 1] = e1[:, 0]
        self.triangle_edge_normals[:, 1, 0] = -e2[:, 1]
        self.triangle_edge_normals[:, 1, 1] = e2[:, 0]
        self.triangle_edge_normals[:, 2, 0] = -e3[:, 1]
        self.triangle_edge_normals[:, 2, 1] = e3[:, 0]
        
        # Normalize normals
        edge_lengths = np.linalg.norm(self.triangle_edge_normals, axis=2, keepdims=True)
        self.triangle_edge_normals = -self.triangle_edge_normals / edge_lengths
        
        # Compute edge lengths
        self.triangle_edge_lengths = np.column_stack([
            vecnorm(e1, axis=1),
            vecnorm(e2, axis=1), 
            vecnorm(e3, axis=1)
        ])
        
        # Boundary handling (specific to rectangular boundary mesh i.e. images)
        self._compute_boundary_info()
        
        # Compute area gradient (dA/dt)
        self._compute_area_gradient()
        
        if not is_triangle_soup:
            self._compute_edge_to_triangle_mapping()
    
    def _compute_boundary_info(self):
        """Compute boundary vertex and edge information."""
        # boundary edges have one adjacent triangle. interior edges have 2.
        edge_counts = np.bincount(self.triangles_to_edges.flatten(), minlength=self.nE)
        self.is_boundary_edge = (edge_counts == 1)
        
        # get edge tangent vectors
        edge_tangs = self.X[self.edges[:, 0], :] - self.X[self.edges[:, 1], :]
        edge_tangs = edge_tangs / vecnorm(edge_tangs, axis=1)
        
        # Identify X and Y aligned edges (Assuming rectangular boundaries)
        self.is_x_edge = np.abs(edge_tangs[:,1]) < 1e-6
        self.is_y_edge = np.abs(edge_tangs[:,0]) < 1e-6
        
        # Identify boundary vertices
        x_vert_inds = np.unique(self.edges[self.is_boundary_edge & self.is_x_edge, :])
        y_vert_inds = np.unique(self.edges[self.is_boundary_edge & self.is_y_edge, :])
        
        self.is_x_vert = np.zeros(self.nX, dtype=bool)
        self.is_y_vert = np.zeros(self.nX, dtype=bool)
        self.is_x_vert[x_vert_inds] = True
        self.is_y_vert[y_vert_inds] = True
        self.is_interior = ~self.is_x_vert & ~self.is_y_vert
        
        # Identify boundary triangles
        self.is_boundary_triangle = np.sum(~self.is_interior[self.T], axis=1) != 0
    
    def _compute_area_gradient(self):
        """Compute gradient of triangle areas with respect to vertex positions."""
        # prepare for vectorized computation
        TX = self.X[self.T, :].transpose([0,2,1])  # (nT, 2, 3)
        
        # vectorized edge vectors
        e123 = TX - np.roll(TX, -1, axis=2)  
        e312 = np.roll(e123, -1, axis=2)     
        
        # get part of e123 orthogonal to e312
        dot_products = np.sum(e123*e312,axis=1, keepdims=True)
        e312_norms_sq = np.sum(e312**2, axis=1, keepdims=True)
        projections = dot_products / e312_norms_sq * e312
        altitudes = e123 - projections
        altitudes = altitudes / np.linalg.norm(altitudes, axis=1, keepdims=True)
        
        # Compute area gradients: dA/dt = "base" |e312| * altitude_direction / 2
        e312_norms = np.linalg.norm(e312, axis=1, keepdims=True)
        area_gradients = e312_norms * altitudes / 2
        self.dA_dt = area_gradients.reshape(self.nT, 6) # nT, 2, 3 -> nT, 6
    
    def _compute_edge_to_triangle_mapping(self):
        """
        Compute mapping from edges to triangles.
        each edge row corresponds to two columnes (triangles)
        boundary edges map to two repeat triangles
        interior edges map to two non-repeat triangles
        """
        # Create sparse matrix to map triangles to edges
        from scipy.sparse import csr_matrix
        row_inds = np.tile(np.arange(self.nT), 3)
        col_inds = self.triangles_to_edges.T.flatten()
        data = np.ones(len(row_inds))
        T2E_mat = csr_matrix((data, (row_inds, col_inds)), shape=(self.nT, self.nE))
        
        # Separate boundary and interior edges
        boundary_mask = self.is_boundary_edge
        boundary_E2T = T2E_mat[:, self.is_boundary_edge].T
        interior_E2T = T2E_mat[:, ~self.is_boundary_edge].T
        
        # Retrieve edge to triangle maps separately between interior and boundary edges
        interior_edge_inds, interior_tri_inds = interior_E2T.nonzero()
        perm = np.argsort(interior_edge_inds)
        interiorEdges2TriInds = interior_tri_inds[perm].reshape(-1,2) # two edges per triangle

        boundary_edge_inds, boundary_tri_inds = boundary_E2T.nonzero()
        perm = np.argsort(boundary_edge_inds)
        boundaryEdges2TriInds = boundary_tri_inds[perm] # one edge per triangle

        self.edges_to_triangles = np.zeros((self.nE, 2), dtype=int)
        self.edges_to_triangles[self.is_boundary_edge,0] = boundaryEdges2TriInds
        self.edges_to_triangles[self.is_boundary_edge,1] = boundaryEdges2TriInds
        self.edges_to_triangles[~self.is_boundary_edge,:] = interiorEdges2TriInds

        """
        # Debug verification. Ensures mesh maps are self compatible
        
        for eind in range(self.nE):
            etris = self.edges_to_triangles[eind,:]
            if self.is_boundary_edge[eind]:
                assert len(np.unique(etris))==1
            else:
                assert len(np.unique(etris))==2
            evs = self.edges[eind,:]
            for t in etris:
                assert eind in self.triangles_to_edges[t,:]
                assert evs[0] in self.T[t,:]
                assert evs[1] in self.T[t,:]
        """

        pass

    def render_to(self, pngfilename: str = 'a.png', return_figax: bool = False):
        """
        Quick and dirty rendering of mesh to png for debug ease
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        fig, ax = plt.subplots(dpi=400)
        ax.set_aspect('equal')
        X, T = self.X, self.T
        triangles = X[T]  # (nT, 3, 2)
        pc = PolyCollection(triangles, facecolors='w', edgecolors='k', alpha=1, linewidth=0.5)
        ax.add_collection(pc)
        plt.plot(0,0) # triggers polycollection draw
        ax.set_ylim(self.X[:,1].max(), 0)
        plt.savefig(pngfilename,dpi=400)
        print(f"Saved mesh render to {pngfilename}")

        if return_figax:
            return (fig,ax)
        plt.close(fig)

    def get_non_butterfly_overlapping_edges(self, candidate_edges: np.ndarray) -> np.ndarray:
        """
        Given a set of edges in a triangle mesh, greedily find the subset of edges that do not share any adjacent triangles: "non butterfly overlap"
        """
        assert max(candidate_edges) < self.nE and min(candidate_edges) >= 0, "Candidate edge isnt a valid edge."
        non_overlapping_edges_to_flip = []
        occupied_triangles = np.zeros(self.nT, dtype=bool)
        for edge in candidate_edges:
            butterflyTinds = self.edges_to_triangles[edge,:]
            # check overlaps
            if not any(occupied_triangles[butterflyTinds]):
                non_overlapping_edges_to_flip.append(edge)
                occupied_triangles[butterflyTinds] = True
        return non_overlapping_edges_to_flip
    
    def get_v12345(self, edge_subset_inds = None):
        """
        Returns v1,v2,v3,v4,v5 satisfying these rules
        # each edge is made of verts v23. opposing vertices are v1 and v3.
        # v5 accounts for one new vertex for every edge
        """
        if edge_subset_inds is None:
            edge_subset_inds = np.arange(self.nE)
        nE = edge_subset_inds.shape[0]

        v123 = self.T[self.edges_to_triangles[edge_subset_inds,0],:];
        v234 = self.T[self.edges_to_triangles[edge_subset_inds,1],:];
        v23 = self.edges[edge_subset_inds,:];
        
        II,JJ = np.where((v123.reshape(nE,1,3)==v23[:,:,np.newaxis]).sum(axis=1) == 0)
        perm = np.argsort(II)
        Ip = II[perm]
        Jp = JJ[perm]
        v1 = v123[Ip, Jp];
        ip1 = (Jp+2)%3; ip1[ip1==0]=3; ip1 -= 1 # convert to 1 indexing for mod, then back to 0 indexing at the end
        ip2 = (Jp+3)%3; ip2[ip2==0]=3; ip2 -= 1
        v2 = v123[Ip, ip1]
        v3 = v123[Ip, ip2]
        II,JJ = np.where((v234.reshape(nE,1,3)==v23[:,:,np.newaxis]).sum(axis=1) == 0)
        perm = np.argsort(II)
        Ip = II[perm]
        Jp = JJ[perm]
        v4 = v234[Ip, Jp];

        """
        # Debug check
        # assert vectorized extraction was correct
        for i in range(nE):
            assert v1[i] in v123[i,:]
            assert v2[i] in v123[i,:]
            assert v3[i] in v123[i,:]

            assert v2[i] in v234[i,:]
            assert v3[i] in v234[i,:]
            assert v4[i] in v234[i,:]

            assert v2[i] in mesh.edges[i,:]
            assert v3[i] in mesh.edges[i,:]
        """

        v5 = self.nX + np.arange(nE)
        return v1,v2,v3,v4,v5

def mesh_from_xt(X: np.ndarray, T: np.ndarray, is_triangle_soup: bool = False) -> Mesh:
    """
    Factory function to create mesh from vertices and triangles.
    Args:
        X: Vertex coordinates (n_vertices, 2)
        T: Triangle connectivity (n_triangles, 3)
        is_triangle_soup: Whether triangles form a soup (no connectivity)
    Returns:
        Mesh object
    """
    return Mesh(X, T, is_triangle_soup)
