import numpy as np

from pyFM.mesh import TriMesh
from pyFM.functional import FunctionalMapping

import meshplot as mp

from IPython.display import display, HTML
mp.offline()

def plot_mesh(myMesh, cmap=None):
    html = mp.plot(myMesh.vertlist, myMesh.facelist, c=cmap).to_html(imports=True, html_frame=False)
    display(HTML(html))



def double_plot(myMesh1, myMesh2, cmap1=None, cmap2=None):
    d = mp.subplot(myMesh1.vertlist, myMesh1.facelist, c=cmap1, s=[2, 2, 0])
    mp.subplot(myMesh2.vertlist, myMesh2.facelist, c=cmap2, s=[2, 2, 1], data=d)


def visu(vertices):
    min_coord, max_coord = np.min(vertices, axis=0, keepdims=True), np.max(vertices, axis=0, keepdims=True)
    cmap = (vertices - min_coord) / (max_coord - min_coord)
    return cmap


"""# 2- Loading and processing a mesh

### Basic Mesh methods

A TriMesh class can be created from a path (to a .off or a .obj file) or simply an array of vertices and an optional array of faces.

The mesh can be centered, area-normalized, rotated or translated when loading.


Vertices and faces are stored in the 'vertlist' and 'facelist' attributes. One can also use 'mesh.vertices' and 'mesh.faces' to access them. While these notations can feel non-intuitive they result in clearer functions as it avoids expressions of the form ```mesh.vertices - vertices```.

A TriMesh class possess multiple attributes like edges, per-face area, per-vertex area, per-face normals, per-vertex normals, ...
"""

mesh1 = TriMesh('data/cat-00.off', area_normalize=True, center=False)
mesh2 = TriMesh(mesh1.vertlist, mesh1.facelist)

# Attributes are computed on the fly and cached
edges = mesh1.edges

area = mesh1.area

face_areas = mesh1.face_areas
vertex_areas = mesh1.vertex_areas
face_normals = mesh1.normals

# AREA WEIGHTED VERTEX NORMALS
vertex_normals_a = mesh1.vertex_normals

# UNIFORM WEIGHTED VERTEX NORMALS
mesh1.set_vertex_normal_weighting('uniform')
vertex_normals_u = mesh1.vertex_normals

"""### Geodesics

We propose three versions to compute geodesics :
- Heat method - based on [potpourri3d](https://github.com/nmwsharp/potpourri3d) using robust laplacian (recommended)
- Heat method - pure python implementation from pyFM (not robust but control on the whole code)
- Dijkstra
"""

# Geodesic distance from a given index
# Set robust to False to obtain result from the Python implementation
dists = mesh1.geod_from(1000, robust=True)

S1_geod = mesh1.get_geodesic(verbose=True)

"""### Laplacian and functions

The spectrum of the LBO can be computed easily.

Eigenvalues and eigenvectors are stored in the ```mesh.eigenvalues``` and ```mesh.eigenvectors``` attributes.

Gradient and divergence can be computed using the associated methods. Using the ```mesh.project``` and ```mesh.unproject``` functions allows to switch between seeing a function in the LBO basis or on the complete shape.

The squared $L^2$ norm and $H^1_0$ norm can be computed via the ```mesh.l2_sqnorm``` and ```mesh.h1_sqnorm``` methods.
"""

# By default does not use the intrinsic delaunay Laplacian
mesh1.process(k=100, intrinsic=False, verbose=True)

# plot the third eigenfunction
plot_mesh(mesh1, mesh1.eigenvectors[:, 2])

"""# 3 - Computing the functional map

**Loading data**
"""

mesh1 = TriMesh('data/cat-00.off')
mesh2 = TriMesh('data/lion-00.off')
print(f'Mesh 1 : {mesh1.n_vertices:4d} vertices, {mesh1.n_faces:5d} faces\n'
      f'Mesh 2 : {mesh2.n_vertices:4d} vertices, {mesh2.n_faces:5d} faces')

double_plot(mesh1, mesh2)

"""**Computing descriptors**"""

process_params = {
    'n_ev': (35, 35),  # Number of eigenvalues on source and Target
    'landmarks': np.loadtxt('data/landmarks.txt', dtype=int)[:5],  # loading 5 landmarks
    'subsample_step': 5,  # In order not to use too many descriptors
    'descr_type': 'WKS',  # WKS or HKS
}

model = FunctionalMapping(mesh1, mesh2)
model.preprocess(**process_params, verbose=True);


fit_params = {
    'w_descr': 1e0,
    'w_lap': 1e-2,
    'w_dcomm': 1e-1,
    'w_orient': 0
}

model.fit(**fit_params, verbose=True)

"""**Visualizing the associated point to point map**"""

p2p_21 = model.get_p2p(n_jobs=1)
cmap1 = visu(mesh1.vertlist);
cmap2 = cmap1[p2p_21]
double_plot(mesh1, mesh2, cmap1, cmap2)

"""# 4 - Refining the Functional Map
```model.FM``` returns the current state of functional map. One can change which one is returned by using ```model.change_FM_type(FM_type)```, as one can see below. 

**ICP**
"""

model.icp_refine(verbose=True)
p2p_21_icp = model.get_p2p()
cmap1 = visu(mesh1.vertlist);
cmap2 = cmap1[p2p_21_icp]
double_plot(mesh1, mesh2, cmap1, cmap2)

"""**Zoomout**"""

model.change_FM_type('classic')  # We refine the first computed map, not the icp-refined one
model.zoomout_refine(nit=15, step=1, verbose=True)
print(model.FM.shape)
p2p_21_zo = model.get_p2p()
cmap1 = visu(mesh1.vertlist);
cmap2 = cmap1[p2p_21_zo]
double_plot(mesh1, mesh2, cmap1, cmap2)

"""# Evaluating Results"""

import pyFM.eval

# Compute geodesic distance matrix on the cat mesh
A_geod = mesh1.get_geodesic(verbose=True)

# Load an approximate ground truth map
gt_p2p = np.loadtxt('data/lion2cat', dtype=int)

acc_base = pyFM.eval.accuracy(p2p_21, gt_p2p, A_geod, sqrt_area=mesh1.sqrtarea)

acc_icp = pyFM.eval.accuracy(p2p_21_icp, gt_p2p, A_geod, sqrt_area=np.sqrt(mesh1.area))

acc_zo = pyFM.eval.accuracy(p2p_21_zo, gt_p2p, A_geod, sqrt_area=np.sqrt(mesh1.area))

print(f'Accuracy results\n'
      f'\tBasic FM : {1e3 * acc_base:.2f}\n'
      f'\tICP refined : {1e3 * acc_icp:.2f}\n'
      f'\tZoomOut refined : {1e3 * acc_zo:.2f}\n')