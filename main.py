from pyFM.functional import FunctionalMapping
import meshplot as mp
import numpy as np

from pyFM.mesh import TriMesh


def plot_mesh(myMesh, cmap=None):
    mp.plot(myMesh.vertlist, myMesh.facelist, c=cmap)


def double_plot(myMesh1, myMesh2, cmap1=None, cmap2=None):
    d = mp.subplot(myMesh1.vertlist, myMesh1.facelist, c=cmap1, s=[2, 2, 0])
    mp.subplot(myMesh2.vertlist, myMesh2.facelist, c=cmap2, s=[2, 2, 1], data=d)


def visu(vertices):
    min_coord, max_coord = np.min(vertices, axis=0, keepdims=True), np.max(vertices, axis=0, keepdims=True)
    cmap = (vertices - min_coord) / (max_coord - min_coord)
    return cmap

mesh1 = TriMesh('data/cat-00.off', area_normalize=True, center=False)
mesh2 = TriMesh('data/lion-00.off')
process_params = {
    'n_ev': (10,10),  # Number of eigenvalues on source and Target
    'landmarks': np.loadtxt('data/landmarks.txt',dtype=int)[:5],  # loading 5 landmarks
    'subsample_step': 4,  # In order not to use too many descriptors
    'descr_type': 'WKS',  # WKS or HKS
    'k_process': 10
}
model = FunctionalMapping(mesh1,mesh2)
model.preprocess(**process_params,verbose=True, is_point_cloud=False)
fit_params = {
    'w_descr': 1e-1,
    'w_lap': 1e-2,
    'w_dcomm': 1e-1,
    'w_orient': 0
}

model.fit(**fit_params, verbose=True)