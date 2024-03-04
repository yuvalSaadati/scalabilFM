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





mesh1 = TriMesh(np.load('data/lion-00_1e4.npy'))
mesh2 = TriMesh(np.load('data/cat-00_1e4.npy'))
# mesh1 = TriMesh('data/cat-00.off', area_normalize=True, center=False)
# mesh2 = TriMesh('data/lion-00.off')

print(f'Mesh 1 : {mesh1.n_vertices:4d} vertices, {mesh1.n_faces:5d} faces\n'
      f'Mesh 2 : {mesh2.n_vertices:4d} vertices, {mesh2.n_faces:5d} faces')

double_plot(mesh1,mesh2)


mesh1 = TriMesh(np.load('data/lion-00_1e4.npy'))
mesh2 = TriMesh(np.load('data/cat-00_1e4.npy'))
# mesh1 = TriMesh('data/cat-00.off', area_normalize=True, center=False)
# mesh2 = TriMesh('data/lion-00.off')

print(f'Mesh 1 : {mesh1.n_vertices:4d} vertices, {mesh1.n_faces:5d} faces\n'
      f'Mesh 2 : {mesh2.n_vertices:4d} vertices, {mesh2.n_faces:5d} faces')

double_plot(mesh1,mesh2)


from pyFM.functional import FunctionalMapping

process_params = {
    'n_ev': (500,500),  # Number of eigenvalues on source and Target
    'landmarks': np.loadtxt('data/landmarks_small.txt',dtype=int)[:5],  # loading 5 landmarks
    'subsample_step': 5,  # In order not to use too many descriptors
    'descr_type': 'WKS',  # WKS or HKS
}

model = FunctionalMapping(mesh1,mesh2)
model.preprocess(**process_params,verbose=True);

model.mesh1.save_eigen_vectors('lion-00_1e4_eigen_vectors.npy')
model.mesh1.save_eigen_vectors('cat-00_1e4_eigen_vectors.npy')
