import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
from sklearn.neighbors import kneighbors_graph
from tqdm import trange
from .._utils import *
from ._trainer import Trainer
from .._losses import SpectralNetLoss
from .._models import SpectralNetModel
import scipy.sparse as sparse
import json
from ....mesh import laplacian


class SpectralTrainer:
    def __init__(self, config: dict, device: torch.device, is_sparse: bool = False):
        """
        Initialize the SpectralNet model trainer.

        Parameters
        ----------
        config : dict
            The configuration dictionary.
        device : torch.device
            The device to use for training.
        is_sparse : bool, optional
            Whether the graph-laplacian obtained from a mini-batch is sparse or not.
            If True, the batch is constructed by taking 1/5 of the original random batch
            and adding 4 of its nearest neighbors to each sample. Defaults to False.

        Notes
        -----
        This class is responsible for training the SpectralNet model.
        The configuration dictionary (`config`) contains various settings for training.
        The device (`device`) specifies the device (CPU or GPU) to be used for training.
        The `is_sparse` flag is used to determine the construction of the batch when the graph-laplacian is sparse.
        """

        self.device = device
        self.is_sparse = is_sparse
        self.spectral_config = config
        self.lr = self.spectral_config["lr"]
        self.n_nbg = self.spectral_config["n_nbg"]
        self.min_lr = self.spectral_config["min_lr"]
        self.epochs = self.spectral_config["epochs"]
        self.scale_k = self.spectral_config["scale_k"]
        self.lr_decay = self.spectral_config["lr_decay"]
        self.patience = self.spectral_config["patience"]
        self.architecture = self.spectral_config["hiddens"]
        self.batch_size = self.spectral_config["batch_size"]
        self.is_local_scale = self.spectral_config["is_local_scale"]
        self.spectral_net = None
        

    # Function to apply geometric transformations for augmentation using PyTorch
    def augment_mesh(self, vertices):
        # Ensure vertices is a PyTorch tensor
        if isinstance(vertices, np.ndarray):
            vertices = torch.tensor(vertices, dtype=torch.float32)
        
        # Rotation around the z-axis
        angle = torch.rand(1) * 2 * np.pi 
        rotation_matrix = torch.tensor([[torch.cos(angle), -torch.sin(angle), 0],
                                        [torch.sin(angle), torch.cos(angle), 0],
                                        [0, 0, 1]])
        
        # Apply rotation
        vertices_rotated = torch.mm(vertices, rotation_matrix)
        
        # Uniform scaling
        scale = torch.tensor(1.0) + (torch.rand(1) - 0.5) * 0.2  # Scale factor between 0.9 and 1.1
        vertices_scaled = vertices_rotated * scale
        
        # Translation
        translation = (torch.rand(3) - 0.5) * 0.2  # Translation vector between -0.1 and 0.1 for each axis
        vertices_translated = vertices_scaled + translation
        return vertices_translated
    def filter_and_reindex_faces(self, faces, batch_indices):
        batch_indices_np = np.array(batch_indices, dtype=int)

        # Create a mask to filter faces where all vertices are in the batch
        mask = np.isin(faces, batch_indices_np).all(axis=1)
        filtered_faces = faces[mask]

        # Prepare the new indices map
        new_indices_map = np.full(np.max(batch_indices_np) + 1, -1, dtype=int)  # Initialize with -1 for non-existent indices
        new_indices_map[batch_indices_np] = np.arange(len(batch_indices_np))

        # Re-index faces: Apply new_indices_map to every vertex in the filtered faces
        if filtered_faces.size > 0:  # Ensure there are faces to process
            reindexed_faces = new_indices_map[filtered_faces]
        else:
            reindexed_faces = np.array([])  # Handle the case where no faces are left after filtering

        return reindexed_faces
    
    def create_mini_batches(self, vertices, batch_size):
        mini_batches = []
        vertex_indices = np.arange(vertices.shape[0])
        for i in range(0, len(vertex_indices), batch_size):
            batch_indices = vertex_indices[i:i + batch_size]
            mini_batches.append((vertices[batch_indices], batch_indices))
        return mini_batches
    
    def train(
        self, X: torch.Tensor, y: torch.Tensor, siamese_net: nn.Module = None, cotangent_weights = None, A = None,facelist=None
    ) -> SpectralNetModel:
        """
        Train the SpectralNet model.

        Parameters
        ----------
        X : torch.Tensor
            The dataset to train on.
        y : torch.Tensor, optional
            The labels of the dataset in case there are any.
        siamese_net : nn.Module, optional
            The siamese network to use for computing the affinity matrix.

        Returns
        -------
        SpectralNetModel
            The trained SpectralNet model.

        Notes
        -----
        This function trains the SpectralNet model using the provided dataset (`X`) and labels (`y`).
        If labels are not provided (`y` is None), unsupervised training is performed.
        The siamese network (`siamese_net`) is an optional parameter used for computing the affinity matrix.
        The trained SpectralNet model is returned as the output.
        """
        # X = self.augment_mesh(X)
        self.X = X.view(X.size(0), -1)
        self.y = y
        self.counter = 0
        self.siamese_net = siamese_net
        if self.spectral_net is None:
            self.spectral_net = SpectralNetModel(
                self.architecture, input_dim=self.X.shape[1]
            ).to(self.device)
        
            self.optimizer = optim.Adam(self.spectral_net.parameters(), lr=self.lr)

            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=self.lr_decay, patience=self.patience
            )
            self.criterion = SpectralNetLoss()


        print("Training SpectralNet:")
        
        ##2: building vertices batches accoring to faces, and calculate w in 1024 batch size
        # FL_loader = torch.chunk(FL, chunks=len(train_loader), dim=0)
        # FL_loader = [FL[i:i+train_loader.batch_size] for i in range(0, FL.size(0), train_loader.batch_size)]
        # vertices_batches = []
        # faces_indexes = []
        # W_batch_array = []
        # for batch_indices in FL_loader:
        #     vertices_batch = self.X[batch_indices].view(batch_indices.shape[0]*3, 3)
        #     W_batch = cotangent_weights[batch_indices].reshape(3*batch_indices.shape[0], cotangent_weights.shape[1])
        #     W_batch = W_batch[:, 0:W_batch.shape[0]]
        #     W_batch_array.append(W_batch)
        #     corresponding_indices = np.arange(batch_indices.shape[0]*3).reshape(batch_indices.shape[0], 3)
        #     faces_indexes.append(corresponding_indices)
        #     vertices_batches.append(vertices_batch)
        t = trange(self.epochs, leave=True)
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        for epoch in t:

            train_loss = 0.0
            batch_size = 32
            #for batch_indices in self.create_mini_batches(self.X, batch_size):
            X_orth = self.X
            X_grad = self.X
            X_grad = X_grad.to(device=self.device)
            X_grad = X_grad.view(X_grad.size(0), -1)
            X_orth = X_orth.to(device=self.device)
            X_orth = X_orth.view(X_orth.size(0), -1)

            if self.is_sparse:
                X_grad = make_batch_for_sparse_grapsh(X_grad)
                X_orth = make_batch_for_sparse_grapsh(X_orth)

            # Orthogonalization step
            self.spectral_net.eval()
            self.spectral_net(X_orth, should_update_orth_weights=True)

            # Gradient step
            self.spectral_net.train()
            self.optimizer.zero_grad()
            Y = self.spectral_net(X_grad, should_update_orth_weights=False)

            if self.siamese_net is not None:
                with torch.no_grad():
                    X_grad = self.siamese_net.forward_once(X_grad)
        

            loss = self.criterion(Y,False, cotangent_weights, A)

            loss.backward()
            self.optimizer.step()
            train_loss = loss.item()

            current_lr = self.optimizer.param_groups[0]["lr"]
            if current_lr <= self.spectral_config["min_lr"]:
                break
            scheduler.step()
            torch.nn.utils.clip_grad_norm_(self.spectral_net.parameters(), max_norm=1.0)
            print(f"Train Loss epoch {epoch}: {train_loss}, LR: {current_lr}")
            t.refresh()
            # print(self.spectral_net.parameters())
            # if epoch == 39:
            #     W = self._get_affinity_matrix(X_grad)
            #     plot_sorted_laplacian(W, Y_grad)    
        return self.spectral_net
    def get_grassman_distance(self, A: np.ndarray, B: np.ndarray) -> float:
        """
        Computes the Grassmann distance between the subspaces spanned by the columns of A and B.

        Parameters
        ----------
        A : np.ndarray
            Numpy ndarray.
        B : np.ndarray
            Numpy ndarray.

        Returns
        -------
        float
            The Grassmann distance.
        """
        A, _ = np.linalg.qr(A)
        B, _ = np.linalg.qr(B)
        M = np.dot(np.transpose(A), B)
        _, s, _ = np.linalg.svd(M, full_matrices=False)
        s = 1 - np.square(s)
        grassmann = np.sum(s)
        return grassmann
    
    def normalize_eigenvectors(self, eigenvectors):
        """
        Normalize a set of eigenvectors.

        Parameters:
        eigenvectors (numpy.ndarray): Array containing eigenvectors as columns.

        Returns:
        numpy.ndarray: Normalized eigenvectors.
        """
        # Compute the norms of each eigenvector
        norms = np.linalg.norm(eigenvectors, axis=0)
        normalized_eigenvectors = eigenvectors/norms
        return normalized_eigenvectors

    def validate(self, valid_loader: DataLoader) -> float:
        valid_loss = 0.0
        self.spectral_net.eval()
        with torch.no_grad():
            for batch in valid_loader:
                X, y = batch
                X, y = X.to(self.device), y.to(self.device)

                if self.is_sparse:
                    X = make_batch_for_sparse_grapsh(X)

                Y = self.spectral_net(X, should_update_orth_weights=False)
                with torch.no_grad():
                    if self.siamese_net is not None:
                        X = self.siamese_net.forward_once(X)

                W = self._get_affinity_matrix(X)

                loss = self.criterion(W, Y)
                valid_loss += loss.item()

        valid_loss /= len(valid_loader)
        return valid_loss

    def _get_affinity_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """
        This function computes the affinity matrix W using the Gaussian kernel.

        Args:
            X (torch.Tensor):   The input data

        Returns:
            torch.Tensor: The affinity matrix W
        """

        is_local = self.is_local_scale
        n_neighbors = self.n_nbg
        scale_k = self.scale_k
        Dx = torch.cdist(X, X)
        Dis, indices = get_nearest_neighbors(X, k=n_neighbors + 1)
        scale = compute_scale(Dis, k=scale_k, is_local=is_local)
        W = get_gaussian_kernel(
            Dx, scale, indices, device=self.device, is_local=is_local
        )
        return W

    def _get_data_loader(self) -> tuple:
        """
        This function returns the data loaders for training, validation and testing.

        Returns:
            tuple:  The data loaders
        """
        if self.y is None:
            self.y = torch.zeros(len(self.X))
        train_size = int(0.9 * len(self.X))
        valid_size = len(self.X) - train_size
        dataset = TensorDataset(self.X, self.y)
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        ortho_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=self.batch_size, shuffle=False
        )
        return train_loader, ortho_loader, valid_loader
