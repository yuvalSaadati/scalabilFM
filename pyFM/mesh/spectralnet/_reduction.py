import torch
import numpy as np
import matplotlib.pyplot as plt

from ._utils import *
from ._cluster import SpectralNet
from sklearn.cluster import KMeans
from ._metrics import Metrics


class SpectralReduction:
    def __init__(
        self,
        n_components: int,
        is_sparse_graph: bool = False,
        spectral_hiddens: list = [1024, 1024, 512, 10],
        spectral_epochs: int = 30,
        spectral_lr: float = 1e-3,
        spectral_lr_decay: float = 0.1,
        spectral_min_lr: float = 1e-6,
        spectral_patience: int = 5,
        spectral_batch_size: int = 1024,
        spectral_n_nbg: int = 30,
        spectral_scale_k: int = 15,
        spectral_is_local_scale: bool = True,
        should_compute_acc: bool = False,
        should_true_eigenvectors: bool = True,
        t: int = 0,
        return_eigenvalues: bool = False,
    ):
        """SpectralNet is a class for implementing a Deep learning model that performs spectral clustering.
        This model optionally utilizes Autoencoders (AE) and Siamese networks for training.

        Parameters
        ----------
        n_components : int
            The number of components to keep.

        should_use_ae : bool, optional (default=False)
            Specifies whether to use the Autoencoder (AE) network as part of the training process.

        should_use_siamese : bool, optional (default=False)
                Specifies whether to use the Siamese network as part of the training process.

        is_sparse_graph : bool, optional (default=False)
            Specifies whether the graph Laplacian created from the data is sparse.

        ae_hiddens : list, optional (default=[512, 512, 2048, 10])
            The number of hidden units in each layer of the Autoencoder network.

        ae_epochs : int, optional (default=30)
            The number of epochs to train the Autoencoder network.

        ae_lr : float, optional (default=1e-3)
            The learning rate for the Autoencoder network.

        ae_lr_decay : float, optional (default=0.1)
            The learning rate decay factor for the Autoencoder network.

        ae_min_lr : float, optional (default=1e-7)
            The minimum learning rate for the Autoencoder network.

        ae_patience : int, optional (default=10)
            The number of epochs to wait before reducing the learning rate for the Autoencoder network.

        ae_batch_size : int, optional (default=256)
            The batch size used during training of the Autoencoder network.

        siamese_hiddens : list, optional (default=[1024, 1024, 512, 10])
            The number of hidden units in each layer of the Siamese network.

        siamese_epochs : int, optional (default=30)
            The number of epochs to train the Siamese network.

        siamese_lr : float, optional (default=1e-3)
            The learning rate for the Siamese network.

        siamese_lr_decay : float, optional (default=0.1)
            The learning rate decay factor for the Siamese network.

        siamese_min_lr : float, optional (default=1e-7)
            The minimum learning rate for the Siamese network.

        siamese_patience : int, optional (default=10)
            The number of epochs to wait before reducing the learning rate for the Siamese network.

        siamese_n_nbg : int, optional (default=2)
            The number of nearest neighbors to consider as 'positive' pairs by the Siamese network.

        siamese_use_approx : bool, optional (default=False)
            Specifies whether to use Annoy instead of KNN for computing nearest neighbors,
            particularly useful for large datasets.

        siamese_batch_size : int, optional (default=256)
            The batch size used during training of the Siamese network.

        spectral_hiddens : list, optional (default=[1024, 1024, 512, 10])
            The number of hidden units in each layer of the Spectral network.

        spectral_epochs : int, optional (default=30)
            The number of epochs to train the Spectral network.

        spectral_lr : float, optional (default=1e-3)
            The learning rate for the Spectral network.

        spectral_lr_decay : float, optional (default=0.1)
            The learning rate decay factor.

        should_compute_acc : bool, optional (default=False)
            Specifies whether to compute the accuracy score of the clustering algorithm.

        should_true_eigenvectors : bool, optional (default=True)
            Specifies whether to compute the true eigenvectors of the Laplacian of the input data.

        t : int, optional (default=0)
            The diffusion time for the diffusion map algorithm.

        return_eigenvalues : bool, optional (default=False)
            Specifies whether to return the eigenvalues of the Laplacian of the input data."""

        self.n_components = n_components
        self.is_sparse_graph = is_sparse_graph
        self.spectral_hiddens = spectral_hiddens
        self.spectral_epochs = spectral_epochs
        self.spectral_lr = spectral_lr
        self.spectral_lr_decay = spectral_lr_decay
        self.spectral_min_lr = spectral_min_lr
        self.spectral_patience = spectral_patience
        self.spectral_n_nbg = spectral_n_nbg
        self.spectral_scale_k = spectral_scale_k
        self.spectral_is_local_scale = spectral_is_local_scale
        self.spectral_batch_size = spectral_batch_size
        self.should_compute_acc = should_compute_acc
        self.X_new = None
        self.ortho_matrix = np.eye(n_components)
        self.eigenvalues = np.ones(n_components)
        self.t = t
        self.return_eigenvalues = return_eigenvalues
        if t > 0 or return_eigenvalues:
            self.should_true_eigenvectors = True
        else:
            self.should_true_eigenvectors = should_true_eigenvectors
        self._spectralnet = None

    def _fit(self, X: torch.Tensor, y: torch.Tensor, W=None, A=None,facelist=None) -> np.ndarray:
        """Fit the SpectralNet model to the input data.

        Parameters
        ----------
        X : torch.Tensor
            The input data of shape (n_samples, n_features).

        y: torch.Tensor
            The labels of the input data of shape (n_samples,).

        Returns
        -------
        np.ndarray
            The fitted embeddings of shape (n_samples, n_components).
        """
        if X.type != torch.FloatTensor:
            X = X.type(torch.FloatTensor)

        # Normalize the input data
        # X = (X - X.mean(axis=0)) / X.std(axis=0)

        self.spectral_batch_size = min(self.spectral_batch_size, X.shape[0])

        n_batches = (X.shape[0] // self.spectral_batch_size)
        if n_batches <= 25:
            self.spectral_patience = 10
        else:
            self.spectral_patience = max(1, 250//n_batches)

        if self._spectralnet is None:
            self._spectralnet = SpectralNet(
                n_clusters=self.n_components,
                spectral_hiddens=self.spectral_hiddens,
                spectral_epochs=self.spectral_epochs,
                spectral_lr=self.spectral_lr,
                spectral_lr_decay=self.spectral_lr_decay,
                spectral_min_lr=self.spectral_min_lr,
                spectral_patience=self.spectral_patience,
                spectral_n_nbg=self.spectral_n_nbg,
                spectral_scale_k=self.spectral_scale_k,
                spectral_is_local_scale=self.spectral_is_local_scale,
                spectral_batch_size=self.spectral_batch_size,
            )

        self._spectralnet.fit(X, y, W, A,facelist)

        if self.should_true_eigenvectors:
            self.compute_ortho_matrix(X, W, A)

    def _predict(self, X: torch.Tensor, W = None, A= None) -> np.ndarray:
        """Predict embeddings for the input data using the fitted SpectralNet model.

        Parameters
        ----------
        X : torch.Tensor
            The input data of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            The predicted embeddings of shape (n_samples, n_components).
        """
        self._spectralnet.predict(X, W, A)
        return self._spectralnet.embeddings_

    def _transform(self, X: torch.Tensor, W, A) -> np.ndarray:
        """Transform the input data into embeddings using the fitted SpectralNet model.

        Parameters
        ----------
        X : torch.Tensor
            The input data of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            The transformed embeddings of shape (n_samples, n_components).
        """
        if X.type != torch.FloatTensor:
            X = X.type(torch.FloatTensor)
        if self.return_eigenvalues:
            return self._predict(X, W, A) @ self.ortho_matrix @ np.diag((1-self.eigenvalues) ** self.t), self.eigenvalues

            #return self._predict(X, W, A) , self.eigenvalues

        return self._predict(X, W, A) @ self.ortho_matrix @ np.diag((1-self.eigenvalues) ** self.t)

    def fit_transform(self, X: torch.Tensor, y: torch.Tensor = None, W=None, A=None,facelist=None) -> np.ndarray:
        """Fit the SpectralNet model to the input data and transform it into embeddings.

        This is a convenience method that combines the fit and transform steps.

        Parameters
        ----------
        X : torch.Tensor
            The input data of shape (n_samples, n_features).

        y: torch.Tensor
            The labels of the input data of shape (n_samples,).

        Returns
        -------
        np.ndarray
            The fitted and transformed embeddings of shape (n_samples, n_components).
        """
        self._fit(X, y, W, A, facelist)
        return self._transform(X, W, A)

    def _get_laplacian_of_small_batch(self, batch: torch.Tensor, W=None, A=None) -> np.ndarray:
        """Get the Laplacian of a small batch of the input data

        Parameters
        ----------

        batch : torch.Tensor
            A small batch of the input data of shape (batch_size, n_features).

        Returns
        -------
        np.ndarray
            The Laplacian of the small batch of the input data.



        """

        # W = get_affinity_matrix(batch, n_nbg=self.spectral_n_nbg, device=self._spectralnet.device)
        # L = get_laplacian(W)
        return np.linalg.inv(A.toarray()) @ W.toarray()

    def _remove_smallest_eigenvector(self, V: np.ndarray) -> np.ndarray:
        """Remove the constant eigenvector from the eigenvectors of the Laplacian of a small batch of the input data.


        Parameters
        ----------
        V : np.ndarray
            The eigenvectors of the Laplacian of a small batch of the input data.


        Returns
        -------
        np.ndarray
            The eigenvectors of the Laplacian of a small batch of the input data without the constant eigenvector.
        """

        batch_raw, batch_encoded = self._spectralnet.get_random_batch()
        L_batch = self._get_laplacian_of_small_batch(batch_encoded)
        V_batch = self._predict(batch_raw)
        eigenvalues = np.diag(V_batch.T @ L_batch @ V_batch)
        indices = np.argsort(eigenvalues)
        smallest_index = indices[0]
        V = V[:, np.arange(V.shape[1]) != smallest_index]
        V = V[
            :,
            (np.arange(V.shape[1]) == indices[1])
            | (np.arange(V.shape[1]) == indices[2]),
        ]

        return V

    def visualize(
        self, V: np.ndarray, y: torch.Tensor = None, n_components: int = 1
    ) -> None:
        """Visualize the embeddings of the input data using the fitted SpectralNet model.

        Parameters
        ----------
        V : torch.Tensor
            The reduced data of shape (n_samples, n_features) to be visualized.
        y : torch.Tensor
            The input labels of shape (n_samples,).
        """
        # V = self._remove_smallest_eigenvector(V)
        # print('V.shape', V.shape)

        # plot_laplacian_eigenvectors(V, y)
        if self.should_compute_acc:
            cluster_labels = self._get_clusters_by_kmeans(V)
            acc = Metrics.acc_score(cluster_labels, y.detach().cpu().numpy(), n_clusters=10)
            print("acc with 2 components: ", acc)

        if n_components > 1:
            x_axis = V[:, 0]
            y_axis = V[:, 1]

        elif n_components == 1:
            x_axis = V
            y_axis = np.zeros_like(V)

        else:
            raise ValueError(
                "n_components must be a positive integer (greater than 0))"
            )

        if y is None:
            plt.scatter(x_axis, y_axis)
        else:
            # plt.scatter(x_axis, y_axis, c=y, cmap="tab10", s=3)
            plt.scatter(x_axis, y_axis, c=y, s=3)

        plt.show()

    def _get_clusters_by_kmeans(self, embeddings: np.ndarray) -> np.ndarray:
        """Performs k-means clustering on the spectral-embedding space.

        Parameters
        ----------
        embeddings : np.ndarray
            The spectral-embedding space.

        Returns
        -------
        np.ndarray
            The cluster assignments for the given data.
        """

        kmeans = KMeans(n_clusters=self.n_components, n_init=10).fit(embeddings)
        cluster_assignments = kmeans.predict(embeddings)
        return cluster_assignments

    def compute_ortho_matrix(self, X: torch.Tensor, W=None, A=None) -> None:
        """Compute the orthogonal matrix for the spectral embeddings.

        Parameters
        ----------
        X : torch.Tensor
            The input data of shape (n_samples, n_features).
        """
        pred = self._predict(X)
        W = W.toarray()
        A = A.toarray()
        numerator = pred.T @ W @ pred
        denominator = pred.T @ A @ pred

        # Ensure denominator stability
        denominator = np.where(np.abs(denominator) < 1e-8, np.ones_like(denominator), denominator)
        L = numerator / denominator

        # Sum of the diagonal (first k eigenvalues for k dimensions of Y)
        L_diag = np.diag(L)
        self.eigenvalues = L_diag

        # pred /= np.linalg.norm(pred, axis=0) ## need to norm?
        # Lambda = self._get_lambda_on_multi_batches(X, W, A)

        # ortho_matrix, eigenvalues_pred, _ = np.linalg.svd(Lambda)
        # eigenvalues_pred = eigenvalues_pred.real
        # self.ortho_matrix = np.array(ortho_matrix.real)
        # indices = np.argsort(eigenvalues_pred)
        # self.ortho_matrix = np.array(self.ortho_matrix[:, indices])
        # self.eigenvalues = eigenvalues_pred[indices]
        # print('Eigenvalues:\n', eigenvalues_pred[indices])

    def _get_lambda_on_multi_batches(self, X, W=None, A=None) -> np.ndarray:
        """Get the mean eigenvalues matrix of the Laplacian of the input data.

        Returns
        -------
        np.ndarray
            The eigenvalues matrix of the Laplacian of the input data.
        """
        n_batches = (X.shape[0] // self.spectral_batch_size)
        Lambda = self._get_lambda_on_batch(None, 1e-6, W, A)
        for i in range(1, n_batches):
            Lambda += self._get_lambda_on_batch(None, 1e-6, W, A)
        Lambda = torch.Tensor(Lambda) / n_batches
        return (Lambda + Lambda.T) / 2

    def _get_lambda_on_batch(self, batch_size = None, eps = 1e-6, W=None, A=None) -> np.ndarray:
        """Get the eigenvalues of the Laplacian of a small batch of the input data.

        Returns
        -------
        np.ndarray
            The eigenvalues of the Laplacian of a small batch of the input data.
        """

        if batch_size is None:
            batch_size = self.spectral_batch_size
        batch_raw, batch_encoded = self._spectralnet.get_random_batch(batch_size)
        L_batch = self._get_laplacian_of_small_batch(batch_encoded, W, A)
        V_batch = self._predict(batch_raw)
        # indices = np.where(np.linalg.norm(V_batch, axis=0) > eps)
        # V_batch[:, indices] = V_batch[:, indices] / np.linalg.norm(V_batch[:, indices], axis=0)
        V_batch = V_batch / np.linalg.norm(V_batch, axis=0)
        Lambda = V_batch.T @ L_batch @ V_batch
        return Lambda
