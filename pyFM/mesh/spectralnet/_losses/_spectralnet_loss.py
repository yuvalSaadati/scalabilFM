import torch
import torch.nn as nn
import scipy.sparse as sparse
import numpy as np
import math
import scipy.sparse.linalg
class SpectralNetLoss(nn.Module):
    def __init__(self):
        super(SpectralNetLoss, self).__init__()
        self.index = 0
    
    def forward(
        self, Y: torch.Tensor, is_normalized: bool = False, cotangent_weights=None, 
         A =None
    ) -> torch.Tensor:
        """
        This function computes the loss of the SpectralNet model.
        The loss is the rayleigh quotient of the Laplacian matrix obtained from W,
        and the orthonormalized output of the network.

        Args:
            W (torch.Tensor):               Affinity matrix
            Y (torch.Tensor):               Output of the network
            is_normalized (bool, optional): Whether to use the normalized Laplacian matrix or not.

        Returns:
            torch.Tensor: The loss
    """
        n, m = Y.shape
        # eigenvalues, eigenvectors = sparse.linalg.eigsh(cotangent_weights, k=m, M=A, sigma=-0.01)
       
        cotangent_weights = torch.tensor(cotangent_weights.toarray(), dtype=torch.float32)
        A = torch.tensor(A.toarray(), dtype=torch.float32)
        numerator = Y.T @ cotangent_weights @ Y
        denominator = Y.T @ A @ Y
        # Ensure denominator stability
        denominator = torch.where(denominator.abs() < 1e-8, torch.ones_like(denominator) * 1e-8, denominator)

        L = numerator / denominator
        # Compute loss as the difference between predicted and target eigenvalues
        # loss = torch.sum((L - torch.tensor(eigenvalues, dtype=torch.float32) )** 2)

        L_diag = torch.diag(L)
        loss = torch.sum(L_diag)
        return loss
        # print(f'Y norm min: {torch.norm(Y, dim=0).min()}, Y norm max: {torch.norm(Y, dim=0).max()}')
        # L = (Y.T @cotangent_weights @ Y)/(Y.T @A @ Y)
        # L_diag = torch.diag(L)
        # loss = torch.sum(L_diag)
        # return loss
        #L = torch.linalg.inv(torch.sqrt(A)) @ cotangent_weights @ torch.linalg.inv(torch.sqrt(A))
        # Create a diagonal matrix with the same shape and add 0.01 to the diagonal elements
        # diagonal_matrix = torch.zeros(m) + 0.001
        # diagonal_matrix = torch.diag(diagonal_matrix)
        #L = scipy.sparse.linalg.inv(A) @ cotangent_weights
        #L = torch.tensor(L.toarray(), dtype=torch.float32)
        # eigenvalues1LR, eigenvectors1LR = sparse.linalg.eigs(scipy.sparse.linalg.inv(A)@cotangent_weights, k=2, M=None, which="SR")
        # eigenvalues1, eigenvectors1 = sparse.linalg.eigs(scipy.sparse.linalg.inv(A)@cotangent_weights, k=10, M=None,sigma=-0.01)
        # eigenvalues2, eigenvectors2 = sparse.linalg.eigs(cotangent_weights, k=10, M=A,sigma=-0.01)

        # cotangent_weights = torch.tril(cotangent_weights) + torch.tril(cotangent_weights, diagonal=0).t()
        # cotangent_weights = torch.abs(0.5*(cotangent_weights + cotangent_weights.T))


        #L = torch.tensor(cotangent_weights.toarray(), dtype=torch.float32)- 0.01*torch.tensor(A.toarray(), dtype=torch.float32)
        #P = torch.linalg.inv(torch.diag(torch.sqrt(torch.diag(L)))) 
        # define preconditioner
        # P = torch.diag(torch.diag(L))    
      
        # Y = Y / math.sqrt(m)
       # F = Y.T @ P.T @L @ P @ Y
        # L_diag = torch.diag(Y.T[:, 1:n] @L[1:n, 1:n] @ Y[1:n, :])
        # L_diag = torch.diag(Y.T @L @ Y)


        # n, m = Y.shape
        # # result now contains the result of Y^T L Y
        # print(result)
        # L_diag = torch.diag(Y.T @L @ Y)
        # np.savetxt("debug.csv", Y[:, 0:1].detach().numpy(), delimiter=',')

        # print(torch.sum(torch.diag(Y.T  @L @ Y)))
        # L_=Y.T[0:1, :]@L[:,:]
        # last_layer=L_.T*Y[:,0:1]
        # print(last_layer.sum())
        # std =torch.std(last_layer) 
        # # return loss
        # torch.sum(last_layer[last_layer>std]).item()
        # torch.sum(last_layer[last_layer<std]).item()
        # print(torch.sum(last_layer[last_layer<std]).item() +torch.sum(last_layer[last_layer>=std]).item())
        #return  torch.trace(Y.T@cotangent_weights@Y)
        # if is_normalized:
        # for i in range(len(Y)):
        #     Y[i] = Y[i] / A[i, i]
        
        # Dy = torch.cdist(Y, Y)
        # loss = torch.sum(cotangent_weights * Dy.pow(2)) / (2 * m)
        # if is_point_cloud or cotangent_weights is None:
        #     return loss
        
        # # if cotangent_weights is not None:
        # #     cotangent_weights= cotangent_weights.asformat("array")
        # #     cotangent_weights = torch.from_numpy(cotangent_weights)
        # #     cotangent_weights = cotangent_weights.to(torch.float32)

        # # 1: using w with adding padding to y 
        # # if cotangent_weights is not None:
        # #     Y_pad =torch.nn.functional.pad(Y, (0,0,0, cotangent_weights.shape[0] - Y.shape[0]), value=0)

        # #     trace = torch.trace(Y_pad.T@cotangent_weights@Y_pad)
        # #     return trace
        
        #  # 2: computing w for every barch
       
        # if cotangent_weights is not None:
        #     #trace = torch.trace(Y.T@torch.diag(1.0 / torch.diag(A))@cotangent_weights@Y)
        #     # W_eigenvalues, W_eigenvectors = sparse.linalg.eigsh(cotangent_weights, k=Y.shape[1], M=None,sigma=-0.01)
        #     # normalized_laplacian_eigenvectors= self.normalize_eigenvectors(W_eigenvectors)
        #     # normalized_spectralreduction_eigenvectors = self.normalize_eigenvectors(Y)

        #     # grassmann = self.get_grassman_distance(normalized_laplacian_eigenvectors, normalized_spectralreduction_eigenvectors)
        #     #return trace 
            
        #     #return  torch.trace(Y.T@torch.diag(1.0 / torch.diag(A))@cotangent_weights@Y)
        # return loss
