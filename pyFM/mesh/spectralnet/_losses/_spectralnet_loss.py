import torch
import torch.nn as nn


class SpectralNetLoss(nn.Module):
    def __init__(self):
        super(SpectralNetLoss, self).__init__()

    def forward(
        self, W: torch.Tensor, Y: torch.Tensor, is_normalized: bool = False, cotangent_weights: torch.Tensor =None, 
        is_point_cloud = True, A: torch.Tensor =None
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
        m = Y.size(0)

        # # Compute pairwise distances using Y
        # Dy = torch.cdist(Y, Y)

        # # Weight the pairwise distances by cotangent weights
        # weighted_Dy = Dy.pow(2) *  cotangent_weights * A

        # # Compute the loss using the weighted distances
        # loss = torch.sum(W * weighted_Dy) / (2 * m)
        L =  torch.inverse(A) @ cotangent_weights
        loss= torch.trace(Y.T@L@Y)/ (2 * m)
        return loss
        #return  torch.trace(Y.T@cotangent_weights@Y)
        # m = Y.size(0)
        # if is_normalized:
        #     D = torch.sum(W, dim=1)
        #     Y = Y / torch.sqrt(D)[:, None]

        # Dy = torch.cdist(Y, Y)
        # loss = torch.sum(W * Dy.pow(2)) / (2 * m)
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
