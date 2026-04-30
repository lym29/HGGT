import trimesh
import numpy as np
import torch

# def get_bbox_size(mesh):
#     bbox = mesh.get_axis_aligned_bounding_box()
#     min_bound = bbox.get_min_bound()
#     max_bound = bbox.get_max_bound()
#     return np.linalg.norm(max_bound - min_bound)

def get_bbox_size(mesh: trimesh.Trimesh) -> float:
    """
    Calculate the norm of the bounding box extents of a trimesh object.
    
    Args:
        mesh (trimesh.Trimesh): The mesh to measure.
        
    Returns:
        float: The diagonal size of the mesh's axis-aligned bounding box.
    """
    if mesh.is_empty:
        return 0.0
    # trimesh.Trimesh.extents provides the size of the AABB along each axis.
    # np.linalg.norm calculates the length of this vector (the diagonal).
    return np.linalg.norm(mesh.extents)

def apply_transform_verts_torch(mesh, rotmat=None, transl=None, device='cpu'):
    """
    Apply a transformation to the mesh.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): The mesh to transform.
        rotmat (torch.Tensor): Rotation matrix of shape (N, 3, 3) where N is the batch size.
        transl (torch.Tensor): Translation vector of shape (N, 3).
        device (str): Device to perform the computation on ('cpu' or 'cuda').
        
    Returns:
        torch.Tensor: Transformed vertices of the mesh.
    """
    verts = torch.from_numpy(np.asarray(mesh.vertices)).float().to(device)
    
    verts = verts.unsqueeze(0).expand(rotmat.shape[0], -1, -1)
    
    if rotmat is not None:
        verts = torch.matmul(verts, rotmat.transpose(1, 2))
    if transl is not None:
        verts = verts + transl[:, None, :]
    
    return verts