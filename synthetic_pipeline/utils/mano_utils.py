import numpy as np
import torch
from typing import Tuple, Dict, Any
import os
import shutil

SEAL_FACES_R = [
    [120, 108, 778],
    [108, 79, 778],
    [79, 78, 778],
    [78, 121, 778],
    [121, 214, 778],
    [214, 215, 778],
    [215, 279, 778],
    [279, 239, 778],
    [239, 234, 778],
    [234, 92, 778],
    [92, 38, 778],
    [38, 122, 778],
    [122, 118, 778],
    [118, 117, 778],
    [117, 119, 778],
    [119, 120, 778],
]

# vertex ids around the ring of the wrist
CIRCLE_V_ID = np.array(
    [108, 79, 78, 121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120],
    dtype=np.int64,
)

MANO_TIPS = {
    "thumb": 744,
    "index": 320,
    "middle": 443,
    "ring": 554,
    "pinky": 671,
}


def seal_mano_mesh(
    v3d: torch.Tensor = None,
    faces: np.ndarray = None,
    tri_uvs: np.ndarray = None,
    is_rhand: bool = True
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Seals the MANO mesh by adding a center vertex at the wrist and connecting it to the wrist vertices.
    Args:
        v3d (torch.Tensor): Vertex coordinates of the MANO mesh, shape (B, 778, 3).
        faces (np.ndarray): Faces of the MANO mesh, shape (1538, 3).
        is_rhand (bool): Whether the mesh is for a right hand.
        tri_uvs (np.ndarray): UV mapping for the mesh, shape (1538*3, 2).
    Returns:
        new_vertices (torch.Tensor): Sealed vertex coordinates, shape (B, 779, 3).
        new_faces (np.ndarray): Updated faces including the seal, shape (1554, 3).
        new_triangle_uvs (np.ndarray): Updated UV mapping, shape (1554*3, 2).
    """
    if v3d is not None:
        centers = v3d[:, CIRCLE_V_ID].mean(dim=1, keepdim=True)  # (B, 1, 3)
        new_vertices = torch.cat((v3d, centers), dim=1)       # (B, 779, 3)
    else:
        new_vertices = None
    
    if faces is not None:
        seal_faces = np.array(SEAL_FACES_R)
        if not is_rhand:
            seal_faces = seal_faces[:, [1, 0, 2]]  # invert face normal
        new_faces = np.concatenate((faces, seal_faces), axis=0)
    else:
        new_faces = None
    
    if tri_uvs is not None:
        assert faces is not None, "faces must be provided if tri_uvs is provided"
        tri_uvs = tri_uvs.reshape(-1, 3, 2)
        circle_uvs = []
        for i in range(len(CIRCLE_V_ID)):
            idx = np.where((faces == CIRCLE_V_ID[i]).any(axis=1))[0][0]
            for j in range(3):
                if faces[idx, j] == CIRCLE_V_ID[i]:
                    circle_uvs.append(tri_uvs[idx, j])
                    break
        circle_uvs = np.stack(circle_uvs, axis=0)
        center_uv = circle_uvs.mean(axis=0)

        new_triangle_uvs = []
        for f in seal_faces:
            for vid in f:
                if vid == 778:
                    new_triangle_uvs.append(center_uv)
                else:
                    idx = np.where((faces == vid).any(axis=1))[0][0]
                    for j in range(3):
                        if faces[idx, j] == vid:
                            new_triangle_uvs.append(tri_uvs[idx, j])
                            break
        new_triangle_uvs = np.stack(new_triangle_uvs, axis=0)
        new_triangle_uvs = np.vstack([tri_uvs.reshape(-1, 2), new_triangle_uvs])
    else:
        new_triangle_uvs = None
    
    return new_vertices, new_faces, new_triangle_uvs


def read_obj_simple(obj_path: str) -> Dict[str, Any]:
    """
    A simple OBJ file reader, read vertices, faces, texture coordinates and normals.
    
    Args:
        obj_path: OBJ file path
        
    Returns:
        dict: A dictionary containing vertices, faces, uvs, normals, face_data
    """
    vertices = []
    faces = []
    face_data_raw = []  # Save original face data
    uvs = []
    normals = []
    
    with open(obj_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                # Vertices coordinates
                coords = [float(x) for x in line.split()[1:4]]
                vertices.append(coords)
            elif line.startswith('vt '):
                # Texture coordinates
                uv = [float(x) for x in line.split()[1:3]]
                uvs.append(uv)
            elif line.startswith('vn '):
                # Normals
                normal = [float(x) for x in line.split()[1:4]]
                normals.append(normal)
            elif line.startswith('f '):
                # Faces, format: f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
                face_line = line.split()[1:4]
                face_indices = []
                face_vertex_data = []
                
                for vertex_data in face_line:
                    parts = vertex_data.split('/')
                    # Vertex index (the first number), OBJ index starts from 1, convert to 0
                    v_idx = int(parts[0]) - 1
                    face_indices.append(v_idx)
                    
                    # Save the complete vertex data for writing
                    face_vertex_data.append(vertex_data)
                
                faces.append(face_indices)
                face_data_raw.append(face_vertex_data)
    
    return {
        'vertices': np.array(vertices),
        'faces': np.array(faces),
        'face_data_raw': face_data_raw,  # 原始的face数据
        'uvs': np.array(uvs) if uvs else None,
        'normals': np.array(normals) if normals else None
    }


def write_obj_with_texture(obj_path: str, vertices: np.ndarray, faces: np.ndarray, 
                          face_data_raw: list = None, uvs: np.ndarray = None, normals: np.ndarray = None, 
                          texture_image = None, texture_filename: str = "hand_texture.png") -> None:
    """
    Export the MANO mesh with texture and normals.
    Args:
        obj_path: Output OBJ file path
        vertices: Vertices coordinates (N, 3)
        faces: Faces indices (M, 3)
        face_data_raw: Original face data, format like [["2/1/1", "3/2/2", "1/3/3"], ...]
        uvs: Texture coordinates (N, 2) or None
        normals: Normals (N, 3) or None
        texture_image: PIL Image object or the source path of the texture image
        texture_filename: The name of the saved texture file
    """
    base_dir = os.path.dirname(obj_path)
    base_name = os.path.splitext(os.path.basename(obj_path))[0]
    
    # 处理纹理图片
    if texture_image is not None:
        dest_texture_path = os.path.join(base_dir, texture_filename)
        
        if hasattr(texture_image, 'save'):
            # If it is a PIL Image object, save it
            texture_image.save(dest_texture_path)
        elif isinstance(texture_image, str) and os.path.exists(texture_image):
            # If it is a string, copy the file
            shutil.copy2(texture_image, dest_texture_path)
        else:
            print(f"Cannot handle texture image: {texture_image}")
            texture_image = None
    
    # Write the OBJ file
    with open(obj_path, 'w') as f:
        f.write(f"# MANO hand mesh\n")
        f.write(f"mtllib {base_name}.mtl\n")
        f.write(f"g default\n\n")
        
        # Write the vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write the texture coordinates
        if uvs is not None:
            f.write("\n")
            for uv in uvs:
                f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        
        # Write the normals
        if normals is not None:
            f.write("\n")
            for normal in normals:
                f.write(f"vn {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
        
        # Write the material usage declaration
        f.write("\nusemtl material_0\n")
        
        # Write the faces
        f.write("\n")
        if face_data_raw is not None:
            # Use the original face data, keep the original format
            for face_data in face_data_raw:
                f.write(f"f {' '.join(face_data)}\n")
        else:
            # If there is no original data, use the simplified format
            for face in faces:
                if uvs is not None and normals is not None:
                    # Full format: f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
                    f.write(f"f {face[0]+1}/{face[0]+1}/{face[0]+1} {face[1]+1}/{face[1]+1}/{face[1]+1} {face[2]+1}/{face[2]+1}/{face[2]+1}\n")
                elif uvs is not None:
                    # Only texture coordinates: f v1/vt1 v2/vt2 v3/vt3
                    f.write(f"f {face[0]+1}/{face[0]+1} {face[1]+1}/{face[1]+1} {face[2]+1}/{face[2]+1}\n")
                elif normals is not None:
                    # Only normals: f v1//vn1 v2//vn2 v3//vn3
                    f.write(f"f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\n")
                else:
                    # Only vertices: f v1 v2 v3
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    # Write the MTL file
    mtl_path = os.path.join(base_dir, f"{base_name}.mtl")
    with open(mtl_path, 'w') as f:
        f.write(f"newmtl material_0\n")
        f.write(f"Ka 1.0 1.0 1.0\n")  # Ambient light
        f.write(f"Kd 1.0 1.0 1.0\n")  # Diffuse light
        f.write(f"Ks 0.0 0.0 0.0\n")  # Specular light
        f.write(f"Ns 0.0\n")          # Specular exponent
        if texture_image is not None:
            f.write(f"map_Kd {texture_filename}\n")  # Diffuse texture

