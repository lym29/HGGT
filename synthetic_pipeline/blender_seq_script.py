"""
Author: Yumeng Liu
Email: lym29@mail.ustc.edu.cn
Blender script to render RGB images and save camera parameters for HGGT training.
"""

import argparse
import json
import math
import os
import random
import sys
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Set, Tuple

import bpy
import numpy as np
from mathutils import Matrix, Vector
import shutil
import glob
import re



# --- Helper Functions Section (mostly unchanged) ---

IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": bpy.ops.import_scene.obj,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.import_mesh.ply,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
}

def reset_cameras() -> None:
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="CAMERA")
    bpy.ops.object.delete()
    bpy.ops.object.camera_add()
    new_camera = bpy.context.active_object
    new_camera.name = "Camera"
    scene.camera = new_camera

def _sample_spherical(radius: float) -> np.ndarray:
    vec = np.random.randn(3)
    vec /= np.linalg.norm(vec, axis=0)
    vec *= radius
    return vec

def randomize_camera(
    target_center: Vector,
    camera_distance: float,
    only_northern_hemisphere: bool = False,
) -> bpy.types.Object:
    offset_vector = _sample_spherical(radius=camera_distance)
    camera = bpy.data.objects["Camera"]
    final_position = target_center + Vector(offset_vector)
    
    if only_northern_hemisphere:
        if final_position.z < target_center.z:
            final_position.z = target_center.z + abs(final_position.z - target_center.z)

    camera.location = final_position
    return camera

def reset_scene() -> None:
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

def force_smooth_all_meshes() -> None:
    """
    Force smooth shading on all mesh objects.
    Fixes cases where shade_smooth() does not take effect.
    """
    print(f"  Force smoothing all mesh objects...")
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            # Select object
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            
            # Recalculate normals
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.normals_make_consistent(inside=False)
            bpy.ops.object.mode_set(mode='OBJECT')
            
            # Apply smooth shading
            bpy.ops.object.shade_smooth()
            
            # Force each face to use smooth shading
            for poly in obj.data.polygons:
                poly.use_smooth = True
            
            print(f"    ✓ {obj.name}")

def apply_smooth_shading(
    use_subdivision: bool = False, 
    subdiv_levels: int = 1,
    objects_dict: Dict[str, List[bpy.types.Object]] = None,
    target_patterns: List[str] = None
) -> None:
    """
    Apply smooth shading to specified mesh objects.
    Enables auto smooth to make full use of normals from OBJ files.
    
    Args:
        use_subdivision: Whether to add a subdivision surface modifier (smoother but slower to render)
        subdiv_levels: Subdivision level (1-3)
        objects_dict: Object dictionary mapping object types to object lists
        target_patterns: List of object types to process, e.g. ["hand_right"]. If None, process all objects
    """
    # Determine which objects to process
    objects_to_process = []
    
    if objects_dict and target_patterns:
        # Process only objects of the specified types
        for pattern in target_patterns:
            if pattern in objects_dict:
                objects_to_process.extend(objects_dict[pattern])
        print(f"  Applying smooth shading to patterns: {target_patterns}")
    elif objects_dict:
        # Process all objects in the dictionary
        for obj_list in objects_dict.values():
            objects_to_process.extend(obj_list)
        print(f"  Applying smooth shading to all objects in dict")
    else:
        # Default behavior: process all selected objects
        bpy.ops.mesh.select_all(action='SELECT')
        objects_to_process = bpy.context.selected_objects
        print(f"  Applying smooth shading to all selected objects")
    
    for obj in objects_to_process:
        if obj.type != 'MESH':
            continue
        
        # Apply smooth shading
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.shade_smooth()
        print(f"    Applied smooth shading to: {obj.name}")
        
        # Apply subdivision if requested (may cause holes; prefer subdividing in trimesh stage)
        if use_subdivision:
            # Important: recalculate normals before subdivision to avoid holes
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.normals_make_consistent(inside=False)
            bpy.ops.object.mode_set(mode='OBJECT')
            print(f"    Recalculated normals to prevent holes")
            
            subsurf = obj.modifiers.new(name="Subdivision", type='SUBSURF')
            subsurf.levels = subdiv_levels
            subsurf.render_levels = subdiv_levels
            subsurf.subdivision_type = 'CATMULL_CLARK'
            print(f"    Applied subdivision (level {subdiv_levels}) to: {obj.name}")
        
        print(f"    Smooth shading complete for: {obj.name}")


def load_object(object_path: str, object_name: str = None) -> None:
    """
    Load a mesh object from file and optionally rename it.
    
    Args:
        object_path: Path to the mesh file
        object_name: Optional name to assign to the loaded mesh object
    """
    # Record existing objects before import
    existing_objects = set(bpy.context.scene.objects)
    
    file_extension = object_path.split(".")[-1].lower()
    import_function = IMPORT_FUNCTIONS[file_extension]
    if file_extension in {"glb", "gltf"}:
        import_function(filepath=object_path, merge_vertices=True)
    # elif file_extension == "obj":
    #     import_function(
    #         filepath=object_path,
    #         use_split_objects=False,
    #         use_split_groups=False,
    #         split_mode='OFF',  # Disable normal splitting so shade_smooth takes effect
    #         use_smooth_groups=True,
    #         use_edges=True,
    #         use_groups_as_vgroups=False
    #     )
    else:
        import_function(filepath=object_path)
    
    # If object_name is provided, rename the imported mesh objects
    if object_name:
        new_objects = set(bpy.context.scene.objects) - existing_objects
        mesh_objects = [obj for obj in new_objects if obj.type == 'MESH']
        
        if len(mesh_objects) == 1:
            # Single mesh object - rename it directly
            mesh_objects[0].name = object_name
            print(f"    Renamed object to: {object_name}")
        elif len(mesh_objects) > 1:
            # Multiple mesh objects - rename them with indices
            for i, obj in enumerate(mesh_objects):
                obj.name = f"{object_name}_{i}"
                print(f"    Renamed object to: {object_name}_{i}")
        else:
            print(f"    Warning: No mesh objects found to rename to {object_name}")
    

def clear_mesh_objects() -> None:
    """Clear only mesh objects from the scene, keeping camera and lights."""
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
    bpy.ops.object.delete()

def scene_bbox() -> Tuple[Vector, Vector]:
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in get_scene_meshes():
        found = True
        for coord in obj.bound_box:
            coord = obj.matrix_world @ Vector(coord)
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

def get_scene_meshes() -> Generator[bpy.types.Object, None, None]:
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def get_4x4_RT_matrix_from_blender(cam: bpy.types.Object) -> Matrix:
    # Blender's cam.matrix_world: camera-to-world in Blender convention
    RT_blender = cam.matrix_world.copy()
    # Blender to OpenCV conversion matrix
    blender_to_opencv = Matrix((
        (1.0, 0.0, 0.0, 0.0),
        (0.0, -1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0)
    ))
    # Convert to OpenCV convention (camera-to-world)
    RT_opencv = RT_blender @ blender_to_opencv
    return RT_opencv

def setup_three_point_lighting(target_center: Vector, camera: bpy.types.Object, target_obj: bpy.types.Object):
    """Sets up a classic three-point lighting system relative to the camera."""
    # Clear old lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()

    # Calculate base light distance and direction vectors
    light_distance = (target_center - camera.location).length * 1.5
    camera_direction = (target_center - camera.location).normalized()
    camera_up = camera.matrix_world.to_quaternion() @ Vector((0.0, 1.0, 0.0))
    camera_right = camera_direction.cross(camera_up).normalized()

    # 1. Key Light 
    key_light_location = camera.location + camera_right * light_distance * 0.7 + camera_up * light_distance * 0.3
    _create_area_light(
        name="KeyLight",
        location=key_light_location,
        energy=random.choice([1.0, 1.5, 2.0]),  # Reduced energy
        size=light_distance / 3,
        target_obj=target_obj
    )

    # 2. Fill Light
    fill_light_location = camera.location - camera_right * light_distance * 0.6 + camera_up * light_distance * 0.2
    _create_area_light(
        name="FillLight",
        location=fill_light_location,
        energy=random.choice([1.0, 1.5, 2.0]),  # Reduced energy
        size=light_distance / 2.5,
        target_obj=target_obj
    )

    # 3. Rim Light
    rim_light_location = target_center - camera_direction * light_distance * 0.8
    _create_area_light(
        name="RimLight",
        location=rim_light_location,
        energy=random.choice([1.0, 1.5, 2.0]),  # Reduced energy
        size=light_distance / 3,
        target_obj=target_obj
    )

def _create_area_light(name: str, location: Vector, energy: float, size: float, target_obj: bpy.types.Object) -> bpy.types.Object:
    """Helper function to create and aim an Area light."""
    light_data = bpy.data.lights.new(name, type='AREA')
    light_data.energy = energy
    light_data.shape = 'RECTANGLE'
    light_data.size = size
    light_data.size_y = size

    light_object = bpy.data.objects.new(name, light_data)
    light_object.location = location
    bpy.context.collection.objects.link(light_object)

    track_to = light_object.constraints.new(type='TRACK_TO')
    track_to.target = target_obj
    track_to.track_axis = 'TRACK_NEGATIVE_Z'
    track_to.up_axis = 'UP_Y'
    
    return light_object

def setup_transparent_world_with_lighting(engine: str) -> None:
    """
    Set up world material for transparent background with proper environment lighting.
    
    Args:
        engine: Render engine ('CYCLES' or 'BLENDER_EEVEE')
    """
    scene = bpy.context.scene
    
    # Enable world nodes
    scene.world.use_nodes = True
    world_nodes = scene.world.node_tree.nodes
    world_links = scene.world.node_tree.links
    
    # Clear existing nodes
    world_nodes.clear()
    
    # Add Background shader for environment lighting
    bg_node = world_nodes.new('ShaderNodeBackground')
    bg_node.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)  # White for good lighting
    
    # Set appropriate strength for environment lighting
    if engine == "CYCLES":
        bg_node.inputs['Strength'].default_value = 2.0  # Good ambient lighting
    else:  # EEVEE
        bg_node.inputs['Strength'].default_value = 3.0
    
    # Add Light Path node to detect camera rays
    light_path_node = world_nodes.new('ShaderNodeLightPath')
    
    # Add Mix Shader to make background transparent to camera but visible for lighting
    mix_node = world_nodes.new('ShaderNodeMixShader')
    
    # Add Transparent BSDF for camera rays
    transparent_node = world_nodes.new('ShaderNodeBsdfTransparent')
    
    # Add World Output node
    output_node = world_nodes.new('ShaderNodeOutputWorld')
    
    # Connect nodes:
    # Use Light Path "Is Camera Ray" to mix between transparent and background
    world_links.new(light_path_node.outputs['Is Camera Ray'], mix_node.inputs['Fac'])
    world_links.new(bg_node.outputs['Background'], mix_node.inputs[2])  # Background for lighting
    world_links.new(transparent_node.outputs['BSDF'], mix_node.inputs[1])  # Transparent for camera
    world_links.new(mix_node.outputs['Shader'], output_node.inputs['Surface'])

def get_sequence_files(object_dir: str) -> List[Tuple[int, str]]:
    """Get all sequence files in the directory, returns list of (time_step, file_path) tuples"""
    sequence_files = []
    if not os.path.exists(object_dir):
        raise ValueError(f"Directory {object_dir} does not exist")
    
    for filename in os.listdir(object_dir):
        if filename.endswith('.obj'):
            try:
                # Extract frame number from filename (e.g., "0123.obj" -> 123)
                frame_num = int(filename.replace('.obj', ''))
                file_path = os.path.join(object_dir, filename)
                sequence_files.append((frame_num, file_path))
            except ValueError:
                print(f"Warning: Skipping file {filename} - cannot parse frame number")
    
    sequence_files.sort(key=lambda x: x[0])  # Sort by frame number
    print(f"Found {len(sequence_files)} sequence files")
    
    return sequence_files

def get_hoi_sequence_files(object_dir: str) -> List[Tuple[int, str]]:
    """
    Get all sequence files in the directory, 
    returns list of (time_step, hand_file_path, object_file_path) tuples
    """
    hand_pattern = re.compile(r'(\d{4})_hand_mesh\.obj$')
    object_pattern = re.compile(r'(\d{4})_object_mesh\.obj$')
    hand_files = {}
    object_files = {}

    for filename in os.listdir(object_dir):
        hand_match = hand_pattern.match(filename)
        object_match = object_pattern.match(filename)
        if hand_match:
            frame = int(hand_match.group(1))
            hand_files[frame] = os.path.join(object_dir, filename)
        elif object_match:
            frame = int(object_match.group(1))
            object_files[frame] = os.path.join(object_dir, filename)

    # Keep only frames where both hand and object meshes exist
    frames = sorted(set(hand_files) & set(object_files))
    sequence_files = [
        (frame, hand_files[frame], object_files[frame]) for frame in frames
    ]

    return sequence_files

def get_camera_parameters(cam: bpy.types.Object) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get camera intrinsic and extrinsic parameters in OpenCV format.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (intrinsic_matrix, extrinsic_matrix)
    """
    # Get render size
    render = bpy.context.scene.render
    w = render.resolution_x
    h = render.resolution_y
    
    # Calculate intrinsic parameters
    focal = w / 2 / np.tan(cam.data.angle / 2)
    cx = w / 2
    cy = h / 2
    
    # Create intrinsic matrix (3x3)
    intrinsic = np.array([
        [focal, 0, cx],
        [0, focal, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Get extrinsic matrix (4x4) - camera-to-world transformation
    RT = get_4x4_RT_matrix_from_blender(cam)  # camera-to-world (4x4)
    extrinsic = np.linalg.inv(RT) # world-to-camera

    rot_x_90 = np.array([
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])
    extrinsic = extrinsic @ rot_x_90
    
    return intrinsic, extrinsic

def get_frame_center_and_size() -> Tuple[Vector, float]:
    """Get current frame object's center position and size"""
    bbox_min, bbox_max = scene_bbox()
    center = (bbox_min + bbox_max) / 2
    size = (bbox_max - bbox_min).length
    return center, size

def get_scene_objects_by_name(name_patterns: List[str]) -> Dict[str, List[bpy.types.Object]]:
    """
    Get scene objects grouped by name patterns.
    
    Args:
        name_patterns: List of name patterns to search for (case-insensitive)
    
    Returns:
        Dict mapping pattern names to lists of matching objects
    """
    result = {pattern: [] for pattern in name_patterns}
    
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj_name_lower = obj.name.lower()
            for pattern in name_patterns:
                if pattern.lower() in obj_name_lower:
                    result[pattern].append(obj)
                    break  # Assign to first matching pattern only
    
    return result

def setup_object_mask_rendering() -> Tuple[bpy.types.CompositorNode, bpy.types.CompositorNode]:
    """
    Set up compositor nodes for object mask rendering.
    
    Returns:
        Tuple of (render_layers_node, file_output_node)
    """
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    
    # Clear existing nodes
    for node in tree.nodes:
        tree.nodes.remove(node)
    
    # Create nodes for mask rendering
    render_layers = tree.nodes.new('CompositorNodeRLayers')
    file_output = tree.nodes.new('CompositorNodeOutputFile')
    
    # Configure file output for masks
    file_output.format.file_format = 'PNG'
    file_output.format.color_mode = 'BW'  # Black and white for masks
    file_output.format.color_depth = '8'
    file_output.base_path = ""
    
    # Ensure we have a file slot for masks
    if len(file_output.file_slots) == 0:
        file_output.file_slots.new("mask")
    
    file_slot = file_output.file_slots[0]
    file_slot.format.file_format = 'PNG'
    file_slot.format.color_mode = 'BW'
    file_slot.use_node_format = False
    
    # Connect IndexOB output for object masks
    tree.links.new(render_layers.outputs['IndexOB'], file_output.inputs[0])
    
    return render_layers, file_output

def render_object_masks(objects_dict: Dict[str, List[bpy.types.Object]], 
                       masks_dir: str, view_id: str) -> None:
    """
    Render masks for different object groups.
    
    Args:
        objects_dict: Dictionary mapping object names to lists of objects
        masks_dir: Directory to save masks
        view_id: Current view identifier
    """
    scene = bpy.context.scene
    
    # Set up mask rendering
    render_layers, file_output = setup_object_mask_rendering()
    
    # Enable Object Index pass
    scene.view_layers["ViewLayer"].use_pass_object_index = True
    
    for obj_name, obj_list in objects_dict.items():
        if not obj_list:
            print(f"    Warning: No objects found for '{obj_name}'")
            continue
            
        print(f"    Rendering mask for '{obj_name}' ({len(obj_list)} objects)")
        
        # Reset all object indices to 0 (background)
        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH':
                obj.pass_index = 0
        
        # Set target objects to index 1 (foreground)
        for obj in obj_list:
            obj.pass_index = 1
            print(f"      - Object: {obj.name}")
        
        # Set output path for this mask
        mask_filename = f"{view_id}_{obj_name}_mask"
        file_output.base_path = masks_dir + "/"
        file_output.file_slots[0].path = mask_filename
        
        # Render the mask
        bpy.ops.render.render(write_still=False)
        
        print(f"    Saved mask: {mask_filename}.png")

def setup_depth_rendering() -> Tuple[bpy.types.CompositorNode, bpy.types.CompositorNode]:
    """
    Set up compositor nodes for depth rendering.
    
    Returns:
        Tuple of (render_layers_node, file_output_node)
    """
    scene = bpy.context.scene
    scene.view_layers["ViewLayer"].use_pass_z = True
    scene.use_nodes = True
    tree = scene.node_tree
    
    # Clear existing nodes
    for node in tree.nodes:
        tree.nodes.remove(node)
    
    # Create nodes for depth rendering
    render_layers = tree.nodes.new('CompositorNodeRLayers')
    file_output = tree.nodes.new('CompositorNodeOutputFile')
    file_output.format.file_format = 'OPEN_EXR'
    file_output.format.color_depth = '32'
    
    # Configure file output settings
    file_output.base_path = ""
    
    # Ensure we have a file slot and configure it
    if len(file_output.file_slots) == 0:
        file_output.file_slots.new("depth")
    
    # Configure the file slot
    file_slot = file_output.file_slots[0]
    file_slot.format.file_format = 'OPEN_EXR'
    file_slot.format.color_depth = '32'
    file_slot.use_node_format = False  # Use slot-specific format
    
    # Connect depth output
    tree.links.new(render_layers.outputs['Depth'], file_output.inputs[0])
    
    return render_layers, file_output

def render_object_depths(objects_dict: Dict[str, List[bpy.types.Object]], 
                        depths_dir: str, view_id: str) -> None:
    """
    Render separate depth maps for different object groups.
    
    Args:
        objects_dict: Dictionary mapping object names to lists of objects
        depths_dir: Directory to save depth maps
        view_id: Current view identifier
    """
    scene = bpy.context.scene
    
    # Store original visibility states
    original_visibility = {}
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            original_visibility[obj.name] = obj.hide_render
    
    for obj_name, obj_list in objects_dict.items():
        if not obj_list:
            print(f"    Warning: No objects found for '{obj_name}' depth")
            continue
            
        print(f"    Rendering depth for '{obj_name}' ({len(obj_list)} objects)")
        
        # Hide all mesh objects first
        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH':
                obj.hide_render = True
        
        # Show only target objects
        for obj in obj_list:
            obj.hide_render = False
            print(f"      - Showing object: {obj.name}")
        
        # Set up depth rendering
        render_layers, file_output = setup_depth_rendering()
        
        # Set output path for this depth
        depth_filename = f"{view_id}_{obj_name}_depth"
        file_output.base_path = depths_dir + "/"
        file_output.file_slots[0].path = depth_filename
        
        # Render the depth
        bpy.ops.render.render(write_still=False)
        
        # Convert EXR to NPY
        depth_pattern = os.path.join(depths_dir, f"{depth_filename}*.exr")
        depth_files = glob.glob(depth_pattern)
        
        if depth_files:
            actual_depth_path = depth_files[0]
            depth_npy_path = os.path.join(depths_dir, f"{view_id}_{obj_name}_depth.npy")
            
            # Load EXR depth file
            depth_image = bpy.data.images.load(actual_depth_path)
            width, height = depth_image.size
            
            # Get pixel data
            pixels = np.array(depth_image.pixels[:])
            
            # Reshape to (height, width, channels) and extract depth (first channel)
            depth_array = pixels.reshape((height, width, 4))[:, :, 0]
            
            # Flip vertically (Blender uses bottom-left origin)
            depth_array = np.flipud(depth_array)
            np.save(depth_npy_path, depth_array)
            
            # Clean up
            bpy.data.images.remove(depth_image)
            os.remove(actual_depth_path)  # Remove EXR file
            
            print(f"    Saved depth: {view_id}_{obj_name}_depth.npy")
        else:
            print(f"    Error: No depth file found for {obj_name}")
    
    # Restore original visibility states
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH' and obj.name in original_visibility:
            obj.hide_render = original_visibility[obj.name]

def render_sequence(
    object_dir: str,
    num_renders: int,
    only_northern_hemisphere: bool,
    output_dir: str,
) -> None:
    """
    Render mesh sequence in HGGT dataset format.
    Each sequence contains images from different views at the same time step.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get sequence files
    # sequence_files = get_sequence_files(object_dir)
    sequence_files = get_hoi_sequence_files(object_dir)
    if not sequence_files:
        print("Error: No sequence files found in directory")
        return
    
    reset_scene()
    
    # Set up camera
    cam = scene.objects["Camera"]
    cam.data.lens = 35
    cam.data.sensor_width = 32
    
    # Process each time step
    for frame_num, hand_file_path, object_file_path in sequence_files:
        frame_id = f"{frame_num:06d}"
        print(f"\nProcessing frame {frame_id}")
        
        # Create directory for this frame using original frame number
        sequence_dir = os.path.join(output_dir, f"sequence_{frame_id}")
        if os.path.exists(os.path.join(sequence_dir,"info.json")):
            continue # skip if info.json exists, all files are already rendered
        
        images_dir = os.path.join(sequence_dir, "images")
        # depths_dir = os.path.join(sequence_dir, "depths")  # Depth rendering disabled
        masks_dir = os.path.join(sequence_dir, "masks")
        camera_dir = os.path.join(sequence_dir, "camera")
        extrinsics_dir = os.path.join(camera_dir, "extrinsics")
        
        os.makedirs(images_dir, exist_ok=True)
        # os.makedirs(depths_dir, exist_ok=True)  # Depth rendering disabled
        os.makedirs(masks_dir, exist_ok=True)
        os.makedirs(extrinsics_dir, exist_ok=True)
        
        # Load mesh for current frame
        clear_mesh_objects()
        load_object(hand_file_path, object_name="hand_right")
        apply_smooth_shading(
            use_subdivision=True,
            subdiv_levels=1,
            objects_dict={"hand_right": [bpy.context.scene.objects["hand_right"]]},
            target_patterns=["hand_right"]
        )
        load_object(object_file_path, object_name="object")
        current_center, current_size = get_frame_center_and_size()
        
        # Get object groups after loading
        object_patterns = ["object", "hand_right"]
        objects_dict = get_scene_objects_by_name(object_patterns)
        
        # Print found objects for debugging
        print(f"Found objects in scene:")
        for pattern, obj_list in objects_dict.items():
            print(f"  {pattern}: {[obj.name for obj in obj_list]}")
        
        
        # Create camera target
        target_empty = bpy.data.objects.new("CameraTarget", None)
        scene.collection.objects.link(target_empty)
        target_empty.location = current_center

        cam_constraint = cam.constraints.new(type="TRACK_TO")
        cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
        cam_constraint.up_axis = "UP_Y"
        cam_constraint.target = target_empty
        
        # Calculate adaptive camera distance
        camera_fov = cam.data.angle
        if math.tan(camera_fov / 2.0) > 1e-6:
            distance = (current_size / 2.0) / math.tan(camera_fov / 2.0)
        else:
            distance = current_size * 2.0
        
        adaptive_camera_distance = distance * 1.1
        print(f"Adaptive camera distance: {adaptive_camera_distance}")
        
        # Generate different views for this frame
        for view_idx in range(num_renders):
            view_id = f"{view_idx:06d}"
            print(f"  Rendering view {view_id}/{num_renders:06d}")
            
            # Randomize camera position for this view
            camera = randomize_camera(
                target_center=current_center,
                camera_distance=adaptive_camera_distance,
                only_northern_hemisphere=only_northern_hemisphere,
            )
            
            # Set up lighting for this view
            setup_three_point_lighting(current_center, camera, target_empty)
        
            # Get camera parameters
            intrinsic, extrinsic = get_camera_parameters(cam)
            
            if view_idx == 0:
                np.savetxt(os.path.join(camera_dir, "intrinsics.txt"), intrinsic)
            np.savetxt(os.path.join(extrinsics_dir, f"{view_id}.txt"), extrinsic)

            # Set output paths for this view
            rgb_path = os.path.join(images_dir, f"{view_id}.png")
            # depth_npy_path = os.path.join(depths_dir, f"{view_id}.npy")  # Depth rendering disabled
            
            # Check if masks exist for all object types
            masks_exist = True
            for obj_name in object_patterns:
                mask_path = os.path.join(masks_dir, f"{view_id}_{obj_name}_mask.png")
                if not os.path.exists(mask_path):
                    masks_exist = False
                    break

            if os.path.exists(rgb_path) and masks_exist:
                print(f"    Skipping view {view_id} - already rendered")
                continue
            
            
            scene.render.filepath = rgb_path
            scene.render.image_settings.file_format = "PNG"
            scene.render.image_settings.color_mode = "RGBA"  # Ensure alpha channel
            bpy.ops.render.render(write_still=True)
            
            # Render object masks
            print(f"    Rendering object masks for view {view_id}")
            render_object_masks(objects_dict, masks_dir, view_id)
            
            # # Depth rendering disabled - using multi-depth GT from render_multi_depth_gt.py instead
            # # Render individual object depths
            # print(f"    Rendering individual object depths for view {view_id}")
            # render_object_depths(objects_dict, depths_dir, view_id)
            # 
            # # Re-setup compositor for combined depth rendering
            # render_layers, file_output = setup_depth_rendering()
            # file_output.base_path = depths_dir + "/"
            # file_output.file_slots[0].path = f"{view_id}"
            # 
            # print(f"    Debug: base_path = '{file_output.base_path}'")
            # print(f"    Debug: file_slot.path = '{file_output.file_slots[0].path}'")
            # 
            # # Clear render filepath to avoid conflicts with file output node
            # # Set to a specific path that we can control and clean up
            # dummy_path = os.path.join(depths_dir, "temp_render")
            # scene.render.filepath = dummy_path
            # scene.render.image_settings.file_format = "OPEN_EXR"  # Reset format
            # 
            # # Render depth using compositor
            # bpy.ops.render.render(write_still=False)  # Don't write the main render file
            # 
            # # Clean up any unwanted render files
            # for ext in [".png", ".exr"]:
            #     unwanted_file = dummy_path + ext
            #     if os.path.exists(unwanted_file):
            #         os.remove(unwanted_file)
            # 
            # # Read the rendered depth and convert to numpy
            # try:
            #     depth_pattern = os.path.join(depths_dir, f"{view_id}*.exr")
            #     depth_files = glob.glob(depth_pattern)
            #     
            #     if not depth_files:
            #         print(f"    Error: No depth file found matching pattern {depth_pattern}")
            #         continue
            #     
            #     actual_depth_path = depth_files[0]  # Take the first match
            #     print(f"    Found depth file: {actual_depth_path}")
            #     
            #     # Load EXR depth file
            #     depth_image = bpy.data.images.load(actual_depth_path)
            #     width, height = depth_image.size
            #     
            #     # Get pixel data
            #     pixels = np.array(depth_image.pixels[:])
            #     
            #     # Reshape to (height, width, channels) and extract depth (first channel)
            #     depth_array = pixels.reshape((height, width, 4))[:, :, 0]
            #     
            #     # Flip vertically (Blender uses bottom-left origin)
            #     depth_array = np.flipud(depth_array)
            #     np.save(depth_npy_path, depth_array)
            #     # Clean up
            #     bpy.data.images.remove(depth_image)
            #     
            #     print(f"    Saved RGB and depth (.npy) for view {view_id}")
            #     
            # except Exception as e:
            #     print(f"    Error processing depth for view {view_id}: {e}")
            
            print(f"    Saved RGB for view {view_id}")
            
            # Save extrinsics for this view
            np.savetxt(os.path.join(extrinsics_dir, f"{view_id}.txt"), extrinsic)

        # Clean up for next frame
        bpy.data.objects.remove(target_empty, do_unlink=True)
        if cam_constraint:
            cam.constraints.remove(cam_constraint)
        
        print(f"Completed frame {frame_id} with {num_renders} views")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_dir", type=str, required=True, 
                       help="Directory containing sequence of meshes named as {t:04d}.obj")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"],
                       help="CYCLES for better quality")
    parser.add_argument("--only_northern_hemisphere", action="store_true", default=False)
    parser.add_argument("--num_renders", type=int, default=10, 
                       help="Number of views to generate for each time step")
    
    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)

    context = bpy.context
    scene = context.scene
    render = scene.render

    # Set up scene for rendering
    scene.render.engine = args.engine
    scene.render.image_settings.file_format = "PNG"  # PNG supports transparency
    scene.render.image_settings.color_mode = "RGBA"  # Enable alpha channel
    scene.render.image_settings.color_depth = "8"
    scene.render.resolution_x = 384 #518
    scene.render.resolution_y = 384 #518
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True  # Enable transparent background
    
    # Set up transparent background with proper environment lighting
    # setup_transparent_world_with_lighting(args.engine)
    
    # Additional settings for better rendering
    if args.engine == "CYCLES":
        scene.cycles.device = "GPU"
        scene.cycles.samples = 64
        scene.cycles.use_denoising = True
        scene.cycles.film_exposure = 1.0  # Keep normal exposure
        scene.cycles.filter_width = 1.5
        # Enable adaptive sampling for faster rendering
        # Adaptive sampling reduces samples in low-noise areas and increases in high-noise areas
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.adaptive_threshold = 0.01  # Noise threshold (lower = more quality, higher = faster)
        scene.cycles.adaptive_min_samples = 16  # Minimum samples before adaptive sampling kicks in
        # Ensure proper color management for white background
        scene.view_settings.view_transform = 'Standard'
        scene.view_settings.look = 'None'
    else:  # EEVEE
        scene.eevee.use_soft_shadows = False
        scene.eevee.use_ssr = False
        scene.eevee.use_ssr_refraction = False
        scene.eevee.use_gtao = False
        scene.eevee.use_bloom = False
        # Ensure proper color management for white background
        scene.view_settings.view_transform = 'Standard'
        scene.view_settings.look = 'None'
    
    render_sequence(
        object_dir=args.object_dir,
        num_renders=args.num_renders,
        only_northern_hemisphere=args.only_northern_hemisphere,
        output_dir=args.output_dir,
    )