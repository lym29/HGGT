import logging
import rootutils
__ROOT__ = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import numpy as np
import trimesh
import os
from typing import Optional
from utils.objaverse import DownloadObjaverse
from utils import mesh_utils, mano_utils
from utils.transform import axisangle2mat
from smplx import MANO
import torch
import pickle
from PIL import Image
from tqdm import trange, tqdm
import glob
import random
from datetime import datetime
import json


class GraspXLLoader:
    def __init__(self, args):
        self.args = args
        self.obj_downloader = DownloadObjaverse(download_path=args.objaverse_dir)
        # Convert GPU ID to PyTorch device format
        self.device = f"cuda:{args.gpu_id}" if args.gpu_id >= 0 else "cpu"
        self.use_handy = args.use_handy
        print(f"[PID {os.getpid()}] GraspXLLoader initialized with device: {self.device}")
        
        # Load MANO beta statistics from HaMeR prior
        self.mano_beta_mean = None
        self.mano_beta_std = args.mano_beta_std  # std per dimension
        if os.path.exists(args.hamer_prior_path):
            hamer_data = np.load(args.hamer_prior_path)
            self.mano_beta_mean = torch.from_numpy(hamer_data['shape']).float()  # shape: (10,)
            # print(f"Loaded MANO beta prior from HaMeR: mean={self.mano_beta_mean.numpy()}, std={self.mano_beta_std} (per dimension)")
        else:
            print(f"Warning: HaMeR prior not found at {args.hamer_prior_path}, will use zero betas")
        
        if self.use_handy is False:
            # 使用简洁的OBJ读取函数，避免法向量数量问题
            template_path = os.path.join(args.dart_dir, "DART_texture_template_mesh/template_mesh/original_mano_template/hand.obj")
            template_data = mano_utils.read_obj_simple(template_path)
            
            # print(f"Loaded MANO template mesh:")
            # print(f"  Vertices: {template_data['vertices'].shape}")
            # print(f"  Faces: {template_data['faces'].shape}")
            # print(f"  UVs: {template_data['uvs'].shape if template_data['uvs'] is not None else 'None'}")
            # print(f"  Normals: {template_data['normals'].shape if template_data['normals'] is not None else 'None'}")
            
            self.template_vertices = template_data['vertices']
            self.template_faces = template_data['faces']
            self.template_face_data_raw = template_data['face_data_raw']
            self.template_uvs = template_data['uvs']
            self.template_normals = template_data['normals']
            
            self.hand_texture_dir = os.path.join(args.dart_dir, "DART_texture_template_mesh/texture&accessories/basic/")
            
            self.available_textures = self._get_available_textures()
            
            self.mano_layer = MANO(
                os.path.join(args.mano_dir, "models"),
                create_transl=False,
                use_pca=False,
                flat_hand_mean=False,
                is_rhand=True,
            ).to(self.device)
            
            # 使用模板网格的面和UV数据
            self.faces_rh = self.template_faces
            # 将UV坐标转换为三角形UV格式（每个面3个UV坐标）
            if self.template_uvs is not None:
                tri_uvs = []
                for face in self.template_faces:
                    for vertex_idx in face:
                        tri_uvs.append(self.template_uvs[vertex_idx])
                self.tri_uv_rh = np.array(tri_uvs).reshape(-1, 2)
            else:
                self.tri_uv_rh = None
            
        else:
            with open(f"{__ROOT__}/assets/handy/models/Right_Hand_Shape.pkl", 'rb') as f:
                hand_model = pickle.load(f)
            img = Image.open(f"{__ROOT__}/handy/output/material_0.png")
            self.hand_texture_image = img 
            
            self.vertex_uv_rh = np.asarray(hand_model['uv_coords'])
            self.faces_rh = np.asarray(hand_model['f_uv']).astype(np.int64)
            
            self.mano_layer = MANO(
                os.path.join(args.handy_dir, "models/HANDY_RIGHT.pkl" ),
                create_transl=False, use_pca=False, flat_hand_mean=False, is_rhand=True,
            ).to(self.device)

            # print(f"Loaded HANDY with {self.vertex_uv_rh.shape[0]} vertices and corresponding vertex UVs.")
            self.transform_to_uv = hand_model['transform_to_uv']
            
            # HANDY 模式下不使用随机纹理
            self.available_textures = []
    
    def _get_available_textures(self) -> list:
        if not os.path.exists(self.hand_texture_dir):
            print(f"Warning: Texture directory not found: {self.hand_texture_dir}")
            return []
        
        texture_files = []
        for ext in ['*.png', '*.PNG']:
            pattern = os.path.join(self.hand_texture_dir, '**', ext)
            texture_files.extend(glob.glob(pattern, recursive=True))
        
        # Exclude files in yellow/basic/ directory, these textures are wrong.
        exclude_dir = os.path.join(self.hand_texture_dir, 'yellow', 'basic')
        texture_files = [f for f in texture_files if not f.startswith(exclude_dir)]
        
        if not texture_files:
            print(f"Warning: No texture files found in {self.hand_texture_dir}")
        # else:
        #     print(f"Found {len(texture_files)} texture files in {self.hand_texture_dir} (including subdirectories)")
        
        return texture_files
    
    def _load_random_texture(self) -> Image.Image:
        if not self.available_textures:
            raise RuntimeError(f"No textures available in {self.hand_texture_dir}. Please check the directory structure.")
        
        selected_texture = random.choice(self.available_textures)
        # print(f"Selected random texture: {os.path.relpath(selected_texture, self.hand_texture_dir)}")
        return Image.open(selected_texture)
            
    def load_motion(self, motion_path: str, sample_stride: int = 50):
        raw_data = np.load(motion_path, allow_pickle=True).item()
        oid = motion_path.split("/")[-2]
        
        pose_r = torch.from_numpy(np.concatenate((
            raw_data['right_hand']['rot'], 
            raw_data['right_hand']['pose']), axis=1)).to(self.device)
        trans_r = torch.from_numpy(raw_data['right_hand']['trans']).to(self.device)
        num_frames = pose_r.shape[0]
        if sample_stride < num_frames:
            sampled_frames = np.arange(sample_stride, num_frames, sample_stride)
        else:
            sampled_frames = np.arange(num_frames-1, num_frames)
        
        # Sample random betas from HaMeR prior
        if self.mano_beta_mean is not None:
            # Sample from normal distribution: N(mean, std)
            # Use same beta for all frames in a sequence for consistency
            random_beta = torch.randn(1, 10) * self.mano_beta_std + self.mano_beta_mean
            betas = random_beta.repeat(len(sampled_frames), 1).to(self.device)
        else:
            betas = torch.zeros((len(sampled_frames), 10)).to(self.device)
        
        # Extract MANO parameters for sampled frames
        global_orient = pose_r[sampled_frames, :3]
        hand_pose = pose_r[sampled_frames, 3:]
        hand_trans = trans_r[sampled_frames, :]
        
        mano_output = self.mano_layer(
            global_orient=global_orient,
            hand_pose=hand_pose,
            betas=betas,
        )

        verts_rh = mano_output.vertices
        # if self.use_handy is False:
        #     verts_rh, _, _ = mano_utils.seal_mano_mesh(v3d=verts_rh, is_rhand=True)

        verts_rh = verts_rh + hand_trans.unsqueeze(1)
        trans_o = torch.from_numpy(raw_data[oid]['trans'][sampled_frames,:]).to(self.device)
        rot_o = torch.from_numpy(raw_data[oid]['rot'][sampled_frames,:]).to(self.device)
        
        object_mesh = self.load_obj(motion_path)
        verts_o = mesh_utils.apply_transform_verts_torch(mesh=object_mesh, 
                                               rotmat=axisangle2mat(rot_o),
                                               transl=trans_o,
                                               device=self.device)
        
        # Create MANO parameters dictionary
        mano_params = {
            'global_orient': global_orient,
            'hand_pose': hand_pose,
            'betas': betas,
            'transl': hand_trans 
        }
        
        return sampled_frames, verts_rh, object_mesh, verts_o, mano_params
    
    def export_hoi_sequence(self, motion_path: str, mesh_output_dir: str):
        if self.use_handy:
            return self.export_hoi_sequence_handy(motion_path, mesh_output_dir)
        else:
            return self.export_hoi_sequence_mano_with_texture(motion_path, mesh_output_dir) 
            
    def export_hoi_sequence_handy(self, motion_path: str, output_dir: str):
        sampled_frames, verts_rh, object_mesh, verts_o, mano_params = self.load_motion(motion_path)
        seq_len = verts_rh.shape[0]
        verts_rh = verts_rh[:, self.transform_to_uv, :]
        hand_visuals = trimesh.visual.TextureVisuals(
            uv=self.vertex_uv_rh,
            image=self.hand_texture_image
        )

        os.makedirs(output_dir, exist_ok=True)
        
        # Save MANO parameters for later use
        mano_params_save = {
            'sampled_frames': sampled_frames,
            'global_orient': mano_params['global_orient'].cpu().numpy(),
            'hand_pose': mano_params['hand_pose'].cpu().numpy(),
            'betas': mano_params['betas'].cpu().numpy(),
            'transl': mano_params['transl'].cpu().numpy()
        }
        np.save(os.path.join(output_dir, 'mano_params.npy'), mano_params_save)

        for t in trange(seq_len):
            # export_path = os.path.join(output_dir, f"{t:04d}.glb")
            export_path = os.path.join(output_dir, f"{t:04d}.obj") # the texture color is brighter
            if os.path.exists(export_path):
                continue
            
            current_object_mesh = object_mesh.copy()
            current_object_mesh.vertices = verts_o[t].cpu().numpy()
            hand_mesh = trimesh.Trimesh(
                vertices=verts_rh[t].cpu().numpy(),
                faces=self.faces_rh,
                visual=hand_visuals,
                process=False
            )
            
            hand_mesh.export(os.path.join(output_dir, f"{t:04d}_hand_mesh.obj"))
            current_object_mesh.export(os.path.join(output_dir, f"{t:04d}_object_mesh.obj"))

        # print(f"Rendered {seq_len} frames to {output_dir}")

        return
    
    def export_hoi_sequence_mano(self, motion_path: str, output_dir: str):
        sampled_frames, verts_rh, object_mesh, verts_o, mano_params = self.load_motion(motion_path)
        seq_len = verts_rh.shape[0]
        
        # Create vertex colors for better compatibility
        # Use a light skin color (e.g., RGB: 0.87, 0.72, 0.53)
        skin_color = [0.87, 0.72, 0.53, 1.0]  # RGBA
        # Create color array for all vertices
        num_vertices = verts_rh.shape[1]  # Number of vertices per frame
        vertex_colors = np.tile(skin_color, (num_vertices, 1))
        hand_visuals = trimesh.visual.ColorVisuals(
            vertex_colors=vertex_colors
        )
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save MANO parameters for later use
        mano_params_save = {
            'sampled_frames': sampled_frames,
            'global_orient': mano_params['global_orient'].cpu().numpy(),
            'hand_pose': mano_params['hand_pose'].cpu().numpy(),
            'betas': mano_params['betas'].cpu().numpy(),
            'transl': mano_params['transl'].cpu().numpy()
        }
        np.save(os.path.join(output_dir, 'mano_params.npy'), mano_params_save)
        
        for t in trange(seq_len):
            export_path = os.path.join(output_dir, f"{t:04d}.obj")
            if os.path.exists(export_path):
                continue
            
            current_object_mesh = object_mesh.copy()
            current_object_mesh.vertices = verts_o[t].cpu().numpy()
            hand_mesh = trimesh.Trimesh(
                vertices=verts_rh[t].cpu().numpy(),
                faces=self.faces_rh,
                visual=hand_visuals,
                process=False
            )
            
            hand_mesh.export(os.path.join(output_dir, f"{t:04d}_hand_mesh.obj"))
            current_object_mesh.export(os.path.join(output_dir, f"{t:04d}_object_mesh.obj"))
            
        # print(f"Rendered {seq_len} frames to {output_dir}")
            
        return

    def export_hoi_sequence_mano_with_texture(self, motion_path: str, output_dir: str):
        sampled_frames, verts_rh, object_mesh, verts_o, mano_params = self.load_motion(motion_path)
        seq_len = verts_rh.shape[0]
        
        random_texture = self._load_random_texture()
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save MANO parameters for later use
        mano_params_save = {
            'sampled_frames': sampled_frames,
            'global_orient': mano_params['global_orient'].cpu().numpy(),
            'hand_pose': mano_params['hand_pose'].cpu().numpy(),
            'betas': mano_params['betas'].cpu().numpy(),
            'transl': mano_params['transl'].cpu().numpy()
        }
        np.save(os.path.join(output_dir, 'mano_params.npy'), mano_params_save)

        for t in trange(seq_len):
            export_path = os.path.join(output_dir, f"{t:04d}.obj")
            if os.path.exists(export_path):
                continue
            
            current_object_mesh = object_mesh.copy()
            current_object_mesh.vertices = verts_o[t].cpu().numpy()
            
            hand_vertices = verts_rh[t][:778].cpu().numpy() 
            hand_obj_path = os.path.join(output_dir, f"{t:04d}_hand_mesh.obj")
            
            mano_utils.write_obj_with_texture(
                hand_obj_path,
                vertices=hand_vertices,
                faces=self.template_faces,
                face_data_raw=self.template_face_data_raw,
                uvs=self.template_uvs,
                normals=self.template_normals,
                texture_image=random_texture,
                texture_filename=f"{t:04d}_hand_texture.png"
            )

            current_object_mesh.export(os.path.join(output_dir, f"{t:04d}_object_mesh.obj"))
            
        # print(f"Rendered {seq_len} frames to {output_dir}")
            
        return

    def export_info(self, motion_path: str, render_output_dir: str, tmp_output_dir: str):
        otype = motion_path.split("/")[-3]
        oid = motion_path.split("/")[-2]
        motion_id = motion_path.split("/")[-1].replace(".npy", "")
        
        # Load saved MANO parameters from tmp directory
        mano_params_path = os.path.join(tmp_output_dir, 'mano_params.npy')
        if not os.path.exists(mano_params_path):
            print(f"Warning: MANO parameters not found at {mano_params_path}, skipping info export")
            return
        
        mano_params_data = np.load(mano_params_path, allow_pickle=True).item()
        sampled_frames = mano_params_data['sampled_frames']
        seq_len = len(sampled_frames)

        objaverse_out = self.obj_downloader.load_objects(
            uids=[oid],
            download_processes=1
        )
        objaverse_obj_path = objaverse_out[oid]
        graspxl_obj_path = os.path.join(self.args.graspxl_dir, f"object_dataset/{otype}/{oid}/{oid}.obj")
        
        for t in trange(seq_len):
            curr_dir = os.path.join(render_output_dir,f"sequence_{t:06d}")
            if not os.path.exists(curr_dir):
                continue
            info_path = os.path.join(curr_dir, "info.json")
            if os.path.exists(info_path):
                continue
            info = {
                'motion_path': motion_path, 
                'frame_id': str(sampled_frames[t]),
                'objaverse_obj_path': objaverse_obj_path,
                'graspxl_obj_path': graspxl_obj_path,
                'mano_params': {
                    'global_orient': mano_params_data['global_orient'][t].tolist(),  # 3D rotation
                    'hand_pose': mano_params_data['hand_pose'][t].tolist(),  # 45D pose
                    'betas': mano_params_data['betas'][t].tolist(),  # 10D shape
                    'transl': mano_params_data['transl'][t].tolist()  # 3D translation
                }
            }
            with open(info_path, "w") as f:
                json.dump(info, f, indent=2)
        
    def load_obj(self, motion_path: str, output_dir:str = None) -> trimesh.Trimesh:
        oid = motion_path.split("/")[-2]
        otype = motion_path.split("/")[-3]
        # print(f"Loading object {oid} of type {otype} from {motion_path}")
        out = self.obj_downloader.load_objects(
            uids=[oid],
            download_processes=1
        )
        glb_path = out[oid]
        obj_path = os.path.join(self.args.graspxl_dir, f"object_dataset/{otype}/{oid}/{oid}.obj")
        
        objaverse_obj = trimesh.load_mesh(glb_path, force='mesh')
        graspxl_obj = trimesh.load_mesh(obj_path, force='mesh')
        
        objaverse_size = mesh_utils.get_bbox_size(objaverse_obj)
        graspxl_size = mesh_utils.get_bbox_size(graspxl_obj)
        
        if objaverse_size > 1e-6:
            scale = graspxl_size / objaverse_size
            objaverse_obj.apply_scale(scale)

        objaverse_obj.apply_translation(-objaverse_obj.bounds.mean(axis=0))
        
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            objaverse_obj.export(os.path.join(output_dir,f"{oid}_texture.obj"))
            graspxl_obj.export(os.path.join(output_dir,f"{oid}.obj"))
        
        return objaverse_obj

